"""Multi-pane TUI replay of an agent audit log.

Built for the BENG 280C demo recording: takes a Stage 3c audit log and
re-plays it in a 3-pane rich layout, making the multi-agent collaboration
visually obvious even though the underlying LLM calls run sequentially on
one local GPU.

Layout:
    ┌──────────────────────────────────────────────────────────────┐
    │ MEDICT Agent Loop — Replay   session: <id>      t+12.3s      │
    ├──────────────────────┬───────────────────────────────────────┤
    │  PLANNER (cyan)      │  COORDINATOR (yellow)                 │
    │                      │                                        │
    │  initial plan        │  streaming tool decisions + reasoning │
    ├──────────────────────┴───────────────────────────────────────┤
    │  PHYSICS & HEMODYNAMIC ANALYSIS (green)                      │
    │  verify/analyze tool results as they land                    │
    ├──────────────────────────────────────────────────────────────┤
    │  AUDIT FOOTER  • call counters • token + latency totals      │
    └──────────────────────────────────────────────────────────────┘

Usage:
    python demos/multi_pane.py logs/demo_session.jsonl
    python demos/multi_pane.py logs/demo_session.jsonl --speed 2.0
    python demos/multi_pane.py logs/demo_session.jsonl --instant   # no delays
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

from agents.audit import read_log


# ============================================================================
# Pane state — what each agent panel is currently displaying
# ============================================================================

class PaneState:
    def __init__(self):
        self.planner_plan: list[str] = []
        self.coord_history: list[tuple[str, str]] = []  # list of (kind, text)
        self.analysis_history: list[tuple[str, str]] = []
        self.audit_history: list[str] = []
        self.session_id: str = "-"
        self.t_start: float = 0.0
        self.last_entry_time: float = 0.0

        # Stats
        self.n_llm_calls = 0
        self.n_tool_calls = 0
        self.n_tool_errors = 0
        self.total_llm_latency_ms = 0
        self.total_tool_latency_ms = 0
        self.tokens_in = 0
        self.tokens_out = 0


# ============================================================================
# Pane renderers
# ============================================================================

def _planner_panel(s: PaneState) -> Panel:
    if not s.planner_plan:
        body = Text("(planner has not yet emitted a plan)", style="dim italic")
    else:
        body = Text()
        for i, step in enumerate(s.planner_plan, 1):
            body.append(f" {i}. ", style="bold cyan")
            body.append(f"{step}\n", style="white")
    return Panel(body, title="[bold cyan]PLANNER[/]",
                 border_style="cyan", padding=(1, 2))


def _coord_panel(s: PaneState, max_items: int = 12) -> Panel:
    if not s.coord_history:
        body = Text("(waiting for coordinator…)", style="dim italic")
    else:
        items = s.coord_history[-max_items:]
        body = Text()
        for kind, text in items:
            if kind == "decision":
                body.append("▸ ", style="bold yellow")
                body.append(f"{text}\n", style="white")
            elif kind == "reasoning":
                body.append("   thinking: ", style="dim yellow")
                body.append(f"{text}\n", style="dim italic")
            elif kind == "result":
                body.append("   ", style="")
                body.append(f"{text}\n", style="green")
            elif kind == "error":
                body.append("   ", style="")
                body.append(f"{text}\n", style="red")
            elif kind == "event":
                body.append("• ", style="bold magenta")
                body.append(f"{text}\n", style="magenta")
    return Panel(body, title="[bold yellow]COORDINATOR[/]",
                 border_style="yellow", padding=(1, 2))


def _analysis_panel(s: PaneState, max_items: int = 8) -> Panel:
    if not s.analysis_history:
        body = Text("(no verifier or analyzer results yet)", style="dim italic")
    else:
        items = s.analysis_history[-max_items:]
        body = Text()
        for kind, text in items:
            if kind == "verify_pass":
                body.append("✓ ", style="bold green")
                body.append(f"{text}\n", style="green")
            elif kind == "verify_warn":
                body.append("⚠ ", style="bold yellow")
                body.append(f"{text}\n", style="yellow")
            elif kind == "verify_fail":
                body.append("✗ ", style="bold red")
                body.append(f"{text}\n", style="red")
            elif kind == "analyze":
                body.append("Σ ", style="bold blue")
                body.append(f"{text}\n", style="cyan")
            else:
                body.append(f"  {text}\n", style="white")
    return Panel(body, title="[bold green]PHYSICS & HEMODYNAMIC ANALYSIS[/]",
                 border_style="green", padding=(1, 2))


def _footer_panel(s: PaneState) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_column(justify="left")

    elapsed = s.last_entry_time
    err_style = "red bold" if s.n_tool_errors else "dim"
    table.add_row(
        Text.from_markup(f"[dim]t+[/]{elapsed:5.1f}s"),
        Text.from_markup(f"[dim]LLM calls[/]: [white]{s.n_llm_calls}[/]"),
        Text.from_markup(f"[dim]Tool calls[/]: [white]{s.n_tool_calls}[/]  "
                         f"[{err_style}]errors: {s.n_tool_errors}[/]"),
        Text.from_markup(f"[dim]Tokens[/]: [white]{s.tokens_in}[/] in / "
                         f"[white]{s.tokens_out}[/] out"),
        Text.from_markup(f"[dim]Latency[/]: "
                         f"LLM [white]{s.total_llm_latency_ms/1000:.1f}s[/]  "
                         f"tool [white]{s.total_tool_latency_ms/1000:.1f}s[/]"),
    )

    if s.audit_history:
        recent = Text()
        for line in s.audit_history[-3:]:
            recent.append(f"  {line}\n", style="dim")
        body = Group(table, Text(""), recent)
    else:
        body = table

    return Panel(body, title=f"[bold]AUDIT[/]  [dim]session {s.session_id[:8]}[/]",
                 border_style="white", padding=(1, 2))


def _build_layout(s: PaneState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=2),
        Layout(name="analysis", ratio=1),
        Layout(name="footer", size=8),
    )
    layout["top"].split_row(
        Layout(name="planner", ratio=1),
        Layout(name="coord", ratio=2),
    )
    layout["planner"].update(_planner_panel(s))
    layout["coord"].update(_coord_panel(s))
    layout["analysis"].update(_analysis_panel(s))
    layout["footer"].update(_footer_panel(s))
    return layout


# ============================================================================
# Entry → state-mutation
# ============================================================================

def _shorten(text: str, n: int = 100) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def _apply_entry(s: PaneState, entry: dict) -> None:
    """Update PaneState in place to reflect one audit-log entry."""
    kind = entry["kind"]
    data = entry["data"]

    # Maintain "elapsed-since-start" based on the log timestamps so the
    # footer clock matches the recorded timing rather than wall-clock.
    if s.t_start == 0.0:
        s.t_start = _parse_iso(entry["t"])
    s.last_entry_time = _parse_iso(entry["t"]) - s.t_start

    if kind == "session_start":
        s.session_id = entry["session_id"]
        meta = data.get("metadata", {})
        goal = meta.get("goal", "")
        s.audit_history.append(f"session_start  {goal}")
        return

    if kind == "llm_call":
        s.n_llm_calls += 1
        resp = data.get("response", {})
        s.total_llm_latency_ms += int(resp.get("latency_ms") or 0)
        s.tokens_in += int(resp.get("prompt_tokens") or 0)
        s.tokens_out += int(resp.get("completion_tokens") or 0)
        purpose = data.get("purpose")
        text = resp.get("text") or ""
        reasoning = resp.get("reasoning")

        if purpose == "planner":
            # Plan should be JSON {"plan": [...]}
            try:
                payload = json.loads(text)
                steps = payload.get("plan") or []
                s.planner_plan = [_shorten(str(x), 80) for x in steps]
            except json.JSONDecodeError:
                s.planner_plan = [_shorten(text, 120)]
            s.audit_history.append(f"planner emitted {len(s.planner_plan)}-step plan")
        else:
            # Coordinator decision — show the parsed action if possible
            try:
                payload = json.loads(text)
                if payload.get("done"):
                    s.coord_history.append(("decision", "DONE — exit loop"))
                else:
                    tool = payload.get("tool", "?")
                    args = json.dumps(payload.get("args", {}), separators=(",", ":"))
                    s.coord_history.append(
                        ("decision", f"call {tool}({_shorten(args, 100)})")
                    )
            except json.JSONDecodeError:
                s.coord_history.append(("decision", _shorten(text, 140)))

            if reasoning:
                s.coord_history.append(("reasoning", _shorten(reasoning, 200)))
        return

    if kind == "tool_call":
        s.n_tool_calls += 1
        s.total_tool_latency_ms += int(data.get("latency_ms") or 0)
        if data.get("is_error"):
            s.n_tool_errors += 1

        name = data.get("name", "?")
        result = data.get("result", {})
        latency = data.get("latency_ms", "—")
        s.audit_history.append(f"tool {name}({latency}ms)")

        # Side-effect display in the coordinator pane: every tool gets a result line
        if data.get("is_error"):
            err = result.get("error", "")
            s.coord_history.append(("error", f"✗ {name} → {_shorten(err, 100)}"))
        else:
            summary = _summarize_tool_result(name, result)
            s.coord_history.append(("result", f"← {summary}"))

        # Mirror verifier + analyzer results into the analysis pane
        if name == "verify":
            verdict = result.get("verdict", "?")
            checks = result.get("checks", {}) or {}
            line = f"{result.get('mask_name', '?')}: {verdict.upper()}  " + " ".join(
                f"{n}={c.get('status', '?')}" for n, c in checks.items()
            )
            tag = {"pass": "verify_pass", "warn": "verify_warn",
                   "fail": "verify_fail"}.get(verdict, "verify_warn")
            s.analysis_history.append((tag, line))
        elif name == "analyze":
            summary = result.get("summary", {}) or {}
            sv = summary.get("mean_stroke_volume_mL")
            pk = summary.get("peak_velocity_m_per_s")
            mq = summary.get("mean_peak_Q_mL_per_s")
            s.analysis_history.append((
                "analyze",
                f"{result.get('mask_name', '?')}: SV={sv}mL  peak={pk}m/s  Q_peak={mq}mL/s",
            ))
        return

    if kind == "event":
        s.audit_history.append(f"event {data.get('name')}")
        s.coord_history.append(("event", data.get("name", "?")))
        return

    if kind == "session_end":
        s.audit_history.append(
            f"session_end  status={data.get('status')}  elapsed={data.get('elapsed_s')}s"
        )
        return


def _parse_iso(ts: str) -> float:
    # Strip subsecond precision differences: just parse + convert to epoch seconds
    from datetime import datetime
    return datetime.fromisoformat(ts).timestamp()


def _summarize_tool_result(name: str, result: dict) -> str:
    """One-liner appropriate to each tool, for the coordinator pane."""
    if name == "load_reconstruction":
        return f"loaded {result.get('shape_ZYXT')} from {Path(result.get('source_path', '')).name}"
    if name == "reconstruct":
        return (f"recon ok  {result.get('shape_ZYXT')}  "
                f"{result.get('matlab_elapsed_minutes', '?')}min "
                f"→ {Path(result.get('output_path', '')).name}")
    if name == "suggest_seeds":
        return f"{result.get('n_returned')} candidates returned"
    if name == "segment_from_seed":
        return (f"mask '{result.get('mask_name')}' = {result.get('size_voxels'):,} vox, "
                f"peak {result.get('peak_speed_m_per_s')} m/s")
    if name == "verify":
        return f"{result.get('mask_name')} → verdict={result.get('verdict')}"
    if name == "analyze":
        sm = result.get("summary", {}) or {}
        return (f"{result.get('mask_name')} → SV={sm.get('mean_stroke_volume_mL')}mL  "
                f"peak={sm.get('peak_velocity_m_per_s')}m/s")
    return json.dumps({k: v for k, v in result.items() if k != "checks"},
                      separators=(",", ":"))[:120]


# ============================================================================
# Public: replay
# ============================================================================

def replay(
    log_path: str | Path,
    *,
    speed: float = 1.0,
    instant: bool = False,
    min_step_s: float = 0.4,
    max_step_s: float = 2.5,
) -> None:
    """Read the audit log and replay it in a live multi-pane TUI.

    Timing:
      - If ``instant`` is True, all entries appear immediately (good for scrubbing).
      - Otherwise, delay between consecutive entries = real gap in the log scaled
        by ``speed``, clamped to [min_step_s, max_step_s] so the pacing reads well
        on camera even when the LLM was fast or slow in the original run.
    """
    entries = read_log(log_path)
    if not entries:
        raise ValueError(f"no entries in {log_path}")

    state = PaneState()
    console = Console()

    with Live(_build_layout(state), console=console, refresh_per_second=15,
              screen=True) as live:
        prev_ts: float | None = None
        for entry in entries:
            if not instant:
                ts = _parse_iso(entry["t"])
                if prev_ts is not None:
                    gap = max(0.0, ts - prev_ts) / max(speed, 1e-6)
                    gap = min(max_step_s, max(min_step_s, gap))
                    time.sleep(gap)
                prev_ts = ts
            _apply_entry(state, entry)
            live.update(_build_layout(state))
        # Hold the final frame for a beat so viewers can read it
        time.sleep(1.5 if not instant else 0.0)


# ============================================================================
# Static render — print the final state once (no TUI, no alternate screen)
# ============================================================================

def render_final(log_path: str | Path, *, width: int = 120) -> str:
    """Apply every entry to a fresh PaneState and render the final layout
    to a string. Useful for snapshot tests and previewing layout without a terminal."""
    entries = read_log(log_path)
    state = PaneState()
    for e in entries:
        _apply_entry(state, e)
    console = Console(record=True, width=width, force_terminal=True)
    console.print(_build_layout(state))
    return console.export_text()


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Replay an agent audit log in a multi-pane TUI")
    p.add_argument("log_path", help="Path to a Stage 3c audit JSONL file")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Replay speed multiplier (>1 = faster)")
    p.add_argument("--instant", action="store_true",
                   help="Skip delays; jump to final state immediately")
    p.add_argument("--min-step", type=float, default=0.4,
                   help="Min delay between entries in seconds")
    p.add_argument("--max-step", type=float, default=2.5,
                   help="Max delay between entries in seconds")
    p.add_argument("--static", action="store_true",
                   help="Render the final frame to stdout once and exit (no TUI)")
    args = p.parse_args()

    if args.static:
        print(render_final(args.log_path))
        return

    replay(args.log_path, speed=args.speed, instant=args.instant,
           min_step_s=args.min_step, max_step_s=args.max_step)


if __name__ == "__main__":
    main()
