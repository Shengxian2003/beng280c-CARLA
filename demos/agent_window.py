"""One-agent viewer window for the MEDICT multi-agent demo.

Each agent role gets its own terminal window. You open as many windows as
agents (4 by default), run this script in each with a different agent name,
and they all tail the same audit log file — filtering for the entries that
belong to that agent and rendering them in a role-specific style.

The audit log is the single source of truth. Windows are pure read-only
viewers; arrange them on your desktop however you like.

Usage in each window:
    python demos/agent_window.py planner      logs/session.jsonl
    python demos/agent_window.py coordinator  logs/session.jsonl
    python demos/agent_window.py verifier     logs/session.jsonl
    python demos/agent_window.py hemodynamic  logs/session.jsonl

The viewer waits for the file to appear, tails it as the agent system
writes new entries, and rerenders. Stop with Ctrl-C.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


# ============================================================================
# Per-agent routing: which audit entries belong in which window
# ============================================================================

def _is_specialist(name: str):
    """Route entries belonging to one specialist: its LLM calls + the tool calls
    it made (we can't easily tell which specialist invoked which tool, so for
    visualization we show every tool call whose name is in that specialist's
    allowed toolset)."""
    tool_allow = SPECIALIST_TOOLS.get(name, set())
    purpose_tag = f"specialist.{name}"
    return lambda e: (
        e["kind"] in ("session_start", "session_end")
        or (e["kind"] == "llm_call" and e["data"].get("purpose") == purpose_tag)
        or (e["kind"] == "tool_call" and e["data"].get("name") in tool_allow)
    )


# Which tools each specialist is allowed to call — used for window routing.
# Must match agents/specialist.py:build_default_specialists.
SPECIALIST_TOOLS: dict[str, set[str]] = {
    "reconstruction": {"load_reconstruction", "reconstruct"},
    "segmentation":   {"suggest_seeds", "segment_from_seed"},
    "verifier":       {"verify"},
    "hemodynamic":    {"analyze"},
}


AGENT_ROUTES: dict[str, Callable[[dict], bool]] = {
    "planner": lambda e: (
        e["kind"] == "session_start"
        or e["kind"] == "session_end"
        or (e["kind"] == "llm_call" and e["data"].get("purpose") == "planner")
        or (e["kind"] == "event"
            and e["data"].get("name") in ("planner_parse_failure", "plan_revision"))
    ),
    "plan_critic": lambda e: (
        e["kind"] in ("session_start", "session_end")
        or (e["kind"] == "llm_call" and e["data"].get("purpose") == "plan_critic")
        or (e["kind"] == "event"
            and e["data"].get("name") in ("plan_critic_parse_failure",
                                            "plan_policy_decision"))
    ),
    "coordinator": lambda e: (
        e["kind"] in ("session_start", "session_end")
        or (e["kind"] == "llm_call" and e["data"].get("purpose") == "coordinator")
        or (e["kind"] == "event"
            and e["data"].get("name") in ("delegation", "coordinator_done",
                                            "max_delegations", "bad_delegation",
                                            "orchestrator_error"))
    ),
    "reconstruction": _is_specialist("reconstruction"),
    "segmentation":   _is_specialist("segmentation"),
    "verifier":       _is_specialist("verifier"),
    "hemodynamic":    _is_specialist("hemodynamic"),
}

# Visual identity for each agent window — title bar + accent colour
AGENT_STYLE: dict[str, dict] = {
    "planner":        {"title": "PLANNER",                  "color": "cyan",
                       "subtitle": "proposes the initial plan from the user goal"},
    "plan_critic":    {"title": "PLAN CRITIC (LLM Auditor)", "color": "bright_red",
                       "subtitle": "best-effort review of the plan before execution"},
    "coordinator":    {"title": "COORDINATOR",              "color": "yellow",
                       "subtitle": "delegates the approved plan to specialists"},
    "reconstruction": {"title": "RECONSTRUCTION OPERATOR",  "color": "magenta",
                       "subtitle": "picks recon params; loads or runs MATLAB recon"},
    "segmentation":   {"title": "SEGMENTATION OPERATOR",    "color": "bright_cyan",
                       "subtitle": "picks seeds and percentile; isolates vessels"},
    "verifier":       {"title": "PHYSICS VERIFIER",         "color": "green",
                       "subtitle": "DETERMINISTIC — divergence, flux, peak velocity, phase wrap"},
    "hemodynamic":    {"title": "HEMODYNAMIC ANALYZER",     "color": "blue",
                       "subtitle": "interprets Q(t), stroke volume, peak flow physiologically"},
}


# ============================================================================
# Rendering
# ============================================================================

def _shorten(text, n=140):
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def _t(entry):
    return entry["t"].split("T")[1][:8]


def _render_session_start(console, agent, entry):
    meta = entry["data"].get("metadata", {})
    body = Text()
    body.append(f"Session started  ", style="bold white")
    body.append(f"({entry['session_id'][:8]})", style="dim")
    body.append("\n")
    for k, v in meta.items():
        body.append(f"  {k}: ", style="dim")
        body.append(f"{v}\n", style="white")
    style = AGENT_STYLE[agent]
    console.print(Panel(body, border_style="dim", title=f"[dim]{_t(entry)}[/]"))


def _render_session_end(console, agent, entry):
    d = entry["data"]
    body = Text()
    body.append(f"Session ended  ", style="bold white")
    body.append(f"status={d.get('status')}  elapsed={d.get('elapsed_s')}s",
                style="green" if d.get("status") == "success" else "red")
    body.append("\n")
    summary = d.get("summary") or {}
    for k, v in summary.items():
        body.append(f"  {k}: ", style="dim")
        body.append(f"{v}\n", style="white")
    console.print(Panel(body, border_style="dim", title=f"[dim]{_t(entry)}[/]"))


def _render_llm_planner(console, entry):
    d = entry["data"]
    resp = d.get("response", {})
    text = resp.get("text") or ""
    reasoning = resp.get("reasoning")
    body = Text()
    # Try to parse as a structured plan first
    try:
        payload = json.loads(text)
        steps = payload.get("plan") or []
        if steps:
            body.append("Plan emitted:\n", style="bold cyan")
            for i, step in enumerate(steps, 1):
                body.append(f"  {i}. ", style="bold cyan")
                body.append(f"{_shorten(str(step), 120)}\n", style="white")
        else:
            body.append(_shorten(text, 400), style="white")
    except json.JSONDecodeError:
        body.append(_shorten(text, 400), style="white")
    if reasoning:
        body.append("\nThinking: ", style="dim italic cyan")
        body.append(_shorten(reasoning, 300), style="dim italic")
    _meta_footer(body, resp)
    console.print(Panel(body, border_style="cyan",
                        title=f"[bold cyan]planner reply[/]  [dim]{_t(entry)}[/]"))


def _render_llm_coordinator(console, entry):
    d = entry["data"]
    resp = d.get("response", {})
    text = resp.get("text") or ""
    reasoning = resp.get("reasoning")
    body = Text()
    try:
        payload = json.loads(text)
        if payload.get("done"):
            body.append("Coordinator: DONE — exit loop\n", style="bold green")
            if "summary" in payload:
                body.append(_shorten(payload["summary"], 300), style="dim")
        else:
            tool = payload.get("tool", "?")
            args = json.dumps(payload.get("args", {}), separators=(",", ":"))
            body.append("Calling: ", style="dim")
            body.append(f"{tool}", style="bold yellow")
            body.append("(", style="dim")
            body.append(_shorten(args, 200), style="white")
            body.append(")", style="dim")
    except json.JSONDecodeError:
        body.append(_shorten(text, 400), style="white")
    if reasoning:
        body.append("\n\nThinking: ", style="dim italic yellow")
        body.append(_shorten(reasoning, 400), style="dim italic")
    _meta_footer(body, resp)
    console.print(Panel(body, border_style="yellow",
                        title=f"[bold yellow]coordinator decision[/]  [dim]{_t(entry)}[/]"))


def _render_specialist_llm(console, agent, entry):
    """Render one LLM reply from a specialist — either a tool call or a final report.

    The specialist replies in JSON; we parse it to show the structure visibly
    (why-rationale, tool name + args, or the final natural-language report).
    """
    d = entry["data"]
    resp = d.get("response", {})
    text = resp.get("text") or ""
    reasoning = resp.get("reasoning")
    style = AGENT_STYLE.get(agent, {})
    color = style.get("color", "white")
    body = Text()

    try:
        action = json.loads(text)
        if action.get("done"):
            body.append("Specialist report:\n", style=f"bold {color}")
            body.append(_shorten(action.get("report", ""), 800), style="white")
        else:
            tool = action.get("tool", "?")
            args = json.dumps(action.get("args", {}), separators=(",", ":"))
            why = action.get("why", "")
            body.append("Calling: ", style="dim")
            body.append(f"{tool}", style=f"bold {color}")
            body.append("(", style="dim")
            body.append(_shorten(args, 200), style="white")
            body.append(")\n", style="dim")
            if why:
                body.append("Why: ", style="dim italic")
                body.append(_shorten(why, 240), style="italic")
    except json.JSONDecodeError:
        body.append(_shorten(text, 600), style="white")

    if reasoning:
        body.append("\n\nThinking: ", style=f"dim italic {color}")
        body.append(_shorten(reasoning, 400), style="dim italic")
    _meta_footer(body, resp)
    console.print(Panel(body, border_style=color,
                        title=f"[bold {color}]{agent} reply[/]  [dim]{_t(entry)}[/]"))


def _render_specialist_tool_result(console, agent, entry):
    """Specialist windows show their own tool-call results plainly."""
    d = entry["data"]
    name = d.get("name", "?")
    latency = d.get("latency_ms", "—")
    style = AGENT_STYLE.get(agent, {})
    color = style.get("color", "white")
    body = Text()
    if d.get("is_error"):
        body.append(f"✗ {name} failed: ", style="bold red")
        body.append(_shorten(d["result"].get("error", ""), 300), style="red")
    else:
        body.append(f"✓ {name} returned\n", style="bold green")
        body.append(_short_result(name, d.get("result", {})), style="white")
    body.append(f"\n\n{latency} ms", style="dim")
    console.print(Panel(body, border_style=color,
                        title=f"[dim {color}]tool result for {agent}[/]  [dim]{_t(entry)}[/]"))


def _render_tool_for_coordinator(console, entry):
    """Coordinator window sees its own tool-call results as feedback."""
    d = entry["data"]
    name = d.get("name", "?")
    latency = d.get("latency_ms", "—")
    body = Text()
    if d.get("is_error"):
        err = d["result"].get("error", "")
        body.append(f"✗ {name} failed: ", style="bold red")
        body.append(_shorten(err, 300), style="red")
    else:
        body.append(f"✓ {name} returned\n", style="bold green")
        body.append(_short_result(name, d.get("result", {})), style="white")
    body.append(f"\n\n{latency} ms", style="dim")
    console.print(Panel(body, border_style="yellow",
                        title=f"[dim yellow]tool feedback[/]  [dim]{_t(entry)}[/]"))


def _render_verifier(console, entry):
    d = entry["data"]
    result = d.get("result", {})
    body = Text()
    if d.get("is_error"):
        body.append(f"✗ verifier could not run: {result.get('error', '')}",
                    style="bold red")
        console.print(Panel(body, border_style="red",
                            title=f"[bold red]verifier error[/]  [dim]{_t(entry)}[/]"))
        return

    verdict = result.get("verdict", "?")
    mask = result.get("mask_name", "?")
    style = {"pass": "bold green", "warn": "bold yellow",
             "fail": "bold red"}.get(verdict, "bold white")

    body.append(f"Mask: ", style="dim")
    body.append(f"{mask}", style="bold white")
    body.append("    verdict: ", style="dim")
    body.append(f"{verdict.upper()}\n\n", style=style)

    for name, check in (result.get("checks") or {}).items():
        s = check.get("status", "?")
        symbol = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(s, "?")
        color = {"pass": "green", "warn": "yellow", "fail": "red"}.get(s, "white")
        body.append(f"  {symbol} {name:14s}  ", style=f"bold {color}")
        # Show the key numeric for each check if available
        for key in ("mean_abs_divergence_per_s", "max_deviation_pct",
                    "peak_m_per_s", "fraction_above_threshold"):
            if key in check:
                body.append(f"{key.replace('_', ' ')} = {check[key]}", style=color)
                break
        body.append("\n")

    latency = d.get("latency_ms", "—")
    body.append(f"\n{latency} ms", style="dim")
    console.print(Panel(body, border_style="green",
                        title=f"[bold green]verifier verdict[/]  [dim]{_t(entry)}[/]"))


def _render_hemodynamic(console, entry):
    d = entry["data"]
    result = d.get("result", {})
    body = Text()
    if d.get("is_error"):
        body.append(f"✗ analyzer failed: {result.get('error', '')}", style="bold red")
        console.print(Panel(body, border_style="red",
                            title=f"[bold red]analyzer error[/]  [dim]{_t(entry)}[/]"))
        return

    mask = result.get("mask_name", "?")
    summary = result.get("summary") or {}
    meta = result.get("metadata") or {}

    body.append(f"Mask: ", style="dim")
    body.append(f"{mask}", style="bold white")
    body.append(f"    {meta.get('n_cross_sections', '?')} cross-sections "
                f"along {meta.get('dominant_axis', '?')}\n\n", style="dim")

    body.append(f"  Stroke volume (forward):  ", style="dim")
    body.append(f"{summary.get('mean_stroke_volume_mL', '?'):>8} mL\n", style="bold white")
    body.append(f"  Peak flow rate:           ", style="dim")
    body.append(f"{summary.get('mean_peak_Q_mL_per_s', '?'):>8} mL/s\n", style="bold white")
    body.append(f"  Max peak flow:            ", style="dim")
    body.append(f"{summary.get('max_peak_Q_mL_per_s', '?'):>8} mL/s\n", style="white")
    body.append(f"  Peak velocity in vessel:  ", style="dim")
    body.append(f"{summary.get('peak_velocity_m_per_s', '?'):>8} m/s\n", style="bold white")
    body.append(f"  Mean velocity in vessel:  ", style="dim")
    body.append(f"{summary.get('mean_velocity_m_per_s', '?'):>8} m/s\n", style="white")

    # Per-section regurgitation in case any section flagged backflow
    secs = result.get("per_section") or []
    rfs = [s.get("regurgitant_fraction_pct", 0) for s in secs]
    if rfs and max(rfs) > 0:
        body.append(f"\n  Regurgitation per section: ", style="dim")
        body.append(", ".join(f"{r}%" for r in rfs), style="yellow")

    latency = d.get("latency_ms", "—")
    body.append(f"\n\n{latency} ms", style="dim")
    console.print(Panel(body, border_style="blue",
                        title=f"[bold blue]hemodynamic report[/]  [dim]{_t(entry)}[/]"))


def _render_event(console, entry):
    d = entry["data"]
    body = Text()
    body.append(f"event ", style="bold magenta")
    body.append(d.get("name", "?"), style="magenta")
    if d.get("data"):
        body.append(f"   {json.dumps(d.get('data'))}", style="dim")
    console.print(Panel(body, border_style="magenta",
                        title=f"[dim]{_t(entry)}[/]"))


def _render_plan_critic_llm(console, entry):
    """Render one Plan Critic LLM reply — the structured critique verdict."""
    d = entry["data"]
    resp = d.get("response", {})
    text = resp.get("text") or ""
    body = Text()
    try:
        payload = json.loads(text)
        verdict = payload.get("verdict", "?")
        concerns = payload.get("concerns", []) or []
        suggestions = payload.get("suggestions", "")
        verdict_color = {"approve": "green", "revise": "yellow",
                         "reject": "red"}.get(verdict, "white")
        body.append("Verdict: ", style="dim")
        body.append(f"{verdict.upper()}\n\n", style=f"bold {verdict_color}")
        if concerns:
            body.append("Concerns:\n", style="dim bold")
            for c in concerns:
                body.append(f"  • ", style="dim")
                body.append(_shorten(str(c), 240) + "\n", style="white")
        if suggestions:
            body.append("\nSuggestions:\n", style="dim bold")
            body.append(_shorten(suggestions, 400), style="italic")
    except json.JSONDecodeError:
        body.append("[parse failure — soft approved]\n", style="dim italic red")
        body.append(_shorten(text, 400), style="white")
    _meta_footer(body, resp)
    console.print(Panel(body, border_style="bright_red",
                        title=f"[bold bright_red]plan critique[/]  [dim]{_t(entry)}[/]"))


def _short_result(name, result):
    """Compact human summary of a tool result for the coordinator feedback panel."""
    if name == "load_reconstruction":
        return (f"  shape: {result.get('shape_ZYXT')}\n"
                f"  voxel size: {result.get('voxel_size_mm')} mm")
    if name == "reconstruct":
        return (f"  shape: {result.get('shape_ZYXT')}\n"
                f"  elapsed: {result.get('matlab_elapsed_minutes')} min")
    if name == "suggest_seeds":
        return f"  {result.get('n_returned')} vessel candidates"
    if name == "segment_from_seed":
        return (f"  mask '{result.get('mask_name')}'  "
                f"{result.get('size_voxels'):,} voxels  "
                f"peak {result.get('peak_speed_m_per_s')} m/s")
    if name == "verify":
        return f"  verdict: {result.get('verdict')} (full breakdown in verifier window)"
    if name == "analyze":
        return f"  hemodynamic report on '{result.get('mask_name')}' (full numbers in analyzer window)"
    return "  " + json.dumps(result, separators=(",", ":"))[:200]


def _meta_footer(body, resp):
    body.append("\n\n")
    body.append(f"{resp.get('latency_ms', '?')} ms", style="dim")
    if resp.get("prompt_tokens") is not None:
        body.append(f"  •  {resp['prompt_tokens']}/{resp.get('completion_tokens', '?')} tok",
                    style="dim")
    body.append(f"  •  {resp.get('model', '?')}", style="dim")


# ============================================================================
# Dispatch
# ============================================================================

SPECIALIST_AGENTS = {"reconstruction", "segmentation", "verifier", "hemodynamic"}


def render_entry(console: Console, agent: str, entry: dict):
    kind = entry["kind"]
    if kind == "session_start":
        _render_session_start(console, agent, entry); return
    if kind == "session_end":
        _render_session_end(console, agent, entry); return

    if agent == "planner":
        if kind == "llm_call":
            _render_llm_planner(console, entry); return
        if kind == "event":
            _render_event(console, entry); return

    if agent == "plan_critic":
        if kind == "llm_call":
            _render_plan_critic_llm(console, entry); return
        if kind == "event":
            _render_event(console, entry); return

    if agent == "coordinator":
        if kind == "llm_call":
            _render_llm_coordinator(console, entry); return
        if kind == "event":
            _render_event(console, entry); return

    if agent in SPECIALIST_AGENTS:
        if kind == "llm_call":
            _render_specialist_llm(console, agent, entry); return
        if kind == "tool_call":
            # For verifier and hemodynamic we still want the rich domain-specific
            # rendering (verdict panel, hemodynamic summary). Reconstruction +
            # segmentation get the generic specialist-tool-result panel.
            if agent == "verifier":
                _render_verifier(console, entry); return
            if agent == "hemodynamic":
                _render_hemodynamic(console, entry); return
            _render_specialist_tool_result(console, agent, entry); return


def render_header(console: Console, agent: str, log_path: Path):
    style = AGENT_STYLE[agent]
    header = Text()
    header.append(f" {style['title']} ", style=f"bold reverse {style['color']}")
    header.append("\n")
    header.append(style["subtitle"], style="dim")
    header.append(f"\nLog file: {log_path}", style="dim")
    console.print(header)
    console.print()


# ============================================================================
# Tail loop
# ============================================================================

def tail_log(agent: str, log_path: Path, *, follow: bool = True,
             poll_interval_s: float = 0.1):
    """Stream entries from log_path, render those matching the agent filter.

    If the file doesn't exist yet, wait for it. If ``follow`` is True, keep
    reading new lines as they're appended; otherwise exit at EOF.
    """
    console = Console()
    render_header(console, agent, log_path)
    filter_fn = AGENT_ROUTES[agent]

    # Wait for the file to appear
    while follow and not log_path.exists():
        time.sleep(poll_interval_s)

    if not log_path.exists():
        console.print(f"[red]Log file not found:[/red] {log_path}")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                if not follow:
                    return
                time.sleep(poll_interval_s)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line — wait for next poll
            if filter_fn(entry):
                render_entry(console, agent, entry)
                if entry["kind"] == "session_end":
                    # natural stopping point — but stay open so the user can read
                    if not follow:
                        return


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Single-agent viewer window — tails an audit log and shows entries for one agent role",
    )
    p.add_argument("agent", choices=list(AGENT_ROUTES),
                   help="Which agent's view to show")
    p.add_argument("log_path", help="Path to a Stage 3c audit JSONL file")
    p.add_argument("--no-follow", action="store_true",
                   help="Read existing entries and exit; don't tail")
    args = p.parse_args()

    try:
        tail_log(args.agent, Path(args.log_path), follow=not args.no_follow)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
