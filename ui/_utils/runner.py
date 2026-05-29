"""
Demo runner — wraps existing demos/*.py via subprocess, streams output to UI,
and tails the audit log in parallel to drive a live pipeline-status panel.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from ui._widgets.pipeline_panel import PipelineState, render_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────
# Config + result types
# ─────────────────────────────────────────────────────────────────────

class InputType(str, Enum):
    REAL_SCAN = "real_scan"
    PHANTOM   = "phantom"


@dataclass
class RunConfig:
    input_type:         InputType
    scan_path:          Optional[str] = None
    venc_m_per_s:       float = 1.5
    voxel_size_mm:      float = 2.0
    llm_backend:        str   = "mock"
    llm_model:          str   = "qwen3.6"
    max_plan_revisions: int   = 0
    max_delegations:    int   = 12
    custom_goal:        Optional[str] = None


@dataclass
class RunResult:
    config:         RunConfig
    exit_code:      int
    elapsed_s:      float
    audit_path:     Path
    stdout_tail:    str
    session_status: str = "unknown"
    n_delegations:  int = 0
    verdicts:       dict = field(default_factory=dict)
    analyses:       dict = field(default_factory=dict)
    pipeline_state: Optional[PipelineState] = None
    aborted:        bool = False


# ─────────────────────────────────────────────────────────────────────
# Goal templates + command construction
# ─────────────────────────────────────────────────────────────────────

REAL_SCAN_GOAL_TEMPLATE = (
    "Analyze the hemodynamics of the 4D flow MRI scan at {path}. "
    "VENC = {venc} m/s, voxel size {vox} mm isotropic. "
    "Pick a vessel, verify the physics, and report flow metrics. "
    "If verification fails, explain why and what should be done next."
)


def build_goal(cfg: RunConfig) -> str:
    if cfg.custom_goal:
        return cfg.custom_goal
    if cfg.input_type == InputType.REAL_SCAN:
        return REAL_SCAN_GOAL_TEMPLATE.format(
            path=cfg.scan_path, venc=cfg.venc_m_per_s, vox=cfg.voxel_size_mm,
        )
    return ""


def _build_command(cfg: RunConfig) -> tuple[list[str], Path]:
    """
    Both phantom and real-scan go through the SAME generic pipeline script
    (single_window_demo.py). The only thing that changes is the input source
    flag and the goal — the LLM is always live, never scripted.
    """
    python = sys.executable

    if cfg.input_type == InputType.PHANTOM:
        log_path = PROJECT_ROOT / "logs" / "ui_phantom.jsonl"
        argv = [
            python, "demos/single_window_demo.py",
            "--llm",   cfg.llm_backend,
            "--model", cfg.llm_model,
            "--phantom",
            "--max-plan-revisions", str(cfg.max_plan_revisions),
            "--max-delegations",    str(cfg.max_delegations),
            "--log-path",           str(log_path),
        ]
        if cfg.custom_goal:
            argv += ["--goal", cfg.custom_goal]
        return argv, log_path

    if cfg.input_type == InputType.REAL_SCAN:
        if not cfg.scan_path:
            raise ValueError("REAL_SCAN requires scan_path")
        log_path = PROJECT_ROOT / "logs" / f"ui_real_{Path(cfg.scan_path).stem}.jsonl"
        return [
            python, "demos/single_window_demo.py",
            "--llm",   cfg.llm_backend,
            "--model", cfg.llm_model,
            "--no-fresh-recon",
            "--max-plan-revisions", str(cfg.max_plan_revisions),
            "--max-delegations",    str(cfg.max_delegations),
            "--log-path",           str(log_path),
            "--goal",               build_goal(cfg),
        ], log_path

    raise ValueError(f"unsupported input_type {cfg.input_type}")


# ─────────────────────────────────────────────────────────────────────
# Stdout → PipelineState (fires BEFORE audit-log entries appear)
# Audit entries are written after each LLM call finishes — too late for
# live highlighting. The demo scripts print rich section headers (▶ NAME)
# and verbose callbacks ([name] step N) BEFORE each agent starts thinking;
# these are our earliest signals.
# ─────────────────────────────────────────────────────────────────────

_STDOUT_TRANSITIONS = [
    (re.compile(r"▶\s*planner\b",             re.I), "planner"),
    (re.compile(r"▶\s*plan\s*critic\b",       re.I), "plan_critic"),
    (re.compile(r"▶\s*coordinator\b",         re.I), "coordinator"),
    (re.compile(r"▶\s*summary\s*agent\b",     re.I), "summarizer"),
    (re.compile(r"\[reconstruction\]\s*step", re.I), "specialist.reconstruction"),
    (re.compile(r"\[segmentation\]\s*step",   re.I), "specialist.segmentation"),
    (re.compile(r"\[verifier\]\s*step",       re.I), "specialist.verifier"),
    (re.compile(r"\[hemodynamic\]\s*step",    re.I), "specialist.hemodynamic"),
    (re.compile(r"\[summarizer\]\s*thinking", re.I), "summarizer"),
    # When coordinator emits its delegation to a specialist, the next visible
    # thing in stdout is "delegating to X" — also treat as transition back to
    # coordinator briefly handing off.
    (re.compile(r"delegating to coordinator", re.I), "coordinator"),
]


def _apply_stdout_transitions(line: str, state: "PipelineState") -> None:
    for pattern, key in _STDOUT_TRANSITIONS:
        if pattern.search(line):
            state.mark_active(key)
            return


# ─────────────────────────────────────────────────────────────────────
# Audit-log → PipelineState dispatcher
# ─────────────────────────────────────────────────────────────────────

def _apply_audit_entry(state: PipelineState, entry: dict) -> None:
    kind = entry.get("kind")
    data = entry.get("data", {})

    if kind == "llm_call":
        purpose = data.get("purpose")
        if not purpose:
            return
        state.mark_active(purpose)
        opts = data.get("options", {}) or {}
        b    = opts.get("budget")
        if b:
            # Coordinator budget snapshot uses delegations_*, specialists use rounds_*
            if "delegations_used" in b:
                state.n_delegations   = b.get("delegations_used", state.n_delegations)
                state.max_delegations = b.get("max_delegations", state.max_delegations)
            else:
                state.update_budget(
                    purpose,
                    rounds_used=b.get("rounds_used", b.get("think_used", 0)),
                    max_rounds=b.get("max_rounds", b.get("think_max", 0)),
                    tool_used=b.get("tool_used", 0),
                )

    elif kind == "event":
        name = data.get("name", "")
        # specialist.X.end → mark that specialist done
        if name.startswith("specialist.") and name.endswith(".end"):
            spec_key = name[: -len(".end")]
            inner    = data.get("data", {})
            b        = inner.get("budget", {})
            state.update_budget(
                spec_key,
                rounds_used=b.get("rounds_used", b.get("think_used", 0)),
                max_rounds=b.get("max_rounds", b.get("think_max", 0)),
                tool_used=b.get("tool_used", 0),
            )
            state.mark_done(spec_key, exhausted=b.get("exhausted", False))
        # Coordinator delegation count — emitted by agents/coordinator.py as
        # `delegation` (see audit.event("delegation", {...}) call).
        # Each new delegation resets per-specialist counters so bars show
        # only the current delegation's progress, not prior ones.
        if name == "delegation":
            state.increment_delegations()
            state.reset_specialists_for_new_delegation()

    elif kind == "session_end":
        # Finalize any agent still marked active
        if state.current_agent:
            state.mark_done(state.current_agent)


def _tail_audit_log(log_path: Path, state: PipelineState, processed: set[int]) -> None:
    """Read all entries in log, apply ones not yet processed (by seq number)."""
    if not log_path.exists():
        return
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seq = entry.get("seq")
                if seq is None or seq in processed:
                    continue
                processed.add(seq)
                _apply_audit_entry(state, entry)
    except (FileNotFoundError, PermissionError):
        pass


# ─────────────────────────────────────────────────────────────────────
# Audit log → structured result (post-run)
# ─────────────────────────────────────────────────────────────────────

def _parse_audit_log(path: Path) -> dict:
    out = {"session_status": "unknown", "n_delegations": 0,
           "verdicts": {}, "analyses": {}}
    if not path.exists():
        return out
    last_sid = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = entry.get("session_id")
            if sid != last_sid:
                last_sid = sid
                out["verdicts"], out["analyses"] = {}, {}
            kind, data = entry.get("kind"), entry.get("data", {})
            if kind == "tool_call":
                name   = data.get("name")
                result = data.get("result", {})
                if name == "verify" and "mask_name" in result:
                    out["verdicts"][result["mask_name"]] = result
                elif name == "analyze" and "mask_name" in result:
                    out["analyses"][result["mask_name"]] = result
            elif kind == "session_end":
                out["session_status"] = data.get("status", "unknown")
                out["n_delegations"]  = data.get("summary", {}).get("n_delegations", 0)
    return out


# ─────────────────────────────────────────────────────────────────────
# Public entry point — with live pipeline panel
# ─────────────────────────────────────────────────────────────────────

def run_demo(cfg: RunConfig, *,
             pipeline_placeholder=None,
             stdout_placeholder=None) -> RunResult:
    argv, log_path = _build_command(cfg)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear stale log so live tail starts fresh
    if log_path.exists():
        log_path.unlink()

    # Default placeholders if caller didn't provide any
    if stdout_placeholder is None:
        st.subheader("Pipeline output")
        st.caption("Live stdout (last 60 lines)")
        stdout_placeholder = st.empty()

    state = PipelineState()
    state.max_delegations = cfg.max_delegations
    state.start_run()
    if pipeline_placeholder is not None:
        render_pipeline(state, pipeline_placeholder)

    proc = subprocess.Popen(
        argv, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    # Expose the running PID to session_state so the Stop button can find it
    # even on a fresh script rerun (Streamlit reruns top-to-bottom on each event).
    st.session_state["_running_pid"] = proc.pid
    st.session_state["_running_log"] = str(log_path)
    st.session_state["_cancel_run"]  = False

    stdout_lines: list[str] = []
    stdout_lock              = threading.Lock()

    def _stdout_reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            with stdout_lock:
                stdout_lines.append(line.rstrip())

    reader = threading.Thread(target=_stdout_reader, daemon=True)
    reader.start()

    processed_seqs:        set[int] = set()
    stdout_parse_offset:   int      = 0
    t0      = time.time()
    aborted = False

    while proc.poll() is None:
        # Cooperative cancel check
        if st.session_state.get("_cancel_run"):
            aborted = True
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            break

        # Parse new stdout lines for early agent-transition signals
        with stdout_lock:
            new_lines = stdout_lines[stdout_parse_offset:]
            stdout_parse_offset = len(stdout_lines)
            tail = "\n".join(stdout_lines[-60:])
        for line in new_lines:
            _apply_stdout_transitions(line, state)

        _tail_audit_log(log_path, state, processed_seqs)
        stdout_placeholder.code(tail or "(starting…)", language="text")
        if pipeline_placeholder is not None:
            render_pipeline(state, pipeline_placeholder)
        time.sleep(0.4)

    reader.join(timeout=3.0)
    # Drain whatever is in the log even after abort
    if not aborted:
        _tail_audit_log(log_path, state, processed_seqs)
    with stdout_lock:
        final_tail = "\n".join(stdout_lines[-60:])
    stdout_placeholder.code(final_tail or "(no output)", language="text")
    if pipeline_placeholder is not None:
        render_pipeline(state, pipeline_placeholder)

    # On abort: discard the partial log to keep the archive clean
    if aborted:
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass

    # Clean up session-state flags
    for key in ("_running_pid", "_running_log", "_cancel_run"):
        st.session_state.pop(key, None)

    elapsed = time.time() - t0
    parsed  = _parse_audit_log(log_path) if not aborted else {
        "session_status": "aborted", "n_delegations": 0,
        "verdicts": {}, "analyses": {},
    }
    return RunResult(
        config=cfg, exit_code=proc.returncode, elapsed_s=elapsed,
        audit_path=log_path, stdout_tail=final_tail,
        session_status=parsed["session_status"],
        n_delegations=parsed["n_delegations"],
        verdicts=parsed["verdicts"], analyses=parsed["analyses"],
        pipeline_state=state, aborted=aborted,
    )


def kill_orphan_run() -> bool:
    """
    Kill any subprocess whose PID is recorded in session_state, and discard
    its log file. Returns True if anything was killed/cleaned.

    Use this when the browser closed mid-run and you want to reset state.
    """
    pid = st.session_state.get("_running_pid")
    log = st.session_state.get("_running_log")
    killed = False
    if pid:
        try:
            import os, signal
            os.kill(pid, signal.SIGTERM)
            killed = True
        except (OSError, ProcessLookupError):
            pass
    if log:
        try:
            Path(log).unlink(missing_ok=True)
        except OSError:
            pass
    for key in ("_running_pid", "_running_log", "_cancel_run"):
        st.session_state.pop(key, None)
    return killed
