"""
Pipeline status panel — right column showing which agent is currently working,
elapsed timer per agent, plus a dedicated 5-bar "Energy" readout at the bottom
that tracks Coordinator delegation count + 4 specialist budgets in real time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st

# (key, label, color) — color is used both for "active" highlight and bar tint.
AGENTS = [
    ("planner",                   "🧠 Planner",        "blue"),
    ("plan_critic",               "🔍 Plan Critic",    "red"),
    ("coordinator",               "🎯 Coordinator",    "orange"),
    ("specialist.reconstruction", "📥 Reconstruction", "violet"),
    ("specialist.segmentation",   "✂️ Segmentation",   "rainbow"),
    ("specialist.verifier",       "🛡 Verifier",       "green"),
    ("specialist.hemodynamic",    "💗 Hemodynamic",    "blue"),
    ("summarizer",                "📋 Summarizer",     "orange"),
]

SPECIALIST_KEYS = [k for k, _, _ in AGENTS if k.startswith("specialist.")]


# Default specialist budget — shown in the UI BEFORE the first run so the
# bar is never "?/?". Real values come back via audit-log budget snapshots.
DEFAULT_MAX_ROUNDS = 8


@dataclass
class AgentState:
    status:       str = "idle"          # idle | active | done | exhausted | error
    elapsed_s:    float = 0.0
    started_at:   Optional[float] = None
    rounds_used:  int = 0                # total LLM rounds spent
    max_rounds:   int = 0                # 0 = unknown until first audit entry
    tool_used:    int = 0                # subset of rounds that called a tool


def _init_agent_dict() -> dict[str, AgentState]:
    """Specialists get the default round budget displayed."""
    out: dict[str, AgentState] = {}
    for key, _, _ in AGENTS:
        a = AgentState()
        if key.startswith("specialist."):
            a.max_rounds = DEFAULT_MAX_ROUNDS
        out[key] = a
    return out


@dataclass
class PipelineState:
    agents:           dict[str, AgentState] = field(default_factory=_init_agent_dict)
    run_started_at:   Optional[float] = None
    current_agent:    Optional[str] = None
    # Coordinator delegation counter (separate from agent rounds)
    n_delegations:    int = 0
    max_delegations:  int = 12

    def total_elapsed(self) -> float:
        if self.run_started_at is None:
            return 0.0
        return time.time() - self.run_started_at

    def start_run(self):
        self.run_started_at = time.time()

    def mark_active(self, key: str):
        if key not in self.agents:
            return
        if self.current_agent and self.current_agent != key:
            self._finalize(self.current_agent)
        self.current_agent = key
        a = self.agents[key]
        if a.status != "active":
            a.status     = "active"
            a.started_at = time.time()

    def update_budget(self, key: str, *, rounds_used, max_rounds, tool_used):
        if key not in self.agents:
            return
        a = self.agents[key]
        a.rounds_used = rounds_used
        a.max_rounds  = max_rounds
        a.tool_used   = tool_used

    def increment_delegations(self):
        self.n_delegations += 1

    def reset_specialists_for_new_delegation(self):
        """
        At the start of every new Coordinator delegation, wipe the
        per-specialist counters so the bars show ONLY the current
        delegation's progress, not accumulated state from prior delegations.

        Keeps `max_rounds` intact; resets used counters, status, and per-agent
        elapsed timer. The top-level total elapsed in PipelineState continues
        to accumulate as normal.
        """
        for key, a in self.agents.items():
            if not key.startswith("specialist."):
                continue
            a.rounds_used = 0
            a.tool_used   = 0
            a.elapsed_s   = 0.0
            a.started_at  = None
            a.status      = "idle"
        # Also drop any stale "current_agent" pointer if it was a specialist
        if self.current_agent and self.current_agent.startswith("specialist."):
            self.current_agent = None

    def mark_done(self, key: str, *, exhausted: bool = False):
        if key not in self.agents:
            return
        # NEW: finishing an invocation reverts the agent to idle so the row
        # doesn't look "permanently complete" — it might be called again.
        # Only `exhausted` is sticky (visible as 🪫 until the run ends).
        self._finalize(key, status=("exhausted" if exhausted else "idle"))
        if self.current_agent == key:
            self.current_agent = None

    def _finalize(self, key: str, status: str = "idle"):
        a = self.agents[key]
        if a.started_at is not None:
            a.elapsed_s += time.time() - a.started_at
            a.started_at = None
        a.status = status


# ─────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────

STATUS_ICON = {
    "idle":      "💤",
    "active":    "🔆",   # currently running this round
    "done":      "💤",   # treated as idle now (kept for backward compat)
    "exhausted": "🪫",   # sticky red-battery once usage capped
    "error":     "❌",
}


def _battery_icon(pct: float) -> str:
    """Battery emoji that tracks remaining %."""
    if pct >= 75: return "🔋"
    if pct >= 25: return "🔋"
    if pct >  0:  return "🪫"
    return "⚠️"


def _overall_battery_pct(state: PipelineState) -> float:
    """
    Overall pipeline 'energy left' as a percentage.
    Formula (per user spec): (total_possible - used) / total_possible
      total_possible = max_delegations  (Coordinator's full budget)
      used           = current n_delegations
    """
    total = max(state.max_delegations, 1)
    used  = min(state.n_delegations, total)
    return 100.0 * (total - used) / total


def render_pipeline(state: PipelineState, placeholder) -> None:
    with placeholder.container():
        # ── Header with overall battery % ──────────────────────────
        battery_pct = _overall_battery_pct(state)
        icon        = _battery_icon(battery_pct)
        col_t, col_b = st.columns([3, 2])
        col_t.markdown("### 🔄 Pipeline")
        col_b.markdown(f"### {icon} **{battery_pct:.0f}%**")
        st.caption(f"⏱ Total: **{state.total_elapsed():.1f} s** · "
                   f"🎯 deleg {state.n_delegations}/{state.max_delegations}")
        st.divider()

        # ── Agent rows — energy data INLINE with each agent ────────
        for key, label, color in AGENTS:
            a = state.agents.get(key)
            if a is None:
                a = AgentState()
            _render_agent_row(state, key, a, label, color)


def _agent_energy_text(state: PipelineState, key: str, a: AgentState) -> str:
    """Inline energy summary for one agent (rounds + tool calls, or delegations)."""
    if key == "coordinator":
        return (f"🎯 {state.n_delegations}/{state.max_delegations} deleg")
    if key.startswith("specialist."):
        if a.max_rounds:
            return f"⚡ {a.rounds_used}/{a.max_rounds} · 🔧 {a.tool_used}"
        return ""
    # planner / plan_critic / summarizer — no budget concept, just a marker
    return ""


def _render_agent_row(state: PipelineState, key: str,
                       a: AgentState, label: str, color: str) -> None:
    icon    = STATUS_ICON.get(a.status, "•")
    elapsed = a.elapsed_s + (time.time() - a.started_at if a.started_at else 0)
    energy  = _agent_energy_text(state, key, a)

    if a.status == "active":
        # Bordered "glow" highlight; RUNNING text is always green for clarity.
        with st.container(border=True):
            top = st.columns([1, 4])
            top[0].markdown(f"### {icon}")
            top[1].markdown(f"**{label}**  :green[**RUNNING**]")
            sub_bits = [f"⏱ {elapsed:.1f}s"]
            if energy:
                sub_bits.append(energy)
            st.caption("  ·  ".join(sub_bits))
    else:
        cols = st.columns([1, 4, 3])
        cols[0].markdown(f"### {icon}")
        cols[1].markdown(f"**{label}**")
        # right column: timer (top) + energy text (below)
        right_bits = []
        if elapsed > 0:
            right_bits.append(f"{elapsed:.1f}s")
        if energy:
            right_bits.append(energy)
        cols[2].caption("  ·  ".join(right_bits) if right_bits else "")
