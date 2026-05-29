"""
Final-report-per-agent panel.

For each agent (Planner, Plan Critic, Coordinator, 4 Specialists),
extract its final reasoning + final structured output from the audit log
and display it in an expander so the user can read what each agent
contributed independently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import streamlit as st

AGENT_DISPLAY = [
    ("planner",                   "🧠 Planner",        "Initial pipeline plan"),
    ("plan_critic",               "🔍 Plan Critic",    "Plan review verdict"),
    ("coordinator",               "🎯 Coordinator",    "Final delegation summary"),
    ("specialist.reconstruction", "📥 Reconstruction", "Recon-loading report"),
    ("specialist.segmentation",   "✂️ Segmentation",   "Vessel mask production report"),
    ("specialist.verifier",       "🛡 Verifier",        "Physics-check verdict + interpretation"),
    ("specialist.hemodynamic",    "💗 Hemodynamic",    "Flow metrics + clinical context"),
]


# ─────────────────────────────────────────────────────────────────────
# Audit-log extraction
# ─────────────────────────────────────────────────────────────────────

def _scan_log(path: Path) -> dict[str, dict]:
    """For each purpose, return its last llm_call and last end-event payload."""
    last_llm:     dict[str, dict] = {}
    last_end:     dict[str, dict] = {}
    if not path.exists():
        return {"llm": last_llm, "end": last_end}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = e.get("kind")
            data = e.get("data", {})
            if kind == "llm_call":
                p = data.get("purpose")
                if p:
                    last_llm[p] = data
            elif kind == "event":
                name = data.get("name", "")
                if name.startswith("specialist.") and name.endswith(".end"):
                    spec = name[: -len(".end")]
                    last_end[spec] = data.get("data", {})
                elif name == "coordinator_done":
                    last_end["coordinator"] = data.get("data", {})
                elif name == "max_delegations":
                    last_end.setdefault("coordinator", {})["max_delegations_hit"] = True
    return {"llm": last_llm, "end": last_end}


# ─────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────

def _try_pretty_json(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return text


def _extract_report_field(text: Optional[str]) -> Optional[str]:
    """If the LLM response has a 'report' or 'summary' field, return it."""
    if not text:
        return None
    try:
        payload = json.loads(text)
        for key in ("report", "summary"):
            if key in payload and payload[key]:
                return str(payload[key])
    except json.JSONDecodeError:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────

def render_agent_reports(audit_path: Path) -> None:
    st.subheader("Final report by agent")
    st.caption(
        "Each section shows the most recent reasoning + final structured output "
        "produced by that agent during the last run."
    )

    scan = _scan_log(audit_path)
    if not scan["llm"]:
        st.info("No agent activity recorded in this log.")
        return

    for key, label, blurb in AGENT_DISPLAY:
        llm_call = scan["llm"].get(key)
        end_evt  = scan["end"].get(key)
        if llm_call is None and end_evt is None:
            continue

        with st.expander(f"{label}  —  {blurb}", expanded=False):
            response = (llm_call or {}).get("response", {}) or {}

            # 1) Human-readable summary (extracted from the JSON 'report' field)
            report_str = _extract_report_field(response.get("text"))
            if report_str:
                st.markdown("**Final report:**")
                st.markdown(report_str)
                st.divider()

            # 2) Chain-of-thought (Qwen's `thinking` field) if present
            reasoning = response.get("reasoning")
            if reasoning and reasoning.strip():
                with st.expander("Show reasoning (chain-of-thought)", expanded=False):
                    st.code(reasoning, language="markdown")

            # 3) Raw JSON output
            with st.expander("Raw JSON output", expanded=False):
                st.code(_try_pretty_json(response.get("text")) or "(no text)",
                        language="json")

            # 4) End-event metadata (budget snapshot, etc.)
            if end_evt:
                with st.expander("End-event metadata", expanded=False):
                    st.json(end_evt)

            # 5) Per-agent stats
            cols = st.columns(4)
            cols[0].metric("Latency", f"{response.get('latency_ms', 0)} ms")
            cols[1].metric("Model",   response.get("model", "?"))
            cols[2].metric("Prompt tokens",
                           response.get("prompt_tokens") or "—")
            cols[3].metric("Completion tokens",
                           response.get("completion_tokens") or "—")
