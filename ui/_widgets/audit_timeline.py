"""Audit-log timeline widget — chronological view of every llm_call, tool_call, event."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

KIND_ICON = {
    "session_start":    "🟢",
    "llm_call":         "🧠",
    "tool_call":        "🔧",
    "event":            "📌",
    "session_end":      "🏁",
}


def render_timeline(log_path: Path, max_entries: int = 200) -> None:
    if not log_path.exists():
        st.warning(f"Audit log not found: `{log_path}`")
        return

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        st.info("Audit log is empty.")
        return

    # Keep most recent session only
    last_sid = entries[-1].get("session_id")
    entries = [e for e in entries if e.get("session_id") == last_sid]

    st.caption(f"Showing {min(len(entries), max_entries)} of {len(entries)} entries "
               f"(session {last_sid[:8] if last_sid else '?'}...)")

    if len(entries) > max_entries:
        entries = entries[-max_entries:]

    for entry in entries:
        kind     = entry.get("kind", "?")
        ts       = entry.get("t", "")[11:19]   # HH:MM:SS
        data     = entry.get("data", {})
        icon     = KIND_ICON.get(kind, "•")
        summary  = _summarize(kind, data)
        with st.container():
            st.markdown(f"`{ts}` {icon} **{kind}** — {summary}")


def _summarize(kind: str, data: dict) -> str:
    if kind == "llm_call":
        purpose = data.get("purpose", "?")
        resp    = data.get("response", {})
        latency = resp.get("latency_ms")
        text    = (resp.get("text") or "").strip().replace("\n", " ")[:100]
        return f"`{purpose}` ({latency} ms) — {text}..."
    if kind == "tool_call":
        name      = data.get("name", "?")
        is_error  = data.get("is_error", False)
        latency   = data.get("latency_ms")
        result    = data.get("result", {})
        status    = result.get("status") or result.get("verdict") or ("error" if is_error else "ok")
        return f"`{name}` ({latency} ms) → **{status}**"
    if kind == "event":
        name = data.get("name", "?")
        return f"`{name}`"
    if kind == "session_start":
        meta = data.get("metadata", {})
        return f"goal: {meta.get('goal', '')[:80]}..."
    if kind == "session_end":
        return f"status: **{data.get('status', '?')}** · elapsed: {data.get('elapsed_s', '?')} s"
    return ""
