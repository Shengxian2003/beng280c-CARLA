"""Browse any historical audit log (JSONL) and inspect its timeline + results."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from ui._utils.runner import _parse_audit_log
from ui._widgets.audit_timeline import render_timeline
from ui._widgets.hemodynamic_panel import render_hemodynamic
from ui._widgets.verifier_panel import render_verifier

st.set_page_config(page_title="MEDICT — Audit Viewer", page_icon="📜", layout="wide")
st.title("Audit Viewer")
st.caption("Browse any past run's audit log.")

logs_dir = PROJECT_ROOT / "logs"
candidates = sorted(logs_dir.glob("*.jsonl")) + sorted((logs_dir / "reference_runs").glob("*.jsonl"))

if not candidates:
    st.warning("No audit logs found under `logs/`.")
    st.stop()

selected = st.selectbox(
    "Select an audit log",
    options=candidates,
    format_func=lambda p: f"{p.relative_to(PROJECT_ROOT)}  ({p.stat().st_size // 1024} KB)",
)

st.divider()
parsed = _parse_audit_log(selected)

c1, c2 = st.columns(2)
with c1: render_verifier(parsed["verdicts"])
with c2: render_hemodynamic(parsed["analyses"])

st.divider()
st.subheader("Timeline")
render_timeline(selected)
