"""Physics Verifier verdict panel — pass/warn/fail badges per check."""
from __future__ import annotations

import streamlit as st

STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
STATUS_COLOR = {"pass": "green", "warn": "orange", "fail": "red"}


def render_verifier(verdicts: dict) -> None:
    st.subheader("Physics Verifier")

    if not verdicts:
        st.info("No verifier verdicts recorded for this run.")
        return

    for mask_name, verdict in verdicts.items():
        overall = verdict.get("verdict", "?")
        icon    = STATUS_ICON.get(overall, "❓")
        color   = STATUS_COLOR.get(overall, "gray")

        st.markdown(
            f"**Mask:** `{mask_name}` &nbsp;·&nbsp; "
            f":{color}[{icon} {overall.upper()}]"
        )

        checks = verdict.get("checks", {})
        for name in ["divergence", "net_flux", "peak_velocity", "phase_unwrap"]:
            c = checks.get(name, {})
            if not c:
                continue
            status = c.get("status", "?")
            ic     = STATUS_ICON.get(status, "❓")
            detail = _check_detail(name, c)
            st.markdown(f"&nbsp;&nbsp;{ic} **{name}** — {detail}")

        st.divider()


def _check_detail(name: str, c: dict) -> str:
    if name == "divergence":
        return (f"mean |∇·v| = {c.get('mean_abs_divergence_per_s', '?')} s⁻¹ "
                f"(threshold {c.get('threshold_warn_per_s')}/{c.get('threshold_fail_per_s')})")
    if name == "net_flux":
        return (f"max deviation = {c.get('max_deviation_pct', '?')}% "
                f"(threshold {c.get('threshold_warn_pct')}/{c.get('threshold_fail_pct')}%)")
    if name == "peak_velocity":
        return (f"peak = {c.get('peak_m_per_s', '?')} m/s "
                f"(physiological range {c.get('physiological_range_m_per_s', '?')})")
    if name == "phase_unwrap":
        return (f"wrap fraction = {c.get('fraction_above_threshold', '?')} "
                f"(threshold {c.get('threshold_warn_fraction')}/{c.get('threshold_fail_fraction')})")
    return ""
