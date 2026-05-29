"""Hemodynamic Analyzer results panel — flow metrics with physiological context."""
from __future__ import annotations

import streamlit as st

PHYSIOLOGICAL_RANGES = {
    "mean_stroke_volume_mL":     (60, 100,    "mL"),
    "mean_peak_Q_mL_per_s":      (400, 600,   "mL/s"),
    "peak_velocity_m_per_s":     (0.8, 2.0,   "m/s"),
}

LABELS = {
    "mean_stroke_volume_mL":     "Mean stroke volume",
    "mean_peak_Q_mL_per_s":      "Mean peak Q",
    "peak_velocity_m_per_s":     "Mean peak velocity",
    "stroke_volume_cv":          "Stroke-volume CV across sections",
}


def render_hemodynamic(analyses: dict) -> None:
    st.subheader("Hemodynamic Analyzer")

    if not analyses:
        st.info("No hemodynamic results — Coordinator did not reach this stage.")
        return

    for mask_name, report in analyses.items():
        st.markdown(f"**Mask:** `{mask_name}`")
        summary = report.get("summary", {})
        if not summary:
            st.caption("(report has no summary block)")
            continue

        for key, label in LABELS.items():
            if key not in summary:
                continue
            val = summary[key]
            unit_range = PHYSIOLOGICAL_RANGES.get(key)
            if unit_range:
                lo, hi, unit = unit_range
                delta = _classify(val, lo, hi)
                st.metric(label, f"{val} {unit}", delta=delta, delta_color="off")
                st.caption(f"Physiological range: {lo}–{hi} {unit}")
            else:
                st.metric(label, f"{val}")

        n_sections = len(report.get("per_section", []))
        if n_sections:
            st.caption(f"Cross-sections analyzed: {n_sections}")
        st.divider()


def _classify(val: float, lo: float, hi: float) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v < lo:
        return f"below range ({v:.2f} < {lo})"
    if v > hi:
        return f"above range ({v:.2f} > {hi})"
    return "within physiological range"
