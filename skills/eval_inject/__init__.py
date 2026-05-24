"""Stage 4 evaluation: error-injection harness.

Provides a clean synthetic flow phantom and four targeted error injectors,
one for each Physics Verifier check. Used by the detection-rate sweep in
``evaluation/run_detection_eval.py`` to build per-check detection curves.

Public API:
    clean_flow_phantom(...)          → dict of clean baseline fields
    velocity_to_phase(vx, vy, vz, venc)  → phase arrays for verify()

    inject_divergence(vx, vy, vz, mask, magnitude_per_s=...)
    inject_flux_imbalance(vx, vy, vz, mask, fractional_imbalance=...)
    inject_peak_velocity(vx, vy, vz, mask, target_peak_m_per_s=...)
    inject_phase_wrap(thetaX, thetaY, thetaZ, mask, fraction=...)
"""
from __future__ import annotations

from ._phantom import clean_flow_phantom, velocity_to_phase
from ._injectors import (
    inject_divergence,
    inject_flux_imbalance,
    inject_peak_velocity,
    inject_phase_wrap,
)

__all__ = [
    "clean_flow_phantom",
    "velocity_to_phase",
    "inject_divergence",
    "inject_flux_imbalance",
    "inject_peak_velocity",
    "inject_phase_wrap",
]
