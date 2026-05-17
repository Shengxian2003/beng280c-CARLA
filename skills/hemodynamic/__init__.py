"""Hemodynamic Analysis Skill (Stage 2d).

Computes volumetric flow rate Q(t), stroke volume, and peak flow at evenly-spaced
cross-sections along a vessel mask, plus peak velocity within the vessel.

Public API:
    analyze(thetaX, thetaY, thetaZ, venc_m_per_s, mask, voxel_size_mm, dt_s, ...) → dict
"""
from __future__ import annotations

import numpy as np

from ._metrics import (
    dominant_flow_axis,
    flow_rate_time_series,
    stroke_volume_and_peak,
    peak_velocity_in_mask,
)


def analyze(
    thetaX: np.ndarray,
    thetaY: np.ndarray,
    thetaZ: np.ndarray,
    *,
    venc_m_per_s: float,
    mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    dt_seconds: float = 0.05,
    n_cross_sections: int = 5,
) -> dict:
    """Run hemodynamic analysis on a 4D flow velocity field.

    Parameters
    ----------
    thetaX, thetaY, thetaZ : ndarray (Z, Y, X, T)
        Background-corrected phase in radians (output of ``reconstruct()``).
    venc_m_per_s : float
        Velocity encoding range. Converts phase → velocity: v = phase * VENC / π.
    mask : ndarray (Z, Y, X) bool
        Vessel mask from the Segmentation Skill. Required (Stage 2c output).
    voxel_size_mm : (dz, dy, dx)
        Voxel size in mm.
    dt_seconds : float
        Time between consecutive cardiac phases. For OSU-MR data (20 phases /
        ~1 s RR cycle) this is ≈ 0.05 s.
    n_cross_sections : int
        Number of evenly-spaced planes along the dominant flow axis.

    Returns
    -------
    dict
        {
          "per_section": [ {position, Q_mL_per_s[T], stroke_volume_*, peak_Q, ...}, ... ],
          "summary":     { mean_stroke_volume_mL, mean_peak_Q_mL_per_s, peak_velocity_m_per_s, ... },
          "metadata":    { dominant_axis, n_phases, dt_s, cycle_duration_s, ... },
        }
    """
    if mask is None or not mask.any():
        raise ValueError("analyze() requires a non-empty vessel mask (from Stage 2c segmentation)")

    # Phase (rad) → velocity (m/s)
    s = venc_m_per_s / np.pi
    vx, vy, vz = thetaX * s, thetaY * s, thetaZ * s

    axis = dominant_flow_axis(vx, vy, vz, mask)
    axis_name = ["Z", "Y", "X"][axis]

    n_along = vx.shape[axis]
    # Skip endpoints — boundary slabs are usually outside the vessel
    indices = np.linspace(int(n_along * 0.1), int(n_along * 0.9),
                          n_cross_sections, dtype=int)

    per_section = []
    for idx in indices:
        Q = flow_rate_time_series(vx, vy, vz, mask, voxel_size_mm, axis, int(idx))
        derived = stroke_volume_and_peak(Q, dt_seconds)
        per_section.append({
            "plane_index":  int(idx),
            "position_pct": round(float(idx) / max(n_along - 1, 1) * 100, 1),
            "Q_mL_per_s":   [round(float(q), 3) for q in Q],
            **derived,
        })

    velocity = peak_velocity_in_mask(vx, vy, vz, mask)

    peak_qs = np.array([s["peak_Q_mL_per_s"] for s in per_section])
    sv_fwds = np.array([s["stroke_volume_forward_mL"] for s in per_section])

    summary = {
        "mean_stroke_volume_mL":     round(float(sv_fwds.mean()), 3),
        "mean_peak_Q_mL_per_s":      round(float(peak_qs.mean()), 3),
        "max_peak_Q_mL_per_s":       round(float(peak_qs.max()), 3),
        "peak_velocity_m_per_s":     velocity["peak_m_per_s"],
        "mean_velocity_m_per_s":     velocity["mean_m_per_s"],
        "peak_velocity_per_phase":   velocity["peak_per_phase_m_per_s"],
    }

    n_phases = thetaX.shape[-1]
    metadata = {
        "dominant_axis":       axis_name,
        "n_cross_sections":    n_cross_sections,
        "n_phases":            n_phases,
        "dt_s":                dt_seconds,
        "cycle_duration_s":    round(n_phases * dt_seconds, 4),
        "voxel_size_mm":       list(voxel_size_mm),
        "voxel_volume_mm3":    round(float(np.prod(voxel_size_mm)), 4),
        "n_mask_voxels":       int(mask.sum()),
        "venc_m_per_s":        venc_m_per_s,
    }

    return {"per_section": per_section, "summary": summary, "metadata": metadata}


__all__ = ["analyze"]
