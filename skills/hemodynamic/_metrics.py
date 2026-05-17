"""Hemodynamic metric computations for 4D flow CMR.

All functions take phase-derived velocity arrays (m/s) shaped (Z, Y, X, T) and
a 3D vessel mask (Z, Y, X). The cross-section convention matches the
Physics Verifier's net-flux check: dominant flow axis from time-averaged |v|,
N evenly-spaced planes from 10% to 90% along that axis.
"""
from __future__ import annotations

import numpy as np


def dominant_flow_axis(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    mask: np.ndarray | None = None,
) -> int:
    """Return 0|1|2 for Z|Y|X — axis with the largest mean |v_axis| (time-averaged).

    Restricting to the mask is preferred; otherwise averages over the whole volume.
    """
    vx_t = vx.mean(axis=-1)
    vy_t = vy.mean(axis=-1)
    vz_t = vz.mean(axis=-1)
    if mask is not None and mask.any():
        means = [np.abs(vz_t[mask]).mean(),
                 np.abs(vy_t[mask]).mean(),
                 np.abs(vx_t[mask]).mean()]
    else:
        means = [np.abs(vz_t).mean(), np.abs(vy_t).mean(), np.abs(vx_t).mean()]
    return int(np.argmax(means))


def flow_rate_time_series(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    axis: int,
    plane_index: int,
) -> np.ndarray:
    """Q(t) in mL/s through a single cross-section perpendicular to ``axis``
    at slice ``plane_index``. Returns a length-T 1-D array.
    """
    dz, dy, dx = [s * 1e-3 for s in voxel_size_mm]      # mm → m
    area_m2 = [dy * dx, dz * dx, dz * dy][axis]          # face perpendicular to axis

    v_axial = [vz, vy, vx][axis]                         # (Z, Y, X, T)

    slc: list = [slice(None)] * 4
    slc[axis] = plane_index
    v_slab = v_axial[tuple(slc)]                         # drop the axis dim → 3-D (..., T)
    m_slab = mask[tuple(slc[:3])]                        # mask is 3-D, same drop

    # v_slab has T as last dim; broadcast mask to mask out voxels per time step
    masked = v_slab * m_slab[..., None]
    Q_m3_per_s = masked.sum(axis=tuple(range(masked.ndim - 1))) * area_m2
    return Q_m3_per_s * 1e6                              # m³/s → mL/s


def stroke_volume_and_peak(Q_mL_per_s: np.ndarray, dt_s: float) -> dict:
    """Derive volumetric metrics from a Q(t) time series.

    forward = the direction of the largest instantaneous |Q|. Stroke volume
    integrates only contributions in that direction; net SV is the signed integral
    over the full cycle.
    """
    Q = np.asarray(Q_mL_per_s, dtype=float)
    if Q.size == 0:
        return {
            "stroke_volume_forward_mL": 0.0,
            "stroke_volume_net_mL":     0.0,
            "peak_Q_mL_per_s":          0.0,
            "time_to_peak_s":           0.0,
            "regurgitant_fraction_pct": 0.0,
        }

    peak_idx = int(np.argmax(np.abs(Q)))
    sign = 1.0 if Q[peak_idx] >= 0 else -1.0

    forward = (sign * Q).clip(min=0.0)       # contributions in the forward direction
    backward = (sign * Q).clip(max=0.0)      # negative values = backflow

    sv_forward  = float(forward.sum() * dt_s)
    sv_backward = float(-backward.sum() * dt_s)
    sv_net      = float(sign * Q.sum() * dt_s)

    rf = (sv_backward / sv_forward * 100.0) if sv_forward > 0 else 0.0

    return {
        "stroke_volume_forward_mL": round(sv_forward, 3),
        "stroke_volume_net_mL":     round(sv_net, 3),
        "peak_Q_mL_per_s":          round(float(np.abs(Q).max()), 3),
        "time_to_peak_s":           round(peak_idx * dt_s, 4),
        "regurgitant_fraction_pct": round(rf, 2),
    }


def peak_velocity_in_mask(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Spatial peak of 3D speed magnitude within the vessel mask, per cardiac phase
    plus overall peak and time-to-peak."""
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)   # (Z, Y, X, T)
    if not mask.any():
        return {"peak_m_per_s": 0.0, "mean_m_per_s": 0.0, "peak_per_phase_m_per_s": []}

    # peak speed at each time step
    per_phase_peak = [float(speed[..., t][mask].max()) for t in range(speed.shape[-1])]
    overall_mean = float(speed[mask].mean())
    return {
        "peak_m_per_s":            round(max(per_phase_peak), 4),
        "mean_m_per_s":            round(overall_mean, 4),
        "peak_per_phase_m_per_s":  [round(p, 4) for p in per_phase_peak],
    }
