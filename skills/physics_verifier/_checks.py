from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def check_divergence(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    mask: np.ndarray | None,
    voxel_size_mm: tuple[float, float, float],
) -> dict:
    """
    Incompressibility check: mean |∇·v| inside vessel interior.

    vx, vy, vz : (Z, Y, X, T) velocity in m/s
    voxel_size_mm : (dz, dy, dx)
    """
    dz, dy, dx = [s * 1e-3 for s in voxel_size_mm]

    vx_t = vx.mean(axis=-1)
    vy_t = vy.mean(axis=-1)
    vz_t = vz.mean(axis=-1)

    div = (
        np.gradient(vx_t, dx, axis=2)
        + np.gradient(vy_t, dy, axis=1)
        + np.gradient(vz_t, dz, axis=0)
    )

    # Erode mask by 1 voxel — boundary voxels sit adjacent to ~0 m/s background,
    # creating large spurious gradients that dominate the divergence estimate.
    if mask is not None:
        interior = binary_erosion(mask, iterations=1)
        roi = div[interior] if interior.any() else div[mask]
    else:
        roi = div.ravel()

    mean_abs = float(np.mean(np.abs(roi)))
    max_abs = float(np.max(np.abs(roi)))

    WARN, FAIL = 5.0, 20.0  # s⁻¹
    status = "fail" if mean_abs > FAIL else "warn" if mean_abs > WARN else "pass"

    return {
        "status": status,
        "mean_abs_divergence_per_s": round(mean_abs, 4),
        "max_abs_divergence_per_s": round(max_abs, 4),
        "threshold_warn_per_s": WARN,
        "threshold_fail_per_s": FAIL,
    }


def check_net_flux(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    mask: np.ndarray | None,
    voxel_size_mm: tuple[float, float, float],
) -> dict:
    """
    Flux conservation: volumetric flow through 5 evenly-spaced cross-sections
    should be consistent (continuity equation).

    Requires a proper connected vessel mask (Stage 2c) to be meaningful on real data.
    """
    dz, dy, dx = [s * 1e-3 for s in voxel_size_mm]

    vx_t = vx.mean(axis=-1)
    vy_t = vy.mean(axis=-1)
    vz_t = vz.mean(axis=-1)

    means = [np.abs(vz_t).mean(), np.abs(vy_t).mean(), np.abs(vx_t).mean()]
    axis = int(np.argmax(means))
    v_axial = [vz_t, vy_t, vx_t][axis]
    area_m2 = [(dy * dx), (dz * dx), (dz * dy)][axis]

    n = v_axial.shape[axis]
    indices = np.linspace(int(n * 0.1), int(n * 0.9), 5, dtype=int)

    flux_values = []
    for idx in indices:
        slc: list = [slice(None)] * 3
        slc[axis] = idx
        v_slc = v_axial[tuple(slc)]
        if mask is not None:
            m_slc = mask[tuple(slc)]
            flux = float(np.sum(v_slc[m_slc]) * area_m2 * 1e6)  # → mL/s
        else:
            flux = float(np.sum(v_slc) * area_m2 * 1e6)
        flux_values.append(round(flux, 3))

    arr = np.array(flux_values)
    ref = float(np.abs(arr).mean()) or 1.0
    max_dev_pct = float(np.max(np.abs(arr - arr.mean())) / ref * 100)

    WARN_PCT, FAIL_PCT = 10.0, 25.0
    status = "fail" if max_dev_pct > FAIL_PCT else "warn" if max_dev_pct > WARN_PCT else "pass"

    return {
        "status": status,
        "flux_mL_per_s": flux_values,
        "max_deviation_pct": round(max_dev_pct, 2),
        "threshold_warn_pct": WARN_PCT,
        "threshold_fail_pct": FAIL_PCT,
        "dominant_axis": ["Z", "Y", "X"][axis],
        "requires_segmentation": True,
    }


def check_peak_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    mask: np.ndarray | None,
) -> dict:
    """
    Physiological plausibility: peak 3D speed should be within cardiac 4D flow range.

    Note: 3D speed = sqrt(vx²+vy²+vz²) can legitimately exceed VENC since it
    combines all three components — do not threshold against VENC directly.
    """
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2).mean(axis=-1)
    roi = speed[mask] if mask is not None else speed.ravel()

    peak = float(roi.max())
    p95 = float(np.percentile(roi, 95))
    mean_v = float(roi.mean())

    LO, WARN_HI, FAIL_HI = 0.05, 3.0, 4.5
    if peak > FAIL_HI:
        status = "fail"
    elif peak > WARN_HI or peak < LO:
        status = "warn"
    else:
        status = "pass"

    return {
        "status": status,
        "peak_m_per_s": round(peak, 4),
        "p95_m_per_s": round(p95, 4),
        "mean_m_per_s": round(mean_v, 4),
        "physiological_range_m_per_s": [LO, WARN_HI],
    }


def check_phase_unwrap(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    venc_m_per_s: float | None,
) -> dict:
    """
    Phase-wrap detection: spatial jumps larger than VENC in any velocity
    component indicate residual wrapping artifacts.

    Checks each component separately — speed magnitude alone would miss wraps
    where the sign flips but magnitude stays the same (e.g. +VENC → -VENC).
    """
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2).mean(axis=-1)
    threshold = venc_m_per_s if venc_m_per_s is not None else float(np.percentile(speed, 99)) * 2.0

    max_jump = 0.0
    n_above = 0
    n_total = 0
    for component in [vx.mean(axis=-1), vy.mean(axis=-1), vz.mean(axis=-1)]:
        for ax in range(3):
            diff = np.abs(np.diff(component, axis=ax))
            max_jump = max(max_jump, float(diff.max()))
            n_above += int((diff > threshold).sum())
            n_total += diff.size

    fraction_above = n_above / n_total if n_total > 0 else 0.0

    WARN_FRAC, FAIL_FRAC = 0.001, 0.01
    status = "fail" if fraction_above > FAIL_FRAC else "warn" if fraction_above > WARN_FRAC else "pass"

    return {
        "status": status,
        "max_spatial_jump_m_per_s": round(max_jump, 4),
        "fraction_above_threshold": round(fraction_above, 6),
        "threshold_m_per_s": round(threshold, 4),
        "venc_m_per_s": round(venc_m_per_s, 4) if venc_m_per_s is not None else None,
        "threshold_warn_fraction": WARN_FRAC,
        "threshold_fail_fraction": FAIL_FRAC,
    }
