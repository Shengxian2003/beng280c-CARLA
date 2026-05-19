"""Error injectors for Stage 4 evaluation.

Each injector deliberately corrupts a velocity (or phase) field in a way
that should be detected by exactly ONE of the four Physics Verifier checks.
The magnitude is parameterised so the detection-rate sweep in
``evaluation/run_detection_eval.py`` can vary error strength and build
detection-vs-magnitude curves.

All injectors are deterministic given the ``seed`` argument.

Design contract:
- input fields are NOT mutated; injectors return new arrays
- error is applied INSIDE the provided mask (errors outside the vessel are
  meaningless for the verifier, which restricts its checks to mask interior)
- magnitude=0 returns the input unchanged (used as a control)
"""
from __future__ import annotations

import numpy as np

from ._phantom import velocity_to_phase


# ============================================================================
# Injection: divergence (target check: check_divergence)
# ============================================================================

def inject_divergence(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    magnitude_per_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add a divergence-creating term inside the mask.

    Adds vx_extra = alpha * x_mm to vx, which contributes ∂vx/∂x = alpha
    everywhere inside the mask. Result: |∇·v| ≈ magnitude_per_s (in s⁻¹).

    Parameters
    ----------
    magnitude_per_s : float
        Target divergence magnitude in s⁻¹. The verifier's thresholds are
        warn > 5, fail > 20. Use 0 for the control (no injection).
    """
    if magnitude_per_s == 0:
        return vx.copy(), vy.copy(), vz.copy()

    dx_m = voxel_size_mm[2] * 1e-3
    Z, Y, X, T = vx.shape

    # x_mm grid centered at zero to keep the perturbation balanced around 0
    x_idx = (np.arange(X) - X / 2) * dx_m              # shape (X,)
    perturbation = magnitude_per_s * x_idx              # m/s, shape (X,)
    perturbation_4d = perturbation[None, None, :, None]  # broadcasts over (Z,Y,X,T)

    vx_out = vx + perturbation_4d * mask[..., None]
    return vx_out, vy.copy(), vz.copy()


# ============================================================================
# Injection: flux imbalance (target: check_net_flux)
# ============================================================================

def inject_flux_imbalance(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    mask: np.ndarray,
    *,
    flow_axis: str = "Z",
    fractional_imbalance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make the flux along ``flow_axis`` vary linearly along that axis.

    Multiplies the axial velocity by (1 + fractional_imbalance * normalized_pos),
    where normalized_pos goes from -0.5 (start) to +0.5 (end) along flow_axis.
    A ``fractional_imbalance`` of 0.5 means flux differs by ~50% from start
    to end of the vessel (the verifier flags > 25% as fail).
    """
    if fractional_imbalance == 0:
        return vx.copy(), vy.copy(), vz.copy()

    axis = {"Z": 0, "Y": 1, "X": 2}[flow_axis]
    N = vx.shape[axis]
    normalized_pos = np.linspace(-0.5, 0.5, N)   # along the flow axis

    # Build a 4D scale array that broadcasts: scale[k] applies to slab k along axis
    shape = [1, 1, 1, 1]
    shape[axis] = N
    scale = (1.0 + fractional_imbalance * normalized_pos).reshape(shape)

    # Apply only inside mask: scale where mask is True, leave alone outside
    mask4 = mask[..., None]
    delta_factor = (scale - 1.0)                  # how much to add as a multiplicative perturbation

    if flow_axis == "X":
        return vx + vx * delta_factor * mask4, vy.copy(), vz.copy()
    if flow_axis == "Y":
        return vx.copy(), vy + vy * delta_factor * mask4, vz.copy()
    return vx.copy(), vy.copy(), vz + vz * delta_factor * mask4


# ============================================================================
# Injection: peak velocity (target: check_peak_velocity)
# ============================================================================

def inject_peak_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    mask: np.ndarray,
    *,
    target_peak_m_per_s: float = 0.0,
    region_size: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert a small hotspot inside the mask with the target speed.

    Picks a random mask voxel as the hotspot centre (seeded) and sets a
    ``region_size``-cubed neighbourhood to a single-component velocity that
    yields the desired 3-D speed magnitude.

    Parameters
    ----------
    target_peak_m_per_s : float
        Desired peak speed magnitude. Verifier warns outside [0.05, 3.0],
        fails > 4.5. Use 0 for the control.
    """
    if target_peak_m_per_s == 0:
        return vx.copy(), vy.copy(), vz.copy()

    rng = np.random.default_rng(seed)
    Z, Y, X, T = vx.shape
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return vx.copy(), vy.copy(), vz.copy()

    cz, cy, cx = coords[rng.integers(len(coords))]
    half = region_size // 2
    z0, z1 = max(0, cz - half), min(Z, cz + half + 1)
    y0, y1 = max(0, cy - half), min(Y, cy + half + 1)
    x0, x1 = max(0, cx - half), min(X, cx + half + 1)

    vx_out, vy_out, vz_out = vx.copy(), vy.copy(), vz.copy()
    # Put the hotspot in vz so speed = |vz| = target
    vz_out[z0:z1, y0:y1, x0:x1, :] = target_peak_m_per_s
    return vx_out, vy_out, vz_out


# ============================================================================
# Injection: phase wrap (target: check_phase_unwrap)
# ============================================================================

def inject_phase_wrap(
    thetaX: np.ndarray,
    thetaY: np.ndarray,
    thetaZ: np.ndarray,
    mask: np.ndarray,
    *,
    fraction: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add ±2π wraps to a fraction of voxels inside the mask.

    Picks ``fraction`` of mask voxels uniformly at random and flips their
    phase by π (which causes a velocity sign-flip = jump ~ 2·VENC), in a
    randomly-chosen velocity component.

    The Physics Verifier's check_phase_unwrap looks at component-wise spatial
    jumps > VENC. Inserting a π flip creates a jump of ~2·VENC between the
    wrapped voxel and its neighbour, which should be detected.

    Parameters
    ----------
    fraction : float
        Fraction (0-1) of mask voxels to wrap. Verifier warns > 0.001, fails > 0.01.
        Use 0 for the control.
    """
    if fraction == 0:
        return thetaX.copy(), thetaY.copy(), thetaZ.copy()

    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask)
    n_wrap = max(1, int(len(coords) * fraction))
    chosen = coords[rng.choice(len(coords), size=n_wrap, replace=False)]

    out = [thetaX.copy(), thetaY.copy(), thetaZ.copy()]
    # For each chosen voxel: pick a random component, add a 2π phase wrap.
    # Why 2π (not π): the verifier converts phase → velocity via v = phase·VENC/π.
    # Adding π yields velocity jump = VENC exactly, which is NOT > VENC (verifier
    # uses strict inequality). Adding 2π yields 2·VENC, which is detected.
    for (cz, cy, cx) in chosen:
        comp = rng.integers(3)
        out[comp][cz, cy, cx, :] += 2 * np.pi
    return out[0], out[1], out[2]
