"""Synthetic clean flow phantom — the baseline for detection-rate evaluation.

The phantom is designed to pass ALL four physics verifier checks:
  - divergence-free (incompressible, uniform flow inside a cylinder)
  - flux-conserved (constant flow along a straight tube)
  - physiological peak velocity (0.5 m/s)
  - no phase wraps (velocity well below VENC, no spatial discontinuities)

Stage 4 evaluation injects controlled errors on top of this clean baseline
and measures whether the verifier detects them.
"""
from __future__ import annotations

import numpy as np


def clean_flow_phantom(
    Z: int = 40,
    Y: int = 40,
    X: int = 40,
    T: int = 8,
    velocity_m_per_s: float = 0.5,
    vessel_radius_voxels: float = 8.0,
    flow_axis: str = "Z",
    venc_m_per_s: float = 1.5,
) -> dict:
    """Return a clean (verifier-passing) 4D flow phantom.

    The vessel is a cylinder oriented along ``flow_axis`` carrying a uniform
    flow at ``velocity_m_per_s``. Default geometry is generous enough that
    even a 1-voxel erosion (used in divergence check) leaves > 5000 voxels.

    Returns
    -------
    dict with keys:
        thetaX, thetaY, thetaZ : (Z, Y, X, T) phase in radians, ready for verify()
        vx, vy, vz             : (Z, Y, X, T) velocity in m/s (for injectors)
        mask                   : (Z, Y, X) bool vessel mask
        venc_m_per_s, voxel_size_mm, dt_seconds  (metadata)
    """
    # Cylindrical vessel mask centred in the transverse plane
    yy, xx = np.meshgrid(np.arange(Y) - Y / 2, np.arange(X) - X / 2, indexing="ij")
    cross = (yy ** 2 + xx ** 2) <= vessel_radius_voxels ** 2  # (Y, X)
    mask = np.broadcast_to(cross, (Z, Y, X)).copy()

    # Velocity field — zero outside mask, uniform inside
    vx = np.zeros((Z, Y, X, T), dtype=np.float64)
    vy = np.zeros_like(vx)
    vz = np.zeros_like(vx)
    target = {"X": vx, "Y": vy, "Z": vz}[flow_axis]
    target[mask] = velocity_m_per_s

    # Convert velocity → phase (radians): theta = v * pi / VENC
    s = np.pi / venc_m_per_s
    return {
        "thetaX": vx * s,
        "thetaY": vy * s,
        "thetaZ": vz * s,
        "vx": vx, "vy": vy, "vz": vz,
        "mask": mask,
        "venc_m_per_s": venc_m_per_s,
        "voxel_size_mm": (2.0, 2.0, 2.0),
        "dt_seconds": 0.05,
        "flow_axis": flow_axis,
    }


def velocity_to_phase(vx, vy, vz, venc_m_per_s):
    """Convert velocity (m/s) → phase (rad) using v = phase * VENC / π."""
    s = np.pi / venc_m_per_s
    return vx * s, vy * s, vz * s
