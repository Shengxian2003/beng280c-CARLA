from __future__ import annotations

import json
import numpy as np

from ._checks import (
    check_divergence,
    check_net_flux,
    check_peak_velocity,
    check_phase_unwrap,
)
from ._mask import threshold_mask, velocity_mask


def verify(
    thetaX: np.ndarray,
    thetaY: np.ndarray,
    thetaZ: np.ndarray,
    venc_m_per_s: float,
    xHat: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    voxel_size_mm: tuple[float, float, float] = (1.5, 1.5, 1.5),
) -> dict:
    """
    Run all four physics checks on a 4D flow velocity field.

    Parameters
    ----------
    thetaX, thetaY, thetaZ : ndarray (Z, Y, X, T), phase in radians [-π, π]
        Differential phase images from reconstruction (angle of velocity encodings).
    venc_m_per_s : float
        Velocity encoding range (m/s). Used to convert phase → velocity and as
        the threshold for the phase-unwrap check.
    xHat : ndarray (Z, Y, X, T), optional
        Reconstructed magnitude image — used to auto-generate a vessel mask.
    mask : ndarray (Z, Y, X) bool, optional
        Pre-computed vessel mask. Overrides auto-generation if provided.
    voxel_size_mm : (dz, dy, dx)
        Voxel dimensions in mm — required for gradient and flux calculations.

    Returns
    -------
    dict  JSON-serialisable verdict:
        {
          "verdict": "pass" | "warn" | "fail",
          "checks": { divergence, net_flux, peak_velocity, phase_unwrap },
          "metadata": { ... }
        }
    """
    # Convert phase (radians) → velocity (m/s)
    scale = venc_m_per_s / np.pi
    vx = thetaX * scale
    vy = thetaY * scale
    vz = thetaZ * scale

    if mask is None:
        # velocity_mask is preferred for 4D flow — vessels are brightest by speed,
        # not necessarily by signal magnitude
        mask = velocity_mask(thetaX, thetaY, thetaZ, venc_m_per_s)

    results = {
        "divergence":    check_divergence(vx, vy, vz, mask, voxel_size_mm),
        "net_flux":      check_net_flux(vx, vy, vz, mask, voxel_size_mm),
        "peak_velocity": check_peak_velocity(vx, vy, vz, mask),
        "phase_unwrap":  check_phase_unwrap(vx, vy, vz, venc_m_per_s),
    }

    statuses = [c["status"] for c in results.values()]
    overall = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"

    return {
        "verdict": overall,
        "checks": results,
        "metadata": {
            "shape_ZYXT": list(thetaX.shape),
            "voxel_size_mm": list(voxel_size_mm),
            "venc_m_per_s": venc_m_per_s,
            "n_mask_voxels": int(mask.sum()) if mask is not None else None,
        },
    }


def verify_from_mat(
    mat_path: str,
    venc_m_per_s: float,
    **kwargs,
) -> dict:
    """
    Load reconstruction output from a .mat file and run verify().

    Expects the .mat to contain an 'outputs' struct with fields:
        thetaX, thetaY, thetaZ  — phase arrays (Z, Y, X, T)
        xHat                     — magnitude array (optional)
    """
    try:
        import scipy.io as sio
        data = sio.loadmat(mat_path, squeeze_me=True)["outputs"]
        thetaX = np.array(data["thetaX"].item())
        thetaY = np.array(data["thetaY"].item())
        thetaZ = np.array(data["thetaZ"].item())
        xHat_raw = data["xHat"].item() if "xHat" in data.dtype.names else None
        xHat = np.array(xHat_raw) if xHat_raw is not None else None
    except NotImplementedError:
        # MATLAB v7.3 HDF5 format — fall back to h5py
        import h5py
        with h5py.File(mat_path, "r") as f:
            outputs = f["outputs"]
            thetaX = np.array(outputs["thetaX"]).T
            thetaY = np.array(outputs["thetaY"]).T
            thetaZ = np.array(outputs["thetaZ"]).T
            xHat = np.array(outputs["xHat"]).T if "xHat" in outputs else None

    return verify(thetaX, thetaY, thetaZ, venc_m_per_s, xHat=xHat, **kwargs)


def verdict_json(result: dict) -> str:
    """Pretty-print a verify() result as JSON."""
    return json.dumps(result, indent=2)
