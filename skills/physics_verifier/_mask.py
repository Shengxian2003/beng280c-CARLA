import numpy as np


def threshold_mask(xHat: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    """
    Coarse vessel mask from signal magnitude via percentile threshold.

    xHat : (Z, Y, X, T) magnitude
    Returns (Z, Y, X) bool mask.

    Note: in 4D flow MRI the vessel lumen may be dimmer than surrounding
    tissue. Prefer velocity_mask() when phase data is available.
    """
    mag = np.abs(xHat).mean(axis=-1)
    return mag > np.percentile(mag, percentile)


def velocity_mask(
    thetaX: np.ndarray,
    thetaY: np.ndarray,
    thetaZ: np.ndarray,
    venc_m_per_s: float,
    percentile: float = 85.0,
) -> np.ndarray:
    """
    PC-MRA-style vessel mask: select voxels by velocity magnitude.

    Flowing blood has high speed; stationary tissue has speed ≈ 0.
    This is more reliable than signal-magnitude thresholding for 4D flow data.

    thetaX/Y/Z : (Z, Y, X, T) phase in radians
    Returns (Z, Y, X) bool mask of the top (100-percentile)% fastest voxels.
    """
    scale = venc_m_per_s / np.pi
    speed = np.sqrt(
        (thetaX * scale) ** 2 +
        (thetaY * scale) ** 2 +
        (thetaZ * scale) ** 2
    ).mean(axis=-1)
    threshold = np.percentile(speed, percentile)
    mask = speed > threshold
    # Degenerate case: uniform or zero speed → fall back to full volume
    if not mask.any():
        return np.ones(speed.shape, dtype=bool)
    return mask
