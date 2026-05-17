from __future__ import annotations

import numpy as np

from ._pcmra import (
    compute_pcmra,
    segment_largest_vessel,
    segment_top_n_vessels,
    segment_from_seed as _segment_from_seed_impl,
    suggest_seed_points as _suggest_seed_points_impl,
)


def segment(
    thetaX: np.ndarray,
    thetaY: np.ndarray,
    thetaZ: np.ndarray,
    xHat: np.ndarray,
    venc_m_per_s: float,
    percentile: float = 85.0,
    closing_iter: int = 2,
    n_vessels: int = 1,
) -> np.ndarray:
    """
    Segment vessel(s) from a 4D flow MRI dataset using PC-MRA.

    Pipeline:
      1. Compute PC-MRA = signal magnitude × speed magnitude (time-averaged)
      2. Threshold at given percentile
      3. Morphological closing to fill lumen gaps
      4. Connected component labeling → keep largest (or top n)

    Parameters
    ----------
    thetaX, thetaY, thetaZ : ndarray (Z, Y, X, T), phase in radians
    xHat                   : ndarray (Z, Y, X, T), magnitude image
    venc_m_per_s           : velocity encoding (m/s)
    percentile             : threshold percentile (default 85 = top 15%)
    closing_iter           : binary_closing iterations (default 2)
    n_vessels              : 1 → largest only; >1 → top-N largest

    Returns
    -------
    (Z, Y, X) bool mask of segmented vessel region.
    """
    pcmra = compute_pcmra(thetaX, thetaY, thetaZ, xHat, venc_m_per_s)

    if n_vessels == 1:
        return segment_largest_vessel(pcmra, percentile, closing_iter)
    return segment_top_n_vessels(pcmra, n_vessels, percentile, closing_iter)


def segment_from_seed(
    thetaX: np.ndarray,
    thetaY: np.ndarray,
    thetaZ: np.ndarray,
    xHat: np.ndarray,
    venc_m_per_s: float,
    seed_point: tuple[int, int, int],
    percentile: float = 85.0,
    closing_iter: int = 2,
) -> np.ndarray:
    """
    Agent-driven segmentation: return the vessel containing seed_point.

    The Stage 3 Coordinator agent calls this with a chosen seed coordinate.
    Seed selection logic (anatomical prior, PC-MRA inspection, iterative
    retry based on verifier feedback) lives in the agent, not here.

    seed_point : (z, y, x) coordinate the agent picked
    Returns (Z, Y, X) bool mask of the connected vessel containing seed_point.
    Returns empty mask if seed lies in background — agent should retry.
    """
    pcmra = compute_pcmra(thetaX, thetaY, thetaZ, xHat, venc_m_per_s)
    return _segment_from_seed_impl(pcmra, seed_point, percentile, closing_iter)


def suggest_seed_points(
    thetaX: np.ndarray,
    thetaY: np.ndarray,
    thetaZ: np.ndarray,
    xHat: np.ndarray,
    venc_m_per_s: float,
    n_candidates: int = 5,
    percentile: float = 85.0,
    closing_iter: int = 2,
) -> list[dict]:
    """
    Propose candidate seed points for the agent to choose from.

    Returns a list of dicts (sorted by brightness × size, descending), each:
      {seed: (z,y,x), size: int, mean_pcmra: float, bbox: (z0,y0,x0,z1,y1,x1)}

    The agent inspects these candidates and picks one based on:
      - Anatomical priors (aorta is usually upper-anterior in cardiac scans)
      - Bounding box geometry (tube-shaped → aorta; compact → chamber)
      - Iterative retry after a failed verify() verdict
    """
    pcmra = compute_pcmra(thetaX, thetaY, thetaZ, xHat, venc_m_per_s)
    return _suggest_seed_points_impl(pcmra, n_candidates, percentile, closing_iter)
