from __future__ import annotations

import numpy as np
from scipy.ndimage import label, binary_closing


def compute_pcmra(
    thetaX: np.ndarray,
    thetaY: np.ndarray,
    thetaZ: np.ndarray,
    xHat: np.ndarray,
    venc_m_per_s: float,
) -> np.ndarray:
    """
    Phase-Contrast Magnetic Resonance Angiography (PC-MRA) image.

    PC-MRA = signal magnitude × speed magnitude, time-averaged.
    Static tissue has speed ≈ 0 → contributes nothing.
    Flowing blood has high speed AND non-zero signal → appears bright.

    thetaX/Y/Z : (Z, Y, X, T) phase in radians
    xHat       : (Z, Y, X, T) reconstructed magnitude
    Returns    : (Z, Y, X) time-averaged PC-MRA image
    """
    scale = venc_m_per_s / np.pi
    speed = np.sqrt(
        (thetaX * scale) ** 2 +
        (thetaY * scale) ** 2 +
        (thetaZ * scale) ** 2
    )
    weighted = np.abs(xHat) * speed
    return weighted.mean(axis=-1)


def segment_largest_vessel(
    pcmra: np.ndarray,
    percentile: float = 85.0,
    closing_iter: int = 2,
) -> np.ndarray:
    """
    Threshold a PC-MRA image and keep the largest connected component.

    The largest connected high-PC-MRA region in cardiac 4D flow is typically
    the aorta — a coherent tube suitable for divergence and flux checks.

    pcmra        : (Z, Y, X) PC-MRA image
    percentile   : threshold (default 85th = top 15% of voxels)
    closing_iter : binary_closing iterations to fill small lumen gaps

    Returns (Z, Y, X) bool mask of the single largest connected vessel.
    """
    threshold = np.percentile(pcmra, percentile)
    binary = pcmra > threshold

    if closing_iter > 0:
        binary = binary_closing(binary, iterations=closing_iter)

    labels, n_labels = label(binary)
    if n_labels == 0:
        return binary

    # Component sizes — skip background label 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    return labels == largest_label


def segment_top_n_vessels(
    pcmra: np.ndarray,
    n: int = 3,
    percentile: float = 85.0,
    closing_iter: int = 2,
) -> np.ndarray:
    """
    Like segment_largest_vessel but keeps the top-N connected components.
    Useful when multiple vessels (aorta + pulmonary arteries) should be analyzed.
    """
    threshold = np.percentile(pcmra, percentile)
    binary = pcmra > threshold

    if closing_iter > 0:
        binary = binary_closing(binary, iterations=closing_iter)

    labels, n_labels = label(binary)
    if n_labels == 0:
        return binary

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    top_labels = np.argsort(sizes)[-n:]
    return np.isin(labels, top_labels[top_labels > 0])


def segment_from_seed(
    pcmra: np.ndarray,
    seed_point: tuple[int, int, int],
    percentile: float = 85.0,
    closing_iter: int = 2,
) -> np.ndarray:
    """
    Region-grow from a seed point: return the connected high-PC-MRA component
    that contains the given voxel.

    pcmra      : (Z, Y, X) PC-MRA image
    seed_point : (z, y, x) coordinate the agent picked to grow from

    Returns (Z, Y, X) bool mask of the connected component containing seed_point.
    Returns an empty mask if the seed falls below the PC-MRA threshold — the
    agent should interpret this as "seed missed the vessel; try another".
    """
    z, y, x = seed_point
    if not (0 <= z < pcmra.shape[0] and 0 <= y < pcmra.shape[1] and 0 <= x < pcmra.shape[2]):
        raise ValueError(f"seed_point {seed_point} is outside volume shape {pcmra.shape}")

    threshold = np.percentile(pcmra, percentile)
    binary = pcmra > threshold

    if closing_iter > 0:
        binary = binary_closing(binary, iterations=closing_iter)

    if not binary[z, y, x]:
        # Seed below threshold — no region to grow
        return np.zeros_like(binary, dtype=bool)

    labels, _ = label(binary)
    seed_label = int(labels[z, y, x])
    return labels == seed_label


def suggest_seed_points(
    pcmra: np.ndarray,
    n_candidates: int = 5,
    percentile: float = 85.0,
    closing_iter: int = 2,
    min_size: int = 50,
) -> list[dict]:
    """
    Propose seed points for the agent to consider.

    Returns the top-N connected high-PC-MRA components, each annotated with:
      - seed     : (z, y, x) centroid coordinate
      - size     : voxel count
      - mean_pcmra : mean PC-MRA brightness inside the component
      - bbox     : (z0, y0, x0, z1, y1, x1) bounding box

    The agent can rank these by anatomical prior (e.g., "aorta is usually in
    the anterior-superior region") or by trial-and-error with the verifier.
    Candidates are pre-sorted by total brightness (size × mean_pcmra) descending.
    """
    threshold = np.percentile(pcmra, percentile)
    binary = pcmra > threshold
    if closing_iter > 0:
        binary = binary_closing(binary, iterations=closing_iter)

    labels, n_labels = label(binary)
    if n_labels == 0:
        return []

    candidates = []
    for lbl in range(1, n_labels + 1):
        comp = labels == lbl
        size = int(comp.sum())
        if size < min_size:
            continue
        coords = np.argwhere(comp)
        # Use the brightest voxel of the component as the seed.
        # Centroid can fall in a gap for non-convex shapes (e.g. branching vessels).
        masked_pcmra = np.where(comp, pcmra, -np.inf)
        peak_idx = np.unravel_index(np.argmax(masked_pcmra), pcmra.shape)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1
        candidates.append({
            "seed": tuple(int(c) for c in peak_idx),
            "size": size,
            "mean_pcmra": float(pcmra[comp].mean()),
            "bbox": (int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)),
        })

    candidates.sort(key=lambda c: c["mean_pcmra"] * c["size"], reverse=True)
    return candidates[:n_candidates]
