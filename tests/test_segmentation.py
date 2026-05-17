import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.segmentation import segment, compute_pcmra, segment_from_seed, suggest_seed_points
from skills.segmentation._pcmra import (
    segment_largest_vessel,
    segment_top_n_vessels,
    segment_from_seed as segment_from_seed_pcmra,
    suggest_seed_points as suggest_seed_points_pcmra,
)

VENC = 1.5


def _synthetic_4d_flow(z_dim=24, y_dim=24, x_dim=24, t_dim=6, seed=0):
    """
    Build a synthetic 4D flow dataset with a known tube-shaped vessel:
      - A "vessel" cylinder along the Z axis at (y=12, x=12), radius 3
      - Inside: high signal, fast flow along Z
      - Outside: noise-level signal, ~0 velocity
    """
    rng = np.random.default_rng(seed)
    shape = (z_dim, y_dim, x_dim, t_dim)

    # Background: low magnitude, near-zero phase
    xHat = rng.uniform(0.1, 0.3, shape).astype(np.float32)
    thetaX = rng.normal(0, 0.05, shape).astype(np.float32)
    thetaY = rng.normal(0, 0.05, shape).astype(np.float32)
    thetaZ = rng.normal(0, 0.05, shape).astype(np.float32)

    # Vessel cylinder (along Z at center)
    yy, xx = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing="ij")
    cyl = ((yy - 12) ** 2 + (xx - 12) ** 2) <= 9  # radius 3
    cyl_3d = np.broadcast_to(cyl[np.newaxis, :, :, np.newaxis], shape)

    xHat = np.where(cyl_3d, 1.0, xHat)
    # Fast Z flow inside vessel: ~0.8 m/s → phase ≈ 0.8 * π / 1.5 ≈ 1.67 rad
    thetaZ = np.where(cyl_3d, 1.67, thetaZ)

    return thetaX, thetaY, thetaZ, xHat


# ── Basic contract ────────────────────────────────────────────────────────────

def test_segment_returns_bool_3d_mask():
    tx, ty, tz, xH = _synthetic_4d_flow()
    mask = segment(tx, ty, tz, xH, VENC)
    assert mask.dtype == bool
    assert mask.ndim == 3
    assert mask.shape == tx.shape[:3]


def test_segment_finds_synthetic_vessel():
    """The cylinder we injected should be captured by the segmentation."""
    tx, ty, tz, xH = _synthetic_4d_flow()
    mask = segment(tx, ty, tz, xH, VENC)
    # The cylinder center at (any z, 12, 12) must be inside the mask
    assert mask[5, 12, 12]
    assert mask[15, 12, 12]


def test_pcmra_brightest_in_vessel():
    """PC-MRA should be highest where flow + signal overlap (the vessel)."""
    tx, ty, tz, xH = _synthetic_4d_flow()
    pcmra = compute_pcmra(tx, ty, tz, xH, VENC)
    # Center of vessel vs corner of volume
    assert pcmra[12, 12, 12] > pcmra[1, 1, 1] * 5


# ── Connected component behaviour ─────────────────────────────────────────────

def test_keeps_only_largest_connected_component():
    """Given two disconnected blobs, segment_largest_vessel keeps the larger one."""
    pcmra = np.zeros((20, 20, 20), dtype=np.float32)
    # Small blob (size ≈ 27 voxels)
    pcmra[2:5, 2:5, 2:5] = 10.0
    # Large blob (size ≈ 125 voxels)
    pcmra[10:15, 10:15, 10:15] = 10.0

    mask = segment_largest_vessel(pcmra, percentile=50, closing_iter=0)
    assert mask[12, 12, 12]      # inside large blob
    assert not mask[3, 3, 3]     # inside small blob — filtered out


def test_top_n_keeps_multiple_components():
    pcmra = np.zeros((20, 20, 20), dtype=np.float32)
    pcmra[2:5, 2:5, 2:5] = 10.0
    pcmra[10:15, 10:15, 10:15] = 10.0

    mask = segment_top_n_vessels(pcmra, n=2, percentile=50, closing_iter=0)
    assert mask[12, 12, 12]
    assert mask[3, 3, 3]


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_field_returns_empty_mask():
    """Zero-everything input should produce no mask (no connected components)."""
    shape = (10, 10, 10, 4)
    z = np.zeros(shape, dtype=np.float32)
    mask = segment(z, z, z, z, VENC)
    assert not mask.any()


def test_mask_shape_matches_spatial_dims():
    tx, ty, tz, xH = _synthetic_4d_flow(z_dim=15, y_dim=22, x_dim=18, t_dim=4)
    mask = segment(tx, ty, tz, xH, VENC)
    assert mask.shape == (15, 22, 18)


# ── Seed-based segmentation (agent-driven) ────────────────────────────────────

def test_segment_from_seed_returns_containing_component():
    """A seed inside the synthetic vessel returns the vessel mask."""
    tx, ty, tz, xH = _synthetic_4d_flow()
    mask = segment_from_seed(tx, ty, tz, xH, VENC, seed_point=(12, 12, 12))
    assert mask[12, 12, 12]
    # The whole cylinder column should be included
    assert mask[5, 12, 12]
    assert mask[20, 12, 12]


def test_segment_from_seed_background_returns_empty():
    """A seed in pure background returns an empty mask (agent retries)."""
    tx, ty, tz, xH = _synthetic_4d_flow()
    # (1, 1, 1) is corner background, far from the cylinder at (z, 12, 12)
    mask = segment_from_seed(tx, ty, tz, xH, VENC, seed_point=(1, 1, 1))
    assert not mask.any()


def test_segment_from_seed_isolates_one_component():
    """
    With two disconnected high-PC-MRA blobs, the seed selects only its own.
    """
    pcmra = np.zeros((20, 20, 20), dtype=np.float32)
    pcmra[2:5, 2:5, 2:5] = 10.0
    pcmra[10:15, 10:15, 10:15] = 10.0

    mask = segment_from_seed_pcmra(pcmra, seed_point=(3, 3, 3),
                                    percentile=50, closing_iter=0)
    assert mask[3, 3, 3]
    assert not mask[12, 12, 12]


def test_segment_from_seed_out_of_bounds_raises():
    tx, ty, tz, xH = _synthetic_4d_flow()
    with pytest.raises(ValueError):
        segment_from_seed(tx, ty, tz, xH, VENC, seed_point=(999, 0, 0))


# ── Seed suggestion (agent decision support) ──────────────────────────────────

def test_suggest_seed_points_returns_list_of_dicts():
    tx, ty, tz, xH = _synthetic_4d_flow()
    candidates = suggest_seed_points(tx, ty, tz, xH, VENC, n_candidates=3)
    assert isinstance(candidates, list)
    assert all("seed" in c and "size" in c and "mean_pcmra" in c for c in candidates)


def test_suggest_seed_points_finds_vessel():
    """The synthetic vessel should appear as a top candidate."""
    tx, ty, tz, xH = _synthetic_4d_flow()
    candidates = suggest_seed_points(tx, ty, tz, xH, VENC, n_candidates=3)
    # Top candidate's seed should fall inside the cylinder (y≈12, x≈12)
    z, y, x = candidates[0]["seed"]
    assert abs(y - 12) < 4 and abs(x - 12) < 4


def test_suggest_seed_points_sorted_by_brightness():
    """Candidates must be ordered by (size × mean_pcmra) descending."""
    tx, ty, tz, xH = _synthetic_4d_flow()
    candidates = suggest_seed_points(tx, ty, tz, xH, VENC, n_candidates=5)
    scores = [c["size"] * c["mean_pcmra"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_suggest_seed_points_respects_n():
    pcmra = np.zeros((20, 20, 20), dtype=np.float32)
    pcmra[2:5, 2:5, 2:5] = 10.0
    pcmra[10:15, 10:15, 10:15] = 10.0
    candidates = suggest_seed_points_pcmra(pcmra, n_candidates=1,
                                            percentile=50, closing_iter=0)
    assert len(candidates) == 1
