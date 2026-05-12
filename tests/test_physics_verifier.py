import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.physics_verifier import verify

VENC = 1.5  # m/s — typical aortic 4D flow
SHAPE = (20, 30, 25, 8)  # (Z, Y, X, T) — small but realistic proportions


def _phase_field(peak_velocity_ms: float, seed: int = 42) -> tuple:
    """Synthetic phase field (radians) for a given peak velocity."""
    rng = np.random.default_rng(seed)
    scale = peak_velocity_ms / VENC * np.pi  # peak phase
    tx = rng.uniform(-scale, scale, SHAPE).astype(np.float32)
    ty = rng.uniform(-scale * 0.3, scale * 0.3, SHAPE).astype(np.float32)
    tz = rng.uniform(-scale * 0.3, scale * 0.3, SHAPE).astype(np.float32)
    return tx, ty, tz


# ── Basic contract ────────────────────────────────────────────────────────────

def test_returns_verdict_key():
    tx, ty, tz = _phase_field(0.8)
    result = verify(tx, ty, tz, VENC)
    assert result["verdict"] in ("pass", "warn", "fail")


def test_all_four_checks_present():
    tx, ty, tz = _phase_field(0.8)
    result = verify(tx, ty, tz, VENC)
    for key in ("divergence", "net_flux", "peak_velocity", "phase_unwrap"):
        assert key in result["checks"]


def test_each_check_has_status():
    tx, ty, tz = _phase_field(0.8)
    result = verify(tx, ty, tz, VENC)
    for name, check in result["checks"].items():
        assert check["status"] in ("pass", "warn", "fail"), f"{name} missing valid status"


def test_metadata_fields():
    tx, ty, tz = _phase_field(0.8)
    result = verify(tx, ty, tz, VENC, voxel_size_mm=(2.0, 2.0, 2.0))
    meta = result["metadata"]
    assert meta["shape_ZYXT"] == list(SHAPE)
    assert meta["venc_m_per_s"] == VENC


# ── Mask handling ─────────────────────────────────────────────────────────────

def test_mask_voxel_count_in_metadata():
    tx, ty, tz = _phase_field(0.8)
    mask = np.zeros(SHAPE[:3], dtype=bool)
    mask[5:15, 10:20, 8:18] = True
    result = verify(tx, ty, tz, VENC, mask=mask)
    assert result["metadata"]["n_mask_voxels"] == int(mask.sum())


def test_auto_mask_generated_when_no_mask_provided():
    """verify() always auto-generates a velocity mask when none is supplied."""
    tx, ty, tz = _phase_field(0.8)
    result = verify(tx, ty, tz, VENC)
    assert result["metadata"]["n_mask_voxels"] is not None
    assert result["metadata"]["n_mask_voxels"] > 0


# ── Error injection: peak velocity ────────────────────────────────────────────

def test_injected_high_velocity_warns_or_fails():
    """Velocity well above WARN_HI (3.0 m/s) must not pass."""
    tx, ty, tz = _phase_field(peak_velocity_ms=4.8)  # exceeds FAIL_HI=4.5
    result = verify(tx, ty, tz, VENC)
    assert result["checks"]["peak_velocity"]["status"] in ("warn", "fail")


def test_physiological_velocity_passes():
    """Normal aortic peak (~0.8 m/s) should pass the velocity check."""
    tx, ty, tz = _phase_field(peak_velocity_ms=0.8)
    result = verify(tx, ty, tz, VENC)
    assert result["checks"]["peak_velocity"]["status"] == "pass"


# ── Error injection: divergence ───────────────────────────────────────────────

def test_injected_divergence_fails():
    """
    Synthetic field: vx = C * x_index, vy = vz = 0.
    Divergence = C / dx (s⁻¹). With C=0.05 m/s per voxel and dx=1mm → 50 s⁻¹ >> FAIL=20.
    """
    Z, Y, X, T = SHAPE
    venc_raw = VENC

    # Build phase arrays that encode a linearly ramping vx
    # vx = 0.05 * i (m/s) at voxel i → phase = vx * π / VENC
    vx_vals = np.arange(X, dtype=np.float32) * 0.05  # m/s
    phase_x = vx_vals * np.pi / venc_raw               # radians
    tx = np.broadcast_to(
        phase_x[np.newaxis, np.newaxis, :, np.newaxis], (Z, Y, X, T)
    ).copy()
    ty = np.zeros((Z, Y, X, T), dtype=np.float32)
    tz = np.zeros((Z, Y, X, T), dtype=np.float32)

    # dvx/dx = 0.05 m/s / 0.001 m = 50 s⁻¹  (voxel_size=1mm)
    result = verify(tx, ty, tz, venc_raw, voxel_size_mm=(1.0, 1.0, 1.0))
    assert result["checks"]["divergence"]["status"] == "fail"


def test_zero_velocity_field_low_divergence():
    """Zero velocity → zero divergence → should pass divergence check."""
    tx = np.zeros(SHAPE, dtype=np.float32)
    ty = np.zeros(SHAPE, dtype=np.float32)
    tz = np.zeros(SHAPE, dtype=np.float32)
    result = verify(tx, ty, tz, VENC)
    assert result["checks"]["divergence"]["status"] == "pass"


# ── Error injection: phase unwrap ─────────────────────────────────────────────

def test_injected_phase_wrap_fails():
    """
    Artificially insert phase jumps larger than VENC in many voxels.
    fraction_above_threshold should exceed FAIL_FRAC (0.01).
    """
    tx, ty, tz = _phase_field(0.5)
    # Inject wrap artifacts: set alternating slices to +π and −π in X
    tx[:, :, ::2, :] = np.pi        # max positive phase
    tx[:, :, 1::2, :] = -np.pi      # max negative phase  → jumps of 2π → 2*VENC in velocity
    result = verify(tx, ty, tz, VENC)
    assert result["checks"]["phase_unwrap"]["status"] in ("warn", "fail")


def test_smooth_field_passes_phase_unwrap():
    """Smoothly varying field should have no large spatial jumps."""
    Z, Y, X, T = SHAPE
    # vx smoothly varies 0 → 0.3 m/s across X
    vx_phase = np.linspace(0, 0.3 * np.pi / VENC, X, dtype=np.float32)
    tx = np.broadcast_to(
        vx_phase[np.newaxis, np.newaxis, :, np.newaxis], (Z, Y, X, T)
    ).copy()
    ty = np.zeros((Z, Y, X, T), dtype=np.float32)
    tz = np.zeros((Z, Y, X, T), dtype=np.float32)
    result = verify(tx, ty, tz, VENC)
    assert result["checks"]["phase_unwrap"]["status"] == "pass"


# ── Overall verdict aggregation ───────────────────────────────────────────────

def test_overall_verdict_is_worst_check():
    """If any check fails, overall verdict must be fail."""
    tx, ty, tz = _phase_field(4.8)  # triggers peak_velocity fail
    result = verify(tx, ty, tz, VENC)
    assert result["verdict"] == "fail"
