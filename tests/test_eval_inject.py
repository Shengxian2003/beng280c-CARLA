"""Tests for the Stage 4 error-injection harness.

Core contract:
  1. The clean phantom passes all four verifier checks (baseline).
  2. Each injector at sufficient magnitude flips its TARGET check from
     pass → warn or fail, without falsely flipping unrelated checks.
  3. Injectors with magnitude=0 are identity functions (controls).

If these tests fail, the detection-rate sweep numbers will be meaningless.
"""
from __future__ import annotations

import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.eval_inject import (
    clean_flow_phantom,
    velocity_to_phase,
    inject_divergence,
    inject_flux_imbalance,
    inject_peak_velocity,
    inject_phase_wrap,
)
from skills.physics_verifier import verify


# ============================================================================
# Helpers
# ============================================================================

def _verify(phantom, thetaX=None, thetaY=None, thetaZ=None):
    """Run the verifier on the phantom (or override phase arrays)."""
    return verify(
        thetaX if thetaX is not None else phantom["thetaX"],
        thetaY if thetaY is not None else phantom["thetaY"],
        thetaZ if thetaZ is not None else phantom["thetaZ"],
        venc_m_per_s=phantom["venc_m_per_s"],
        mask=phantom["mask"],
        voxel_size_mm=phantom["voxel_size_mm"],
    )


# ============================================================================
# 1. The clean phantom passes everything (baseline)
# ============================================================================

class TestCleanPhantomBaseline:
    def test_clean_phantom_passes_all_checks(self):
        ph = clean_flow_phantom(velocity_m_per_s=0.5)
        verdict = _verify(ph)
        assert verdict["verdict"] == "pass", (
            f"clean phantom should pass; got {verdict['verdict']}. "
            f"checks: {[(n, c['status']) for n, c in verdict['checks'].items()]}"
        )

    def test_clean_phantom_mask_is_reasonable(self):
        ph = clean_flow_phantom()
        n_vox = int(ph["mask"].sum())
        assert n_vox > 5000, f"mask too small for verifier: {n_vox} voxels"

    def test_velocity_phase_round_trip(self):
        ph = clean_flow_phantom()
        tx, ty, tz = velocity_to_phase(ph["vx"], ph["vy"], ph["vz"], ph["venc_m_per_s"])
        np.testing.assert_allclose(tx, ph["thetaX"])
        np.testing.assert_allclose(ty, ph["thetaY"])
        np.testing.assert_allclose(tz, ph["thetaZ"])


# ============================================================================
# 2a. Divergence injector
# ============================================================================

class TestDivergenceInjector:
    def test_zero_magnitude_is_identity(self):
        ph = clean_flow_phantom()
        vx2, vy2, vz2 = inject_divergence(
            ph["vx"], ph["vy"], ph["vz"], ph["mask"], magnitude_per_s=0,
        )
        np.testing.assert_allclose(vx2, ph["vx"])
        np.testing.assert_allclose(vy2, ph["vy"])
        np.testing.assert_allclose(vz2, ph["vz"])

    def test_large_magnitude_triggers_divergence_fail(self):
        ph = clean_flow_phantom()
        vx2, vy2, vz2 = inject_divergence(
            ph["vx"], ph["vy"], ph["vz"], ph["mask"],
            voxel_size_mm=ph["voxel_size_mm"], magnitude_per_s=50.0,
        )
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, ph["venc_m_per_s"])
        v = _verify(ph, thetaX=tx, thetaY=ty, thetaZ=tz)
        assert v["checks"]["divergence"]["status"] == "fail"

    def test_small_magnitude_triggers_warn_but_not_fail(self):
        ph = clean_flow_phantom()
        vx2, vy2, vz2 = inject_divergence(
            ph["vx"], ph["vy"], ph["vz"], ph["mask"],
            voxel_size_mm=ph["voxel_size_mm"], magnitude_per_s=10.0,
        )
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, ph["venc_m_per_s"])
        v = _verify(ph, thetaX=tx, thetaY=ty, thetaZ=tz)
        # 10 s⁻¹ is between warn (5) and fail (20)
        assert v["checks"]["divergence"]["status"] == "warn"


# ============================================================================
# 2b. Flux imbalance injector
# ============================================================================

class TestFluxImbalanceInjector:
    def test_zero_is_identity(self):
        ph = clean_flow_phantom()
        out = inject_flux_imbalance(ph["vx"], ph["vy"], ph["vz"], ph["mask"],
                                     flow_axis="Z", fractional_imbalance=0)
        for orig, got in zip([ph["vx"], ph["vy"], ph["vz"]], out):
            np.testing.assert_allclose(orig, got)

    def test_large_imbalance_triggers_flux_fail(self):
        ph = clean_flow_phantom()
        vx2, vy2, vz2 = inject_flux_imbalance(
            ph["vx"], ph["vy"], ph["vz"], ph["mask"],
            flow_axis="Z", fractional_imbalance=1.5,
        )
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, ph["venc_m_per_s"])
        v = _verify(ph, thetaX=tx, thetaY=ty, thetaZ=tz)
        assert v["checks"]["net_flux"]["status"] == "fail", \
            f'got max_dev={v["checks"]["net_flux"]["max_deviation_pct"]}%'


# ============================================================================
# 2c. Peak velocity injector
# ============================================================================

class TestPeakVelocityInjector:
    def test_zero_is_identity(self):
        ph = clean_flow_phantom()
        out = inject_peak_velocity(ph["vx"], ph["vy"], ph["vz"], ph["mask"],
                                    target_peak_m_per_s=0)
        for orig, got in zip([ph["vx"], ph["vy"], ph["vz"]], out):
            np.testing.assert_allclose(orig, got)

    def test_high_peak_triggers_fail(self):
        ph = clean_flow_phantom()
        vx2, vy2, vz2 = inject_peak_velocity(
            ph["vx"], ph["vy"], ph["vz"], ph["mask"],
            target_peak_m_per_s=6.0,    # > 4.5 fail threshold
            seed=0,
        )
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, ph["venc_m_per_s"])
        v = _verify(ph, thetaX=tx, thetaY=ty, thetaZ=tz)
        assert v["checks"]["peak_velocity"]["status"] == "fail"

    def test_moderately_high_triggers_warn(self):
        ph = clean_flow_phantom()
        vx2, vy2, vz2 = inject_peak_velocity(
            ph["vx"], ph["vy"], ph["vz"], ph["mask"],
            target_peak_m_per_s=3.5,    # between warn (3.0) and fail (4.5)
            seed=0,
        )
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, ph["venc_m_per_s"])
        v = _verify(ph, thetaX=tx, thetaY=ty, thetaZ=tz)
        assert v["checks"]["peak_velocity"]["status"] == "warn"


# ============================================================================
# 2d. Phase wrap injector
# ============================================================================

class TestPhaseWrapInjector:
    def test_zero_is_identity(self):
        ph = clean_flow_phantom()
        tx, ty, tz = inject_phase_wrap(
            ph["thetaX"], ph["thetaY"], ph["thetaZ"], ph["mask"],
            fraction=0, seed=0,
        )
        np.testing.assert_allclose(tx, ph["thetaX"])
        np.testing.assert_allclose(ty, ph["thetaY"])
        np.testing.assert_allclose(tz, ph["thetaZ"])

    def test_many_wraps_trigger_fail(self):
        # Verifier counts (jump > VENC) fraction across ALL voxel-pair diffs
        # in 3 components × 3 axes ≈ 1.7M total comparisons per check. Need
        # quite aggressive wrap fraction to clear the 1% fail threshold.
        ph = clean_flow_phantom()
        tx, ty, tz = inject_phase_wrap(
            ph["thetaX"], ph["thetaY"], ph["thetaZ"], ph["mask"],
            fraction=0.5, seed=0,
        )
        v = _verify(ph, thetaX=tx, thetaY=ty, thetaZ=tz)
        assert v["checks"]["phase_unwrap"]["status"] == "fail", \
            f'got fraction_above={v["checks"]["phase_unwrap"]["fraction_above_threshold"]}'


# ============================================================================
# 3. Selectivity — each injector should NOT spuriously fail unrelated checks
# ============================================================================

class TestSelectivity:
    """A divergence injection should not, e.g., make peak_velocity fail."""

    def test_divergence_injection_does_not_break_phase_unwrap(self):
        ph = clean_flow_phantom()
        vx2, vy2, vz2 = inject_divergence(
            ph["vx"], ph["vy"], ph["vz"], ph["mask"],
            voxel_size_mm=ph["voxel_size_mm"], magnitude_per_s=10.0,
        )
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, ph["venc_m_per_s"])
        v = _verify(ph, thetaX=tx, thetaY=ty, thetaZ=tz)
        # Divergence injection is smooth → no phase jumps → unwrap should still pass
        assert v["checks"]["phase_unwrap"]["status"] == "pass"

    def test_phase_wrap_injection_does_not_break_net_flux_drastically(self):
        ph = clean_flow_phantom()
        tx, ty, tz = inject_phase_wrap(
            ph["thetaX"], ph["thetaY"], ph["thetaZ"], ph["mask"],
            fraction=0.02, seed=0,
        )
        v = _verify(ph, thetaX=tx, thetaY=ty, thetaZ=tz)
        # Flux check should still pass (or at worst warn) — phase wraps are
        # local artifacts, they shouldn't change bulk flux conservation much
        assert v["checks"]["net_flux"]["status"] in ("pass", "warn")
