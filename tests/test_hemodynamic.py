"""Unit tests for the Hemodynamic Analysis Skill (Stage 2d).

Tests use synthetic velocity fields with known ground-truth flow values to
verify the metric calculations. All quantities are unit-checked end-to-end:
phase (rad) → velocity (m/s) → flow (mL/s) → stroke volume (mL).
"""
from __future__ import annotations

import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.hemodynamic import analyze
from skills.hemodynamic._metrics import (
    dominant_flow_axis,
    flow_rate_time_series,
    stroke_volume_and_peak,
    peak_velocity_in_mask,
)


# ----- Helpers ---------------------------------------------------------------

VENC = 1.5  # m/s


def make_constant_flow_phantom(Z=20, Y=10, X=10, T=8,
                               velocity_m_per_s=0.5,
                               flow_axis="Z",
                               vessel_radius_voxels=3):
    """Build a synthetic vessel with constant velocity along ``flow_axis``.

    Returns (thetaX, thetaY, thetaZ, mask).
    """
    shape = (Z, Y, X, T)
    tx = np.zeros(shape, dtype=np.float64)
    ty = np.zeros(shape, dtype=np.float64)
    tz = np.zeros(shape, dtype=np.float64)

    # Cylindrical mask centred in the transverse plane
    yy, xx = np.meshgrid(
        np.arange(Y) - Y / 2,
        np.arange(X) - X / 2,
        indexing="ij",
    )
    cross_section = (yy ** 2 + xx ** 2) <= vessel_radius_voxels ** 2  # (Y, X)
    mask = np.broadcast_to(cross_section, (Z, Y, X)).copy()

    # Constant velocity along flow_axis inside the mask
    phase = velocity_m_per_s * np.pi / VENC   # invert v = phase·VENC/π
    target = {"X": tx, "Y": ty, "Z": tz}[flow_axis]
    target[mask] = phase

    return tx, ty, tz, mask


def make_pulsatile_flow(Z=20, Y=10, X=10, T=20, peak_v=0.8, vessel_radius=3):
    """Vessel with sinusoidal forward flow along Z (single forward half-cycle)."""
    tx, ty, tz, mask = make_constant_flow_phantom(
        Z, Y, X, T, velocity_m_per_s=0, flow_axis="Z", vessel_radius_voxels=vessel_radius
    )
    # half-cosine waveform: 0 → peak → 0 over T phases
    waveform = peak_v * np.sin(np.linspace(0, np.pi, T))
    phase_wave = waveform * np.pi / VENC
    for t in range(T):
        tz[..., t][mask] = phase_wave[t]
    return tx, ty, tz, mask


# ----- dominant_flow_axis ---------------------------------------------------

class TestDominantAxis:
    def test_picks_z(self):
        tx, ty, tz, mask = make_constant_flow_phantom(flow_axis="Z")
        vx, vy, vz = tx * VENC / np.pi, ty * VENC / np.pi, tz * VENC / np.pi
        assert dominant_flow_axis(vx, vy, vz, mask) == 0

    def test_picks_y(self):
        tx, ty, tz, mask = make_constant_flow_phantom(flow_axis="Y")
        vx, vy, vz = tx * VENC / np.pi, ty * VENC / np.pi, tz * VENC / np.pi
        assert dominant_flow_axis(vx, vy, vz, mask) == 1

    def test_picks_x(self):
        tx, ty, tz, mask = make_constant_flow_phantom(flow_axis="X")
        vx, vy, vz = tx * VENC / np.pi, ty * VENC / np.pi, tz * VENC / np.pi
        assert dominant_flow_axis(vx, vy, vz, mask) == 2


# ----- flow_rate_time_series ------------------------------------------------

class TestFlowRate:
    def test_constant_flow_value(self):
        """v=0.5 m/s, vessel cross-section area = 29 voxels * (2×2) mm² each.
        Q should equal v * area, constant in time, with no fluctuation."""
        v = 0.5
        tx, ty, tz, mask = make_constant_flow_phantom(
            velocity_m_per_s=v, flow_axis="Z", vessel_radius_voxels=3
        )
        vx, vy, vz = tx * VENC / np.pi, ty * VENC / np.pi, tz * VENC / np.pi

        # Pick middle Z slice for the cross-section
        Q = flow_rate_time_series(vx, vy, vz, mask, voxel_size_mm=(2, 2, 2),
                                  axis=0, plane_index=10)
        # Constant in time
        assert np.allclose(Q, Q[0], atol=1e-9)
        # Expected: n_vessel_voxels * v * area_per_voxel
        n_vox = int(mask[10].sum())
        area_m2 = (2e-3) * (2e-3)
        expected_mL_per_s = n_vox * v * area_m2 * 1e6
        assert np.isclose(Q[0], expected_mL_per_s, rtol=1e-6)

    def test_zero_outside_vessel(self):
        # If the plane has no mask voxels, flow must be zero
        tx, ty, tz, mask = make_constant_flow_phantom(flow_axis="Z")
        empty_mask = np.zeros_like(mask)
        vx, vy, vz = tx * VENC / np.pi, ty * VENC / np.pi, tz * VENC / np.pi
        Q = flow_rate_time_series(vx, vy, vz, empty_mask, (2, 2, 2), axis=0, plane_index=10)
        assert np.allclose(Q, 0)

    def test_units_mL_per_s(self):
        """Sanity check on units: 1 m/s through 1 mm² → 1 mL/s.

        Working in SI: 1 m/s · (1e-3 m)² = 1e-6 m³/s = 1 mL/s
        (since 1 m³ = 1e6 mL).
        """
        Z, Y, X, T = 5, 5, 5, 2
        vz = np.ones((Z, Y, X, T))
        vx = np.zeros_like(vz)
        vy = np.zeros_like(vz)
        mask = np.zeros((Z, Y, X), bool)
        mask[2, 2, 2] = True
        Q = flow_rate_time_series(vx, vy, vz, mask, voxel_size_mm=(1, 1, 1),
                                  axis=0, plane_index=2)
        assert np.allclose(Q, 1.0, rtol=1e-9)


# ----- stroke_volume_and_peak -----------------------------------------------

class TestStrokeVolume:
    def test_constant_positive_flow(self):
        # Q = 5 mL/s for 1 second → SV = 5 mL
        Q = np.full(10, 5.0)
        dt = 0.1
        d = stroke_volume_and_peak(Q, dt)
        assert d["stroke_volume_forward_mL"] == pytest.approx(5.0)
        assert d["stroke_volume_net_mL"]     == pytest.approx(5.0)
        assert d["peak_Q_mL_per_s"]          == 5.0
        assert d["regurgitant_fraction_pct"] == 0.0

    def test_sinusoidal_forward_only(self):
        # Half-sine, peak=10. Sample odd T so peak lands exactly on π/2.
        T, dt = 21, 0.05
        Q = 10 * np.sin(np.linspace(0, np.pi, T))
        d = stroke_volume_and_peak(Q, dt)
        # Result is rounded to 3 decimals in the output dict
        expected = float(Q.sum() * dt)
        assert d["stroke_volume_forward_mL"] == pytest.approx(expected, abs=5e-4)
        assert d["regurgitant_fraction_pct"] == 0.0
        assert d["peak_Q_mL_per_s"] == pytest.approx(10, rel=1e-3)

    def test_regurgitation(self):
        # Forward 8 mL/s for 4 steps, then backward -2 mL/s for 4 steps
        Q = np.array([8, 8, 8, 8, -2, -2, -2, -2], dtype=float)
        dt = 0.1
        d = stroke_volume_and_peak(Q, dt)
        # Forward = 8 * 4 * 0.1 = 3.2 mL ; Backward = 2 * 4 * 0.1 = 0.8 mL
        assert d["stroke_volume_forward_mL"] == pytest.approx(3.2)
        assert d["regurgitant_fraction_pct"] == pytest.approx(25.0)

    def test_empty_series(self):
        d = stroke_volume_and_peak(np.array([]), 0.05)
        assert d["stroke_volume_forward_mL"] == 0.0
        assert d["peak_Q_mL_per_s"] == 0.0

    def test_sign_of_peak_defines_forward(self):
        # Predominantly negative Q → forward direction is negative
        Q = np.array([-5.0, -5.0, 1.0, 1.0])
        d = stroke_volume_and_peak(Q, 1.0)
        # Forward = |-5|*2 = 10; backward = |1|*2 = 2
        assert d["stroke_volume_forward_mL"] == pytest.approx(10.0)
        assert d["regurgitant_fraction_pct"] == pytest.approx(20.0)


# ----- peak_velocity_in_mask -----------------------------------------------

class TestPeakVelocity:
    def test_matches_constant_input(self):
        tx, ty, tz, mask = make_constant_flow_phantom(velocity_m_per_s=0.7, flow_axis="Z")
        vx, vy, vz = tx * VENC / np.pi, ty * VENC / np.pi, tz * VENC / np.pi
        d = peak_velocity_in_mask(vx, vy, vz, mask)
        assert d["peak_m_per_s"] == pytest.approx(0.7, rel=1e-6)
        assert d["mean_m_per_s"] == pytest.approx(0.7, rel=1e-6)

    def test_empty_mask_zero(self):
        z = np.zeros((4, 4, 4, 2))
        m = np.zeros((4, 4, 4), bool)
        d = peak_velocity_in_mask(z, z, z, m)
        assert d["peak_m_per_s"] == 0.0


# ----- analyze() end-to-end -------------------------------------------------

class TestAnalyzeEndToEnd:
    def test_constant_flow_phantom(self):
        v = 0.5
        tx, ty, tz, mask = make_constant_flow_phantom(velocity_m_per_s=v, flow_axis="Z")
        result = analyze(tx, ty, tz, venc_m_per_s=VENC, mask=mask,
                         voxel_size_mm=(2, 2, 2), dt_seconds=0.05)

        assert result["metadata"]["dominant_axis"] == "Z"
        assert result["metadata"]["n_mask_voxels"] == int(mask.sum())
        # All sections see the same constant flow → all peak_Q identical
        peak_qs = [s["peak_Q_mL_per_s"] for s in result["per_section"]]
        assert max(peak_qs) - min(peak_qs) < 1e-6
        # Peak velocity matches input
        assert result["summary"]["peak_velocity_m_per_s"] == pytest.approx(v, rel=1e-6)

    def test_pulsatile_stroke_volume(self):
        # Odd T so a sample lands exactly at π/2 — otherwise discrete max underestimates
        T = 21
        peak_v = 0.8
        tx, ty, tz, mask = make_pulsatile_flow(T=T, peak_v=peak_v)
        result = analyze(tx, ty, tz, venc_m_per_s=VENC, mask=mask,
                         voxel_size_mm=(2, 2, 2), dt_seconds=0.05)

        # No backflow in pure half-sine
        for sec in result["per_section"]:
            assert sec["regurgitant_fraction_pct"] == 0.0

        # Peak velocity should match the input peak
        assert result["summary"]["peak_velocity_m_per_s"] == pytest.approx(peak_v, rel=1e-3)

    def test_rejects_empty_mask(self):
        tx, ty, tz, _ = make_constant_flow_phantom(flow_axis="Z")
        empty = np.zeros(tx.shape[:3], bool)
        with pytest.raises(ValueError, match="non-empty vessel mask"):
            analyze(tx, ty, tz, venc_m_per_s=VENC, mask=empty)

    def test_metadata_completeness(self):
        tx, ty, tz, mask = make_constant_flow_phantom(flow_axis="Z")
        result = analyze(tx, ty, tz, venc_m_per_s=VENC, mask=mask,
                         dt_seconds=0.05, voxel_size_mm=(2, 2, 2))
        m = result["metadata"]
        for key in ("dominant_axis", "n_cross_sections", "n_phases", "dt_s",
                    "cycle_duration_s", "voxel_size_mm", "voxel_volume_mm3",
                    "n_mask_voxels", "venc_m_per_s"):
            assert key in m, f"missing metadata key: {key}"

    def test_per_section_shape(self):
        tx, ty, tz, mask = make_constant_flow_phantom(flow_axis="Z", T=8)
        result = analyze(tx, ty, tz, venc_m_per_s=VENC, mask=mask,
                         n_cross_sections=5)
        assert len(result["per_section"]) == 5
        for sec in result["per_section"]:
            # Q time series length must equal number of phases
            assert len(sec["Q_mL_per_s"]) == 8

    def test_n_cross_sections_param(self):
        tx, ty, tz, mask = make_constant_flow_phantom(flow_axis="Z")
        for n in (3, 5, 7):
            r = analyze(tx, ty, tz, venc_m_per_s=VENC, mask=mask, n_cross_sections=n)
            assert len(r["per_section"]) == n
