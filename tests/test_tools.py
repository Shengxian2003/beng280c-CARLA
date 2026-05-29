"""Unit tests for the agent tool layer (Stage 3b)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utility.tools import (
    Workspace,
    ToolError,
    ToolSpec,
    TOOLS,
    TOOLS_BY_NAME,
    call_tool,
    tools_prompt_block,
    to_json_safe,
    _validate_args,
)


# ============================================================================
# Helpers — synthetic 4D flow phantom usable by every tool
# ============================================================================

def _synthetic_recon(Z=12, Y=10, X=10, T=4, peak_v=0.8):
    """Build a workspace with a small synthetic vessel along Z."""
    # Phase (rad), background zero, vessel = constant flow along Z
    tx = np.zeros((Z, Y, X, T), dtype=np.float64)
    ty = np.zeros((Z, Y, X, T), dtype=np.float64)
    tz = np.zeros((Z, Y, X, T), dtype=np.float64)
    xHat = np.full((Z, Y, X, T), 0.1, dtype=np.complex128)

    # Cylindrical vessel
    yy, xx = np.meshgrid(np.arange(Y) - Y / 2, np.arange(X) - X / 2, indexing="ij")
    cross = (yy ** 2 + xx ** 2) <= 2.5 ** 2
    vessel = np.broadcast_to(cross, (Z, Y, X))

    # Constant velocity inside vessel; bright signal so PC-MRA picks it up
    venc = 1.5
    phase = peak_v * np.pi / venc
    for z in range(Z):
        tz[z][cross] = phase
        xHat[z][cross] = 1.0
    return {"xHat": xHat, "thetaX": tx, "thetaY": ty, "thetaZ": tz}, vessel.copy(), venc


def _ws_with_recon():
    recon, _, venc = _synthetic_recon()
    ws = Workspace()
    ws.recon = recon
    ws.venc_m_per_s = venc
    ws.voxel_size_mm = (2.0, 2.0, 2.0)
    return ws


# ============================================================================
# Validation
# ============================================================================

class TestValidator:
    schema = {
        "type": "object",
        "properties": {
            "name":  {"type": "string"},
            "n":     {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            "mode":  {"type": "string", "enum": ["a", "b"]},
            "flag":  {"type": "boolean", "default": False},
        },
        "required": ["name"],
    }

    def test_applies_defaults(self):
        out = _validate_args({"name": "x"}, self.schema)
        assert out == {"name": "x", "n": 5, "flag": False}

    def test_explicit_overrides_default(self):
        out = _validate_args({"name": "x", "n": 7}, self.schema)
        assert out["n"] == 7

    def test_missing_required(self):
        with pytest.raises(ToolError, match="missing required"):
            _validate_args({"n": 5}, self.schema)

    def test_unknown_key(self):
        with pytest.raises(ToolError, match="unknown argument"):
            _validate_args({"name": "x", "bogus": 1}, self.schema)

    def test_bad_type(self):
        with pytest.raises(ToolError, match="expected integer, got str"):
            _validate_args({"name": "x", "n": "five"}, self.schema)

    def test_bool_is_not_integer(self):
        # Catch the Python quirk: True is technically isinstance(int) but reject it
        with pytest.raises(ToolError, match="bool"):
            _validate_args({"name": "x", "n": True}, self.schema)

    def test_below_minimum(self):
        with pytest.raises(ToolError, match="below minimum"):
            _validate_args({"name": "x", "n": 0}, self.schema)

    def test_above_maximum(self):
        with pytest.raises(ToolError, match="above maximum"):
            _validate_args({"name": "x", "n": 99}, self.schema)

    def test_enum_violation(self):
        with pytest.raises(ToolError, match="must be one of"):
            _validate_args({"name": "x", "mode": "c"}, self.schema)


# ============================================================================
# Sanitizer
# ============================================================================

class TestSanitize:
    def test_numpy_scalars(self):
        out = to_json_safe({
            "i": np.int64(7), "f": np.float32(1.5), "b": np.bool_(True),
        })
        assert out == {"i": 7, "f": 1.5, "b": True}
        # Must be JSON-encodable
        json.dumps(out)

    def test_small_array_inlined(self):
        a = np.array([1.0, 2.0, 3.0])
        assert to_json_safe(a) == [1.0, 2.0, 3.0]

    def test_large_array_summarized(self):
        a = np.zeros((10, 10))  # 100 > 64 inline limit
        out = to_json_safe(a)
        assert isinstance(out, str)
        assert "ndarray" in out and "shape=[10, 10]" in out

    def test_nan_becomes_null(self):
        out = to_json_safe({"v": np.float64("nan")})
        assert out == {"v": None}

    def test_nested(self):
        out = to_json_safe({"meta": {"shape": np.array([1, 2, 3]),
                                     "items": [np.int32(1), np.int32(2)]}})
        assert out == {"meta": {"shape": [1, 2, 3], "items": [1, 2]}}

    def test_complex_round_trip(self):
        out = to_json_safe(np.complex128(1 + 2j))
        assert out == {"real": 1.0, "imag": 2.0}


# ============================================================================
# Tool registry sanity
# ============================================================================

class TestRegistry:
    def test_unique_names(self):
        names = [t.name for t in TOOLS]
        assert len(names) == len(set(names)), f"duplicate tool names: {names}"

    def test_each_tool_has_object_schema(self):
        for t in TOOLS:
            assert t.parameters.get("type") == "object", f"{t.name} schema not object"
            assert "properties" in t.parameters

    def test_each_tool_has_description(self):
        for t in TOOLS:
            assert t.description.strip(), f"{t.name} has empty description"

    def test_prompt_block_renders(self):
        block = tools_prompt_block()
        for t in TOOLS:
            assert f"### {t.name}" in block
        # And it's valid for prompting (no exception)
        assert "## Tools available" in block


# ============================================================================
# Dispatcher
# ============================================================================

class TestDispatcher:
    def test_unknown_tool_returns_error_dict(self):
        ws = Workspace()
        out = call_tool(ws, "bogus_tool", {})
        assert out["error_type"] == "UnknownTool"
        assert "known_tools" in out

    def test_missing_required_arg(self):
        ws = Workspace()
        out = call_tool(ws, "verify", {})  # missing mask_name
        assert out["error_type"] == "ToolError"
        assert "mask_name" in out["error"]

    def test_function_exception_wrapped(self):
        ws = Workspace()
        # No recon loaded → require_recon raises ToolError
        out = call_tool(ws, "suggest_seeds", {})
        assert out["error_type"] == "ToolError"
        assert "reconstruction" in out["error"].lower()


# ============================================================================
# Tools — exercised with synthetic recon (no MATLAB, no real data needed)
# ============================================================================

class TestToolsWithSyntheticRecon:
    def test_load_reconstruction(self, tmp_path):
        import scipy.io as sio
        recon, _, _ = _synthetic_recon()
        mat = tmp_path / "synthetic_recon.mat"
        sio.savemat(mat, {"outputs": {
            "xHat":   recon["xHat"],
            "thetaX": recon["thetaX"],
            "thetaY": recon["thetaY"],
            "thetaZ": recon["thetaZ"],
        }})

        ws = Workspace()
        out = call_tool(ws, "load_reconstruction", {"mat_path": str(mat)})
        assert out["status"] == "loaded"
        assert out["shape_ZYXT"] == list(recon["xHat"].shape)
        assert ws.recon is not None

    def test_load_reconstruction_bad_file(self, tmp_path):
        import scipy.io as sio
        bad = tmp_path / "no_outputs.mat"
        sio.savemat(bad, {"something_else": np.zeros(3)})
        ws = Workspace()
        out = call_tool(ws, "load_reconstruction", {"mat_path": str(bad)})
        assert out["error_type"] == "ToolError"
        assert "outputs" in out["error"]

    def test_suggest_seeds_then_segment_then_verify(self):
        ws = _ws_with_recon()

        # 1) Suggest seeds — lower percentile because the phantom is tiny
        # and high-percentile + closing leaves too few connected voxels
        suggestions = call_tool(ws, "suggest_seeds",
                                {"n_candidates": 3, "percentile": 70.0})
        assert suggestions["n_returned"] >= 1
        c0 = suggestions["candidates"][0]
        assert {"index", "seed_zyx", "size_voxels", "mean_pcmra", "bbox_dims"} <= set(c0)

        # 2) Segment from that seed
        z, y, x = c0["seed_zyx"]
        seg = call_tool(ws, "segment_from_seed", {
            "seed_z": z, "seed_y": y, "seed_x": x, "mask_name": "vessel_1",
            "percentile": 70.0,
        })
        assert seg["status"] == "ok"
        assert seg["size_voxels"] > 0
        assert "vessel_1" in ws.masks

        # 3) Verify
        verdict = call_tool(ws, "verify", {"mask_name": "vessel_1"})
        assert verdict["mask_name"] == "vessel_1"
        assert verdict["verdict"] in ("pass", "warn", "fail")
        # JSON-safe (no numpy types should remain)
        json.dumps(verdict)

    def test_segment_empty_mask_returns_warning(self):
        ws = _ws_with_recon()
        # A seed deep in background should produce an empty mask
        out = call_tool(ws, "segment_from_seed", {
            "seed_z": 0, "seed_y": 0, "seed_x": 0, "mask_name": "should_be_empty",
        })
        assert out["status"] == "empty_mask"
        assert "should_be_empty" not in ws.masks  # rejected before storing

    def test_verify_unknown_mask(self):
        ws = _ws_with_recon()
        out = call_tool(ws, "verify", {"mask_name": "never_segmented"})
        assert out["error_type"] == "ToolError"
        assert "never_segmented" in out["error"]

    def test_analyze_full_pipeline(self):
        ws = _ws_with_recon()
        # Plant a mask manually so we don't depend on segmentation finding it
        ws.masks["test_vessel"] = (np.abs(ws.recon["thetaZ"][..., 0]) > 0.1)
        out = call_tool(ws, "analyze", {"mask_name": "test_vessel"})
        assert out["mask_name"] == "test_vessel"
        assert "per_section" in out and len(out["per_section"]) == 5
        assert "summary" in out
        json.dumps(out)  # fully JSON-safe

    def test_full_chain_load_to_analyze(self, tmp_path):
        """End-to-end: simulate what the agent will do — load, suggest, segment, verify, analyze."""
        import scipy.io as sio
        recon, _, _ = _synthetic_recon()
        mat = tmp_path / "e2e.mat"
        sio.savemat(mat, {"outputs": {
            "xHat": recon["xHat"], "thetaX": recon["thetaX"],
            "thetaY": recon["thetaY"], "thetaZ": recon["thetaZ"],
        }})
        ws = Workspace()

        assert call_tool(ws, "load_reconstruction",
                         {"mat_path": str(mat)})["status"] == "loaded"

        seeds = call_tool(ws, "suggest_seeds", {"n_candidates": 3, "percentile": 70.0})
        assert seeds["n_returned"] >= 1
        s = seeds["candidates"][0]

        seg = call_tool(ws, "segment_from_seed", {
            "seed_z": s["seed_zyx"][0],
            "seed_y": s["seed_zyx"][1],
            "seed_x": s["seed_zyx"][2],
            "mask_name": "aorta_v1",
            "percentile": 70.0,
        })
        assert seg["status"] == "ok"

        verdict = call_tool(ws, "verify", {"mask_name": "aorta_v1"})
        assert verdict["verdict"] in ("pass", "warn", "fail")

        report = call_tool(ws, "analyze", {"mask_name": "aorta_v1"})
        assert "summary" in report

    def test_argument_with_wrong_type_caught(self):
        ws = _ws_with_recon()
        out = call_tool(ws, "segment_from_seed", {
            "seed_z": "five",  # should be integer
            "seed_y": 5, "seed_x": 5, "mask_name": "m",
        })
        assert out["error_type"] == "ToolError"
        assert "seed_z" in out["error"]
