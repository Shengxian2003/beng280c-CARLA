"""Tests for the per-agent viewer window (demos/agent_window.py)."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demos.agent_window import (
    AGENT_ROUTES,
    AGENT_STYLE,
    render_entry,
    tail_log,
    _short_result,
)
from rich.console import Console


# ============================================================================
# Routing rules — which entries each agent sees
# ============================================================================

class TestRouting:
    def test_all_agents_have_styles(self):
        assert set(AGENT_ROUTES) == set(AGENT_STYLE)

    def test_planner_sees_only_planner_llm_calls(self):
        planner = AGENT_ROUTES["planner"]
        assert planner({"kind": "llm_call", "data": {"purpose": "planner"}})
        assert not planner({"kind": "llm_call", "data": {"purpose": "coordinator"}})
        assert not planner({"kind": "tool_call", "data": {"name": "verify"}})

    def test_coordinator_sees_its_llm_calls_and_events(self):
        coord = AGENT_ROUTES["coordinator"]
        assert coord({"kind": "llm_call", "data": {"purpose": "coordinator"}})
        assert not coord({"kind": "llm_call", "data": {"purpose": "planner"}})
        # Coordinator no longer sees tool calls — those route to the specialist windows
        assert not coord({"kind": "tool_call", "data": {"name": "verify"}})
        assert coord({"kind": "event", "data": {"name": "delegation"}})

    def test_verifier_sees_its_llm_calls_and_verify_tool(self):
        v = AGENT_ROUTES["verifier"]
        assert v({"kind": "llm_call", "data": {"purpose": "specialist.verifier"}})
        assert v({"kind": "tool_call", "data": {"name": "verify"}})
        assert not v({"kind": "tool_call", "data": {"name": "analyze"}})
        assert not v({"kind": "llm_call", "data": {"purpose": "specialist.hemodynamic"}})

    def test_hemodynamic_sees_its_llm_calls_and_analyze_tool(self):
        h = AGENT_ROUTES["hemodynamic"]
        assert h({"kind": "llm_call", "data": {"purpose": "specialist.hemodynamic"}})
        assert h({"kind": "tool_call", "data": {"name": "analyze"}})
        assert not h({"kind": "tool_call", "data": {"name": "verify"}})

    def test_reconstruction_sees_its_llm_calls_and_recon_tools(self):
        r = AGENT_ROUTES["reconstruction"]
        assert r({"kind": "llm_call", "data": {"purpose": "specialist.reconstruction"}})
        assert r({"kind": "tool_call", "data": {"name": "load_reconstruction"}})
        assert r({"kind": "tool_call", "data": {"name": "reconstruct"}})
        assert not r({"kind": "tool_call", "data": {"name": "verify"}})

    def test_segmentation_sees_its_llm_calls_and_seg_tools(self):
        s = AGENT_ROUTES["segmentation"]
        assert s({"kind": "llm_call", "data": {"purpose": "specialist.segmentation"}})
        assert s({"kind": "tool_call", "data": {"name": "suggest_seeds"}})
        assert s({"kind": "tool_call", "data": {"name": "segment_from_seed"}})
        assert not s({"kind": "tool_call", "data": {"name": "verify"}})

    def test_all_agents_see_session_lifecycle(self):
        # Every viewer should know when the session begins and ends
        for name, fn in AGENT_ROUTES.items():
            assert fn({"kind": "session_start", "data": {"metadata": {}}}), \
                f"{name} missed session_start"
            assert fn({"kind": "session_end", "data": {"status": "ok"}}), \
                f"{name} missed session_end"


# ============================================================================
# Tool-result one-liners
# ============================================================================

class TestShortResult:
    def test_load(self):
        s = _short_result("load_reconstruction",
                          {"shape_ZYXT": [77, 96, 72, 20], "voxel_size_mm": [2, 2, 2]})
        assert "[77, 96, 72, 20]" in s

    def test_segment(self):
        s = _short_result("segment_from_seed",
                          {"mask_name": "aorta", "size_voxels": 33618,
                           "peak_speed_m_per_s": 2.59})
        assert "aorta" in s and "33,618" in s

    def test_verify(self):
        s = _short_result("verify", {"verdict": "fail"})
        assert "fail" in s


# ============================================================================
# Rendering doesn't crash on real entries
# ============================================================================

class TestRendering:
    """Each render path is exercised on a representative entry — output is
    captured and checked for the key strings rather than exact layout."""

    def _capture(self, agent, entry):
        # Console.record + export_text — write to an in-memory buffer
        console = Console(record=True, width=120, force_terminal=True)
        render_entry(console, agent, entry)
        return console.export_text()

    def test_render_planner_plan(self):
        entry = {
            "kind": "llm_call", "t": "2026-05-17T07:00:01+00:00", "seq": 1,
            "session_id": "x",
            "data": {"purpose": "planner",
                     "response": {"text": json.dumps({"plan": ["alpha", "beta"]}),
                                  "latency_ms": 100, "model": "qwen3.6"}},
        }
        out = self._capture("planner", entry)
        assert "alpha" in out and "beta" in out
        assert "qwen3.6" in out

    def test_render_coordinator_decision(self):
        entry = {
            "kind": "llm_call", "t": "2026-05-17T07:00:01+00:00", "seq": 2,
            "session_id": "x",
            "data": {"purpose": "coordinator",
                     "response": {"text": json.dumps({"tool": "verify",
                                                       "args": {"mask_name": "v1"}}),
                                  "reasoning": "checking the mask",
                                  "latency_ms": 200, "model": "qwen3.6"}},
        }
        out = self._capture("coordinator", entry)
        assert "verify" in out and "v1" in out
        assert "checking the mask" in out

    def test_render_coordinator_done(self):
        entry = {
            "kind": "llm_call", "t": "2026-05-17T07:00:01+00:00", "seq": 3,
            "session_id": "x",
            "data": {"purpose": "coordinator",
                     "response": {"text": json.dumps({"done": True, "summary": "complete"}),
                                  "latency_ms": 50, "model": "m"}},
        }
        out = self._capture("coordinator", entry)
        assert "DONE" in out and "complete" in out

    def test_coordinator_does_not_render_tool_calls(self):
        """Tool calls now belong to the specialist windows, not the coordinator's."""
        entry = {
            "kind": "tool_call", "t": "2026-05-17T07:00:02+00:00", "seq": 4,
            "session_id": "x",
            "data": {"name": "segment_from_seed", "is_error": False,
                     "result": {"mask_name": "v1", "size_voxels": 1000,
                                "peak_speed_m_per_s": 1.5},
                     "latency_ms": 250},
        }
        out = self._capture("coordinator", entry)
        # No render → empty output
        assert out.strip() == ""

    def test_specialist_renders_its_llm_reply(self):
        entry = {
            "kind": "llm_call", "t": "2026-05-17T07:00:02+00:00", "seq": 5,
            "session_id": "x",
            "data": {"purpose": "specialist.verifier",
                     "response": {"text": json.dumps({"tool": "verify",
                                                       "args": {"mask_name": "v1"},
                                                       "why": "need numerical results"}),
                                  "latency_ms": 200, "model": "qwen3.6"}},
        }
        out = self._capture("verifier", entry)
        assert "verify" in out
        assert "v1" in out
        assert "need numerical results" in out

    def test_specialist_renders_done_report(self):
        entry = {
            "kind": "llm_call", "t": "2026-05-17T07:00:02+00:00", "seq": 6,
            "session_id": "x",
            "data": {"purpose": "specialist.hemodynamic",
                     "response": {"text": json.dumps({
                         "done": True,
                         "report": "SV of 93.9 mL is within normal adult range.",
                     }), "latency_ms": 300, "model": "qwen3.6"}},
        }
        out = self._capture("hemodynamic", entry)
        assert "Specialist report" in out
        assert "93.9" in out and "normal adult" in out

    def test_reconstruction_specialist_renders(self):
        entry = {
            "kind": "tool_call", "t": "2026-05-17T07:00:02+00:00", "seq": 7,
            "session_id": "x",
            "data": {"name": "load_reconstruction", "is_error": False,
                     "result": {"status": "loaded", "shape_ZYXT": [77, 96, 72, 20],
                                "voxel_size_mm": [2, 2, 2]},
                     "latency_ms": 50},
        }
        out = self._capture("reconstruction", entry)
        assert "load_reconstruction" in out
        assert "[77, 96, 72, 20]" in out

    def test_render_verifier_pass(self):
        entry = {
            "kind": "tool_call", "t": "2026-05-17T07:00:03+00:00", "seq": 5,
            "session_id": "x",
            "data": {"name": "verify", "is_error": False,
                     "result": {"mask_name": "v1", "verdict": "pass",
                                "checks": {"divergence": {"status": "pass",
                                                          "mean_abs_divergence_per_s": 2.0},
                                           "phase_unwrap": {"status": "pass",
                                                            "fraction_above_threshold": 0.0001}}},
                     "latency_ms": 80},
        }
        out = self._capture("verifier", entry)
        assert "PASS" in out
        assert "divergence" in out and "phase_unwrap" in out

    def test_render_verifier_fail(self):
        entry = {
            "kind": "tool_call", "t": "2026-05-17T07:00:03+00:00", "seq": 5,
            "session_id": "x",
            "data": {"name": "verify", "is_error": False,
                     "result": {"mask_name": "v1", "verdict": "fail",
                                "checks": {"divergence": {"status": "fail",
                                                          "mean_abs_divergence_per_s": 43.0}}},
                     "latency_ms": 80},
        }
        out = self._capture("verifier", entry)
        assert "FAIL" in out and "43" in out

    def test_render_verifier_error(self):
        entry = {
            "kind": "tool_call", "t": "2026-05-17T07:00:03+00:00", "seq": 5,
            "session_id": "x",
            "data": {"name": "verify", "is_error": True,
                     "result": {"error": "mask missing", "error_type": "ToolError"}},
        }
        out = self._capture("verifier", entry)
        assert "mask missing" in out

    def test_render_hemodynamic_report(self):
        entry = {
            "kind": "tool_call", "t": "2026-05-17T07:00:04+00:00", "seq": 6,
            "session_id": "x",
            "data": {"name": "analyze", "is_error": False,
                     "result": {
                         "mask_name": "v1",
                         "summary": {"mean_stroke_volume_mL": 93.9,
                                     "mean_peak_Q_mL_per_s": 438.3,
                                     "max_peak_Q_mL_per_s": 821.4,
                                     "peak_velocity_m_per_s": 2.59,
                                     "mean_velocity_m_per_s": 1.0},
                         "per_section": [{"regurgitant_fraction_pct": 130.6},
                                         {"regurgitant_fraction_pct": 26.1}],
                         "metadata": {"n_cross_sections": 5, "dominant_axis": "X"}},
                     "latency_ms": 120},
        }
        out = self._capture("hemodynamic", entry)
        assert "93.9" in out and "2.59" in out
        assert "Regurgitation" in out  # because at least one section > 0

    def test_session_start_end_render(self):
        start = {
            "kind": "session_start", "t": "2026-05-17T07:00:00+00:00", "seq": 0,
            "session_id": "abc-123",
            "data": {"metadata": {"goal": "test demo"}},
        }
        end = {
            "kind": "session_end", "t": "2026-05-17T07:00:30+00:00", "seq": 99,
            "session_id": "abc-123",
            "data": {"status": "success", "elapsed_s": 30.0,
                     "summary": {"verdict": "pass"}},
        }
        # Every agent should render these without crashing
        for name in AGENT_ROUTES:
            out = self._capture(name, start)
            assert "test demo" in out
            out = self._capture(name, end)
            assert "success" in out


# ============================================================================
# Tail loop (file existing, --no-follow mode)
# ============================================================================

class TestTailLog:
    def _populate(self, path):
        from utility.audit import AuditLog
        from utility.llm import LLMResponse
        log = AuditLog(path)
        log.llm_call(
            messages=[],
            response=LLMResponse(text=json.dumps({"plan": ["a", "b"]}),
                                 model="m", latency_ms=10),
            purpose="planner",
        )
        log.tool_call("verify", {"mask_name": "x"},
                      {"mask_name": "x", "verdict": "pass",
                       "checks": {"divergence": {"status": "pass"}}},
                      latency_ms=20)
        log.tool_call("analyze", {"mask_name": "x"},
                      {"mask_name": "x", "summary": {"mean_stroke_volume_mL": 50}},
                      latency_ms=30)
        log.close(status="success")

    def test_no_follow_reads_existing(self, tmp_path, capsys):
        log_path = tmp_path / "session.jsonl"
        self._populate(log_path)
        tail_log("verifier", log_path, follow=False)
        out = capsys.readouterr().out
        assert "PASS" in out

    def test_planner_window_filters_correctly(self, tmp_path, capsys):
        log_path = tmp_path / "session.jsonl"
        self._populate(log_path)
        tail_log("planner", log_path, follow=False)
        out = capsys.readouterr().out
        # Planner saw the planner llm_call
        assert "a" in out and "b" in out
        # And NOT the verify or analyze tool calls
        assert "Stroke volume" not in out

    def test_hemodynamic_window_only_sees_analyze(self, tmp_path, capsys):
        log_path = tmp_path / "session.jsonl"
        self._populate(log_path)
        tail_log("hemodynamic", log_path, follow=False)
        out = capsys.readouterr().out
        assert "Stroke volume" in out  # analyze rendered
        # No verifier checks
        assert "divergence" not in out

    def test_missing_file_message(self, tmp_path, capsys):
        tail_log("planner", tmp_path / "nope.jsonl", follow=False)
        out = capsys.readouterr().out
        assert "not found" in out
