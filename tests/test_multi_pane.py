"""Tests for the multi-pane audit-log replay TUI (demos/multi_pane.py).

The Live TUI itself can't be unit-tested headlessly (rich.live uses alternate
screen), so we test the state-mutation pipeline and the static render path.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demos.multi_pane import (
    PaneState,
    _apply_entry,
    _summarize_tool_result,
    render_final,
    replay,
)
from utility.audit import AuditLog
from utility.llm import LLMResponse


# ============================================================================
# Helpers
# ============================================================================

def _resp(text="ok", reasoning=None, latency=10, in_tok=5, out_tok=2):
    return LLMResponse(
        text=text, reasoning=reasoning, model="mock",
        latency_ms=latency, prompt_tokens=in_tok, completion_tokens=out_tok,
    )


def _populate_log(path):
    """Write a small but representative audit log to disk."""
    log = AuditLog(path, session_metadata={"goal": "test"})
    log.llm_call(
        messages=[{"role": "user", "content": "plan"}],
        response=_resp(text=json.dumps({"plan": ["step one", "step two"]}),
                       latency=100, in_tok=20, out_tok=10),
        purpose="planner",
    )
    log.llm_call(
        messages=[],
        response=_resp(text=json.dumps({"tool": "load_reconstruction",
                                         "args": {"mat_path": "/tmp/x.mat"}}),
                       reasoning="I should load the recon first",
                       latency=200, in_tok=30, out_tok=15),
        purpose="coordinator",
    )
    log.tool_call("load_reconstruction", {"mat_path": "/tmp/x.mat"},
                  {"status": "loaded", "shape_ZYXT": [77, 96, 72, 20],
                   "source_path": "/tmp/x.mat"}, latency_ms=300)
    log.tool_call("verify", {"mask_name": "v1"},
                  {"mask_name": "v1", "verdict": "fail",
                   "checks": {"divergence": {"status": "fail"},
                              "phase_unwrap": {"status": "pass"}}},
                  latency_ms=50)
    log.tool_call("analyze", {"mask_name": "v1"},
                  {"mask_name": "v1",
                   "summary": {"mean_stroke_volume_mL": 93.9,
                               "peak_velocity_m_per_s": 2.59,
                               "mean_peak_Q_mL_per_s": 438.3}},
                  latency_ms=80)
    log.tool_call("verify", {"mask_name": "missing"},
                  {"error": "mask not found", "error_type": "ToolError"})
    log.event("loop_exit", {"reason": "done"})
    log.close(status="success", summary={"verdict": "fail"})


# ============================================================================
# Tool-result summaries
# ============================================================================

class TestSummarizeToolResult:
    def test_load_reconstruction(self):
        s = _summarize_tool_result("load_reconstruction", {
            "shape_ZYXT": [77, 96, 72, 20],
            "source_path": "/mnt/g/medict_tmp/recon_cs_5iter.mat",
        })
        assert "[77, 96, 72, 20]" in s and "recon_cs_5iter.mat" in s

    def test_segment_from_seed(self):
        s = _summarize_tool_result("segment_from_seed", {
            "mask_name": "aorta_v1", "size_voxels": 33618, "peak_speed_m_per_s": 2.59,
        })
        assert "aorta_v1" in s and "33,618" in s and "2.59" in s

    def test_verify(self):
        s = _summarize_tool_result("verify", {"mask_name": "v1", "verdict": "fail"})
        assert "v1" in s and "fail" in s

    def test_analyze(self):
        s = _summarize_tool_result("analyze", {
            "mask_name": "v1",
            "summary": {"mean_stroke_volume_mL": 93.9, "peak_velocity_m_per_s": 2.59},
        })
        assert "93.9" in s and "2.59" in s

    def test_unknown_falls_back_to_json(self):
        s = _summarize_tool_result("mystery_tool", {"foo": 1, "bar": [1, 2]})
        assert "foo" in s and "bar" in s


# ============================================================================
# State mutation
# ============================================================================

class TestApplyEntry:
    def test_session_start_sets_id(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "session_start", "session_id": "abc-123",
            "t": "2026-05-17T07:00:00+00:00", "seq": 0,
            "data": {"metadata": {"goal": "test"}},
        })
        assert state.session_id == "abc-123"
        assert any("session_start" in line for line in state.audit_history)

    def test_planner_llm_call_populates_plan(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "llm_call", "session_id": "a", "seq": 1,
            "t": "2026-05-17T07:00:01+00:00",
            "data": {
                "purpose": "planner",
                "response": {"text": json.dumps({"plan": ["one", "two", "three"]}),
                             "latency_ms": 10, "prompt_tokens": 5, "completion_tokens": 2},
            },
        })
        assert state.planner_plan == ["one", "two", "three"]
        assert state.n_llm_calls == 1
        assert state.tokens_in == 5 and state.tokens_out == 2

    def test_planner_non_json_falls_back_to_text(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "llm_call", "session_id": "a", "seq": 1,
            "t": "2026-05-17T07:00:01+00:00",
            "data": {"purpose": "planner",
                     "response": {"text": "free-form plan text", "latency_ms": 5}},
        })
        assert len(state.planner_plan) == 1
        assert "free-form" in state.planner_plan[0]

    def test_coordinator_decision_parsed(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "llm_call", "session_id": "a", "seq": 1,
            "t": "2026-05-17T07:00:01+00:00",
            "data": {
                "purpose": "coordinator",
                "response": {"text": json.dumps({"tool": "verify", "args": {"mask_name": "v1"}}),
                             "reasoning": "I need to check this",
                             "latency_ms": 100},
            },
        })
        # Should have at least one decision and one reasoning line
        kinds = [k for k, _ in state.coord_history]
        assert "decision" in kinds and "reasoning" in kinds
        assert any("verify" in t for k, t in state.coord_history if k == "decision")

    def test_coordinator_done_signal(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "llm_call", "session_id": "a", "seq": 1,
            "t": "2026-05-17T07:00:01+00:00",
            "data": {"purpose": "coordinator",
                     "response": {"text": json.dumps({"done": True, "summary": "fin"}),
                                  "latency_ms": 50}},
        })
        assert any("DONE" in t for k, t in state.coord_history)

    def test_tool_call_success_updates_counters(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "tool_call", "session_id": "a", "seq": 2,
            "t": "2026-05-17T07:00:02+00:00",
            "data": {"name": "verify", "args": {}, "is_error": False,
                     "result": {"verdict": "pass", "mask_name": "v1", "checks": {}},
                     "latency_ms": 75},
        })
        assert state.n_tool_calls == 1
        assert state.n_tool_errors == 0
        assert state.total_tool_latency_ms == 75

    def test_tool_call_error_increments_error_count(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "tool_call", "session_id": "a", "seq": 3,
            "t": "2026-05-17T07:00:03+00:00",
            "data": {"name": "verify", "args": {}, "is_error": True,
                     "result": {"error": "missing mask", "error_type": "ToolError"}},
        })
        assert state.n_tool_errors == 1
        assert any(k == "error" for k, _ in state.coord_history)

    def test_verify_result_mirrored_to_analysis_pane(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "tool_call", "session_id": "a", "seq": 2,
            "t": "2026-05-17T07:00:02+00:00",
            "data": {"name": "verify", "args": {}, "is_error": False,
                     "result": {"mask_name": "v1", "verdict": "pass",
                                "checks": {"divergence": {"status": "pass"}}}},
        })
        assert state.analysis_history
        tag, _ = state.analysis_history[-1]
        assert tag == "verify_pass"

    def test_analyze_result_mirrored_to_analysis_pane(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "tool_call", "session_id": "a", "seq": 2,
            "t": "2026-05-17T07:00:02+00:00",
            "data": {"name": "analyze", "args": {}, "is_error": False,
                     "result": {"mask_name": "v1",
                                "summary": {"mean_stroke_volume_mL": 50.0,
                                            "peak_velocity_m_per_s": 2.0,
                                            "mean_peak_Q_mL_per_s": 300.0}}},
        })
        tag, line = state.analysis_history[-1]
        assert tag == "analyze"
        assert "50.0" in line and "2.0" in line

    def test_elapsed_clock_from_log_timestamps(self):
        state = PaneState()
        _apply_entry(state, {
            "kind": "session_start", "session_id": "a", "seq": 0,
            "t": "2026-05-17T07:00:00+00:00", "data": {"metadata": {}},
        })
        _apply_entry(state, {
            "kind": "event", "session_id": "a", "seq": 1,
            "t": "2026-05-17T07:00:05+00:00", "data": {"name": "x", "data": {}},
        })
        # Should be ~5 seconds elapsed
        assert 4.0 < state.last_entry_time < 6.0


# ============================================================================
# End-to-end with a real audit log
# ============================================================================

class TestRenderFinal:
    def test_renders_known_log(self, tmp_path):
        log_path = tmp_path / "session.jsonl"
        _populate_log(log_path)
        text = render_final(log_path, width=140)
        # All pane titles present
        assert "PLANNER" in text
        assert "COORDINATOR" in text
        assert "PHYSICS & HEMODYNAMIC ANALYSIS" in text
        assert "AUDIT" in text
        # Key content present
        assert "step one" in text
        assert "load_reconstruction" in text
        assert "fail" in text or "FAIL" in text
        assert "93.9" in text and "2.59" in text

    def test_empty_log_raises(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError, match="no entries"):
            replay(empty)

    def test_render_handles_existing_demo_log(self):
        """If the bundled demo log exists, render it and confirm it doesn't crash."""
        demo_path = os.path.join(os.path.dirname(__file__), "..", "logs", "demo_session.jsonl")
        if not os.path.exists(demo_path):
            pytest.skip("demo log not present; run notebooks/test_audit_demo.py first")
        text = render_final(demo_path, width=140)
        assert "PLANNER" in text and "AUDIT" in text


# ============================================================================
# Replay timing
# ============================================================================

class TestReplayInstant:
    def test_instant_mode_runs_without_sleeping(self, tmp_path, monkeypatch):
        """Use instant=True so we don't actually sleep through the replay."""
        log_path = tmp_path / "session.jsonl"
        _populate_log(log_path)

        # Stub Live so we don't enter alternate screen mode during the test
        import demos.multi_pane as mp

        class FakeLive:
            def __init__(self, layout, **kwargs):
                self.layout = layout
                self.updates = 0
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def update(self, layout): self.updates += 1

        fake = FakeLive(None)
        monkeypatch.setattr(mp, "Live", lambda *a, **k: fake)

        # Also stub time.sleep so the held-final-frame delay doesn't slow tests
        monkeypatch.setattr(mp.time, "sleep", lambda s: None)

        mp.replay(log_path, instant=True)
        # Should have called update once per entry
        assert fake.updates >= 5
