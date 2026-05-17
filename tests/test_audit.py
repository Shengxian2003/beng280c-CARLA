"""Unit tests for the audit log (Stage 3c)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.audit import (
    AuditLog,
    read_log,
    iter_log,
    filter_log,
    summarize,
    format_entry,
    pretty_print,
)
from agents.llm import LLMResponse


# ============================================================================
# Helpers
# ============================================================================

def _fake_response(text="hello", reasoning=None, latency_ms=42,
                   in_tokens=10, out_tokens=5, model="mock"):
    return LLMResponse(
        text=text, reasoning=reasoning, model=model,
        latency_ms=latency_ms,
        prompt_tokens=in_tokens, completion_tokens=out_tokens,
    )


# ============================================================================
# Session lifecycle + basic writes
# ============================================================================

class TestSessionLifecycle:
    def test_creates_file_and_writes_start(self, tmp_path):
        p = tmp_path / "session.jsonl"
        log = AuditLog(p, session_metadata={"goal": "test"})
        log.close()
        assert p.exists()
        entries = read_log(p)
        assert entries[0]["kind"] == "session_start"
        assert entries[0]["data"]["metadata"] == {"goal": "test"}
        assert entries[-1]["kind"] == "session_end"

    def test_creates_parent_directory(self, tmp_path):
        p = tmp_path / "nested" / "subdir" / "session.jsonl"
        log = AuditLog(p)
        log.close()
        assert p.exists()

    def test_assigns_unique_session_id(self, tmp_path):
        a = AuditLog(tmp_path / "a.jsonl"); a.close()
        b = AuditLog(tmp_path / "b.jsonl"); b.close()
        assert read_log(tmp_path / "a.jsonl")[0]["session_id"] \
            != read_log(tmp_path / "b.jsonl")[0]["session_id"]

    def test_explicit_session_id(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl", session_id="custom-id-123")
        log.close()
        assert read_log(tmp_path / "s.jsonl")[0]["session_id"] == "custom-id-123"

    def test_close_marks_log_closed(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        log.close()
        with pytest.raises(RuntimeError, match="already closed"):
            log.event("after_close")

    def test_session_end_reports_elapsed(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        time.sleep(0.02)
        log.close()
        end = [e for e in read_log(tmp_path / "s.jsonl") if e["kind"] == "session_end"][0]
        assert end["data"]["elapsed_s"] >= 0.02


# ============================================================================
# Entry structure
# ============================================================================

class TestEntryStructure:
    def test_sequence_numbers_monotonic(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        for i in range(5):
            log.event(f"step_{i}")
        log.close()
        entries = read_log(tmp_path / "s.jsonl")
        assert [e["seq"] for e in entries] == list(range(len(entries)))

    def test_all_entries_have_required_fields(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        log.llm_call(messages=[{"role": "user", "content": "hi"}],
                     response=_fake_response(), purpose="planner")
        log.tool_call("verify", {"mask_name": "m"}, {"verdict": "pass"}, latency_ms=12)
        log.event("done")
        log.close()
        for e in read_log(tmp_path / "s.jsonl"):
            assert "t" in e and "seq" in e and "session_id" in e and "kind" in e and "data" in e

    def test_timestamps_iso_utc(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl"); log.close()
        entries = read_log(tmp_path / "s.jsonl")
        # ISO 8601 with UTC offset (Python emits "+00:00")
        for e in entries:
            assert "T" in e["t"]
            assert e["t"].endswith("+00:00") or e["t"].endswith("Z")


# ============================================================================
# LLM call logging
# ============================================================================

class TestLLMCall:
    def test_captures_response_fields(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        resp = _fake_response(text="answer", reasoning="thought process",
                              latency_ms=123, in_tokens=20, out_tokens=10)
        log.llm_call(
            messages=[{"role": "user", "content": "q"}],
            response=resp,
            purpose="coordinator",
            options={"temperature": 0.0, "json_mode": True},
        )
        log.close()
        e = [x for x in read_log(tmp_path / "s.jsonl") if x["kind"] == "llm_call"][0]
        r = e["data"]["response"]
        assert r["text"] == "answer"
        assert r["reasoning"] == "thought process"
        assert r["latency_ms"] == 123
        assert r["prompt_tokens"] == 20
        assert r["completion_tokens"] == 10
        assert e["data"]["purpose"] == "coordinator"
        assert e["data"]["options"]["temperature"] == 0.0

    def test_preserves_message_history(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        msgs = [
            {"role": "system", "content": "you are X"},
            {"role": "user", "content": "do Y"},
        ]
        log.llm_call(messages=msgs, response=_fake_response())
        log.close()
        e = [x for x in read_log(tmp_path / "s.jsonl") if x["kind"] == "llm_call"][0]
        assert e["data"]["messages"] == msgs


# ============================================================================
# Tool call logging
# ============================================================================

class TestToolCall:
    def test_success(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        log.tool_call("verify", {"mask_name": "v1"},
                      {"verdict": "pass", "mask_name": "v1"}, latency_ms=80)
        log.close()
        e = [x for x in read_log(tmp_path / "s.jsonl") if x["kind"] == "tool_call"][0]
        assert e["data"]["name"] == "verify"
        assert e["data"]["is_error"] is False
        assert e["data"]["latency_ms"] == 80

    def test_error_marked(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        log.tool_call("verify", {},
                      {"error": "missing mask_name", "error_type": "ToolError"})
        log.close()
        e = [x for x in read_log(tmp_path / "s.jsonl") if x["kind"] == "tool_call"][0]
        assert e["data"]["is_error"] is True

    def test_numpy_in_args_sanitized(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        # If the orchestrator accidentally puts numpy types into args, the log
        # must still be JSON-serializable.
        log.tool_call("segment_from_seed",
                      {"seed_z": np.int64(54), "seed_y": np.int32(32), "mask_name": "m"},
                      {"status": "ok", "size_voxels": np.int64(33618)})
        log.close()
        # The pure round-trip already confirms json.dumps worked; check values:
        e = [x for x in read_log(tmp_path / "s.jsonl") if x["kind"] == "tool_call"][0]
        assert e["data"]["args"]["seed_z"] == 54
        assert e["data"]["result"]["size_voxels"] == 33618


# ============================================================================
# Event logging
# ============================================================================

class TestEvent:
    def test_basic(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        log.event("plan_revised", {"reason": "verifier failed", "new_step": 3})
        log.close()
        e = [x for x in read_log(tmp_path / "s.jsonl") if x["kind"] == "event"][0]
        assert e["data"]["name"] == "plan_revised"
        assert e["data"]["data"]["reason"] == "verifier failed"

    def test_event_without_data(self, tmp_path):
        log = AuditLog(tmp_path / "s.jsonl")
        log.event("retry")
        log.close()
        e = [x for x in read_log(tmp_path / "s.jsonl") if x["kind"] == "event"][0]
        assert e["data"]["data"] == {}


# ============================================================================
# Readers
# ============================================================================

class TestReaders:
    def _populate(self, path):
        log = AuditLog(path)
        log.llm_call(messages=[], response=_fake_response(latency_ms=100), purpose="planner")
        log.tool_call("suggest_seeds", {}, {"n_returned": 3}, latency_ms=20)
        log.llm_call(messages=[], response=_fake_response(latency_ms=200), purpose="coordinator")
        log.tool_call("verify", {"mask_name": "m"}, {"verdict": "fail"}, latency_ms=15)
        log.event("retrying")
        log.tool_call("verify", {"mask_name": "m2"}, {"verdict": "pass"}, latency_ms=14)
        log.close(status="success", summary={"final_verdict": "pass"})

    def test_iter_log_streams(self, tmp_path):
        p = tmp_path / "s.jsonl"
        self._populate(p)
        items = list(iter_log(p))
        assert len(items) >= 5

    def test_filter_by_kind(self, tmp_path):
        p = tmp_path / "s.jsonl"
        self._populate(p)
        entries = read_log(p)
        tool_calls = filter_log(entries, kind="tool_call")
        assert len(tool_calls) == 3
        assert all(e["kind"] == "tool_call" for e in tool_calls)

    def test_filter_by_purpose(self, tmp_path):
        p = tmp_path / "s.jsonl"
        self._populate(p)
        entries = read_log(p)
        planner = filter_log(entries, purpose="planner")
        assert len(planner) == 1

    def test_filter_by_tool_name(self, tmp_path):
        p = tmp_path / "s.jsonl"
        self._populate(p)
        entries = read_log(p)
        verifies = filter_log(entries, tool="verify")
        assert len(verifies) == 2

    def test_corrupt_line_raises(self, tmp_path):
        p = tmp_path / "broken.jsonl"
        p.write_text('{"kind":"event"}\nnot-json-at-all\n')
        with pytest.raises(ValueError, match="corrupt log entry"):
            read_log(p)


# ============================================================================
# Summary
# ============================================================================

class TestSummary:
    def test_aggregate_counts_and_latency(self, tmp_path):
        p = tmp_path / "s.jsonl"
        log = AuditLog(p)
        log.llm_call(messages=[], response=_fake_response(latency_ms=100, in_tokens=10, out_tokens=5))
        log.llm_call(messages=[], response=_fake_response(latency_ms=200, in_tokens=20, out_tokens=10))
        log.tool_call("verify", {}, {"verdict": "pass"}, latency_ms=30)
        log.tool_call("analyze", {}, {"summary": {}}, latency_ms=70)
        log.tool_call("verify", {}, {"error": "x", "error_type": "ToolError"})
        log.close(status="success")

        s = summarize(p)
        assert s["n_llm_calls"] == 2
        assert s["n_tool_calls"] == 3
        assert s["total_llm_latency_ms"] == 300
        assert s["total_tool_latency_ms"] == 100
        assert s["total_input_tokens"] == 30
        assert s["total_output_tokens"] == 15
        assert s["tool_call_counts"] == {"verify": 2, "analyze": 1}
        assert s["n_tool_errors"] == 1
        assert s["session_status"] == "success"

    def test_summary_accepts_entries_list(self, tmp_path):
        p = tmp_path / "s.jsonl"
        log = AuditLog(p); log.close()
        s = summarize(read_log(p))
        assert s["session_status"] == "ok"

    def test_incomplete_session(self, tmp_path):
        # Simulate a crashed session — no close()
        p = tmp_path / "crashed.jsonl"
        log = AuditLog(p)
        log.event("step_1")
        # Deliberately don't call close()
        s = summarize(p)
        assert s["session_status"] == "incomplete"


# ============================================================================
# Pretty-print
# ============================================================================

class TestPrettyPrint:
    def test_format_entry_truncates(self):
        e = {
            "t": "2026-05-16T12:34:56+00:00", "seq": 0, "session_id": "abc",
            "kind": "llm_call",
            "data": {"purpose": "planner",
                     "response": {"text": "x" * 500, "latency_ms": 10,
                                  "prompt_tokens": 5, "completion_tokens": 3}}
        }
        line = format_entry(e, max_chars=50)
        assert "..." in line and len(line) < 250

    def test_format_entry_marks_tool_error(self):
        e = {
            "t": "2026-05-16T12:34:56+00:00", "seq": 1, "session_id": "abc",
            "kind": "tool_call",
            "data": {"name": "verify", "is_error": True,
                     "result": {"error": "bad mask", "error_type": "ToolError"}}
        }
        line = format_entry(e)
        assert "✗" in line and "bad mask" in line

    def test_pretty_print_runs(self, tmp_path, capsys):
        p = tmp_path / "s.jsonl"
        log = AuditLog(p)
        log.llm_call(messages=[], response=_fake_response(), purpose="planner")
        log.tool_call("verify", {"mask_name": "m"}, {"verdict": "pass"}, latency_ms=5)
        log.event("done")
        log.close()
        pretty_print(p)
        captured = capsys.readouterr().out
        # Every entry produced a line
        assert captured.count("\n") == 5  # session_start + llm + tool + event + session_end
        assert "session_start" in captured
        assert "llm  " in captured
        assert "tool" in captured
        assert "event done" in captured
        assert "session_end" in captured
