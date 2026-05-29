"""Tests for the Planner agent."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utility.audit import AuditLog, read_log, filter_log
from utility.llm import MockLLM
from agents.planner import Plan, Planner


SAMPLE_PLAN = {
    "plan": [
        {"step": 1, "specialist": "reconstruction",
         "action": "Load existing 5-iter recon",
         "success_criterion": "shape matches (77,96,72,20)"},
        {"step": 2, "specialist": "segmentation",
         "action": "Segment largest vessel",
         "success_criterion": "mask size > 5000 voxels"},
        {"step": 3, "specialist": "verifier",
         "action": "Run physics verification",
         "success_criterion": "verdict in {pass, warn}"},
        {"step": 4, "specialist": "hemodynamic",
         "action": "Compute flow metrics",
         "success_criterion": "stroke volume reported"},
    ],
    "rationale": "Standard pipeline order, load existing recon to skip 12-min MATLAB run.",
}


class TestPlanParsing:
    def test_valid_plan_parses(self):
        plan = Plan.from_llm_text(json.dumps(SAMPLE_PLAN))
        assert len(plan.steps) == 4
        assert plan.steps[0]["specialist"] == "reconstruction"
        assert plan.rationale.startswith("Standard")

    def test_empty_plan_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Plan.from_llm_text(json.dumps({"plan": []}))

    def test_missing_field_rejected(self):
        bad = {"plan": [{"step": 1, "specialist": "verifier"}]}  # missing action
        with pytest.raises(ValueError, match="missing required field"):
            Plan.from_llm_text(json.dumps(bad))

    def test_invalid_json_rejected(self):
        with pytest.raises(json.JSONDecodeError):
            Plan.from_llm_text("not json")

    def test_prompt_block_contains_steps(self):
        plan = Plan.from_llm_text(json.dumps(SAMPLE_PLAN))
        block = plan.to_prompt_block()
        assert "reconstruction" in block and "verifier" in block
        assert "mask size > 5000" in block
        assert "Standard pipeline order" in block


class TestPlannerPropose:
    def test_propose_returns_plan(self, tmp_path):
        llm = MockLLM(responses=[json.dumps(SAMPLE_PLAN)])
        log = AuditLog(tmp_path / "log.jsonl")
        plan = Planner(llm).propose("Analyze a 4D flow scan.", audit=log)
        log.close()
        assert isinstance(plan, Plan)
        assert plan.steps[0]["specialist"] == "reconstruction"
        assert plan.revision_count == 0

    def test_propose_audits_under_planner_purpose(self, tmp_path):
        llm = MockLLM(responses=[json.dumps(SAMPLE_PLAN)])
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        Planner(llm).propose("goal", audit=log)
        log.close()
        entries = read_log(log_path)
        llm_calls = filter_log(entries, kind="llm_call")
        assert any(e["data"]["purpose"] == "planner" for e in llm_calls)

    def test_bad_json_raises_runtime_error(self, tmp_path):
        llm = MockLLM(responses=["definitely not JSON"])
        log = AuditLog(tmp_path / "log.jsonl")
        with pytest.raises(RuntimeError, match="could not parse"):
            Planner(llm).propose("goal", audit=log)
        log.close()


class TestPlannerRevise:
    def test_revise_increments_count(self, tmp_path):
        llm = MockLLM(responses=[json.dumps(SAMPLE_PLAN), json.dumps(SAMPLE_PLAN)])
        log = AuditLog(tmp_path / "log.jsonl")
        planner = Planner(llm)
        first = planner.propose("goal", audit=log)
        assert first.revision_count == 0
        second = planner.revise("goal", first, ["concern A"], audit=log)
        assert second.revision_count == 1
        third = planner.revise("goal", second, ["concern B"], audit=log) \
            if False else None  # skip; just one revise to keep test focused
        log.close()

    def test_revise_message_history_contains_previous_plan(self, tmp_path):
        llm = MockLLM(responses=[json.dumps(SAMPLE_PLAN)])
        log = AuditLog(tmp_path / "log.jsonl")
        planner = Planner(llm)
        first_plan = Plan(steps=SAMPLE_PLAN["plan"], rationale="x",
                          raw_text=json.dumps(SAMPLE_PLAN))
        planner.revise("goal", first_plan, ["pipeline order wrong"], audit=log)
        log.close()
        # The recorded call should include the previous plan's raw text + the concern
        msgs = llm.calls[0][0]
        joined = "\n".join(m["content"] for m in msgs)
        assert "pipeline order wrong" in joined
        assert "reconstruction" in joined  # came from previous plan
