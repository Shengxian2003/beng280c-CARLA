"""Tests for the Plan Critic agent."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utility.audit import AuditLog, read_log, filter_log
from utility.llm import MockLLM
from agents.plan_critic import PlanCritic, PlanCritique
from agents.planner import Plan


def _make_plan():
    raw = json.dumps({
        "plan": [{"step": 1, "specialist": "reconstruction",
                  "action": "load recon",
                  "success_criterion": "shape matches"}],
        "rationale": "single-step demo",
    })
    return Plan.from_llm_text(raw)


class TestPlanCritiqueParsing:
    def test_approve_with_no_concerns(self):
        c = PlanCritique.from_llm_text(json.dumps({
            "verdict": "approve", "concerns": [], "suggestions": "",
        }))
        assert c.verdict == "approve"
        assert c.is_blocking() is False

    def test_approve_with_advisory_concerns(self):
        c = PlanCritique.from_llm_text(json.dumps({
            "verdict": "approve",
            "concerns": ["minor: rationale could explain why CS vs CORe"],
            "suggestions": "",
        }))
        assert c.verdict == "approve"
        assert len(c.concerns) == 1
        assert not c.is_blocking()

    def test_revise_requires_concerns(self):
        with pytest.raises(ValueError, match="requires at least one concern"):
            PlanCritique.from_llm_text(json.dumps({
                "verdict": "revise", "concerns": [], "suggestions": "x",
            }))

    def test_reject_requires_concerns(self):
        with pytest.raises(ValueError, match="requires at least one concern"):
            PlanCritique.from_llm_text(json.dumps({
                "verdict": "reject", "concerns": [], "suggestions": "x",
            }))

    def test_invalid_verdict(self):
        with pytest.raises(ValueError, match="verdict must be"):
            PlanCritique.from_llm_text(json.dumps({
                "verdict": "maybe", "concerns": ["x"], "suggestions": "",
            }))

    def test_revise_is_blocking(self):
        c = PlanCritique.from_llm_text(json.dumps({
            "verdict": "revise", "concerns": ["needs verifier step"], "suggestions": "",
        }))
        assert c.is_blocking() is True

    def test_reject_is_blocking(self):
        c = PlanCritique.from_llm_text(json.dumps({
            "verdict": "reject", "concerns": ["impossible"], "suggestions": "",
        }))
        assert c.is_blocking() is True


class TestPlanCriticReview:
    def test_review_approve_round_trips(self, tmp_path):
        llm = MockLLM(responses=[json.dumps({
            "verdict": "approve", "concerns": [], "suggestions": "",
        })])
        log = AuditLog(tmp_path / "log.jsonl")
        critique = PlanCritic(llm).review(_make_plan(), user_goal="goal", audit=log)
        log.close()
        assert critique.verdict == "approve"

    def test_review_audits_under_plan_critic_purpose(self, tmp_path):
        llm = MockLLM(responses=[json.dumps({
            "verdict": "approve", "concerns": [], "suggestions": "",
        })])
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        PlanCritic(llm).review(_make_plan(), user_goal="g", audit=log)
        log.close()
        entries = read_log(log_path)
        llm_calls = filter_log(entries, kind="llm_call")
        assert any(e["data"]["purpose"] == "plan_critic" for e in llm_calls)

    def test_unparseable_output_degrades_to_soft_approve(self, tmp_path):
        # If the Critic emits broken JSON, we don't want to block execution on
        # its silence — degrade to approve, record an event in the audit log.
        llm = MockLLM(responses=["not json at all"])
        log_path = tmp_path / "log.jsonl"
        log = AuditLog(log_path)
        critique = PlanCritic(llm).review(_make_plan(), user_goal="g", audit=log)
        log.close()
        assert critique.verdict == "approve"
        # The parse failure should have been logged as an event
        events = filter_log(read_log(log_path), kind="event")
        assert any(e["data"]["name"] == "plan_critic_parse_failure" for e in events)

    def test_review_passes_plan_text_to_llm(self, tmp_path):
        llm = MockLLM(responses=[json.dumps({
            "verdict": "approve", "concerns": [], "suggestions": "",
        })])
        log = AuditLog(tmp_path / "log.jsonl")
        plan = _make_plan()
        PlanCritic(llm).review(plan, user_goal="analyze a scan", audit=log)
        log.close()
        # The user message should contain the plan's raw text + the user goal
        msgs = llm.calls[0][0]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        assert "analyze a scan" in user_msg
        assert "reconstruction" in user_msg  # came from the plan
