"""Tests for the deterministic Plan Policy gate.

These tests pin the documented rules in plan_policy.py. If any test fails,
the policy rules have changed and the docs should be updated to match.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.plan_critic import PlanCritique
from agents.plan_policy import PolicyAction, apply_plan_policy


def _critique(verdict, concerns=None, suggestions=""):
    return PlanCritique(
        verdict=verdict,
        concerns=concerns or [],
        suggestions=suggestions,
        raw_text="<test>",
    )


class TestApprove:
    def test_approve_with_no_concerns_proceeds(self):
        d = apply_plan_policy(_critique("approve", []))
        assert d.action == PolicyAction.PROCEED
        assert d.halt_reason is None
        assert d.warning_reason is None
        assert d.concerns == []

    def test_approve_with_advisory_concerns_still_proceeds(self):
        d = apply_plan_policy(_critique("approve", ["minor note"]))
        assert d.action == PolicyAction.PROCEED
        assert "minor note" in d.concerns
        # Advisory concerns are carried forward but do not block


class TestReject:
    def test_reject_halts_regardless_of_revision_count(self):
        for rc in [0, 1, 2, 5]:
            d = apply_plan_policy(_critique("reject", ["impossible"]),
                                  revision_count=rc)
            assert d.action == PolicyAction.HALT, f"revision_count={rc} should still halt"
            assert "impossible" in (d.halt_reason or "")


class TestRevise:
    def test_revise_below_cap_returns_revise(self):
        d = apply_plan_policy(_critique("revise", ["needs verifier step"]),
                              revision_count=0, max_revisions=2)
        assert d.action == PolicyAction.REVISE
        assert "needs verifier step" in d.concerns

    def test_revise_one_below_cap_still_revises(self):
        d = apply_plan_policy(_critique("revise", ["x"]),
                              revision_count=1, max_revisions=2)
        assert d.action == PolicyAction.REVISE

    def test_revise_at_cap_degrades_to_proceed_with_warning(self):
        d = apply_plan_policy(_critique("revise", ["unresolved"]),
                              revision_count=2, max_revisions=2)
        assert d.action == PolicyAction.PROCEED_WITH_WARNING
        assert "unresolved" in (d.warning_reason or "")

    def test_revise_over_cap_degrades(self):
        d = apply_plan_policy(_critique("revise", ["still concerned"]),
                              revision_count=10, max_revisions=2)
        assert d.action == PolicyAction.PROCEED_WITH_WARNING

    def test_custom_max_revisions(self):
        d = apply_plan_policy(_critique("revise", ["x"]),
                              revision_count=4, max_revisions=5)
        assert d.action == PolicyAction.REVISE
        d = apply_plan_policy(_critique("revise", ["x"]),
                              revision_count=5, max_revisions=5)
        assert d.action == PolicyAction.PROCEED_WITH_WARNING


class TestReasonsAreInformative:
    def test_approve_reason_mentions_advisory_count(self):
        d = apply_plan_policy(_critique("approve", ["note 1", "note 2"]))
        assert "advisory" in d.reason
        assert "2" in d.reason

    def test_revise_reason_mentions_revision_count(self):
        d = apply_plan_policy(_critique("revise", ["x"]),
                              revision_count=0, max_revisions=2)
        assert "1 of 2" in d.reason

    def test_proceed_with_warning_reason_explains_why(self):
        d = apply_plan_policy(_critique("revise", ["unresolved x"]),
                              revision_count=2, max_revisions=2)
        assert "max_revisions" in d.reason


class TestUnknownVerdict:
    def test_unknown_verdict_raises(self):
        bad = PlanCritique(verdict="???",  # bypass the from_llm_text validation
                           concerns=[], suggestions="", raw_text="x")
        with pytest.raises(ValueError, match="unknown PlanCritique verdict"):
            apply_plan_policy(bad)
