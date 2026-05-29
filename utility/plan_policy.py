"""Plan Policy — deterministic gate that decides what to do with a critique.

This is NOT an LLM. It is plain Python code that converts a PlanCritique into
a concrete action: proceed, revise, or halt. The reason this exists as a
separate module is so the rules can be read, reviewed, and tested independently
of any LLM behavior.

## The rules (read these — they are the documented logic)

1. **approve** → action = PROCEED. Critique concerns (if any) are recorded in
   the audit log as advisory notes but do not block execution.

2. **revise** with at least one concern, and revision_count < max_revisions:
   → action = REVISE. Critique concerns are handed back to the Planner.
   Default max_revisions = 2 (i.e. one initial plan + at most two revisions
   before giving up).

3. **revise** with revision_count >= max_revisions:
   → action = PROCEED_WITH_WARNING. We refuse to spin forever; the plan goes
   to the Coordinator with the unresolved concerns logged as warnings. The
   Coordinator can still execute, but Stage 4 evaluation will see the
   warning in the audit log.

4. **reject** at any revision_count:
   → action = HALT. Execution does not start. The concerns are surfaced to
   the caller via PlanPolicyDecision.halt_reason.

## Why these specific rules

- Cap on revisions: an LLM Critic that keeps demanding revisions is a denial-
  of-service vector. Two revisions is enough to catch real fixable issues
  without burning unbounded tokens.
- "approve" does not require empty concerns: the Critic is allowed to note
  minor things without making them blocking. We trust the verdict.
- "reject" always halts: this is the one verdict the Critic can use to stop
  the system. We use it sparingly (the Critic prompt says so) and we surface
  the reasoning so the user can decide whether to override.

## What this gate CANNOT do
- It cannot override the deterministic Physics Verifier. The Physics Verifier
  runs LATER, during execution, on actual velocity fields. By the time
  Verifier runs, this gate has already exited. There is no code path here
  that touches Verifier verdicts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


@dataclass
class PlanCritique:
    """The Plan Critic's verdict on a plan. Produced by `agents.plan_critic`
    and consumed by `apply_plan_policy` below. Lives here (utility/) rather
    than next to the agent so the policy gate can reference it without
    creating a utility → agents dependency."""
    verdict:     str            # "approve" | "revise" | "reject"
    concerns:    list[str]
    suggestions: str
    raw_text:    str            # for the audit log

    @classmethod
    def from_llm_text(cls, text: str) -> "PlanCritique":
        payload = json.loads(text)
        verdict = payload.get("verdict")
        if verdict not in ("approve", "revise", "reject"):
            raise ValueError(f"verdict must be approve|revise|reject, got: {verdict!r}")
        concerns = payload.get("concerns", []) or []
        if not isinstance(concerns, list):
            raise ValueError(f"concerns must be a list, got: {type(concerns).__name__}")
        if verdict in ("revise", "reject") and not concerns:
            raise ValueError(f"{verdict!r} requires at least one concern")
        return cls(
            verdict=verdict,
            concerns=[str(c) for c in concerns],
            suggestions=str(payload.get("suggestions", "")),
            raw_text=text,
        )

    def is_blocking(self) -> bool:
        """Returns True if execution should not proceed without action."""
        return self.verdict in ("revise", "reject")


class PolicyAction(str, Enum):
    PROCEED              = "proceed"
    PROCEED_WITH_WARNING = "proceed_with_warning"
    REVISE               = "revise"
    HALT                 = "halt"


@dataclass
class PlanPolicyDecision:
    action:          PolicyAction
    reason:          str                  # human-readable explanation of the decision
    concerns:        list[str]            # concerns from the critique, carried forward
    halt_reason:     str | None = None    # populated only when action == HALT
    warning_reason:  str | None = None    # populated only when action == PROCEED_WITH_WARNING


def apply_plan_policy(
    critique: PlanCritique,
    *,
    revision_count: int = 0,
    max_revisions: int = 2,
) -> PlanPolicyDecision:
    """Deterministic mapping from PlanCritique → action.

    Parameters
    ----------
    critique : PlanCritique
        The Plan Critic's verdict on the most recent plan.
    revision_count : int
        How many revisions the Planner has already performed for this goal.
        The first plan is revision_count=0; each Planner.revise() increments.
    max_revisions : int
        Hard cap. Past this, "revise" verdicts degrade to PROCEED_WITH_WARNING.

    Returns
    -------
    PlanPolicyDecision
        Concrete action the caller (the orchestrator) should take.
    """
    if critique.verdict == "approve":
        return PlanPolicyDecision(
            action=PolicyAction.PROCEED,
            reason=("plan approved by Plan Critic"
                    + (f" with {len(critique.concerns)} advisory note(s)"
                       if critique.concerns else "")),
            concerns=list(critique.concerns),
        )

    if critique.verdict == "reject":
        return PlanPolicyDecision(
            action=PolicyAction.HALT,
            reason="Plan Critic rejected the plan; halting before execution",
            concerns=list(critique.concerns),
            halt_reason="; ".join(critique.concerns) or "(no specific concerns provided)",
        )

    if critique.verdict == "revise":
        if revision_count < max_revisions:
            return PlanPolicyDecision(
                action=PolicyAction.REVISE,
                reason=(f"Plan Critic asked for revision "
                        f"({revision_count + 1} of {max_revisions} allowed)"),
                concerns=list(critique.concerns),
            )
        # Out of revisions — escalate to "proceed with warning" rather than
        # halting, so we always make some progress for the user.
        return PlanPolicyDecision(
            action=PolicyAction.PROCEED_WITH_WARNING,
            reason=(f"Plan Critic asked for revision but max_revisions={max_revisions} "
                    f"already used; proceeding with unresolved concerns recorded"),
            concerns=list(critique.concerns),
            warning_reason=("unresolved Plan Critic concerns: "
                            + "; ".join(critique.concerns)),
        )

    # Defensive — PlanCritique.from_llm_text already validates the verdict
    raise ValueError(f"unknown PlanCritique verdict: {critique.verdict!r}")
