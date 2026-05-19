"""Planner agent — proposes an ordered plan from a free-form user goal.

The Planner is the FIRST LLM in the pipeline. It reads the user goal and the
project context (canonical paths, acquisition defaults) and emits a structured
plan: ordered steps, each tagged with the specialist who should execute it and
a success criterion the Coordinator can check.

The plan is then handed to the Plan Critic for review. The Coordinator only
sees plans that have passed the Plan Policy gate.

Design notes:
- The Planner does NOT execute anything. It only proposes.
- The plan is structured so the Coordinator can use it directly as guidance
  but is still allowed to deviate if execution surfaces unexpected conditions.
- The Planner is allowed to revise its own plan when given concerns from the
  Plan Critic (revise() method).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from .audit import AuditLog
from .llm import LLM
from .project_context import PROJECT_CONTEXT


PLANNER_SYSTEM = """\
You are the **Planner** of a multi-agent 4D flow MRI analysis pipeline.

Your job: read the user goal and produce an ordered plan that the Coordinator
will execute. Each step in your plan must name:
  - the specialist who will run it (one of: reconstruction, segmentation,
    verifier, hemodynamic)
  - a one-sentence action description
  - a measurable success criterion the Coordinator can check before moving
    on (e.g. "mask size > 5000 voxels", "verifier returns pass or warn",
    "stroke volume within 60-100 mL adult range")

You are NOT executing anything. You only propose.

Respect the canonical pipeline order:
  reconstruction → segmentation → verifier → hemodynamic

Skipping a step requires explicit justification in the rationale field.
"""

PLANNER_PROTOCOL = """\

## Response protocol
Reply with ONE JSON object of the form:

{
  "plan": [
    {"step": 1,
     "specialist": "<name>",
     "action":     "<short imperative>",
     "success_criterion": "<measurable condition>"},
    ...
  ],
  "rationale": "<one-paragraph explanation of why this ordering>"
}
"""

PLANNER_REVISION_PROTOCOL = """\

## Revision protocol
The previous plan was reviewed and concerns were raised. Re-emit the plan
with the same JSON shape, addressing the concerns listed below. Keep what
works; change only what was flagged.
"""


@dataclass
class Plan:
    """A structured plan the Coordinator can read directly."""
    steps: list[dict]                       # list of {step, specialist, action, success_criterion}
    rationale: str = ""
    revision_count: int = 0                 # how many times this plan has been revised
    raw_text: str = ""                      # original LLM output for the audit log

    @classmethod
    def from_llm_text(cls, text: str) -> "Plan":
        """Parse the LLM's JSON output into a Plan. Raises on bad shape."""
        payload = json.loads(text)
        steps = payload.get("plan")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"plan must be a non-empty list, got: {steps!r}")
        for i, s in enumerate(steps):
            for key in ("specialist", "action", "success_criterion"):
                if key not in s:
                    raise ValueError(f"step {i} missing required field {key!r}: {s}")
        return cls(steps=steps, rationale=payload.get("rationale", ""), raw_text=text)

    def to_prompt_block(self) -> str:
        """Render the plan as a markdown block for the Coordinator's system prompt."""
        lines = ["## Approved plan (from Planner, reviewed by Plan Critic)\n"]
        for s in self.steps:
            lines.append(
                f"{s['step']}. **{s['specialist']}** — {s['action']}  "
                f"_(success: {s['success_criterion']})_"
            )
        if self.rationale:
            lines.append(f"\n**Rationale:** {self.rationale}")
        lines.append(
            "\nFollow this plan unless execution reveals a condition that requires deviating. "
            "If you deviate, briefly justify it in your `why` field."
        )
        return "\n".join(lines)


class Planner:
    """Wrapper around an LLM that proposes and revises plans."""

    def __init__(self, llm: LLM, *, project_context: str = PROJECT_CONTEXT,
                 max_tokens: int = 8192,
                 verbose_callback=None):
        self.llm = llm
        self.project_context = project_context
        self.max_tokens = max_tokens
        self.verbose = verbose_callback

    def propose(self, user_goal: str, audit: AuditLog) -> Plan:
        """First-pass plan from the user goal."""
        if self.verbose:
            self.verbose("[planner] thinking…")
        messages = [
            {"role": "system",
             "content": self.project_context + "\n\n" + PLANNER_SYSTEM + PLANNER_PROTOCOL},
            {"role": "user", "content": f"User goal: {user_goal}"},
        ]
        plan = self._call_and_parse(messages, audit, purpose="planner",
                                    failure_reason=f"could not parse Planner output for goal: {user_goal[:80]}")
        if self.verbose:
            self.verbose(f"[planner] proposed plan with {len(plan.steps)} steps")
        return plan

    def revise(self, user_goal: str, previous_plan: Plan, concerns: list[str],
               audit: AuditLog) -> Plan:
        if self.verbose:
            self.verbose(f"[planner] revising (round {previous_plan.revision_count + 1})…")
        """Revise a plan in response to Plan Critic concerns.

        The Planner is given its previous plan, the concerns raised, and is
        asked to re-emit a corrected plan.
        """
        concern_block = "\n".join(f"  - {c}" for c in concerns) or "  (no specific concerns provided)"
        messages = [
            {"role": "system",
             "content": (self.project_context + "\n\n" + PLANNER_SYSTEM
                         + PLANNER_PROTOCOL + PLANNER_REVISION_PROTOCOL)},
            {"role": "user", "content": f"User goal: {user_goal}"},
            {"role": "assistant", "content": previous_plan.raw_text},
            {"role": "user",
             "content": ("The Plan Critic raised these concerns:\n" + concern_block
                         + "\n\nRe-emit the plan addressing each concern. Use the same JSON shape.")},
        ]
        plan = self._call_and_parse(messages, audit, purpose="planner",
                                    failure_reason="could not parse revised plan")
        plan.revision_count = previous_plan.revision_count + 1
        return plan

    def _call_and_parse(self, messages: list[dict], audit: AuditLog, *,
                        purpose: str, failure_reason: str) -> Plan:
        resp = self.llm.chat(messages, json_mode=True, max_tokens=self.max_tokens)
        audit.llm_call(messages=messages, response=resp, purpose=purpose,
                       options={"json_mode": True})
        try:
            return Plan.from_llm_text(resp.text)
        except (json.JSONDecodeError, ValueError) as e:
            audit.event("planner_parse_failure",
                        {"error": str(e), "raw_text": resp.text[:400]})
            raise RuntimeError(f"{failure_reason}: {e}") from e
