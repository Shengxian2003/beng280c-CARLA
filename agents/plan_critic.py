"""Plan Critic agent — best-effort LLM auditor of the proposed plan.

The Plan Critic is the LLM portion of the Auditor. It reviews the Planner's
proposed plan for **obvious methodological issues** before any execution
begins, and emits a structured verdict the Plan Policy gate can act on.

## Known limitations (read this — it shapes how to use the output)

1. The Critic operates only on the plan text, not on data. It cannot verify
   that any step will succeed. It can only flag whether the plan structure
   makes sense.

2. The Critic is best-effort. Like all LLMs, it may miss real issues and may
   flag non-issues. Plan Policy treats its verdict as advisory: only the
   `reject` verdict can halt the pipeline, and even that is logged with the
   concerns for human review.

3. **The Critic CANNOT override the deterministic Physics Verifier.** The
   Physics Verifier operates on actual velocity fields during execution; the
   Critic operates on plan text before execution. They never operate on the
   same artifact, so there is no path for the Critic to disagree with a
   Verifier verdict. This is enforced both architecturally (Critic runs and
   exits before Coordinator starts) and prompt-wise (the Critic is explicitly
   told it is reviewing the plan, not the data).

4. The Critic is told to check methodology, not clinical interpretation. It
   flags things like "plan tries to analyze without verifying first" or
   "plan re-runs reconstruction when the existing one might be acceptable."
   It does not flag things like "the stroke volume number looks too high"
   — that's the Hemodynamic Analyzer's job.

## Verdict semantics
  approve  → plan is methodologically sound; proceed to execution
  revise   → plan has fixable issues (listed in `concerns`); ask Planner
             to re-emit. Bounded by `max_revisions` in plan_policy.
  reject   → plan is fundamentally wrong (e.g. asks the system to do
             something it cannot do, or skips required safety steps).
             Halts execution; surfaces concerns to the user.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .audit import AuditLog
from .llm import LLM
from .planner import Plan
from .project_context import PROJECT_CONTEXT


PLAN_CRITIC_SYSTEM = """\
You are the **Plan Critic** — the LLM portion of the Auditor. Your job is to
review a proposed plan for obvious methodological issues before execution.

## What you check
- Does the plan respect the pipeline order? (reconstruction → segmentation
  → verifier → hemodynamic)
- Are there missing steps that downstream specialists depend on? (e.g. you
  cannot analyze without first segmenting a vessel)
- Does the plan account for known quality issues? (e.g. trying to compute
  clinical metrics from a 5-iteration recon without acknowledging the limit)
- Are success criteria measurable and appropriate?
- Are there redundant or contradictory steps?

## What you do NOT check
- You CANNOT predict whether a step will succeed at runtime — that's the
  Physics Verifier's job, which runs later on actual data.
- You CANNOT override the Physics Verifier's verdicts. The Verifier runs
  AFTER you, on real velocity fields. You see only the plan text, never the
  data. These are different artifacts at different times.
- You do NOT make clinical judgments about specific numbers — those come
  from the Hemodynamic Analyzer during execution.

## Best-effort framing
You are an LLM. You may miss real issues or flag non-issues. The Plan Policy
treats your verdict as advisory. Be specific about your concerns so the
Planner can revise concretely; vague concerns waste a revision cycle.

If the plan is sound, approve it — do not invent concerns to seem thorough.
If you reject, the pipeline halts and surfaces your reasoning to the user.
Only reject for fundamental issues, not stylistic ones.
"""

PLAN_CRITIC_PROTOCOL = """\

## Response protocol
Reply with ONE JSON object of the form:

{
  "verdict":  "approve" | "revise" | "reject",
  "concerns": ["<specific issue 1>", "<specific issue 2>", ...],
  "suggestions": "<concrete changes you'd propose, or empty string>"
}

- approve → concerns may be empty (or list minor non-blocking notes)
- revise  → concerns MUST list specific, fixable issues
- reject  → concerns MUST explain why no revision can fix this plan
"""


@dataclass
class PlanCritique:
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


class PlanCritic:
    """Wrapper around an LLM that reviews plans."""

    def __init__(self, llm: LLM, *, project_context: str = PROJECT_CONTEXT,
                 max_tokens: int = 8192,
                 verbose_callback=None):
        self.llm = llm
        self.project_context = project_context
        self.max_tokens = max_tokens
        self.verbose = verbose_callback

    def review(self, plan: Plan, *, user_goal: str, audit: AuditLog) -> PlanCritique:
        if self.verbose:
            self.verbose("[plan_critic] reviewing the plan…")
        messages = [
            {"role": "system",
             "content": (self.project_context + "\n\n"
                         + PLAN_CRITIC_SYSTEM + PLAN_CRITIC_PROTOCOL)},
            {"role": "user",
             "content": (f"User goal: {user_goal}\n\n"
                         f"Proposed plan (from Planner):\n```json\n{plan.raw_text}\n```\n\n"
                         "Review this plan for methodological issues and emit your verdict.")},
        ]
        resp = self.llm.chat(messages, json_mode=True, max_tokens=self.max_tokens)
        audit.llm_call(messages=messages, response=resp, purpose="plan_critic",
                       options={"json_mode": True})
        try:
            critique = PlanCritique.from_llm_text(resp.text)
            if self.verbose:
                self.verbose(f"[plan_critic] verdict: {critique.verdict}")
            return critique
        except (json.JSONDecodeError, ValueError) as e:
            audit.event("plan_critic_parse_failure",
                        {"error": str(e), "raw_text": resp.text[:400]})
            # Parse failure is treated as a soft approve — the Critic could not
            # produce a valid verdict, so we don't block execution on its silence.
            # Logged as an event so it shows up in Stage 4 auditability metrics.
            return PlanCritique(
                verdict="approve",
                concerns=[f"(plan critic returned unparseable output: {e}) — treated as approve"],
                suggestions="",
                raw_text=resp.text,
            )
