"""Coordinator agent — top-level orchestrator that delegates to specialists.

The Coordinator runs the outer loop. At each turn it picks ONE specialist to
hand a sub-task to, waits for the specialist's natural-language report, and
decides what to do next (delegate again, retry with different parameters, or
finish).

This is the LLM-pick-LLM pattern: the Coordinator never calls a tool directly
— it always goes through a specialist who understands the domain.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .audit import AuditLog
from .llm import LLM
from .project_context import PROJECT_CONTEXT
from .specialist import Specialist
from .tools import Workspace


COORDINATOR_SYSTEM = """\
You are the **Coordinator** of a multi-agent 4D flow MRI analysis pipeline.

You delegate work to four specialist agents:
  - reconstruction   — loads or runs CS/CORe reconstructions of k-space
  - segmentation     — finds vessel masks from PC-MRA
  - verifier         — runs deterministic physics checks (divergence, flux,
                       peak velocity, phase unwrap) and judges trustworthiness
  - hemodynamic      — computes flow Q(t), stroke volume, peak flow, and
                       interprets them physiologically

Each specialist has its own expertise and toolset. You do NOT call tools
yourself. You ONLY delegate.

A typical workflow:
  1. delegate to reconstruction → ask it to load or run a recon
  2. delegate to segmentation   → ask it to produce a clean vessel mask
  3. delegate to verifier       → ask if the mask is trustworthy
  4. if verifier returns 'fail', either re-delegate to segmentation
     (try different parameters) or to reconstruction (re-run with more
     iterations), then re-verify
  5. once verifier returns pass/warn, delegate to hemodynamic for the
     final clinical metrics
  6. finish with done=true and a final summary

You can re-delegate to the same specialist if its previous attempt failed
or the result was insufficient. You can also finish early if verification
fails repeatedly and you want to report partial results.
"""

COORDINATOR_PROTOCOL = """\

## Response protocol
At every turn, reply with ONE JSON object. Two shapes are valid:

  To delegate to a specialist:
    {"delegate_to": "<specialist name>",
     "task":        "<natural-language instruction to that specialist>",
     "why":         "<one-sentence rationale for picking this specialist>"}

  To finish the session:
    {"done":    true,
     "summary": "<final report describing what was accomplished and any
                 caveats the user should know about>"}

Always end with a done response within a reasonable number of steps.
"""


@dataclass
class CoordinatorResult:
    status: str            # "success" | "max_steps" | "error"
    summary: str
    n_delegations: int
    specialists_used: list[str] = field(default_factory=list)
    last_decision: dict | None = None


class Coordinator:
    """Outer-loop orchestrator that delegates to specialists turn by turn."""

    def __init__(
        self,
        llm: LLM,
        specialists: dict[str, Specialist],
        workspace: Workspace,
        audit: AuditLog,
        *,
        max_delegations: int = 12,
        verbose_callback: Optional[Callable[[str], None]] = None,
        project_context: str = PROJECT_CONTEXT,
    ):
        self.llm = llm
        self.specialists = specialists
        self.workspace = workspace
        self.audit = audit
        self.max_delegations = max_delegations
        self.verbose = verbose_callback
        self.project_context = project_context

    def _initial_messages(self, user_goal: str) -> list[dict]:
        names = sorted(self.specialists)
        roster = "\n".join(f"  - {n}" for n in names)
        system = (
            self.project_context + "\n\n"
            + COORDINATOR_SYSTEM
            + COORDINATOR_PROTOCOL
            + f"\n\n## Specialists actually available this session\n{roster}\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"User goal: {user_goal}"},
        ]

    def run(self, user_goal: str) -> CoordinatorResult:
        history = self._initial_messages(user_goal)
        used: list[str] = []
        last: dict | None = None

        for step in range(self.max_delegations):
            resp = self.llm.chat(history, json_mode=True, max_tokens=2048)
            self.audit.llm_call(
                messages=history, response=resp,
                purpose="coordinator",
                options={"json_mode": True, "step": step},
            )
            if self.verbose:
                self.verbose(f"[coordinator] step {step}: {(resp.text or '')[:140]}")

            # Parse decision
            try:
                decision = json.loads(resp.text)
            except json.JSONDecodeError:
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({
                    "error": "your reply was not valid JSON; reply with one JSON object per the protocol"
                })})
                continue

            last = decision

            # Done signal
            if decision.get("done"):
                self.audit.event("coordinator_done", {"step": step})
                return CoordinatorResult(
                    status="success",
                    summary=decision.get("summary", ""),
                    n_delegations=step,
                    specialists_used=used,
                    last_decision=decision,
                )

            # Delegation
            spec_name = decision.get("delegate_to")
            task = decision.get("task", "")
            specialist = self.specialists.get(spec_name)
            if specialist is None:
                err = (f"specialist {spec_name!r} not in roster "
                       f"{sorted(self.specialists)}; pick one of those or done=true")
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({"error": err})})
                self.audit.event("bad_delegation", {"requested": spec_name})
                continue

            self.audit.event("delegation", {"to": spec_name, "task": task[:200]})
            report = specialist.handle(task,
                                       workspace=self.workspace,
                                       audit=self.audit,
                                       verbose_callback=self.verbose)
            used.append(spec_name)

            # Feed report back into the conversation
            history.append({"role": "assistant", "content": resp.text})
            history.append({"role": "user", "content": json.dumps({
                "from_specialist": spec_name,
                "report":          report.get("report", ""),
                "done":            report.get("done", False),
                "tools_called":    report.get("tools_called", []),
                "n_steps":         report.get("n_steps"),
            })})

        # Max delegations exhausted
        self.audit.event("max_delegations", {})
        return CoordinatorResult(
            status="max_steps",
            summary=f"reached max_delegations={self.max_delegations}",
            n_delegations=self.max_delegations,
            specialists_used=used,
            last_decision=last,
        )
