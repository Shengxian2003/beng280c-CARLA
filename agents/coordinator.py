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

from utility.audit import AuditLog
from utility.llm import LLM
from .planner import Plan
from utility.project_context import PROJECT_CONTEXT
from .specialist import Specialist
from utility.tools import Workspace


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

## ⚠ Error classification — distinguish addressing errors from quality verdicts

When a specialist reports back, the cause of "verification failed" matters:

  (a) **Addressing error** — the specialist's tool returned
      `"error": "mask 'X' not found — known masks: [...]"`.
      The data is fine; the LOOKUP was wrong. The specialist named a mask
      that does not exist in the workspace.
      → CORRECT RESPONSE: re-delegate to the SAME specialist with the
        correct `mask_name` (read it from the known-masks list or from the
        most recent segmentation report). Do NOT re-delegate upstream;
        re-segmentation will not help, it will simply produce the same
        mask under the same name.

  (b) **Data-quality verdict** — the verifier's `verify()` returned a real
      verdict dict with `"verdict": "fail"` and numeric reasons (high
      divergence, large flux deviation, etc.).
      → CORRECT RESPONSE: re-delegate upstream (segmentation with tighter
        parameters, or reconstruction with more iterations) and re-verify.

  (c) **Empty mask / skip** — `verify()` returned `"verdict": "skip"` because
      the named mask has zero voxels.
      → CORRECT RESPONSE: re-delegate to segmentation to produce a real mask,
        OR if STL was supposed to be loaded, surface the upstream loader
        failure honestly.

Treating (a) as (b) — re-running segmentation in response to a naming bug —
wastes the delegation budget and never converges, since every retry
produces the same mask under the same name. This was the dominant failure
mode in past sessions. Read the specialist's tools_called and the latest
verify tool_call error TYPE before deciding.
"""

COORDINATOR_PROTOCOL = """\

## Response protocol
At every turn, reply with ONE JSON object. Two shapes are valid:

  To delegate to a specialist:
    {"delegate_to": "<specialist name>",
     "task":        "<natural-language instruction to that specialist>",
     "mask_name":   "<the exact mask identifier the specialist must act on>",
     "why":         "<one-sentence rationale for picking this specialist>"}

  The `mask_name` field is REQUIRED when delegating to `verifier` or
  `hemodynamic` if at least one mask exists in the workspace. It MUST be
  one of the existing mask identifiers (e.g. `aorta_v1`) — never a path,
  never a description ("the segmented mask"), never the name of a verdict
  category. When delegating to `reconstruction` or `segmentation`, the
  field is optional. The dispatcher rejects verifier/hemodynamic
  delegations that omit it when masks are available.

  To finish the session:
    {"done":    true,
     "summary": "<final report describing what was accomplished and any
                 caveats the user should know about>"}

Always end with a done response within a reasonable number of steps.
"""


def _coord_decision_summary(text: str) -> str:
    """One-liner describing the Coordinator's decision for the status bar."""
    try:
        payload = json.loads(text)
        if payload.get("done"):
            return "done — emitting summary"
        if "delegate_to" in payload:
            return f"delegating to {payload['delegate_to']}"
        return text[:120]
    except (json.JSONDecodeError, TypeError):
        return (text or "")[:120].replace("\n", " ")


@dataclass
class CoordinatorResult:
    status: str            # "success" | "partial" | "error"
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
        max_delegations: int = 6,                   # was 12 — lowered to fail
                                                    # faster on hard datasets
        max_tokens: int = 16384,                    # raised so reasoning models
                                                    # (qwen3.x with thinking)
                                                    # have room for thinking
                                                    # + JSON delegation reply.
        verbose_callback: Optional[Callable[[str], None]] = None,
        project_context: str = PROJECT_CONTEXT,
        supports_reconstruction_retry: bool = True,
        max_verify_failures_per_mask: int = 2,
    ):
        self.llm = llm
        self.specialists = specialists
        self.workspace = workspace
        self.audit = audit
        self.max_delegations = max_delegations
        self.max_tokens = max_tokens
        self.verbose = verbose_callback
        self.project_context = project_context
        # Locks out reconstruction re-delegation for input modes whose load
        # is one-shot (phantom, AS4DF, real_scan in load-only mode). Set
        # from the active InputProfile by the demo entry point.
        self.supports_reconstruction_retry = supports_reconstruction_retry
        # Force-done after the same mask has been verified-failed this many
        # times. Stops the "seg → verify fail → seg → verify fail" loop
        # from consuming the entire delegation budget without converging.
        self.max_verify_failures_per_mask = max_verify_failures_per_mask

    def _initial_messages(
        self, user_goal: str, plan: Plan | None = None, warning: str | None = None,
    ) -> list[dict]:
        names = sorted(self.specialists)
        roster = "\n".join(f"  - {n}" for n in names)
        system_parts = [
            self.project_context,
            COORDINATOR_SYSTEM,
            COORDINATOR_PROTOCOL,
            f"\n\n## Specialists actually available this session\n{roster}\n",
        ]
        if not self.supports_reconstruction_retry:
            system_parts.append(
                "\n\n## Reconstruction is ONE-SHOT for this input mode\n"
                "The current input mode loads its data directly (synthetic\n"
                "phantom, AS4DF DICOM, or a pre-saved .mat reconstruction).\n"
                "There is NO parameter to tune by re-running it. Re-delegating\n"
                "to the reconstruction specialist after the initial load CANNOT\n"
                "produce different data — it will either re-load the same bytes\n"
                "or no-op.\n\n"
                "Therefore: after the FIRST successful reconstruction\n"
                "delegation, you MUST NOT delegate to reconstruction again.\n"
                "If verification fails, your only useful recovery moves are:\n"
                "  (a) re-delegate to segmentation with a DIFFERENT seed or\n"
                "      tighter parameters (not the same seed with new params),\n"
                "  (b) emit done=true with an honest caveat that the data\n"
                "      cannot be analyzed with the available tools.\n"
                "The dispatcher will reject a second reconstruction delegation\n"
                "and record the attempt as a protocol violation."
            )
        system_parts.append(
            "\n\n## Verify-fail loop cap (avoid burning the budget)\n"
            f"If the SAME `mask_name` returns `verdict: fail` from `verify` "
            f"{self.max_verify_failures_per_mask} times within this session, "
            f"you MUST emit done=true with a partial-results summary on the "
            f"next turn. Re-segmenting the same vessel another time will "
            f"almost certainly produce another failing mask; honest escalation "
            f"is correct."
        )
        if plan is not None:
            system_parts.append("\n\n" + plan.to_prompt_block())
        if warning:
            system_parts.append(
                "\n\n## Plan Policy warning (carry forward)\n"
                f"{warning}\n\nProceed, but the audit log will record this warning."
            )
        return [
            {"role": "system", "content": "".join(system_parts)},
            {"role": "user",   "content": f"User goal: {user_goal}"},
        ]

    def run(
        self,
        user_goal: str,
        *,
        plan: Plan | None = None,
        warning: str | None = None,
    ) -> CoordinatorResult:
        """Run the delegation loop. If ``plan`` is provided, it is injected into
        the system prompt as approved guidance; the Coordinator may still
        deviate if execution surfaces unexpected conditions."""
        history = self._initial_messages(user_goal, plan=plan, warning=warning)
        used: list[str] = []
        last: dict | None = None
        # Track recon delegations + per-mask verify-fail counts for the
        # one-shot recon lock-out (Fix 5) and the verify-fail loop cap (Fix 7).
        recon_delegations_completed = 0
        verify_fail_count: dict[str, int] = {}

        # Pipeline progress tracker — visible to the LLM in every energy block
        STAGES = ["reconstruction", "segmentation", "verifier", "hemodynamic"]
        stage_status: dict[str, str] = {s: "pending" for s in STAGES}

        def _refresh_stage_status() -> None:
            """Snapshot what the workspace currently shows."""
            if self.workspace.recon is not None:
                stage_status["reconstruction"] = "ok"
            if self.workspace.masks:
                stage_status["segmentation"]   = "ok"
            if self.workspace.verdicts:
                # any verify call ran (pass/warn/fail all count as "done")
                stage_status["verifier"]       = "ok"
            if self.workspace.analyses:
                stage_status["hemodynamic"]    = "ok"

        for step in range(self.max_delegations):
            _refresh_stage_status()
            # ── Inject a live energy block so the LLM sees its delegation budget
            remaining = self.max_delegations - step
            is_final  = (remaining == 1)
            # Detailed stage status: explicitly marked PENDING when missing,
            # so the LLM cannot pretend a stage is done from a text report alone.
            stage_truth_lines = []
            for s in STAGES:
                if stage_status[s] == "ok":
                    stage_truth_lines.append(f"  - {s}: ✓ DONE (verified in workspace)")
                else:
                    stage_truth_lines.append(f"  - {s}: ✗ NOT YET DONE (workspace is empty)")
            stage_block = "\n".join(stage_truth_lines)
            untouched = [s for s in STAGES if stage_status[s] != "ok"]

            ground_truth_warning = (
                "⚠ GROUND TRUTH: the workspace state above is the ONLY authoritative "
                "record of what has happened. Specialists' natural-language reports may "
                "claim work was done when it wasn't (e.g., a specialist may say "
                "'verified the physics' without the Verifier specialist actually running). "
                "A stage is only DONE when the workspace shows it (mask, verdict, or "
                "analysis exists). If the Verifier hasn't run, you MUST delegate to it "
                "before emitting done."
            )

            if is_final:
                energy_msg = (
                    "[ENERGY] ⚠ FINAL DELEGATION — this is your last chance.\n"
                    f"Pipeline progress (workspace state):\n{stage_block}\n\n"
                    f"{ground_truth_warning}\n\n"
                    "Your next reply MUST be done=true with a best-effort summary. "
                    "After this round the loop terminates regardless of what you emit."
                )
            else:
                hint = ""
                if remaining <= 3 and untouched:
                    hint = (
                        f"\n⚠ Budget is low ({remaining} left). Stages NOT YET DONE: "
                        f"{untouched}. PRIORITIZE delegating to the next pending stage. "
                        f"A degraded mask + caveated hemodynamic report is better than "
                        f"hitting the limit with no downstream stage attempted."
                    )
                energy_msg = (
                    f"[ENERGY] You have {remaining}/{self.max_delegations} delegations "
                    f"remaining.\n"
                    f"Pipeline progress (workspace state):\n{stage_block}\n\n"
                    f"{ground_truth_warning}\n"
                    f"Every reply (delegate or done) consumes 1 delegation."
                    f"{hint}"
                )
            # Replace any prior [ENERGY] block on the last user turn
            if history and history[-1].get("role") == "user":
                content = history[-1]["content"] or ""
                marker  = "\n\n[ENERGY]"
                idx     = content.rfind(marker)
                if idx >= 0:
                    content = content[:idx]
                history[-1]["content"] = content + "\n\n" + energy_msg

            if self.verbose:
                self.verbose(
                    f"[coordinator] step {step}: deciding "
                    f"(delegations {remaining}/{self.max_delegations})"
                )
            resp = self.llm.chat(history, json_mode=True, max_tokens=self.max_tokens)
            self.audit.llm_call(
                messages=history, response=resp,
                purpose="coordinator",
                options={
                    "json_mode": True, "step": step,
                    "budget": {
                        "delegations_used":      step,
                        "max_delegations":       self.max_delegations,
                        "delegations_remaining": remaining,
                        "is_final_round":        is_final,
                    },
                },
            )
            if self.verbose:
                # Brief summary of what was decided, not the full JSON
                summary = _coord_decision_summary(resp.text or "")
                self.verbose(f"[coordinator] step {step}: {summary}")

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
            mask_name = decision.get("mask_name")
            specialist = self.specialists.get(spec_name)
            if specialist is None:
                err = (f"specialist {spec_name!r} not in roster "
                       f"{sorted(self.specialists)}; pick one of those or done=true")
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({"error": err})})
                self.audit.event("bad_delegation", {"requested": spec_name})
                continue

            # ── Mask-name handoff enforcement (Fix 1) ────────────────────
            # Verifier and Hemodynamic operate on a specific named mask. If
            # any masks exist in the workspace, the Coordinator must name
            # the one to act on so the downstream specialist does not have
            # to guess.
            requires_mask = spec_name in ("verifier", "hemodynamic")
            known_masks   = sorted(self.workspace.masks)
            if requires_mask and known_masks and not mask_name:
                err = (
                    f"delegations to {spec_name!r} REQUIRE a `mask_name` field "
                    f"when masks exist in the workspace. Known masks: "
                    f"{known_masks}. Re-emit your delegation with "
                    f"`mask_name` set to one of those identifiers."
                )
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({
                    "error":       err,
                    "known_masks": known_masks,
                })})
                self.audit.event("missing_mask_name_in_delegation", {
                    "to":          spec_name,
                    "known_masks": known_masks,
                })
                continue
            if mask_name and known_masks and mask_name not in known_masks:
                err = (
                    f"`mask_name`={mask_name!r} is not a known mask in the "
                    f"workspace. Known masks: {known_masks}. Either name an "
                    f"existing mask or first delegate to segmentation to "
                    f"produce a new one."
                )
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({
                    "error":       err,
                    "known_masks": known_masks,
                })})
                self.audit.event("unknown_mask_name_in_delegation", {
                    "to":          spec_name,
                    "mask_name":   mask_name,
                    "known_masks": known_masks,
                })
                continue

            # ── Reconstruction one-shot lock-out (Fix 5) ────────────────
            # When the active input mode's data load is one-shot (phantom,
            # AS4DF, real-scan no-fresh-recon), a second reconstruction
            # delegation cannot help — reject it without consuming the
            # specialist's energy budget. The Coordinator gets the rejection
            # message back and is forced to pick a different action.
            if (spec_name == "reconstruction"
                    and not self.supports_reconstruction_retry
                    and recon_delegations_completed >= 1):
                err = (
                    "Reconstruction is one-shot for this input mode (no "
                    "iteration parameter to tune). Further reconstruction "
                    "delegations CANNOT change the data. Pick a different "
                    "specialist or emit done=true with caveats."
                )
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({
                    "error": err,
                    "supports_reconstruction_retry": False,
                })})
                self.audit.event("recon_retry_blocked", {
                    "input_mode_is_one_shot": True,
                })
                continue

            # ── Verify-fail loop cap (Fix 7) ────────────────────────────
            # If the Coordinator wants to re-verify a mask that has already
            # failed N times this session, block it. Re-verifying the same
            # mask cannot change the verdict; the Coordinator must either
            # produce a NEW mask first or escalate to done.
            if (spec_name == "verifier"
                    and mask_name
                    and verify_fail_count.get(mask_name, 0)
                        >= self.max_verify_failures_per_mask):
                err = (
                    f"Mask {mask_name!r} has already failed verification "
                    f"{verify_fail_count[mask_name]} times this session. "
                    f"Re-verifying it CANNOT change the verdict. Either "
                    f"delegate to segmentation to produce a new mask under "
                    f"a different name, or emit done=true with an honest "
                    f"summary of the failed verifications."
                )
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({
                    "error":              err,
                    "verify_fail_count":  verify_fail_count,
                })})
                self.audit.event("verify_fail_cap_blocked", {
                    "mask_name": mask_name,
                    "count":     verify_fail_count[mask_name],
                })
                continue

            # Embed the resolved mask_name into the specialist's task so it
            # is the very first thing the specialist's LLM reads. This is the
            # architectural answer to past sessions where the Verifier
            # specialist invented a mask name because the task prose did not
            # include one.
            specialist_task = task
            if mask_name:
                specialist_task = (
                    f"[Coordinator handoff] mask_name = \"{mask_name}\"\n\n"
                    + task
                )

            self.audit.event("delegation", {
                "to":        spec_name,
                "task":      task[:200],
                "mask_name": mask_name,
            })
            report = specialist.handle(specialist_task,
                                       workspace=self.workspace,
                                       audit=self.audit,
                                       verbose_callback=self.verbose)
            used.append(spec_name)
            if spec_name == "reconstruction":
                recon_delegations_completed += 1
            # Inspect the verifier's verdict via the workspace (the ground-
            # truth state) to count fails toward the loop cap. We use the
            # workspace not the specialist's report, because the report is
            # LLM-prose subject to grounding violations.
            if spec_name == "verifier" and mask_name in self.workspace.verdicts:
                vd = self.workspace.verdicts[mask_name].get("verdict")
                if vd == "fail":
                    verify_fail_count[mask_name] = verify_fail_count.get(mask_name, 0) + 1

            # Feed report back into the conversation
            history.append({"role": "assistant", "content": resp.text})
            history.append({"role": "user", "content": json.dumps({
                "from_specialist": spec_name,
                "report":          report.get("report", ""),
                "done":            report.get("done", False),
                "tools_called":    report.get("tools_called", []),
                "n_steps":         report.get("n_steps"),
            })})

        # ---- Max delegations exhausted — summarize what we DID learn -----
        # Pull workspace state to build a "partial results" report so the user
        # gets something actionable instead of a bare "max reached" message.
        self.audit.event("max_delegations", {"used": used})

        masks      = sorted(self.workspace.masks)
        verdicts   = {k: v.get("verdict") for k, v in self.workspace.verdicts.items()}
        analyses   = sorted(self.workspace.analyses)

        partial_lines = [
            f"Reached max_delegations={self.max_delegations} without an explicit done.",
            f"Specialists used: {', '.join(used) or 'none'}.",
        ]
        if self.workspace.recon is not None:
            shape = tuple(self.workspace.recon["xHat"].shape)
            partial_lines.append(f"Reconstruction loaded (shape {shape}).")
        if masks:
            partial_lines.append(f"Masks produced: {masks}.")
        if verdicts:
            verdict_summary = ", ".join(f"{k}={v}" for k, v in verdicts.items())
            partial_lines.append(f"Verifier verdicts: {verdict_summary}.")
        if analyses:
            partial_lines.append(f"Hemodynamic reports produced for: {analyses}.")
        partial_lines.append(
            "The Coordinator did not converge on a clean result; the most likely "
            "cause is a data-quality issue (e.g. segmentation cannot cleanly "
            "isolate a single vessel) rather than an agent bug. The workspace "
            "state above is preserved — partial results may still be useful."
        )

        return CoordinatorResult(
            status="partial",
            summary="\n".join(partial_lines),
            n_delegations=self.max_delegations,
            specialists_used=used,
            last_decision=last,
        )
