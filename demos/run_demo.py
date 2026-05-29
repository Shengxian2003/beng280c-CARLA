"""End-to-end multi-window demo runner — Coordinator delegates to specialist LLMs.

Architecture (LLM-pick-LLM):

    Coordinator LLM
        │
        └── delegates to one of:
              ├── Reconstruction Operator (LLM)  → calls load_reconstruction / reconstruct
              ├── Segmentation Operator   (LLM)  → calls suggest_seeds / segment_from_seed
              ├── Physics Verifier        (LLM)  → calls verify, interprets the verdict
              └── Hemodynamic Analyzer    (LLM)  → calls analyze, contextualizes the numbers

Each specialist is itself an LLM with a focused system prompt. Specialists
share one Ollama+Qwen backend (one GPU on this host) but behave like distinct
experts because their prompts differ.

Workflow:
  1. Run this script. It prints the 6 viewer commands to copy into windows.
  2. Open 6 terminal windows, paste a command into each.
  3. Press ENTER here. The Coordinator runs, delegating to specialists.
     All 6 windows update live.

Usage:
    python demos/run_demo.py                       # default = real Qwen via Ollama
    python demos/run_demo.py --llm ollama          # explicit
    python demos/run_demo.py --llm mock            # scripted MockLLM, no Ollama needed

The MockLLM script is short and deterministic — useful when Ollama is down or
you want a fast smoke test of the wiring before recording.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from utility.audit import AuditLog
from agents.coordinator import Coordinator
from utility.llm import MockLLM, OllamaLLM
from agents.plan_critic import PlanCritic
from utility.plan_policy import PolicyAction, apply_plan_policy
from agents.planner import Plan, Planner
from agents.specialist import build_default_specialists
from utility.tools import Workspace


DEFAULT_GOAL = (
    "Analyze the hemodynamics of the 4D flow MRI scan at "
    "/mnt/g/medict_tmp/recon_cs_5iter.mat. "
    "The reconstruction was run with 5 iterations of CS. "
    "Pick a vessel, verify the physics, and report flow metrics. "
    "If verification fails, explain why and what should be done next."
)


# ============================================================================
# Scripted responses for --llm mock (deterministic, no Ollama needed)
# ============================================================================

# The Coordinator delegates in this fixed order; each specialist is given a
# single tool call to make and then returns its report. All responses are JSON
# strings matching the protocol described in coordinator.py / specialist.py.

MOCK_RESPONSES: list[str] = [
    # --- Planner: propose the initial plan ---
    json.dumps({
        "plan": [
            {"step": 1, "specialist": "reconstruction",
             "action": "Load the existing 5-iter CS reconstruction",
             "success_criterion": "shape matches (77,96,72,20)"},
            {"step": 2, "specialist": "segmentation",
             "action": "Segment the largest vessel via PC-MRA seed-based growing",
             "success_criterion": "mask size > 5000 voxels, peak speed > 0.5 m/s"},
            {"step": 3, "specialist": "verifier",
             "action": "Run the four physics checks on the segmented mask",
             "success_criterion": "report verdict and per-check status to Coordinator"},
            {"step": 4, "specialist": "hemodynamic",
             "action": "Compute flow Q(t), stroke volume, peak velocity",
             "success_criterion": "report falls within physiological ranges or is flagged"},
        ],
        "rationale": "Standard pipeline. Loading existing 5-iter recon avoids a 12-min MATLAB run; "
                     "verifier may flag the under-converged data but hemodynamic specialist will "
                     "contextualize.",
    }),
    # --- Plan Critic: approve ---
    json.dumps({
        "verdict": "approve",
        "concerns": ["minor: rationale acknowledges that 5-iter recon may fail verifier; "
                     "Coordinator should still proceed and let Hemodynamic Analyzer flag the issue"],
        "suggestions": "",
    }),

    # --- Coordinator turn 1: delegate to reconstruction ---
    json.dumps({
        "delegate_to": "reconstruction",
        "task": "Load the existing recon at /mnt/g/medict_tmp/recon_cs_5iter.mat (VENC=1.5, voxel=2mm).",
        "why":  "Need data loaded before any other analysis.",
    }),
    # Reconstruction specialist
    json.dumps({
        "tool": "load_reconstruction",
        "args": {"mat_path": "/mnt/g/medict_tmp/recon_cs_5iter.mat"},
        "why":  "User said the recon already exists; loading is faster than rerunning.",
    }),
    json.dumps({
        "done": True,
        "report": "Loaded 5-iter CS reconstruction, shape [77,96,72,20], VENC=1.5 m/s. "
                  "Quality is smoke-test grade; expect divergence/flux verification to fail.",
    }),

    # --- Coordinator turn 2: delegate to segmentation ---
    json.dumps({
        "delegate_to": "segmentation",
        "task": "Find the largest vessel in the loaded reconstruction. Use percentile=90 to start.",
        "why":  "Need a vessel mask before physics verification or hemodynamics.",
    }),
    # Segmentation specialist
    json.dumps({
        "tool": "suggest_seeds",
        "args": {"n_candidates": 3, "percentile": 90},
        "why":  "Inspect candidates before committing to a specific seed.",
    }),
    json.dumps({
        "tool": "segment_from_seed",
        "args": {"seed_z": 54, "seed_y": 32, "seed_x": 56, "mask_name": "aorta_v1",
                  "percentile": 90},
        "why":  "Candidate 0 is the biggest by mean PC-MRA × size; likely the aorta.",
    }),
    json.dumps({
        "done": True,
        "report": "Mask 'aorta_v1' = 33,618 voxels, peak speed 2.59 m/s. Likely the aorta plus "
                  "some neighboring structures that the 90th percentile couldn't fully separate.",
    }),

    # --- Coordinator turn 3: delegate to verifier ---
    json.dumps({
        "delegate_to": "verifier",
        "task": "Run physics verification on mask 'aorta_v1' and tell me if it's trustworthy for hemodynamics.",
        "why":  "Need to know whether the mask survives the divergence and flux checks.",
    }),
    # Verifier specialist
    json.dumps({
        "tool": "verify",
        "args": {"mask_name": "aorta_v1"},
        "why":  "Standard four-check suite; need numerical results to interpret.",
    }),
    json.dumps({
        "done": True,
        "report": "Verdict: FAIL. Divergence is ~63 s⁻¹ (well over 20 threshold) and net flux "
                  "deviates ~200%. Peak velocity and phase unwrap pass. Two failures are consistent "
                  "with two known causes: 5-iter recon is under-converged AND the mask spans "
                  "multiple vessels. Coordinator should either accept partial results or request "
                  "a 50-iter reconstruction + tighter segmentation. For a demo, proceeding with "
                  "hemodynamics is acceptable as long as the report flags the verification failure.",
    }),

    # --- Coordinator turn 4: delegate to hemodynamic anyway, with caveat ---
    json.dumps({
        "delegate_to": "hemodynamic",
        "task": "Compute flow metrics for mask 'aorta_v1'. Note: verifier returned FAIL "
                "due to under-converged recon — please contextualize accordingly.",
        "why":  "Demo value: still want to show physiological metrics even though verification failed.",
    }),
    # Hemodynamic specialist
    json.dumps({
        "tool": "analyze",
        "args": {"mask_name": "aorta_v1"},
        "why":  "Standard hemodynamic analysis; will interpret the numbers next.",
    }),
    json.dumps({
        "done": True,
        "report": "Mean stroke volume: 93.9 mL (normal adult range 60–100 mL). Peak velocity: "
                  "2.59 m/s (elevated; normal aorta ~1.5 m/s; this could be physiological or "
                  "artifact from the multi-vessel mask). Mean peak flow: 438 mL/s (normal aortic "
                  "range 400–600 mL/s). Per-section regurgitation reaches 130% in some sections, "
                  "which is non-physiological — almost certainly a mask-merging artifact, not a "
                  "real valve problem. Bottom line: bulk metrics are physiologically reasonable, "
                  "but per-section results are not trustworthy until segmentation is improved.",
    }),

    # --- Coordinator turn 5: done ---
    json.dumps({
        "done": True,
        "summary": "Analyzed 4D flow MRI scan end-to-end. Verifier flagged the recon quality "
                   "(5-iter CS is under-converged + multi-vessel mask). Hemodynamic Analyzer "
                   "produced bulk metrics within physiological ranges (SV 93.9 mL, peak flow "
                   "438 mL/s) but warned that per-section results aren't trustworthy. "
                   "Recommendation to user: re-run reconstruction at 50 iterations and segment "
                   "with a tighter percentile (95+) before treating per-section numbers clinically.",
    }),
]


# ============================================================================
# Runner
# ============================================================================

def _print_instructions(console: Console, log_path: Path):
    """Show the user which commands to run in each of the seven viewer windows."""
    body = Text()
    body.append("Open 7 terminal windows and paste one command into each.\n", style="bold white")
    body.append("They'll wait for the log file to appear, then stream entries as the agents run.\n\n",
                style="dim")

    activate = "cd ~/projects/medict && conda activate medict"
    cmds = [
        ("PLANNER",                       "cyan",         "planner"),
        ("PLAN CRITIC (LLM Auditor)",     "bright_red",   "plan_critic"),
        ("COORDINATOR",                   "yellow",       "coordinator"),
        ("RECONSTRUCTION OPERATOR",       "magenta",      "reconstruction"),
        ("SEGMENTATION OPERATOR",         "bright_cyan",  "segmentation"),
        ("PHYSICS VERIFIER",              "green",        "verifier"),
        ("HEMODYNAMIC ANALYZER",          "blue",         "hemodynamic"),
    ]
    for title, color, name in cmds:
        body.append(f"  ── {title} ──\n", style=f"bold {color}")
        body.append(f"  {activate}\n", style="white")
        body.append(f"  python demos/agent_window.py {name} {log_path}\n\n", style="white")

    console.print(Panel(body, border_style="bold white",
                        title="[bold]Multi-Agent Demo — open viewer windows first[/]",
                        subtitle="[dim]press ENTER here when ready[/]"))


def _run_plan_phase(
    llm,
    log: AuditLog,
    user_goal: str,
    console: Console,
    *,
    max_revisions: int = 2,
) -> tuple[Plan | None, str | None, str | None]:
    """Planner → Plan Critic → Plan Policy gate.

    Shows a live status spinner while the LLM is thinking; transitions to a
    permanent printed line whenever a phase finishes. Returns
    ``(plan, warning_for_coordinator, halt_reason)``.
    """
    # Live spinner that updates as each agent acts. The verbose_callback on
    # Planner and PlanCritic fires both before (thinking…) and after (result).
    with console.status("[cyan]Planner thinking…[/]", spinner="dots") as status:
        def update_status(msg: str):
            status.update(msg)

        planner = Planner(llm, verbose_callback=update_status)
        critic  = PlanCritic(llm, verbose_callback=update_status)

        plan = planner.propose(user_goal, audit=log)
    console.print(f"[cyan]✓ planner: proposed plan with {len(plan.steps)} steps[/]")

    for attempt in range(max_revisions + 1):
        with console.status("[bright_red]Plan Critic reviewing…[/]", spinner="dots") as status:
            critic.verbose = lambda msg: status.update(msg)
            critique = critic.review(plan, user_goal=user_goal, audit=log)
        decision = apply_plan_policy(critique, revision_count=plan.revision_count,
                                     max_revisions=max_revisions)
        log.event("plan_policy_decision", {
            "action":          decision.action.value,
            "revision_count":  plan.revision_count,
            "concerns":        decision.concerns,
            "reason":          decision.reason,
        })
        verdict_color = {"approve": "green", "revise": "yellow",
                         "reject": "red"}.get(critique.verdict, "white")
        console.print(f"[{verdict_color}]✓ plan critic: {critique.verdict}[/]  "
                      f"[dim]({decision.action.value})[/]")

        if decision.action == PolicyAction.PROCEED:
            return plan, None, None
        if decision.action == PolicyAction.PROCEED_WITH_WARNING:
            return plan, decision.warning_reason, None
        if decision.action == PolicyAction.HALT:
            return None, None, decision.halt_reason
        if decision.action == PolicyAction.REVISE:
            log.event("plan_revision", {"revision_count": plan.revision_count + 1,
                                         "concerns": decision.concerns})
            with console.status(f"[cyan]Planner revising (round {plan.revision_count + 1})…[/]",
                                spinner="dots") as status:
                planner.verbose = lambda msg: status.update(msg)
                plan = planner.revise(user_goal, plan, decision.concerns, audit=log)
            console.print(f"[cyan]✓ planner: emitted revision {plan.revision_count}[/]")

    # Shouldn't get here — apply_plan_policy caps at max_revisions
    return plan, "exited plan loop without an explicit decision", None


def _build_llm(backend: str, model: str):
    if backend == "mock":
        return MockLLM(responses=MOCK_RESPONSES)
    return OllamaLLM(model=model)


def main():
    p = argparse.ArgumentParser(description="Run the multi-agent demo (Coordinator + 4 specialists).")
    p.add_argument("--llm", choices=["ollama", "mock"], default="ollama",
                   help="LLM backend (default: ollama). Use 'mock' for a scripted dry-run "
                        "without Ollama.")
    p.add_argument("--model", default="qwen3.6", help="Ollama model name")
    p.add_argument("--log-path", default=None,
                   help="Audit log path (default: logs/agent_demo.jsonl)")
    p.add_argument("--goal", default=DEFAULT_GOAL, help="User goal handed to the Coordinator")
    p.add_argument("--max-delegations", type=int, default=12)
    p.add_argument("--no-fresh-recon", action="store_true",
                   help="Disable the Reconstruction specialist's ability to call MATLAB. "
                        "By default fresh recon is enabled — a live elapsed-time counter "
                        "in [recon] lines shows MATLAB progress so it doesn't look stuck.")
    args = p.parse_args()
    demo_mode = args.no_fresh_recon

    console = Console()

    log_path = Path(args.log_path) if args.log_path else Path("logs") / "agent_demo.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)

    _print_instructions(console, log_path.absolute())

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[red]cancelled[/]")
        return

    # ---- Construct LLM + workspace + audit log + specialists --------------
    llm = _build_llm(args.llm, args.model)
    ws = Workspace()
    log = AuditLog(log_path, session_metadata={
        "goal":         args.goal,
        "llm_backend":  args.llm,
        "llm_model":    getattr(llm, "model", "?"),
        "architecture": "Planner → PlanCritic → PlanPolicy → Coordinator → 4 specialists",
        "demo_mode":    demo_mode,
    })

    if demo_mode:
        console.print("[dim italic]--no-fresh-recon active: Reconstruction specialist "
                      "can only load existing recons.[/]\n")
    else:
        console.print("[dim italic]Fresh MATLAB reconstruction enabled. If Qwen "
                      "delegates to it, expect 1–12 min per recon; [recon] lines "
                      "will show elapsed time + latest MATLAB output.[/]\n")

    # ---- Phase 1: Planner + Plan Critic + Plan Policy gate --------------
    try:
        plan, warning, halt_reason = _run_plan_phase(llm, log, args.goal, console)
    except Exception as e:
        log.event("plan_phase_error", {"error_type": type(e).__name__, "msg": str(e)})
        log.close(status="error", summary={"error": str(e), "phase": "plan"})
        raise

    if halt_reason is not None:
        log.close(status="halted_by_plan_critic", summary={
            "halt_reason": halt_reason,
            "phase": "plan_critic_rejected",
        })
        console.print()
        console.print(Panel(
            Text(halt_reason, style="bold red"),
            title="[bold red]Plan Critic halted execution[/]",
            subtitle="[dim]see audit log for full critique[/]",
            border_style="red",
        ))
        return

    # ---- Phase 2: Coordinator delegates to specialists ------------------
    specialists = build_default_specialists(llm, demo_mode=demo_mode)

    try:
        with console.status("[yellow]Coordinator starting…[/]", spinner="dots") as status:
            coord = Coordinator(
                llm=llm,
                specialists=specialists,
                workspace=ws,
                audit=log,
                max_delegations=args.max_delegations,
                verbose_callback=lambda msg: status.update(msg),
            )
            result = coord.run(args.goal, plan=plan, warning=warning)
        log.close(status=result.status, summary={
            "n_delegations":     result.n_delegations,
            "specialists_used":  result.specialists_used,
            "final_summary":     result.summary,
            "n_masks":           len(ws.masks),
            "verdicts":          {k: v.get("verdict") for k, v in ws.verdicts.items()},
            "plan_revisions":    plan.revision_count if plan else 0,
            "plan_warning":      warning,
        })
        console.print()
        console.print(Panel(
            Text(result.summary or "(no summary)", style="white"),
            title=f"[bold green]Coordinator finished[/]  ({result.status})",
            subtitle=f"[dim]{result.n_delegations} delegations, "
                     f"specialists: {', '.join(result.specialists_used) or 'none'}[/]",
            border_style="green",
        ))
    except Exception as e:
        log.event("orchestrator_error", {"error_type": type(e).__name__, "msg": str(e)})
        log.close(status="error", summary={"error": str(e), "phase": "coordinator"})
        raise

    console.print(f"\n[bold]Audit log:[/] [white]{log_path}[/]")
    console.print(f"[dim]Replay any window with: python demos/agent_window.py <agent> {log_path} --no-follow[/]")


if __name__ == "__main__":
    main()
