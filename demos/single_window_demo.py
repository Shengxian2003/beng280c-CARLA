"""Single-terminal demo — every agent prints inline; thinking summary at the end.

Use when you don't want to open 7 separate viewer windows for the multi-agent
pipeline. All agent activity streams into the single terminal where you ran
this script, color-coded by agent. After the run finishes, a "Thinking
Process" section dumps Qwen's chain-of-thought for every LLM call, grouped
by agent, so you can read the reasoning post-hoc.

Same underlying agents as ``demos/run_demo.py`` — Planner → PlanCritic →
PlanPolicy → Coordinator → 4 specialists. Same audit log shape too.

Usage:
    python demos/single_window_demo.py                      # real Qwen via Ollama
    python demos/single_window_demo.py --llm mock           # scripted dry-run
    python demos/single_window_demo.py --no-fresh-recon     # disable MATLAB call
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
from rich.rule import Rule
from rich.text import Text

from agents.audit import AuditLog, extract_thinking_by_agent
from agents.coordinator import Coordinator
from agents.llm import MockLLM, OllamaLLM
from agents.plan_critic import PlanCritic
from agents.plan_policy import PolicyAction, apply_plan_policy
from agents.planner import Plan, Planner
from agents.specialist import build_default_specialists
from agents.tools import Workspace

# Reuse the canonical goal + mock script from run_demo.py
from demos.run_demo import DEFAULT_GOAL, MOCK_RESPONSES


# ============================================================================
# Per-agent visual identity (must match agent_window.py for consistency)
# ============================================================================

AGENT_COLORS = {
    "planner":                    "cyan",
    "plan_critic":                "bright_red",
    "coordinator":                "yellow",
    "specialist.reconstruction":  "magenta",
    "specialist.segmentation":    "bright_cyan",
    "specialist.verifier":        "green",
    "specialist.hemodynamic":     "blue",
    "unknown":                    "white",
}

AGENT_LABELS = {
    "planner":                    "PLANNER",
    "plan_critic":                "PLAN CRITIC",
    "coordinator":                "COORDINATOR",
    "specialist.reconstruction":  "RECONSTRUCTION",
    "specialist.segmentation":    "SEGMENTATION",
    "specialist.verifier":        "PHYSICS VERIFIER",
    "specialist.hemodynamic":     "HEMODYNAMIC ANALYZER",
}


# ============================================================================
# Inline rendering during the run
# ============================================================================

def _make_inline_callback(console: Console, agent_color: str):
    """Return a verbose_callback that prints one indented line per update."""
    def cb(msg: str):
        console.print(f"   [dim {agent_color}]│ {msg}[/]")
    return cb


def _section_header(console: Console, title: str, color: str):
    """Print a clear visual break announcing a new phase."""
    console.print()
    console.print(Rule(f"[bold {color}]▶ {title}[/]", style=color))


def _print_plan(console: Console, plan: Plan):
    body = Text()
    for s in plan.steps:
        body.append(f"  {s['step']}. ", style="bold cyan")
        body.append(f"{s['specialist']}", style="bold")
        body.append(f" — {s['action']}\n", style="white")
        body.append(f"     [success: {s['success_criterion']}]\n", style="dim italic")
    if plan.rationale:
        body.append(f"\n  Rationale: ", style="dim bold")
        body.append(plan.rationale, style="italic")
    console.print(Panel(body, border_style="cyan",
                        title="[bold cyan]Plan emitted[/]"))


def _print_critique(console: Console, critique, decision):
    color = {"approve": "green", "revise": "yellow", "reject": "red"}.get(
        critique.verdict, "white")
    body = Text()
    body.append("Verdict: ", style="dim")
    body.append(critique.verdict.upper() + "\n", style=f"bold {color}")
    body.append("Policy action: ", style="dim")
    body.append(decision.action.value + "\n", style=f"bold {color}")
    if critique.concerns:
        body.append("\nConcerns:\n", style="dim bold")
        for c in critique.concerns:
            body.append(f"  • {c}\n", style="white")
    if critique.suggestions:
        body.append(f"\nSuggestions: ", style="dim bold")
        body.append(critique.suggestions, style="italic")
    console.print(Panel(body, border_style="bright_red",
                        title="[bold bright_red]Plan critic verdict[/]"))


def _print_run_summary(console: Console, result):
    color = "green" if result.status == "success" else "red"
    console.print()
    console.print(Panel(
        Text(result.summary or "(no summary)", style="white"),
        title=f"[bold {color}]Run complete — {result.status}[/]",
        subtitle=(f"[dim]{result.n_delegations} delegations · "
                  f"{', '.join(result.specialists_used) or 'no specialists used'}[/]"),
        border_style=color,
    ))


def _print_thinking_summary(console: Console, log_path: Path):
    """End-of-run dump: Qwen's reasoning for every LLM call, grouped by agent."""
    by_agent = extract_thinking_by_agent(log_path)

    console.print()
    console.print(Rule("[bold]THINKING PROCESS — Qwen reasoning grouped by agent[/]",
                       style="white"))

    if not any(any(call.get("reasoning") for call in calls)
               for calls in by_agent.values()):
        console.print(
            "\n[dim italic]No reasoning fields captured. This happens when "
            "the LLM backend doesn't expose chain-of-thought (e.g. --llm mock "
            "scripted responses, or non-reasoning models). Try --llm ollama "
            "to see Qwen 3.6's thinking.[/]\n"
        )
        return

    # Split each agent's calls into:
    #   - "conclusion": the call where the LLM emitted {"done": true, ...}
    #   - "steps":      everything else (intermediate tool decisions / planning)
    # The conclusion is the agent's final reasoned answer; steps are the work
    # it did to get there.
    for purpose, calls in by_agent.items():
        color = AGENT_COLORS.get(purpose, "white")
        label = AGENT_LABELS.get(purpose, purpose.upper())
        conclusions, steps = _split_done_vs_steps(calls)

        console.print()
        console.print(Rule(f"[bold {color}]── {label} ──[/]  "
                           f"[dim]{len(conclusions)} conclusion(s), "
                           f"{len(steps)} intermediate step(s)[/]",
                           style=color))

        # ---- Conclusions: prominent panel per done report ----------------
        for call in conclusions:
            reasoning = (call.get("reasoning") or "").strip()
            text      = (call.get("text") or "").strip()
            report    = _extract_done_report(text)

            body = Text()
            if report:
                body.append("Report to Coordinator:\n", style="dim bold")
                for line in report.splitlines():
                    body.append(f"  {line}\n", style="white")
            if reasoning:
                if report:
                    body.append("\nReasoning:\n", style="dim bold italic")
                for line in reasoning.splitlines():
                    body.append(f"  {line}\n", style=f"italic {color}")

            console.print(Panel(
                body,
                title=f"[bold {color}]✓ Conclusion (step {call['step']})[/]",
                subtitle=f"[dim]{call.get('latency_ms', '?')} ms[/]",
                border_style=color,
            ))

        # ---- Steps: compact one-line summary per intermediate decision ---
        if steps:
            console.print(f"   [dim]intermediate steps:[/]")
            for call in steps:
                action = _short_step_action(call.get("text") or "")
                reasoning_chars = len((call.get("reasoning") or "").strip())
                console.print(
                    f"     [{color}]step {call['step']:>2d}[/]  "
                    f"[white]{action}[/]  "
                    f"[dim]({call.get('latency_ms', '?')} ms, "
                    f"{reasoning_chars} chars thinking)[/]"
                )


def _split_done_vs_steps(calls):
    """Partition a list of llm calls into (conclusions, intermediate_steps).

    A call is a "conclusion" if its response.text parses to {"done": true, ...}.
    Anything else (tool call decisions, parse failures, planning) is a step.
    """
    conclusions, steps = [], []
    for call in calls:
        try:
            payload = json.loads(call.get("text") or "")
            if isinstance(payload, dict) and payload.get("done"):
                conclusions.append(call)
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        steps.append(call)
    return conclusions, steps


def _extract_done_report(text):
    """If text is a {"done": true, "report"/"summary": "..."} JSON, return the report string."""
    try:
        payload = json.loads(text)
        return payload.get("report") or payload.get("summary") or ""
    except (json.JSONDecodeError, TypeError):
        return ""


def _short_step_action(text):
    """Compact one-line description of an intermediate (non-done) LLM action."""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            if "tool" in payload:
                return f"→ called {payload['tool']}"
            if "delegate_to" in payload:
                return f"→ delegated to {payload['delegate_to']}"
            if "plan" in payload:
                return f"→ emitted plan with {len(payload['plan'])} steps"
            if "verdict" in payload:
                return f"→ critique verdict: {payload['verdict']}"
    except (json.JSONDecodeError, TypeError):
        pass
    text = (text or "").replace("\n", " ").strip()
    return text[:80] + ("…" if len(text) > 80 else "")


# ============================================================================
# Run phase wrappers
# ============================================================================

def _run_plan_phase_inline(llm, log, user_goal, console, max_revisions=2):
    """Same plan phase as run_demo.py but with inline prints instead of spinners."""
    planner = Planner(llm, verbose_callback=_make_inline_callback(console, "cyan"))
    critic  = PlanCritic(llm, verbose_callback=_make_inline_callback(console, "bright_red"))

    _section_header(console, "PLANNER", "cyan")
    plan = planner.propose(user_goal, audit=log)
    _print_plan(console, plan)

    for attempt in range(max_revisions + 1):
        _section_header(console, "PLAN CRITIC (LLM Auditor)", "bright_red")
        critique = critic.review(plan, user_goal=user_goal, audit=log)
        decision = apply_plan_policy(critique, revision_count=plan.revision_count,
                                     max_revisions=max_revisions)
        log.event("plan_policy_decision", {
            "action":         decision.action.value,
            "revision_count": plan.revision_count,
            "concerns":       decision.concerns,
            "reason":         decision.reason,
        })
        _print_critique(console, critique, decision)

        if decision.action == PolicyAction.PROCEED:
            return plan, None, None
        if decision.action == PolicyAction.PROCEED_WITH_WARNING:
            return plan, decision.warning_reason, None
        if decision.action == PolicyAction.HALT:
            return None, None, decision.halt_reason
        if decision.action == PolicyAction.REVISE:
            log.event("plan_revision", {"revision_count": plan.revision_count + 1,
                                         "concerns": decision.concerns})
            _section_header(console, f"PLANNER (revision {plan.revision_count + 1})",
                            "cyan")
            plan = planner.revise(user_goal, plan, decision.concerns, audit=log)
            _print_plan(console, plan)

    return plan, "exited plan loop without explicit decision", None


def _build_llm(backend, model):
    if backend == "mock":
        return MockLLM(responses=MOCK_RESPONSES)
    return OllamaLLM(model=model)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Single-terminal multi-agent demo. All agents print inline; "
                    "thinking summary at the end.")
    p.add_argument("--llm", choices=["ollama", "mock"], default="ollama")
    p.add_argument("--model", default="qwen3.6")
    p.add_argument("--log-path", default=None,
                   help="Audit log path (default: logs/single_window_demo.jsonl)")
    p.add_argument("--goal", default=DEFAULT_GOAL)
    p.add_argument("--max-delegations", type=int, default=12)
    p.add_argument("--max-plan-revisions", type=int, default=0,
                   help="Max times Planner can rewrite its plan after Critic feedback. "
                        "Default 0 = accept first plan (avoids Qwen 'forgot JSON' crashes "
                        "on long revision reasoning chains).")
    p.add_argument("--no-fresh-recon", action="store_true",
                   help="Disable the Reconstruction specialist's MATLAB tool. "
                        "If you keep MATLAB enabled, [recon] elapsed-time lines "
                        "will print to stderr during the call.")
    args = p.parse_args()
    demo_mode = args.no_fresh_recon

    console = Console()
    log_path = Path(args.log_path) if args.log_path else Path("logs") / "single_window_demo.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)

    console.print(Panel(
        Text.from_markup(
            f"[bold]Multi-agent demo — single terminal[/]\n"
            f"[dim]LLM: {args.llm} ({args.model if args.llm == 'ollama' else 'scripted'})  ·  "
            f"Log: {log_path}[/]"
        ),
        border_style="bold white",
    ))
    if demo_mode:
        console.print("[dim italic]--no-fresh-recon active: Reconstruction "
                      "specialist can only load existing recons.[/]")
    else:
        console.print("[dim italic]Fresh MATLAB reconstruction enabled. If Qwen "
                      "calls it, expect 1–12 min with live [recon] elapsed-time "
                      "lines on stderr.[/]")

    llm = _build_llm(args.llm, args.model)
    ws = Workspace()
    log = AuditLog(log_path, session_metadata={
        "goal":         args.goal,
        "llm_backend":  args.llm,
        "llm_model":    getattr(llm, "model", "?"),
        "architecture": "Planner → PlanCritic → Coordinator → 4 specialists",
        "view":         "single_window",
        "demo_mode":    demo_mode,
    })

    # ---- Phase 1 + 2 (plan + critic + policy) ----------------------------
    try:
        plan, warning, halt_reason = _run_plan_phase_inline(llm, log, args.goal, console,
                                                              max_revisions=args.max_plan_revisions)
    except Exception as e:
        log.event("plan_phase_error", {"error_type": type(e).__name__, "msg": str(e)})
        log.close(status="error", summary={"error": str(e), "phase": "plan"})
        console.print(f"\n[bold red]Plan phase crashed:[/] {e}")
        _print_thinking_summary(console, log_path)
        raise

    if halt_reason is not None:
        log.close(status="halted_by_plan_critic",
                  summary={"halt_reason": halt_reason})
        console.print()
        console.print(Panel(
            Text(halt_reason, style="bold red"),
            title="[bold red]Plan Critic halted execution[/]",
            border_style="red",
        ))
        _print_thinking_summary(console, log_path)
        return

    # ---- Phase 3 (Coordinator + specialists) -----------------------------
    _section_header(console, "COORDINATOR", "yellow")

    specialists = build_default_specialists(llm, demo_mode=demo_mode)
    coord = Coordinator(
        llm=llm, specialists=specialists, workspace=ws, audit=log,
        max_delegations=args.max_delegations,
        verbose_callback=_make_inline_callback(console, "yellow"),
    )

    try:
        result = coord.run(args.goal, plan=plan, warning=warning)
        log.close(status=result.status, summary={
            "n_delegations":     result.n_delegations,
            "specialists_used":  result.specialists_used,
            "final_summary":     result.summary,
            "n_masks":           len(ws.masks),
            "verdicts":          {k: v.get("verdict") for k, v in ws.verdicts.items()},
            "plan_revisions":    plan.revision_count,
            "plan_warning":      warning,
        })
        _print_run_summary(console, result)
    except Exception as e:
        log.event("orchestrator_error", {"error_type": type(e).__name__, "msg": str(e)})
        log.close(status="error", summary={"error": str(e), "phase": "coordinator"})
        console.print(f"\n[bold red]Coordinator crashed:[/] {e}")
        _print_thinking_summary(console, log_path)
        raise

    # ---- Phase 4 (thinking summary) --------------------------------------
    _print_thinking_summary(console, log_path)

    console.print(f"\n[dim]Audit log: {log_path}[/]")


if __name__ == "__main__":
    main()
