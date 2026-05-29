"""
终极版 V1 Demo — Good Case on Curved-Tapered Phantom.

完整 multi-agent pipeline 在受控的 phantom 输入上跑：
  Planner → Plan Critic → Coordinator → 4 Specialists
  Reconstruction (load_phantom) → Segmentation (passthrough) →
  Verifier (all 4 ✓) → Hemodynamic (analytically-correct numbers)

这是 V1 的 "good case" 半边：
  - Bad case  : demos/single_window_demo.py  (real OSU-MR data, verifier
                 correctly flags merged segmentation)
  - Good case : 本脚本 (clean phantom, full pipeline 干净跑通)

两边对照展示 audit 系统 "对输入质量敏感、诚实报告" 的核心论点。

用法:
    python demos/good_case_demo.py                    # 默认: real Qwen
    python demos/good_case_demo.py --llm mock         # 不需要 Ollama
    python demos/good_case_demo.py --llm ollama       # 显式 Qwen
    python demos/good_case_demo.py --no-napari        # 不弹 napari 验证窗口
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from utility.audit       import AuditLog, extract_thinking_by_agent
from agents.coordinator import Coordinator
from utility.llm         import MockLLM, OllamaLLM
from agents.plan_critic import PlanCritic
from utility.plan_policy import PolicyAction, apply_plan_policy
from agents.planner     import Plan, Planner
from agents.specialist  import build_default_specialists
from utility.tools       import Workspace


# ============================================================================
# Phantom-specific goal
# ============================================================================

PHANTOM_GOAL = (
    "Analyze the hemodynamics of the synthetic curved-tapered aorta phantom. "
    "This is a controlled good-case run with a known incompressible velocity "
    "field and a ground-truth vessel mask. "
    "Step 1: load the phantom via load_phantom (this also places the mask in "
    "the workspace as 'aorta_phantom'). "
    "Step 2: skip segmentation — the mask is already loaded. "
    "Step 3: run the four physics checks on 'aorta_phantom'. "
    "Step 4: compute and report flow Q(t), stroke volume, peak velocity. "
    "Confirm the verifier passes all four checks and the hemodynamic numbers "
    "are physiologically plausible."
)


# ============================================================================
# Scripted responses for --llm mock
# ============================================================================

MOCK_RESPONSES: list[str] = [
    # Planner
    json.dumps({
        "plan": [
            {"step": 1, "specialist": "reconstruction",
             "action": "Load the curved-tapered phantom via load_phantom",
             "success_criterion": "phantom_loaded with mask 'aorta_phantom' in workspace"},
            {"step": 2, "specialist": "segmentation",
             "action": "Confirm phantom mask is present (passthrough; no segmentation needed)",
             "success_criterion": "mask 'aorta_phantom' confirmed in workspace"},
            {"step": 3, "specialist": "verifier",
             "action": "Run 4 physics checks on 'aorta_phantom'",
             "success_criterion": "all 4 checks return 'pass'"},
            {"step": 4, "specialist": "hemodynamic",
             "action": "Compute Q(t), SV, peak velocity on 'aorta_phantom'",
             "success_criterion": "report physiologically plausible numbers"},
        ],
        "rationale": "Phantom is by-construction clean — verifier should pass, hemodynamic "
                     "should produce numbers consistent with the analytic ground truth.",
    }),
    # Plan Critic — approve
    json.dumps({"verdict": "approve", "concerns": [], "suggestions": ""}),
    # Coordinator: delegate reconstruction
    json.dumps({"delegate_to": "reconstruction",
                "task": "Load the phantom via load_phantom (mask_name='aorta_phantom').",
                "why": "Phantom must be loaded before any analysis."}),
    # Reconstruction specialist
    json.dumps({"tool": "load_phantom", "args": {"mask_name": "aorta_phantom"},
                "why": "Load synthetic curved-tapered phantom with ground-truth mask."}),
    json.dumps({"done": True,
                "report": "Phantom loaded. Shape (60,60,60,8), VENC 1.5 m/s, "
                          "ground-truth mask 'aorta_phantom' (~4700 voxels) placed in workspace."}),
    # Coordinator: delegate segmentation
    json.dumps({"delegate_to": "segmentation",
                "task": "Confirm 'aorta_phantom' mask is in the workspace.",
                "why": "Verify the phantom mask is accessible to verifier."}),
    # Segmentation specialist — passthrough
    json.dumps({"done": True,
                "report": "Phantom mask 'aorta_phantom' is present in workspace. "
                          "No segmentation needed; passing through to Verifier."}),
    # Coordinator: delegate verifier
    json.dumps({"delegate_to": "verifier",
                "task": "Run all 4 physics checks on 'aorta_phantom'.",
                "why": "Confirm the phantom satisfies physical invariants."}),
    # Verifier specialist
    json.dumps({"tool": "verify", "args": {"mask_name": "aorta_phantom"},
                "why": "Run divergence / flux / peak / phase-unwrap checks."}),
    json.dumps({"done": True,
                "report": "All 4 checks PASS. Divergence: pass; net_flux: pass; "
                          "peak_velocity: pass; phase_unwrap: pass. Mask is trustworthy."}),
    # Coordinator: delegate hemodynamic
    json.dumps({"delegate_to": "hemodynamic",
                "task": "Compute Q(t), SV, peak velocity on 'aorta_phantom'.",
                "why": "Verifier passed — produce final clinical metrics."}),
    # Hemodynamic specialist
    json.dumps({"tool": "analyze", "args": {"mask_name": "aorta_phantom"},
                "why": "Compute volumetric flow rate, stroke volume, peak velocity."}),
    json.dumps({"done": True,
                "report": "Hemodynamic analysis complete. Numbers consistent with "
                          "phantom's analytic ground truth (pulsatile uniform flow, "
                          "mean velocity ~0.5–1.0 m/s)."}),
    # Coordinator: finish
    json.dumps({"done": True,
                "summary": "Good case demo successful. Pipeline ran end-to-end on the "
                           "curved-tapered phantom; all 4 physics checks passed; "
                           "hemodynamic produced physiologically-plausible numbers. "
                           "This demonstrates the agent system correctly handles clean "
                           "inputs and produces validated clinical metrics."}),
]


# ============================================================================
# Visual identity
# ============================================================================

AGENT_COLORS = {
    "planner":                    "cyan",
    "plan_critic":                "bright_red",
    "coordinator":                "yellow",
    "specialist.reconstruction":  "magenta",
    "specialist.segmentation":    "bright_cyan",
    "specialist.verifier":        "green",
    "specialist.hemodynamic":     "blue",
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


def _make_inline_callback(console: Console, agent_color: str):
    def cb(msg: str):
        console.print(f"   [dim {agent_color}]│ {msg}[/]")
    return cb


def _section_header(console: Console, title: str, color: str):
    console.print()
    console.print(Rule(f"[bold {color}]▶ {title}[/]", style=color))


# ============================================================================
# Final summary panel
# ============================================================================

def _print_final_panel(console: Console, workspace: Workspace) -> None:
    """Print the key results from workspace after the pipeline finishes."""
    console.print()
    console.print(Rule("[bold green]✓ Good Case Demo Complete[/]", style="green"))
    console.print()

    # Verifier verdict
    if workspace.verdicts:
        for mask_name, verdict in workspace.verdicts.items():
            checks = verdict.get("checks", {})
            body = Text()
            body.append(f"Mask: {mask_name}\n", style="bold")
            body.append(f"Overall: ", style="bold")
            v = verdict.get("verdict", "?")
            color = {"pass": "green", "warn": "yellow", "fail": "red"}.get(v, "white")
            body.append(f"{v.upper()}\n", style=f"bold {color}")
            for name, c in checks.items():
                status = c.get("status", "?")
                icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(status, "?")
                body.append(f"  {icon} {name}: {status}\n", style=color)
            console.print(Panel(body, border_style="green",
                                title="[bold]Physics Verifier[/]"))

    # Hemodynamic results — analyze() nests numbers under "summary"
    if workspace.analyses:
        for mask_name, report in workspace.analyses.items():
            body = Text()
            body.append(f"Mask: {mask_name}\n\n", style="bold")
            summary = report.get("summary", {})
            for label, key in [
                ("Mean stroke volume",        "mean_stroke_volume_mL"),
                ("Mean peak Q",               "mean_peak_Q_mL_per_s"),
                ("Mean peak velocity",        "peak_velocity_m_per_s"),
                ("Stroke-volume CV (across sections)", "stroke_volume_cv"),
            ]:
                if key in summary:
                    val = summary[key]
                    unit = ""
                    if "mL" in key and "_per_s" not in key: unit = " mL"
                    elif "_per_s" in key:                   unit = " mL/s" if "Q" in key else " m/s"
                    body.append(f"  {label:<38} {val}{unit}\n", style="white")
            n_sections = len(report.get("per_section", []))
            if n_sections:
                body.append(f"\n  cross-sections analyzed: {n_sections}\n", style="dim")
            console.print(Panel(body, border_style="blue",
                                title="[bold]Hemodynamic Analyzer[/]"))


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="MEDICT Good Case Demo (phantom)")
    p.add_argument("--llm", choices=["mock", "ollama"], default="ollama")
    p.add_argument("--model", default="qwen3.6",
                   help="Ollama model name (ignored for --llm mock)")
    p.add_argument("--llm-host", default=None,
                   help="Override Ollama host URL (default: localhost:11434)")
    p.add_argument("--goal", default=PHANTOM_GOAL)
    p.add_argument("--log", default="logs/good_case_demo.jsonl")
    p.add_argument("--napari", action="store_true",
                   help="Open a napari viewer at the end to visualize the phantom + mask")
    args = p.parse_args()

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]MEDICT — V1 Good Case Demo[/]\n"
        f"LLM backend: {args.llm}  ·  Audit log: {args.log}\n"
        "Pipeline: Planner → Critic → Coordinator → 4 Specialists\n"
        "Input: Curved-tapered phantom (analytically clean, ∇·v = 0)",
        border_style="cyan",
    ))

    # ── LLM backend ──────────────────────────────────────────────
    if args.llm == "mock":
        llm = MockLLM(MOCK_RESPONSES)
    else:
        llm = OllamaLLM(model=args.model,
                        host=args.llm_host or "http://localhost:11434")

    # ── Pipeline ────────────────────────────────────────────────
    workspace = Workspace()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(args.log, session_metadata={
        "goal":         args.goal,
        "llm_backend":  args.llm,
        "llm_model":    getattr(llm, "model", "mock"),
        "architecture": "Planner → PlanCritic → Coordinator → 4 specialists",
        "view":         "good_case",
        "phantom_mode": True,
    })

    specialists = build_default_specialists(llm, phantom_mode=True)
    planner     = Planner(llm, verbose_callback=_make_inline_callback(
                    console, AGENT_COLORS["planner"]))
    critic      = PlanCritic(llm, verbose_callback=_make_inline_callback(
                    console, AGENT_COLORS["plan_critic"]))

    # Planner
    _section_header(console, "Planner", AGENT_COLORS["planner"])
    plan = planner.propose(args.goal, audit=audit)
    console.print(Panel(json.dumps([s for s in plan.steps], indent=2),
                        border_style=AGENT_COLORS["planner"], title="Plan"))

    # Plan Critic
    _section_header(console, "Plan Critic", AGENT_COLORS["plan_critic"])
    critique = critic.review(plan, user_goal=args.goal, audit=audit)
    decision = apply_plan_policy(critique, revision_count=plan.revision_count,
                                 max_revisions=2)
    audit.event("plan_policy_decision", {
        "action":         decision.action.value,
        "revision_count": plan.revision_count,
        "concerns":       decision.concerns,
        "reason":         decision.reason,
    })
    color = "green" if decision.action == PolicyAction.PROCEED else "red"
    console.print(Panel(f"Verdict: {critique.verdict}  →  {decision.action.value}",
                        border_style=color))

    if decision.action not in (PolicyAction.PROCEED, PolicyAction.PROCEED_WITH_WARNING):
        console.print(f"[red]Plan policy halted: {decision.halt_reason or decision.reason}[/]")
        audit.close(status="halted_by_plan_critic",
                    summary={"halt_reason": decision.halt_reason or decision.reason})
        return
    warning = decision.warning_reason if decision.action == PolicyAction.PROCEED_WITH_WARNING else None

    # Coordinator + specialists
    _section_header(console, "Coordinator + Specialists", AGENT_COLORS["coordinator"])
    coordinator = Coordinator(
        llm=llm, specialists=specialists, workspace=workspace, audit=audit,
        max_delegations=8,
        verbose_callback=_make_inline_callback(console, AGENT_COLORS["coordinator"]),
    )

    t0 = time.time()
    result = coordinator.run(args.goal, plan=plan, warning=warning)
    elapsed = time.time() - t0
    audit.close(status=result.status, summary={
        "n_delegations":     result.n_delegations,
        "specialists_used":  result.specialists_used,
        "final_summary":     result.summary,
        "n_masks":           len(workspace.masks),
        "verdicts":          {k: v.get("verdict") for k, v in workspace.verdicts.items()},
    })
    console.print()
    console.print(Panel(result.summary or "(no summary)",
                        border_style=AGENT_COLORS["coordinator"],
                        title=f"Coordinator final ({elapsed:.1f}s, "
                              f"{result.n_delegations} delegations)"))

    # Results panel
    _print_final_panel(console, workspace)

    # Audit summary
    console.print()
    console.print(f"[dim]Audit log: {args.log}[/]")
    console.print(f"[dim]Run: python evaluation/audit_metrics.py {args.log}[/]")

    # Optional napari view
    if args.napari and workspace.recon is not None:
        try:
            import napari
            mask = None
            for m in workspace.masks.values():
                mask = m
                break
            if mask is not None:
                console.print("\n[cyan]Opening napari viewer...[/]")
                vx = workspace.recon["thetaX"] * workspace.venc_m_per_s / np.pi
                vy = workspace.recon["thetaY"] * workspace.venc_m_per_s / np.pi
                vz = workspace.recon["thetaZ"] * workspace.venc_m_per_s / np.pi
                speed_mip = np.sqrt(vx**2 + vy**2 + vz**2).max(axis=-1).astype(np.float32)
                viewer = napari.Viewer(title="MEDICT — Good Case Result")
                viewer.add_image(speed_mip, name="speed (max-over-time)",
                                 colormap="hot")
                viewer.add_image(mask.astype(np.float32),
                                 name="ground-truth mask",
                                 colormap="green", opacity=0.5)
                napari.run()
        except ImportError:
            console.print("[dim](napari not available; skip viz)[/]")


if __name__ == "__main__":
    main()
