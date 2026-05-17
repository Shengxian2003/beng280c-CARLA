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

from agents.audit import AuditLog
from agents.coordinator import Coordinator
from agents.llm import MockLLM, OllamaLLM
from agents.specialist import build_default_specialists
from agents.tools import Workspace


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
    """Show the user which commands to run in each of the six viewer windows."""
    body = Text()
    body.append("Open 6 terminal windows and paste one command into each.\n", style="bold white")
    body.append("They'll wait for the log file to appear, then stream entries as the agents run.\n\n",
                style="dim")

    activate = "cd ~/projects/medict && conda activate medict"
    cmds = [
        ("PLANNER (optional)",            "cyan",         "planner"),
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

    body.append("Note: the Planner window will be empty in the new architecture — the "
                "Coordinator does its own planning. You can skip that one.\n", style="dim italic")

    console.print(Panel(body, border_style="bold white",
                        title="[bold]Multi-Agent Demo — open viewer windows first[/]",
                        subtitle="[dim]press ENTER here when ready[/]"))


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
    args = p.parse_args()

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
        "architecture": "Coordinator + 4 specialists (LLM-pick-LLM)",
    })

    specialists = build_default_specialists(llm)
    coord = Coordinator(
        llm=llm,
        specialists=specialists,
        workspace=ws,
        audit=log,
        max_delegations=args.max_delegations,
        verbose_callback=lambda msg: console.print(f"[dim]{msg}[/]"),
    )

    # ---- Drive the agent loop --------------------------------------------
    try:
        result = coord.run(args.goal)
        log.close(status=result.status, summary={
            "n_delegations":     result.n_delegations,
            "specialists_used":  result.specialists_used,
            "final_summary":     result.summary,
            "n_masks":           len(ws.masks),
            "verdicts":          {k: v.get("verdict") for k, v in ws.verdicts.items()},
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
        log.close(status="error", summary={"error": str(e)})
        raise

    console.print(f"\n[bold]Audit log:[/] [white]{log_path}[/]")
    console.print(f"[dim]Replay any window with: python demos/agent_window.py <agent> {log_path} --no-follow[/]")


if __name__ == "__main__":
    main()
