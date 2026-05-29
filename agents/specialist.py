"""Specialist agents — each is an LLM with a focused role and a restricted toolset.

A specialist is conceptually an "expert" the Coordinator can delegate work to.
Instead of the Coordinator calling tools directly, it asks a specialist
("Verifier, please check mask aorta_v1 is trustworthy"), and the specialist
internally calls its allowed tools, interprets the raw numerical output, and
sends back a natural-language report.

This gives the multi-agent system real LLM dialogue across roles, not just
tool dispatch from a single Coordinator brain.

Specialists share one Qwen backend on the local GPU (multiple Ollama
instances would serialize on the same model anyway), but each gets its own
system prompt — so they behave like distinct experts with different
priorities and language styles.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from utility.audit import AuditLog
from utility.input_modes import REGISTRY, InputMode, InputProfile, with_no_fresh_recon
from utility.llm import LLM
from utility.project_context import PROJECT_CONTEXT
from utility.tools import TOOLS_BY_NAME, ToolSpec, Workspace, call_tool


# ============================================================================
# Energy budget — visible-to-LLM stamina system
# ============================================================================

@dataclass
class EnergyBudget:
    """
    Bounded compute budget for one specialist task, visible to the LLM.

    Each LLM round consumes 1 unit of `rounds`, regardless of whether it
    produces a tool call, a done, a parse error, or a rejected tool. Tool
    calls are tracked separately for display only — they do NOT have a
    hard cap (the round cap already bounds total work).

    The LLM sees its remaining rounds at every turn and is told explicitly
    when it has hit its final round. This replaces the older 20-step hard
    cap that silently truncated specialists mid-reasoning.
    """
    max_rounds:  int = 8           # hard cap on total LLM calls
    rounds_used: int = 0           # incremented every LLM round
    tool_used:   int = 0           # subset that called tools (display only)

    # Backward-compatible aliases so existing audit/render code still works.
    @property
    def think_max(self) -> int:     return self.max_rounds
    @property
    def think_used(self) -> int:    return self.rounds_used
    @property
    def tool_max(self) -> int:      return self.max_rounds   # soft, == round cap

    @property
    def rounds_remaining(self) -> int:
        return max(0, self.max_rounds - self.rounds_used)

    # Aliases for old UI code expecting these names
    @property
    def think_remaining(self) -> int: return self.rounds_remaining
    @property
    def tool_remaining(self) -> int:  return self.rounds_remaining

    def exhausted(self) -> bool:
        return self.rounds_remaining == 0

    def is_final_round(self) -> bool:
        """True when this is the LAST round the LLM gets."""
        return self.rounds_remaining == 1

    def spend_round(self, *, was_tool: bool = False) -> None:
        self.rounds_used += 1
        if was_tool:
            self.tool_used += 1

    # Backward-compat methods used by older specialist code paths
    def spend_think(self) -> None: self.spend_round(was_tool=False)
    def spend_tool(self)  -> None: self.spend_round(was_tool=True)

    def prompt_block(self) -> str:
        """Compact status string injected at every round."""
        if self.is_final_round():
            return (
                "[ENERGY] ⚠ FINAL ROUND — your next reply MUST be done=true with "
                "a best-effort report. After this round the loop terminates "
                "regardless of what you emit."
            )
        return (
            f"[ENERGY] You have {self.rounds_remaining}/{self.max_rounds} rounds "
            f"remaining. (Tool calls made so far: {self.tool_used}.)\n"
            f"Every reply consumes 1 round, whether you call a tool, retry, or "
            f"emit done. When rounds run low, wrap up with done=true and a "
            f"best-effort report."
        )

    def snapshot(self) -> dict:
        """JSON-safe summary for audit log + UI."""
        return {
            "rounds_used":      self.rounds_used,
            "max_rounds":       self.max_rounds,
            "rounds_remaining": self.rounds_remaining,
            "tool_used":        self.tool_used,
            # Backward-compat fields so older UI code still reads OK
            "think_used":       self.rounds_used,
            "think_max":        self.max_rounds,
            "tool_max":         self.max_rounds,
            "exhausted":        self.exhausted(),
        }


def _replace_or_append_budget(history: list[dict], budget: "EnergyBudget") -> None:
    """
    Keep the LLM's most-recent user turn carrying the live energy block.

    The simplest correct way: append the budget to the LAST user message
    if not already there, otherwise replace the trailing [ENERGY] line.
    """
    if not history or history[-1].get("role") != "user":
        return
    msg = history[-1]
    content = msg["content"] or ""
    # Strip any prior energy block we might have appended
    marker = "\n\n[ENERGY]"
    idx = content.rfind(marker)
    if idx >= 0:
        content = content[:idx]
    msg["content"] = content + "\n\n" + budget.prompt_block()


def _short_action_summary(text: str) -> str:
    """Compress a JSON action into a one-line human summary for the status bar."""
    try:
        payload = json.loads(text)
        if payload.get("done"):
            return "done — report written"
        if "tool" in payload:
            return f"calling {payload['tool']}"
        return text[:80]
    except (json.JSONDecodeError, TypeError):
        return (text or "")[:80].replace("\n", " ")


# ============================================================================
# Specialist
# ============================================================================

@dataclass
class Specialist:
    """One specialist LLM agent with a focused role and a tool allowlist."""

    name: str                         # e.g. "verifier"
    system_prompt: str                # explains role + how to respond
    tool_names: list[str]             # subset of TOOLS_BY_NAME the specialist may call
    llm: LLM
    # Energy budget — total LLM rounds allowed. Each LLM call consumes 1
    # round regardless of outcome (tool, done, retry). LLM sees remaining
    # budget every round and is warned explicitly on the final round.
    max_rounds:  int = 8
    max_tokens:  int = 8192           # Qwen 3.6 reasoning eats 500–2000 tokens
    project_context: str = PROJECT_CONTEXT

    def _tools_block(self) -> str:
        """Render this specialist's allowed tools as a markdown block for its prompt."""
        lines = ["## Tools you can call\n"]
        for n in self.tool_names:
            spec = TOOLS_BY_NAME.get(n)
            if spec is None:
                continue
            lines.append(f"### {n}\n{spec.description.strip()}\n")
            lines.append("Parameters: ```json\n"
                         + json.dumps(spec.parameters, indent=2)
                         + "\n```\n")
        return "\n".join(lines)

    def _response_protocol(self) -> str:
        return (
            "\n\n## Response protocol\n"
            "At every turn, reply with ONE JSON object. Two shapes are valid:\n\n"
            "  To call one of your tools:\n"
            '    {\"tool\": \"<name>\", \"args\": {...}, \"why\": \"<one-sentence rationale>\"}\n\n'
            "  To finish and send a report back to the Coordinator:\n"
            '    {\"done\": true, \"report\": \"<your natural-language report>\"}\n\n'
            "Always end with a 'done' response; do not call tools forever. "
            "When you write the report, explain WHAT you found and WHAT the Coordinator "
            "should do with it. Quote specific numbers from the tool results."
        )

    def handle(
        self,
        task: str,
        *,
        workspace: Workspace,
        audit: AuditLog,
        verbose_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Receive a task, run a bounded internal tool-using loop.

        Budget: `think_budget` non-tool-call rounds + `tool_budget` tool calls.
        The LLM sees its remaining budget every round; when exhausted the loop
        terminates and the last partial report is returned.
        """
        purpose       = f"specialist.{self.name}"
        budget        = EnergyBudget(max_rounds=self.max_rounds)
        # ── Tool-call deduplication ────────────────────────────────
        # Same (tool, args) cannot be invoked twice in one specialist task.
        # Maps (tool_name, args-json-key) → previous result (for the warning).
        prior_calls: dict[tuple[str, str], dict] = {}
        system_msg = (
            self.project_context + "\n\n"
            + self.system_prompt + "\n\n"
            + self._tools_block()
            + self._response_protocol()
        )
        history: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"{task}\n\n{budget.prompt_block()}"},
        ]
        audit.event(f"specialist.{self.name}.start", {
            "task": task[:200],
            "budget": budget.snapshot(),
        })

        last_report: dict = {
            "done":         False,
            "report":       f"specialist {self.name!r} exhausted its energy budget without emitting done",
            "tools_called": [],
        }
        tools_called: list[str] = []
        step = 0

        while not budget.exhausted():
            # Refresh the energy block on the LAST user turn so the LLM always
            # sees the *current* remaining rounds (including the FINAL ROUND
            # warning). We re-render the budget block here instead of in the
            # initial user message above.
            _replace_or_append_budget(history, budget)

            if verbose_callback:
                verbose_callback(
                    f"[{self.name}] step {step}: deciding "
                    f"(rounds {budget.rounds_remaining}/{budget.max_rounds}, "
                    f"tools used: {budget.tool_used})"
                )
            resp = self.llm.chat(history, json_mode=True, max_tokens=self.max_tokens)
            audit.llm_call(
                messages=history, response=resp,
                purpose=purpose,
                options={"json_mode": True, "step": step, "budget": budget.snapshot()},
            )
            if verbose_callback:
                verbose_callback(
                    f"[{self.name}] step {step}: {_short_action_summary(resp.text or '')}"
                )

            # ── Parse JSON ────────────────────────────────────────────
            try:
                action = json.loads(resp.text)
            except json.JSONDecodeError:
                budget.spend_round(was_tool=False)
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({
                    "error":  "your previous reply was not valid JSON; "
                              "reply with one JSON object per the protocol",
                    "budget": budget.snapshot(),
                })})
                step += 1
                continue

            # ── Done ──────────────────────────────────────────────────
            if action.get("done"):
                budget.spend_round(was_tool=False)
                last_report = {
                    "done":         True,
                    "report":       action.get("report", ""),
                    "tools_called": tools_called,
                    "n_steps":      step + 1,
                    "budget":       budget.snapshot(),
                }
                break

            # ── Tool call ────────────────────────────────────────────
            tool_name = action.get("tool")
            tool_args = action.get("args", {})
            if tool_name not in self.tool_names:
                budget.spend_round(was_tool=False)
                err = (f"tool {tool_name!r} is not in your allowed list "
                       f"{self.tool_names}; re-plan with one of those tools or send done=true")
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({
                    "error":  err,
                    "budget": budget.snapshot(),
                })})
                step += 1
                continue

            # ── Deduplication: same (tool, args) twice = waste ──────
            call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_key in prior_calls:
                budget.spend_round(was_tool=False)
                cached = prior_calls[call_key]
                audit.event(f"specialist.{self.name}.duplicate_call_blocked", {
                    "tool":  tool_name,
                    "args":  tool_args,
                })
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({
                    "error": (
                        f"DUPLICATE: you already called {tool_name} with these exact "
                        f"args earlier in this task. The previous result is below. "
                        f"Do NOT call it again — interpret the cached result and emit "
                        f"done=true with your report now."
                    ),
                    "cached_result": cached,
                    "budget":        budget.snapshot(),
                })})
                step += 1
                continue

            t_tool = time.time()
            result = call_tool(workspace, tool_name, tool_args)
            audit.tool_call(tool_name, tool_args, result,
                            latency_ms=int((time.time() - t_tool) * 1000))
            tools_called.append(tool_name)
            budget.spend_round(was_tool=True)
            prior_calls[call_key] = result

            history.append({"role": "assistant", "content": resp.text})
            history.append({"role": "user", "content": json.dumps({
                "tool_result": result,
                "budget":      budget.snapshot(),
            })})
            step += 1

        audit.event(f"specialist.{self.name}.end", {
            "completed_with_done": last_report.get("done", False),
            "n_steps":             step + (1 if last_report.get("done") else 0),
            "budget":              budget.snapshot(),
            "tools_called":        tools_called,
        })
        return last_report


# ============================================================================
# Standard specialist configurations
# ============================================================================

VERIFIER_PROMPT = """\
You are the **Physics Verifier** — the deterministic-trust specialist for
4D flow MRI analyses.

Your job: when the Coordinator gives you a mask name, run the Physics Verifier
on it and judge whether the velocity field is trustworthy enough for downstream
clinical metrics.

Domain knowledge you bring:
- ∇·v = 0 (divergence) should hold inside incompressible blood. Typical good values
  are < 5 s⁻¹ (warn) or < 20 s⁻¹ (fail).
- Cross-section flux should be conserved (continuity); deviations > 25 % usually
  mean the mask spans multiple vessels with different flow directions.
- Peak speed magnitude in cardiac vessels is typically 0.5–3 m/s. Can legitimately
  exceed VENC since it combines all three encoding components.
- Phase-unwrap artifacts produce voxel-pair jumps > VENC. Fraction > 1 % = fail.

If the verifier fails, your report should *interpret* WHY (under-converged
reconstruction? merged vessel? wrap-around artifact?) and tell the Coordinator
what to do (re-run reconstruction at more iterations, re-segment with tighter
percentile, etc.).

Be terse. Quote the numerical values that drove your judgment.
"""

HEMODYNAMIC_PROMPT = """\
You are the **Hemodynamic Analyzer** — the clinical interpretation specialist
for 4D flow MRI metrics.

Your job: when the Coordinator gives you a verified mask, compute volumetric
flow and stroke volume, then contextualize the numbers physiologically.

Domain knowledge you bring:
- Adult stroke volume at rest: ~60–100 mL per beat.
- Adult cardiac output: typically 4–8 L/min.
- Peak aortic flow rate: typically 400–600 mL/s.
- Peak velocity in normal aorta: ~1.5 m/s; > 2.5 m/s suggests stenosis or
  flow acceleration.
- Regurgitation > 20 % is clinically significant; > 50 % is severe.

Your report should call out anything physiologically unusual and explain what
that suggests, distinguishing between "the metric is real and clinically
notable" vs "the metric is unusual because of upstream pipeline quality
(e.g. merged vessel inflating SV)".

Be terse. Quote the numerical values that drove your judgment.
"""

SEGMENTATION_PROMPT = """\
You are the **Segmentation Operator** — the vessel-isolation specialist.

Your job: given a reconstruction, find a clean vessel mask to feed downstream
analyzers. You can ask for seed candidates and try region-grow segmentation
with adjustable percentile/closing parameters.

Domain knowledge you bring:
- A good vessel mask: > 5000 voxels, peak speed > 0.5 m/s, single connected
  component, not bridging into the heart chambers.
- Higher percentile (95+) breaks apart merged vessels but may lose lumen voxels.
- Lower percentile (75–85) keeps lumen but can bridge unrelated vessels.
- Closing_iter=1 is usually enough; > 2 causes excessive merging.

If a first attempt is clearly bad (empty, tiny, multi-vessel by inspection),
try a different seed or different percentile before reporting back.

Be terse. Quote the numbers (size, peak speed) for your chosen mask.
"""

RECONSTRUCTION_PROMPT = """\
You are the **Reconstruction Operator** — the data-prep specialist.

⚠ **SCOPE**: You ONLY load or run reconstructions. You DO NOT:
  - run physics verification (that's the Verifier specialist's job)
  - segment vessels (that's the Segmentation specialist's job)
  - compute hemodynamic metrics (that's the Hemodynamic Analyzer's job)

If the task description mentions verification, segmentation, or analysis,
those are downstream stages the Coordinator will handle by delegating to
OTHER specialists. Your report MUST NOT claim to have done any of those —
only what you actually did with your own tools.

⚠ **TOOL DISCIPLINE**: Use ONLY tools listed in your allowed-tools block.
If `reconstruct` is not in your list, DO NOT call it — calling tools you
do not have wastes your energy budget. When in load-only mode, the right
choice is `load_reconstruction` or `load_phantom` (whichever matches your
task). Pick ONE, call it once, then emit done.

Your job: choose reconstruction parameters and either load an existing
reconstruction or run a new one if quality is insufficient.

Domain knowledge you bring:
- 5 iterations: fast (~1–2 min) but under-converged. OK for smoke tests, not
  for clinical metrics.
- 50 iterations: full quality (~12 min on RTX 5090). Recommended for divergence
  and flux verification to pass.
- "cs" (compressed sensing): standard. "core" (CORe): adds outlier rejection,
  better for motion-corrupted scans.
- If a reconstruction is already loaded, calling load again is usually a
  no-op unless the user changed datasets.

If the Coordinator says results look noisy or divergence is failing, the right
call is usually to re-run at 50 iterations.

Be terse. Quote the iteration count, method, and elapsed time in your report.
"""


def build_default_specialists(
    llm: LLM,
    *,
    input_profile: InputProfile | None = None,
    demo_mode: bool = False,
    phantom_mode: bool = False,    # legacy kwarg — kept for backward compat
) -> dict[str, Specialist]:
    """Construct the four standard specialists, all sharing one LLM backend.

    Parameters
    ----------
    input_profile : InputProfile, optional
        Preferred way to choose what input the recon specialist is set up
        for. Drives `recon_tools`, recon prompt suffix, and segmentation
        passthrough behavior. See ``agents/input_modes.py``.
    demo_mode : bool
        When True, the Reconstruction specialist loses access to the
        ``reconstruct`` tool (MATLAB takes 12 minutes; not suitable for a
        live demo). Only applies to REAL_SCAN.
    phantom_mode : bool
        Legacy kwarg. ``phantom_mode=True`` is equivalent to
        ``input_profile=REGISTRY[InputMode.PHANTOM]``. Cannot be combined
        with an explicit ``input_profile``.
    """
    # ── Resolve which profile to use ──────────────────────────────────────
    if input_profile is not None and phantom_mode:
        raise ValueError("pass either input_profile= or phantom_mode=, not both")
    if input_profile is None:
        input_profile = REGISTRY[InputMode.PHANTOM] if phantom_mode \
                        else REGISTRY[InputMode.REAL_SCAN]
    if demo_mode:
        input_profile = with_no_fresh_recon(input_profile)

    recon_tools  = list(input_profile.recon_tools)
    recon_prompt = RECONSTRUCTION_PROMPT + input_profile.recon_prompt_suffix
    seg_prompt   = SEGMENTATION_PROMPT  + input_profile.seg_prompt_suffix

    # Per-specialist round budget — task complexity differs:
    #   Recon/Verifier/Hemo: load → done, ~2 useful rounds + 1 slack
    #   Segmentation:        may legitimately try multiple seeds + parameters
    return {
        "reconstruction": Specialist(
            name="reconstruction",
            system_prompt=recon_prompt,
            tool_names=recon_tools,
            llm=llm,
            max_rounds=3,
        ),
        "segmentation": Specialist(
            name="segmentation",
            system_prompt=seg_prompt,
            tool_names=["suggest_seeds", "segment_from_seed"],
            llm=llm,
            max_rounds=8,
        ),
        "verifier": Specialist(
            name="verifier",
            system_prompt=VERIFIER_PROMPT,
            tool_names=["verify"],
            llm=llm,
            max_rounds=3,
        ),
        "hemodynamic": Specialist(
            name="hemodynamic",
            system_prompt=HEMODYNAMIC_PROMPT,
            tool_names=["analyze"],
            llm=llm,
            max_rounds=3,
        ),
    }
