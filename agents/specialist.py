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

from .audit import AuditLog
from .llm import LLM
from .project_context import PROJECT_CONTEXT
from .tools import TOOLS_BY_NAME, ToolSpec, Workspace, call_tool


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
    max_steps: int = 5                # safety cap on internal tool-calling loop
    project_context: str = PROJECT_CONTEXT  # canonical paths + acquisition defaults
                                            # injected into the system prompt

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
        """Receive a task, run an internal tool-using loop, return the final report dict.

        The internal loop is audited under purpose=``specialist.<name>`` so the
        per-specialist viewer window can filter to just that agent's reasoning.
        """
        purpose = f"specialist.{self.name}"
        system_msg = (
            self.project_context + "\n\n"
            + self.system_prompt + "\n\n"
            + self._tools_block()
            + self._response_protocol()
        )
        history: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": task},
        ]

        last_report: dict = {
            "done": False,
            "report": f"specialist {self.name!r} did not produce a report within {self.max_steps} steps",
            "tools_called": [],
        }
        tools_called: list[str] = []

        for step in range(self.max_steps):
            t0 = time.time()
            resp = self.llm.chat(history, json_mode=True, max_tokens=2048)
            audit.llm_call(
                messages=history, response=resp,
                purpose=purpose,
                options={"json_mode": True, "step": step},
            )
            if verbose_callback:
                verbose_callback(f"[{self.name}] step {step}: {(resp.text or '')[:120]}")

            # Parse the JSON action
            try:
                action = json.loads(resp.text)
            except json.JSONDecodeError:
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user",
                                "content": json.dumps({
                                    "error": "your previous reply was not valid JSON; reply with one JSON object per the protocol"
                                })})
                continue

            # Done signal
            if action.get("done"):
                last_report = {
                    "done": True,
                    "report": action.get("report", ""),
                    "tools_called": tools_called,
                    "n_steps": step + 1,
                }
                break

            # Tool call
            tool_name = action.get("tool")
            tool_args = action.get("args", {})
            if tool_name not in self.tool_names:
                err = (f"tool {tool_name!r} is not in your allowed list "
                       f"{self.tool_names}; re-plan with one of those tools or send done=true")
                history.append({"role": "assistant", "content": resp.text})
                history.append({"role": "user", "content": json.dumps({"error": err})})
                continue

            t_tool = time.time()
            result = call_tool(workspace, tool_name, tool_args)
            audit.tool_call(tool_name, tool_args, result,
                            latency_ms=int((time.time() - t_tool) * 1000))
            tools_called.append(tool_name)

            history.append({"role": "assistant", "content": resp.text})
            history.append({"role": "user", "content": json.dumps(result)})

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


def build_default_specialists(llm: LLM) -> dict[str, Specialist]:
    """Construct the four standard specialists, all sharing one LLM backend."""
    return {
        "reconstruction": Specialist(
            name="reconstruction",
            system_prompt=RECONSTRUCTION_PROMPT,
            tool_names=["load_reconstruction", "reconstruct"],
            llm=llm,
        ),
        "segmentation": Specialist(
            name="segmentation",
            system_prompt=SEGMENTATION_PROMPT,
            tool_names=["suggest_seeds", "segment_from_seed"],
            llm=llm,
        ),
        "verifier": Specialist(
            name="verifier",
            system_prompt=VERIFIER_PROMPT,
            tool_names=["verify"],
            llm=llm,
        ),
        "hemodynamic": Specialist(
            name="hemodynamic",
            system_prompt=HEMODYNAMIC_PROMPT,
            tool_names=["analyze"],
            llm=llm,
        ),
    }
