"""
Summary Agent — the pipeline's final reviewer.

Runs after the Coordinator finishes (success or partial). Reads:
  - what each agent reported
  - the live workspace state (verdicts, analyses, masks)

Emits a single plain-language summary aimed at the end user: what
happened, what numbers came out, whether they're trustworthy, and what
the user should do next.

The summary is also logged under purpose="summarizer" so the UI can
display it prominently in the Results tab.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from utility.audit import AuditLog
from utility.llm import LLM
from utility.project_context import PROJECT_CONTEXT


SUMMARIZER_SYSTEM = """\
You are the **Summary Agent** — the final reviewer of a multi-agent 4D flow
MRI analysis pipeline. The user just ran a pipeline; below you have the
report from each agent and the final workspace state.

## Your job
Produce a concise, plain-language summary for the end user (a clinician or
researcher who is NOT going to read the raw audit log) that answers:

1. **What was the input** (real scan path / synthetic phantom).
2. **Did the pipeline succeed end-to-end, or where did it fail?**
3. **What numerical results were produced** (verifier verdicts + hemodynamic
   metrics) — quote actual numbers, not generic descriptions.
4. **Are those numbers trustworthy?** Why or why not? (Use the verifier's
   verdict + interpretation.)
5. **What should the user do next?**

## Tone + structure
- Clinical reviewer reading a colleague's report. Specific. Honest about
  failures. Quote actual numbers, not "the verifier said it failed".
- Use this markdown structure:

  ### Outcome
  One sentence: success / partial / failed + one-line "why".

  ### Pipeline trace
  - **Reconstruction**: …
  - **Segmentation**: …
  - **Verifier**: …
  - **Hemodynamic**: …

  ### Numbers (if any)
  Stroke volume, peak Q, peak velocity, etc., with physiological-range context.

  ### Caveats
  Bullet list of anything the user should know before trusting / acting on
  the numbers.

  ### Recommendation
  One short paragraph on what to do next.

Keep the whole report **under 400 words**. Skip empty sections rather than
filling them with "N/A".
"""

SUMMARIZER_PROTOCOL = """\

## Response protocol
Reply with ONE JSON object of the form:

{
  "summary": "<your markdown summary>"
}

Nothing else. The markdown will be rendered directly to the user.
"""


@dataclass
class Summarizer:
    llm: LLM
    # Qwen 3.6 35B reasoning easily eats 2000+ tokens before emitting the
    # JSON `summary` field. Allow plenty of headroom so reasoning + output
    # both fit. Output itself is < 400 words; the rest is reasoning.
    max_tokens: int = 8192

    def summarize(
        self,
        *,
        user_goal:        str,
        coordinator_summary: str,
        coordinator_status:  str,
        per_agent_reports:   dict[str, str],   # agent_key → final report text
        workspace_state:     dict,             # verdicts, analyses, masks
        audit:               AuditLog,
        verbose_callback=None,
    ) -> str:
        """Build the prompt, call the LLM, log it, return the summary string."""
        if verbose_callback:
            verbose_callback("[summarizer] thinking…")

        context = self._build_context_block(
            user_goal=user_goal,
            coordinator_summary=coordinator_summary,
            coordinator_status=coordinator_status,
            per_agent_reports=per_agent_reports,
            workspace_state=workspace_state,
        )

        messages = [
            {"role": "system",
             "content": PROJECT_CONTEXT + "\n\n" + SUMMARIZER_SYSTEM + SUMMARIZER_PROTOCOL},
            {"role": "user",
             "content": context},
        ]

        t0 = time.time()
        resp = self.llm.chat(messages, json_mode=True, max_tokens=self.max_tokens)
        audit.llm_call(
            messages=messages, response=resp,
            purpose="summarizer",
            options={"json_mode": True, "elapsed_s": round(time.time() - t0, 2)},
        )

        # Extract the summary text from the JSON
        summary_text = self._extract_summary(resp.text)
        audit.event("summarizer.done", {
            "summary":   summary_text[:400],
            "n_chars":   len(summary_text),
            "elapsed_s": round(time.time() - t0, 2),
        })

        if verbose_callback:
            verbose_callback(f"[summarizer] done ({len(summary_text)} chars)")

        return summary_text

    def _build_context_block(self, *,
                             user_goal: str,
                             coordinator_summary: str,
                             coordinator_status:  str,
                             per_agent_reports:   dict[str, str],
                             workspace_state:     dict) -> str:
        lines: list[str] = []
        lines.append(f"## User goal\n{user_goal}\n")
        lines.append(f"## Coordinator outcome\n"
                     f"Status: {coordinator_status}\n"
                     f"Summary: {coordinator_summary or '(none)'}\n")

        lines.append("## Agent reports")
        for key in ["planner", "plan_critic", "specialist.reconstruction",
                    "specialist.segmentation", "specialist.verifier",
                    "specialist.hemodynamic"]:
            report = per_agent_reports.get(key)
            if not report:
                continue
            lines.append(f"\n### {key}\n{report}\n")

        lines.append("\n## Workspace state (raw)")
        lines.append("```json")
        lines.append(json.dumps(workspace_state, indent=2, default=str))
        lines.append("```")

        lines.append(
            "\nProduce the markdown summary now per the protocol. Keep it under "
            "400 words; quote actual numbers from the workspace state above."
        )
        return "\n".join(lines)

    @staticmethod
    def _extract_summary(text: str) -> str:
        if not text or not text.strip():
            return (
                "(Summary Agent hit the response-token limit before emitting "
                "its JSON output — its reasoning was logged but no summary "
                "could be produced. Raise `max_tokens` in agents/summarizer.py.)"
            )
        try:
            payload = json.loads(text)
            return payload.get("summary") or text
        except json.JSONDecodeError:
            return text
