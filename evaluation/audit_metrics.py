"""Stage 4c — Auditability metrics over an agent audit log.

For a course deliverable that claims "auditable multi-agent pipeline", we
need to *measure* auditability. This module mines an audit log JSONL file
and reports concrete metrics:

  - traceability:    can every tool result be traced back to an LLM decision
                     that explains why it was called?
  - reasoning_coverage: what fraction of LLM calls carry chain-of-thought
                       (reasoning field) in the log?
  - decision_density:  total LLM calls per tool call (proxy for "how much
                       thought went into each action")
  - error_explanation: when a verifier returned fail/warn, was the immediately
                      following specialist LLM call's reasoning long enough
                      to plausibly explain it?
  - latency_breakdown: total time, time spent in LLM vs in tools

Usage:
    python evaluation/audit_metrics.py logs/agent_demo.jsonl
    python evaluation/audit_metrics.py logs/*.jsonl --out results/audit_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys, os
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.audit import read_log, filter_log


# Minimum reasoning length to count as an "explanation". Tunable — the value
# 80 chars is roughly "one sentence", which is the smallest reasoning that
# could plausibly justify a non-trivial decision.
MIN_REASONING_CHARS = 80


def compute_auditability(log_path: str | Path) -> dict:
    """Compute auditability metrics for one audit log."""
    entries = read_log(log_path)

    llm_calls  = filter_log(entries, kind="llm_call")
    tool_calls = filter_log(entries, kind="tool_call")
    events     = filter_log(entries, kind="event")
    start = next((e for e in entries if e["kind"] == "session_start"), None)
    end   = next((e for e in entries if e["kind"] == "session_end"),   None)

    # ---- 1. Reasoning coverage ---------------------------------------------
    n_llm_with_reasoning = sum(
        1 for e in llm_calls
        if (e["data"].get("response", {}).get("reasoning") or "").strip()
    )
    n_llm_with_long_reasoning = sum(
        1 for e in llm_calls
        if len((e["data"].get("response", {}).get("reasoning") or "").strip())
           >= MIN_REASONING_CHARS
    )
    reasoning_coverage = n_llm_with_reasoning / len(llm_calls) if llm_calls else 0.0
    long_reasoning_coverage = (
        n_llm_with_long_reasoning / len(llm_calls) if llm_calls else 0.0
    )

    # ---- 2. Decision density (LLM calls per tool call) ---------------------
    decision_density = (len(llm_calls) / len(tool_calls)) if tool_calls else 0.0

    # ---- 3. Traceability: every tool_call has a preceding llm_call ---------
    # Walk entries in order — for each tool_call, find the most recent llm_call
    # before it. A tool call is "traceable" if such an llm_call exists and has
    # a non-empty response.text.
    n_traceable = 0
    last_llm_with_text = None
    for e in entries:
        if e["kind"] == "llm_call":
            text = (e["data"].get("response", {}).get("text") or "").strip()
            if text:
                last_llm_with_text = e
        elif e["kind"] == "tool_call" and last_llm_with_text is not None:
            n_traceable += 1
    traceability = (n_traceable / len(tool_calls)) if tool_calls else 0.0

    # ---- 4. Verifier explanation coverage ---------------------------------
    # For every verify tool_call that returned warn/fail, was the NEXT LLM
    # call (within 2 entries) a specialist with at least one sentence of
    # reasoning explaining the verdict?
    verify_failures = [
        i for i, e in enumerate(entries)
        if e["kind"] == "tool_call"
        and e["data"].get("name") == "verify"
        and (e["data"].get("result", {}).get("verdict") in ("warn", "fail"))
    ]
    n_explained = 0
    for idx in verify_failures:
        for follow_idx in range(idx + 1, min(idx + 4, len(entries))):
            follow = entries[follow_idx]
            if follow["kind"] != "llm_call":
                continue
            reasoning = (follow["data"].get("response", {}).get("reasoning") or "").strip()
            text      = (follow["data"].get("response", {}).get("text") or "").strip()
            if len(reasoning) >= MIN_REASONING_CHARS or len(text) >= MIN_REASONING_CHARS:
                n_explained += 1
                break
    verifier_explanation_coverage = (
        n_explained / len(verify_failures) if verify_failures else None
    )

    # ---- 5. Latency breakdown ---------------------------------------------
    llm_total_ms  = sum(
        int(e["data"].get("response", {}).get("latency_ms") or 0)
        for e in llm_calls
    )
    tool_total_ms = sum(int(e["data"].get("latency_ms") or 0) for e in tool_calls)
    tool_error_count = sum(
        1 for e in tool_calls if e["data"].get("is_error")
    )

    # ---- 6. Tool diversity -------------------------------------------------
    tool_counts: dict[str, int] = {}
    for e in tool_calls:
        n = e["data"].get("name", "?")
        tool_counts[n] = tool_counts.get(n, 0) + 1

    # ---- 7. Per-purpose llm call counts ------------------------------------
    purpose_counts: dict[str, int] = {}
    for e in llm_calls:
        p = e["data"].get("purpose") or "unknown"
        purpose_counts[p] = purpose_counts.get(p, 0) + 1

    return {
        "log_path": str(log_path),
        "session_id": start["session_id"] if start else None,
        "session_status": end["data"]["status"] if end else "incomplete",
        "session_elapsed_s": end["data"]["elapsed_s"] if end else None,

        "n_entries":    len(entries),
        "n_llm_calls":  len(llm_calls),
        "n_tool_calls": len(tool_calls),
        "n_events":     len(events),
        "n_verify_failures":   len(verify_failures),
        "n_tool_errors":       tool_error_count,

        # The headline auditability numbers
        "reasoning_coverage":              round(reasoning_coverage, 4),
        "long_reasoning_coverage":         round(long_reasoning_coverage, 4),
        "decision_density_llm_per_tool":   round(decision_density, 3),
        "traceability":                    round(traceability, 4),
        "verifier_explanation_coverage":   round(verifier_explanation_coverage, 4)
                                             if verifier_explanation_coverage is not None
                                             else None,

        # Latency
        "total_llm_latency_ms":  llm_total_ms,
        "total_tool_latency_ms": tool_total_ms,
        "llm_fraction_of_time":  round(llm_total_ms / max(llm_total_ms + tool_total_ms, 1), 4),

        # Diversity
        "tool_call_counts":     tool_counts,
        "llm_purpose_counts":   purpose_counts,
    }


def aggregate(metrics_list: list[dict]) -> dict:
    """Aggregate per-log metrics into a corpus-level summary."""
    if not metrics_list:
        return {"n_logs": 0}

    keys = [
        "reasoning_coverage", "long_reasoning_coverage",
        "decision_density_llm_per_tool", "traceability",
        "verifier_explanation_coverage", "llm_fraction_of_time",
    ]

    def _mean(key):
        vals = [m[key] for m in metrics_list if m.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n_logs":                len(metrics_list),
        "total_llm_calls":       sum(m["n_llm_calls"]  for m in metrics_list),
        "total_tool_calls":      sum(m["n_tool_calls"] for m in metrics_list),
        "total_verify_failures": sum(m["n_verify_failures"] for m in metrics_list),
        "mean": {k: _mean(k) for k in keys},
    }


def main():
    p = argparse.ArgumentParser(description="Compute Stage 4c auditability metrics")
    p.add_argument("logs", nargs="+", help="audit log JSONL path(s) or glob")
    p.add_argument("--out", default=None, help="JSON output path (default: stdout only)")
    args = p.parse_args()

    # Expand globs
    log_paths = []
    for pat in args.logs:
        log_paths.extend(sorted(glob(pat)))
    if not log_paths:
        sys.exit(f"No logs found from patterns: {args.logs}")

    per_log = [compute_auditability(p) for p in log_paths]
    summary = {
        "n_logs": len(per_log),
        "per_log": per_log,
        "aggregate": aggregate(per_log),
    }

    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
