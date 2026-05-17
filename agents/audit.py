"""Append-only JSONL audit log for the Stage 3 agent.

Captures every LLM call and every tool call with enough fidelity to
reconstruct the decision sequence after the fact. This is the load-bearing
auditability deliverable for Stage 4 evaluation — without it, "the agent
decided X" is unverifiable.

Design constraints:
  - Append-only: a crash loses at most the last entry.
  - JSONL: one record per line, no nested arrays of entries — easy to
    tail, easy to grep, easy to load with `pd.read_json(..., lines=True)`.
  - Self-contained: every entry carries session_id, sequence number, and
    timestamp so logs from concurrent runs can be merged later.
  - Numpy-safe: payloads are sanitized through tools.to_json_safe so the
    log never accidentally contains an ndarray.

Usage:
    from agents.audit import AuditLog
    from agents.llm import OllamaLLM
    from agents.tools import Workspace, call_tool

    log = AuditLog("logs/session_demo.jsonl",
                   session_metadata={"goal": "analyze hemodynamics"})
    llm = OllamaLLM()
    ws  = Workspace()

    resp = llm.chat([{"role": "user", "content": "plan steps"}])
    log.llm_call(messages=[...], response=resp, purpose="planner")

    result = call_tool(ws, "load_reconstruction", {"mat_path": "..."})
    log.tool_call("load_reconstruction", {"mat_path": "..."}, result, latency_ms=500)

    log.event("plan_revised", {"reason": "verifier failed"})
    log.close(status="success", summary={"final_verdict": "pass"})
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .llm import LLMResponse
from .tools import to_json_safe


# ============================================================================
# Writer
# ============================================================================

class AuditLog:
    """Append-only JSONL writer scoped to one agent session."""

    def __init__(
        self,
        path: str | Path,
        *,
        session_metadata: dict | None = None,
        session_id: str | None = None,
    ):
        self.path = Path(path)
        self.session_id = session_id or str(uuid.uuid4())
        self.seq = 0
        self._closed = False
        self._t_start = time.time()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write("session_start", {"metadata": session_metadata or {}})

    # ---- low-level writer ---------------------------------------------------

    def _write(self, kind: str, data: dict) -> None:
        if self._closed:
            raise RuntimeError(f"AuditLog already closed: {self.path}")
        entry = {
            "t": datetime.now(timezone.utc).isoformat(),
            "seq": self.seq,
            "session_id": self.session_id,
            "kind": kind,
            "data": to_json_safe(data),
        }
        self.seq += 1
        # Append + flush so a kill -9 still leaves a valid (truncated) log.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

    # ---- public API ---------------------------------------------------------

    def llm_call(
        self,
        *,
        messages: list[dict],
        response: LLMResponse,
        purpose: str | None = None,
        options: dict | None = None,
    ) -> None:
        """Log a single LLM round-trip.

        ``purpose`` is a free-text tag the orchestrator sets (e.g. "planner",
        "coordinator", "critic") so post-hoc analysis can slice by role.
        """
        self._write("llm_call", {
            "purpose": purpose,
            "options": options or {},
            "messages": messages,
            "response": {
                "text":              response.text,
                "reasoning":         response.reasoning,
                "model":             response.model,
                "latency_ms":        response.latency_ms,
                "prompt_tokens":     response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
        })

    def tool_call(
        self,
        name: str,
        args: dict,
        result: dict,
        *,
        latency_ms: int | None = None,
    ) -> None:
        """Log one tool invocation: what was called, with what arguments, what it returned."""
        self._write("tool_call", {
            "name":       name,
            "args":       args,
            "result":     result,
            "latency_ms": latency_ms,
            "is_error":   "error_type" in result,
        })

    def event(self, name: str, data: dict | None = None) -> None:
        """Log an out-of-band event (plan revisions, retries, user interventions)."""
        self._write("event", {"name": name, "data": data or {}})

    def close(self, *, status: str = "ok", summary: dict | None = None) -> None:
        """Mark the session complete. Further writes raise."""
        elapsed = round(time.time() - self._t_start, 2)
        self._write("session_end", {
            "status":          status,
            "elapsed_s":       elapsed,
            "n_entries_total": self.seq,  # includes this end record after seq += 1
            "summary":         summary or {},
        })
        self._closed = True


# ============================================================================
# Reader + summary
# ============================================================================

def read_log(path: str | Path) -> list[dict]:
    """Load all entries from a JSONL audit log into a list of dicts."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"corrupt log entry at {path}:{i}: {e}") from None
    return entries


def iter_log(path: str | Path) -> Iterator[dict]:
    """Stream entries one at a time without loading the whole file (for tail -f-style tools)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def filter_log(entries: list[dict], *, kind: str | None = None,
               purpose: str | None = None, tool: str | None = None) -> list[dict]:
    """Slice entries by attributes. All filters combine with AND."""
    out = entries
    if kind:
        out = [e for e in out if e["kind"] == kind]
    if purpose:
        out = [e for e in out if e["kind"] == "llm_call" and e["data"].get("purpose") == purpose]
    if tool:
        out = [e for e in out if e["kind"] == "tool_call" and e["data"].get("name") == tool]
    return out


def summarize(entries_or_path) -> dict:
    """Compute aggregate stats over a log: call counts, latencies, token usage, errors.

    Accepts either a list of entries (from ``read_log``) or a path to a JSONL file.
    """
    if isinstance(entries_or_path, (str, Path)):
        entries = read_log(entries_or_path)
    else:
        entries = list(entries_or_path)

    llm = [e for e in entries if e["kind"] == "llm_call"]
    tools = [e for e in entries if e["kind"] == "tool_call"]
    events = [e for e in entries if e["kind"] == "event"]
    start = next((e for e in entries if e["kind"] == "session_start"), None)
    end = next((e for e in entries if e["kind"] == "session_end"), None)

    def _sum_field(items, path):
        total = 0
        for item in items:
            v = item["data"]
            for key in path:
                v = v.get(key) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                total += v
        return total

    tool_names: dict[str, int] = {}
    for t in tools:
        name = t["data"].get("name", "?")
        tool_names[name] = tool_names.get(name, 0) + 1

    return {
        "session_id":             start["session_id"] if start else None,
        "n_entries":              len(entries),
        "n_llm_calls":            len(llm),
        "n_tool_calls":           len(tools),
        "n_events":               len(events),
        "total_llm_latency_ms":   _sum_field(llm, ["response", "latency_ms"]),
        "total_tool_latency_ms":  _sum_field(tools, ["latency_ms"]),
        "total_input_tokens":     _sum_field(llm, ["response", "prompt_tokens"]),
        "total_output_tokens":    _sum_field(llm, ["response", "completion_tokens"]),
        "tool_call_counts":       tool_names,
        "n_tool_errors":          sum(1 for t in tools if t["data"].get("is_error")),
        "session_status":         end["data"]["status"] if end else "incomplete",
        "session_elapsed_s":      end["data"]["elapsed_s"] if end else None,
    }


# ============================================================================
# Pretty-print (for stdout / quick debugging)
# ============================================================================

def format_entry(entry: dict, *, max_chars: int = 200) -> str:
    """Render one log entry as a single human-friendly line."""
    seq = entry["seq"]
    kind = entry["kind"]
    t = entry["t"].split("T")[1][:8]  # HH:MM:SS
    d = entry["data"]

    if kind == "session_start":
        meta = d.get("metadata", {})
        return f"[{t}] #{seq:03d} session_start  metadata={meta}"

    if kind == "session_end":
        return (f"[{t}] #{seq:03d} session_end    "
                f"status={d.get('status')} elapsed={d.get('elapsed_s')}s "
                f"entries={d.get('n_entries_total')}")

    if kind == "llm_call":
        r = d.get("response", {})
        purpose = d.get("purpose") or "—"
        tokens = f"{r.get('prompt_tokens', '?')}/{r.get('completion_tokens', '?')}"
        text = (r.get("text") or "").replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return f"[{t}] #{seq:03d} llm  ({purpose:11s}) {tokens:>11s}tok {r.get('latency_ms', '?')}ms → {text}"

    if kind == "tool_call":
        marker = "✗" if d.get("is_error") else "✓"
        result = d.get("result", {})
        if d.get("is_error"):
            tail = result.get("error", "")
        else:
            # Show a compact summary of the result
            shown = {k: v for k, v in result.items()
                     if k in ("status", "verdict", "mask_name", "size_voxels",
                              "n_returned", "peak_speed_m_per_s", "shape_ZYXT")}
            tail = json.dumps(shown, separators=(",", ":")) if shown else ""
        if len(tail) > max_chars:
            tail = tail[:max_chars] + "..."
        latency = f"{d.get('latency_ms')}ms" if d.get('latency_ms') is not None else "—"
        return f"[{t}] #{seq:03d} tool {marker} {d.get('name'):22s} {latency:>7s}  {tail}"

    if kind == "event":
        return f"[{t}] #{seq:03d} event {d.get('name')}  {json.dumps(d.get('data', {}))}"

    return f"[{t}] #{seq:03d} {kind}  {json.dumps(d)[:max_chars]}"


def pretty_print(path: str | Path, *, max_chars: int = 200) -> None:
    """Dump a log to stdout one line per entry."""
    for e in read_log(path):
        print(format_entry(e, max_chars=max_chars))
