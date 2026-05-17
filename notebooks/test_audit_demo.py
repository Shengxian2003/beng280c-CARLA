"""Stage 3c demo: simulate an agent loop and produce a real audit log.

Uses MockLLM (scripted responses) so the demo is deterministic and doesn't
need Ollama. The recorded log is the same shape the real orchestrator will
produce in Stage 3d.

Run:
    python notebooks/test_audit_demo.py

After:
    cat logs/demo_session.jsonl              # raw log
    python -c "from agents.audit import pretty_print; pretty_print('logs/demo_session.jsonl')"
    python -c "from agents.audit import summarize; import json; print(json.dumps(summarize('logs/demo_session.jsonl'), indent=2))"
"""
import os, sys, json, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.llm import MockLLM
from agents.tools import Workspace, call_tool
from agents.audit import AuditLog, summarize, pretty_print

LOG_PATH = Path(__file__).parent.parent / "logs" / "demo_session.jsonl"
LOG_PATH.unlink(missing_ok=True)

# ---- Scripted LLM responses ------------------------------------------------
# This mimics what a real Planner→Coordinator loop would emit. Each chunk
# corresponds to one .chat() call from the orchestrator.

planner_plan = json.dumps({
    "plan": [
        "Load the existing reconstruction at /mnt/g/medict_tmp/recon_cs_5iter.mat",
        "Get vessel seed candidates",
        "Segment the largest vessel",
        "Verify physics on the segmentation",
        "If verifier passes (or warn), run hemodynamic analysis",
    ]
})

coord_responses = [
    json.dumps({"tool": "load_reconstruction",
                "args": {"mat_path": "/mnt/g/medict_tmp/recon_cs_5iter.mat"}}),
    json.dumps({"tool": "suggest_seeds",
                "args": {"n_candidates": 3}}),
    json.dumps({"tool": "segment_from_seed",
                "args": {"seed_z": 54, "seed_y": 32, "seed_x": 56,
                         "mask_name": "aorta_v1"}}),
    json.dumps({"tool": "verify", "args": {"mask_name": "aorta_v1"}}),
    json.dumps({"tool": "analyze", "args": {"mask_name": "aorta_v1"}}),
    json.dumps({"done": True,
                "summary": "Hemodynamic analysis complete; verifier flagged divergence + flux (expected for 5-iter recon)."}),
]

llm = MockLLM(responses=[planner_plan] + coord_responses)
ws  = Workspace()
log = AuditLog(LOG_PATH, session_metadata={
    "goal": "Demo agent loop end-to-end with audit logging",
    "data_path": "/mnt/g/medict_tmp/recon_cs_5iter.mat",
    "llm": "MockLLM (scripted)",
})

# ---- Planner turn ----------------------------------------------------------
print("=== planner ===")
plan_msgs = [
    {"role": "system", "content": "You are the Planner. Output JSON with a 'plan' field."},
    {"role": "user", "content": "Analyze hemodynamics of recon_cs_5iter.mat"},
]
resp = llm.chat(plan_msgs, json_mode=True, max_tokens=2048)
log.llm_call(messages=plan_msgs, response=resp, purpose="planner",
             options={"json_mode": True})
print(json.dumps(json.loads(resp.text), indent=2))

# ---- Coordinator loop ------------------------------------------------------
print("\n=== coordinator loop ===")
history = [
    {"role": "system", "content": "You are the Coordinator. Pick the next tool to call."},
    {"role": "user", "content": "Execute the plan."},
]

for step in range(10):
    resp = llm.chat(history, json_mode=True, max_tokens=2048)
    decision = json.loads(resp.text)
    log.llm_call(messages=history, response=resp, purpose="coordinator",
                 options={"json_mode": True})

    if decision.get("done"):
        print(f"\n[step {step}] coordinator: done — {decision.get('summary', '')[:80]}")
        log.event("loop_exit", {"reason": "coordinator_done"})
        break

    name, args = decision["tool"], decision["args"]
    print(f"\n[step {step}] coordinator → {name}({json.dumps(args)})")
    t0 = time.time()
    result = call_tool(ws, name, args)
    latency_ms = int((time.time() - t0) * 1000)
    log.tool_call(name, args, result, latency_ms=latency_ms)

    # Show a compact echo of what the LLM would see back
    preview = json.dumps(result, separators=(",", ":"))[:160]
    print(f"           ← {preview}...")

    # Feed result back into the conversation (mimics what the real orchestrator does)
    history.append({"role": "assistant", "content": resp.text})
    history.append({"role": "user", "content": json.dumps(result)})

log.close(status="success", summary={
    "n_masks_created": len(ws.masks),
    "mask_names": list(ws.masks),
    "verdicts": {k: v.get("verdict") for k, v in ws.verdicts.items()},
})

# ---- Inspect ---------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Audit log written: {LOG_PATH}")
print(f"{'='*70}\n")
pretty_print(LOG_PATH)

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(json.dumps(summarize(LOG_PATH), indent=2))
