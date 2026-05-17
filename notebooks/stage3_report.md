# Stage 3 Report — Agent Foundations
**Project:** MEDICT — Multi-Agent 4D Flow MRI Pipeline
**Course:** BENG 280C, UCSD
**Stages covered:** 3a (LLM Backend) + 3b (Skill-as-Tool Wrappers)
**Date:** 2026-05-16

---

## Executive Summary

Stage 3 builds the substrate the LLM agent runs on. Two pieces:

1. **Stage 3a — LLM Backend Abstraction** (`agents/llm.py`).
   A vendor-neutral `LLM` interface with three concrete implementations: `OllamaLLM` (local Qwen 3.6 via HTTP), `ClaudeLLM` (Anthropic API for evaluation), and `MockLLM` (scripted responses for tests). All return the same `LLMResponse` dataclass carrying text, internal reasoning, latency, and token counts.

2. **Stage 3b — Tool Wrappers** (`agents/tools.py`).
   Six tools expose the four Stage-2 skills (`reconstruct`, `segment`, `verify`, `analyze`) to the LLM as JSON-schema-described functions. A `Workspace` keeps ndarrays in Python while the LLM reasons over compact JSON handles. A custom validator + sanitizer keeps the bridge type-safe and prevents 8 MB ndarrays from being dumped into prompts.

Combined test coverage: **47 new tests** (17 LLM + 30 Tools), all passing. Real-data smoke test confirms the tool layer reproduces direct-skill numbers exactly (33,618 vox, 2.59 m/s, 93.9 mL SV).

Together these two stages unblock Stage 3c (audit log) and Stage 3d (the orchestrator loop) — both of which sit on top of an `LLM` instance and call `call_tool(ws, name, args)`.

---

## How 3a + 3b Compose

```
                    ┌──────────────────────────────────┐
                    │  agents.llm                      │  Stage 3a
                    │  ┌────────────────────────────┐  │
                    │  │   LLM (abstract)           │  │
                    │  │     ↑                      │  │
                    │  │  OllamaLLM   (Qwen 3.6)    │  │
                    │  │  ClaudeLLM   (Claude API)  │  │
                    │  │  MockLLM     (tests)       │  │
                    │  └────────────────────────────┘  │
                    │  returns LLMResponse{            │
                    │    text, reasoning, latency_ms,  │
                    │    prompt_tokens, …, raw         │
                    │  }                               │
                    └──────────────┬───────────────────┘
                                   │
                            future Stage 3d orchestrator
                            chooses { tool, args } from
                            LLM JSON output
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  agents.tools                    │  Stage 3b
                    │                                  │
                    │  call_tool(workspace, name, args)│
                    │     │                            │
                    │     ├─ validate args (JSON Sch)  │
                    │     ├─ dispatch to ToolSpec.fn   │
                    │     └─ sanitize result for JSON  │
                    │                                  │
                    │  Workspace {                     │
                    │    recon: dict | None,           │
                    │    masks: dict[str, ndarray],    │
                    │    suggested_seeds: list | None, │
                    │    verdicts, analyses,           │
                    │    venc, voxel_size, dt          │
                    │  }                               │
                    │                                  │
                    │  Six ToolSpecs:                  │
                    │    load_reconstruction           │
                    │    reconstruct                   │
                    │    suggest_seeds                 │
                    │    segment_from_seed             │
                    │    verify                        │
                    │    analyze                       │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  Stage 2 skills (unchanged)      │
                    │  • skills.reconstruction         │
                    │  • skills.segmentation           │
                    │  • skills.physics_verifier       │
                    │  • skills.hemodynamic            │
                    └──────────────────────────────────┘
```

Key separation: the LLM never touches the Stage-2 skills directly, never sees ndarrays, and never blocks on validation logic. Everything goes through `call_tool()`. This means swapping Qwen for Claude is one line of config; switching tool implementations doesn't change the LLM call site.

---

## Stage 3a — LLM Backend Abstraction

### Files
| File | Purpose |
|---|---|
| `agents/llm.py` | `LLM` ABC + `OllamaLLM` + `ClaudeLLM` + `MockLLM` + `LLMResponse` + `get_llm()` factory |
| `tests/test_llm.py` | 17 tests (14 unit + 3 live Qwen integration) |
| `notebooks/llm_playground.py` | Interactive REPL with slash-commands |

### Interface

A single abstract base class — all backends are interchangeable behind it:

```python
class LLM(ABC):
    model: str

    @abstractmethod
    def chat(self, messages, *, temperature=0.2, max_tokens=2048,
             json_mode=False) -> LLMResponse: ...

    def chat_json(self, messages, *, temperature=0.2, max_tokens=2048):
        # convenience wrapper: chat() with json_mode=True + json.loads
        return self.chat(messages, ..., json_mode=True).json()
```

`messages` is the OpenAI convention `[{"role": "system|user|assistant", "content": "..."}]` — both Ollama and Anthropic accept this form (Claude requires `system` split out, but `ClaudeLLM` handles that internally).

### LLMResponse

```python
@dataclass
class LLMResponse:
    text: str                       # the answer (for conversation history)
    model: str                      # model name as returned by the backend
    latency_ms: int                 # wall-clock time
    reasoning: str | None = None    # chain-of-thought (Qwen 3.6, future Claude extended thinking)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict = field(default_factory=dict)   # backend-specific payload for the audit log
```

Critical design choice: **reasoning is separate from text.** The agent's conversation history forwards only `text`, while the audit log keeps `reasoning` for transparency. Without this separation, multi-turn conversations would either include 500+ tokens of "thinking" in every history snapshot (token waste + confused replies) or lose auditability of how the model decided.

### The three backends

#### OllamaLLM — local development

Talks to a running Ollama daemon over HTTP (`POST /api/chat`). **No `ollama` Python package required** — uses `requests` directly, which is already in the project. Default host `http://localhost:11434`, default timeout 600 seconds (long enough for 50-iteration reconstructions during the agent loop).

```python
llm = OllamaLLM(model="qwen3.6")
resp = llm.chat([{"role": "user", "content": "What is 2 + 2?"}])
print(resp.text)        # "4"
print(resp.reasoning)   # "The user is asking a simple arithmetic question..."
```

Used for: all of Stage 3 development (free, runs entirely on the user's RTX 5090 host).

#### ClaudeLLM — final evaluation runs

Wraps the `anthropic` Python SDK. The SDK is an **optional** dependency — `ClaudeLLM`'s import of `anthropic` is deferred until the first call, with a clear error if it's not installed. Requires `ANTHROPIC_API_KEY` either passed explicitly or via environment variable.

Used for: Stage 4 final benchmarks where we want Anthropic's strongest model for reproducibility against the literature.

#### MockLLM — tests and dry-runs

Returns scripted responses in order. Records every call (messages + options) for inspection. Raises `RuntimeError` if exhausted.

```python
mock = MockLLM(responses=["Hi", '{"action": "verify"}'])
mock.chat([{"role": "user", "content": "..."}])      # → "Hi"
mock.chat_json([{"role": "user", "content": "..."}]) # → {"action": "verify"}
assert mock.calls[0][1]["json_mode"] is False        # call records persist
```

Used for: every Stage 3d orchestrator test (deterministic, no LLM dependency, ~ms not seconds).

### The Qwen 3.6 reasoning-budget gotcha (load-bearing)

During first integration testing, both live Qwen tests failed with empty responses. Initial hypothesis: bug in the HTTP wrapper. Root cause was different:

**Qwen 3.6 is a reasoning model.** Ollama returns its replies in the form:
```json
{
  "message": {
    "role": "assistant",
    "content": "OK",                          // ← the actual answer
    "thinking": "The user wants me to reply with just OK..."   // ← chain-of-thought
  }
}
```

The `thinking` field consumes tokens against the `num_predict` budget. With `max_tokens=20` in the test, the model spent all 20 tokens on internal reasoning and emitted `content=""`. Ollama is silent about this — `done_reason` was `"stop"`, not anything indicating budget exhaustion.

**Fix:** `OllamaLLM` reads both `content` (→ `text`) and `thinking` (→ `reasoning`). The default `max_tokens` was bumped to 2048 (was 512). Test budgets are 500-800 tokens to leave room for both reasoning and answer.

**Why this matters going forward:** Every agent prompt must budget for reasoning + answer. The orchestrator (Stage 3d) will set `max_tokens >= 1500` for any Qwen call that needs a JSON response. If we ever measure "the agent froze," check `resp.reasoning` length first — it's usually the same problem.

### Playground REPL

`notebooks/llm_playground.py` is the developer-facing tool for sanity-checking Qwen behavior before wiring prompts into the agent. CLI flags:

```bash
python notebooks/llm_playground.py [--backend ollama|claude|mock]
                                   [--model qwen3.6]
                                   [--system "You are ..."]
                                   [--json]
                                   [--temp 0.2]
                                   [--think]
```

Inside the REPL, slash-commands let you adjust state mid-conversation:

| Command | Purpose |
|---|---|
| `/system <text>` | Set or replace the system prompt |
| `/json on\|off` | Toggle JSON output mode |
| `/temp <float>` | Sampling temperature (0.0 = deterministic) |
| `/think on\|off` | Show internal reasoning inline |
| `/reset` | Clear history (keeps system message) |
| `/show` | Print current conversation |
| `/save <path>` | Write conversation to JSON |
| `/quit` | Exit |

Each turn prints `[latency_ms ms, in=N out=N tok]` plus `thinking=N chars` if reasoning was produced but not shown.

### Tests

| Test class | Count | Description |
|---|---|---|
| `TestLLMResponse` | 2 | `.json()` helper round-trip and error |
| `TestMockLLM` | 5 | Scripted responses in order, call recording, exhaustion, deep-copy safety |
| `TestClaudeLLM` | 3 | API-key required, env var fallback, explicit override (no live call) |
| `TestGetLLM` | 4 | Factory wiring, unknown backend rejected |
| `TestOllamaLive` (skip if Ollama down) | 3 | Basic chat, JSON mode, reasoning field captured |

**17/17 passing.** Live tests auto-skip via `_ollama_up()` probe — CI on machines without Ollama still gets clean unit-test results.

---

## Stage 3b — Skill-as-Tool Wrappers

### Files
| File | Purpose |
|---|---|
| `agents/tools.py` | 6 `ToolSpec`s + `Workspace` + `call_tool()` + validator + sanitizer + `tools_prompt_block()` |
| `tests/test_tools.py` | 30 tests (validation, sanitization, dispatcher, each tool end-to-end) |

### Workspace — why it exists

The naive design would have tools take ndarrays as arguments and return ndarrays. This breaks the moment you try to JSON-serialize: a single `xHat` array is 77×96×72×20 complex128 = ~85 MB. Dumping that into an LLM prompt is impossible, dumping its `.tolist()` representation is still 85 MB of text.

The solution: ndarrays stay in process, the LLM refers to them by string handles it picks itself. The container is `Workspace`:

```python
@dataclass
class Workspace:
    # Current reconstruction (None until loaded)
    recon: dict | None = None     # {xHat, thetaX, thetaY, thetaZ, meta, ...}

    # Named vessel masks the LLM has segmented this session
    masks: dict[str, np.ndarray] = field(default_factory=dict)

    # Cached PC-MRA seed candidates (LLM asks once, refers back by index)
    suggested_seeds: list[dict] | None = None

    # Per-mask results
    verdicts: dict[str, dict] = field(default_factory=dict)
    analyses: dict[str, dict] = field(default_factory=dict)

    # Acquisition config shared across tools
    venc_m_per_s: float = 1.5
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0)
    dt_seconds: float = 0.05
```

The LLM-visible flow becomes:

```
LLM → call_tool("load_reconstruction", {"mat_path": "..."})
        → Workspace gets ndarrays
        ← returns 200-byte JSON summary: shape, venc, voxel_size

LLM → call_tool("segment_from_seed", {"seed_z": 54, "seed_y": 32, "seed_x": 56,
                                       "mask_name": "aorta_v1"})
        → Workspace.masks["aorta_v1"] = ndarray
        ← returns: {status: "ok", size_voxels: 33618, peak_speed_m_per_s: 2.59, ...}

LLM → call_tool("verify", {"mask_name": "aorta_v1"})
        → looks up workspace.masks["aorta_v1"]
        ← returns full 4-check verdict dict (~1 KB JSON)

LLM → call_tool("verify", {"mask_name": "aorta_v2"})
        → ERROR: mask not found → returns {"error": ..., "known_masks": ["aorta_v1"]}
        → LLM reads the error and tries again with the right name
```

This design has three nice properties:
1. **Compact context.** Every tool result is small (<2 KB JSON typically). 10-turn conversations stay well under the model's context window.
2. **Reproducibility.** The Workspace is the agent's full state at any moment — easy to snapshot for debugging or to resume.
3. **Composability.** Tools that depend on prior state (`verify` needs a mask, `analyze` needs a mask) reference it by name. The LLM can try `aorta_v1`, `aorta_v2`, `aorta_v3` in one conversation and compare verdicts side-by-side.

### The six tools

#### 1. `load_reconstruction` — fast path

Loads a previously-saved Stage 2a output `.mat` and populates `Workspace.recon`. Used during agent development to skip the 1-12 minute MATLAB reconstruction.

| Argument | Type | Default | Notes |
|---|---|---|---|
| `mat_path` | string | required | Path to a Stage 2a output |
| `venc_m_per_s` | number | 1.5 | Stored in workspace for downstream tools |
| `voxel_size_mm_dz/dy/dx` | number | 2.0 each | Voxel dimensions (separate fields because JSON Schema arrays are clunkier to validate) |

Returns `{status, source_path, shape_ZYXT, venc_m_per_s, voxel_size_mm}`.

#### 2. `reconstruct` — slow path

Wraps `skills.reconstruction.reconstruct()`. Same workspace effect as `load_reconstruction` but runs the full Stage 2a pipeline.

Distinct from `load_reconstruction` because: (a) the LLM's prompt should make it clear this one is slow and only worth calling when justified, and (b) the arguments are different (`kspace_path` + `method` + `n_iterations` vs. just a path).

#### 3. `suggest_seeds`

Calls `skills.segmentation.suggest_seed_points()`. Returns a compact list of candidate vessel regions sorted by brightness × size:

```json
{
  "candidates": [
    {"index": 0, "seed_zyx": [54, 32, 56], "size_voxels": 36356,
     "mean_pcmra": 0.034, "bbox_dims": [44, 73, 70]},
    {"index": 1, "seed_zyx": [46, 20, 39], "size_voxels": 16232, ...}
  ],
  "n_returned": 3
}
```

Cached in `Workspace.suggested_seeds` so the LLM can refer back to "candidate 0" later without re-computing PC-MRA. The bbox is reduced to `bbox_dims` (Δz, Δy, Δx) rather than the full 6-tuple — the LLM only needs the dimensions to reason about "is this a big vessel or a small one."

#### 4. `segment_from_seed`

Calls `skills.segmentation.segment_from_seed()`. Stores the resulting mask under the LLM-chosen `mask_name`. Returns size + peak speed so the LLM can immediately decide "is this a real vessel?" without a separate verify call:

```json
{
  "status": "ok",
  "mask_name": "aorta_v1",
  "seed_zyx": [54, 32, 56],
  "size_voxels": 33618,
  "peak_speed_m_per_s": 2.588,
  "percentile": 90.0,
  "closing_iter": 1
}
```

If segmentation returns an empty mask (seed in background or threshold too high), the tool returns `{"status": "empty_mask", "warning": "..."}` and **does not store** the mask. The LLM gets a clear "try again" signal.

#### 5. `verify`

Calls `skills.physics_verifier.verify()` on a named mask. Returns the full 4-check verdict dict (divergence, net flux, peak velocity, phase unwrap) with per-check status and numerical values. The verdict is also cached in `Workspace.verdicts[mask_name]`.

#### 6. `analyze`

Calls `skills.hemodynamic.analyze()` on a named mask. Returns per-cross-section flow + summary metrics + metadata. Cached in `Workspace.analyses[mask_name]`.

### Custom JSON Schema validator

The agent needs argument validation but adding `jsonschema` as a dependency is overkill for the handful of features we use. The inline validator in `_validate_args()` supports:

| Feature | Behavior |
|---|---|
| `"type": "string\|integer\|number\|boolean\|array\|object\|null"` | Standard typecheck |
| `"enum": [...]` | Value must be in list |
| `"minimum": N` / `"maximum": N` | Range check on numbers |
| `"default": ...` | Applied if argument missing |
| `"required": [...]` | Listed keys must be present |
| Unknown keys | **Rejected** — catches typos in LLM output |
| `bool` ≠ `integer` | Explicitly rejected (Python's `isinstance(True, int) == True` quirk would otherwise pass `True` as an integer) |

Not supported: `pattern`, `format`, `additionalProperties`, `$ref`, conditionals. None are needed for our tool schemas. Total validator: ~50 lines.

Validation errors raise `ToolError`, which the dispatcher catches and wraps into a structured response the LLM can read.

### JSON sanitization at the boundary

Tool internals operate on numpy types freely; the output goes through `to_json_safe()` before crossing the LLM boundary:

| Input | Output |
|---|---|
| `np.int64`, `np.bool_` | Python `int`, `bool` (`.item()`) |
| `np.float32`, `np.float64` | Python `float` (NaN → `None`) |
| `np.complex128` | `{"real": float, "imag": float}` |
| `np.ndarray` (≤ 64 items) | `list` via `.tolist()` |
| `np.ndarray` (> 64 items) | `"<ndarray shape=[...] dtype=...>"` string |
| `dict`, `list`, `tuple` | Recursively sanitized |

The 64-item threshold catches the edge case where a tool would accidentally return a velocity field. If `analyze()` returned `Q_mL_per_s` for 100 time phases, it would be summarized as a string — but our schemas keep T = 20 so the time series passes through inline. This is a safety net, not a routine path.

### Error contract — never raise into the LLM loop

`call_tool()` **always** returns a dict — never raises. Errors look like:

```json
// Unknown tool
{"error": "unknown tool 'verify_mask'", "error_type": "UnknownTool",
 "known_tools": ["analyze", "load_reconstruction", "reconstruct",
                 "segment_from_seed", "suggest_seeds", "verify"]}

// Schema violation
{"error": "missing required argument 'mask_name'",
 "error_type": "ToolError", "tool": "verify"}

// Tool function raised
{"error": "mask 'aort_v1' not found — known masks: ['aorta_v1']",
 "error_type": "ToolError", "tool": "verify"}

// Underlying skill crashed
{"error": "Image data of dtype complex128 cannot be converted to float",
 "error_type": "TypeError", "tool": "analyze"}
```

The LLM gets a structured, machine-readable response in every case. The orchestrator (Stage 3d) simply feeds this back into the next turn's prompt — no special-case crash handling at the orchestrator level. The Qwen-3.6-with-reasoning loop should naturally re-plan in response.

### Prompt block

`tools_prompt_block()` renders all six tools as a markdown block for the system prompt:

```
## Tools available

### load_reconstruction
Load a previously-saved Stage 2a reconstruction (.mat) into the workspace. ...

Parameters (JSON Schema):
```json
{
  "type": "object",
  "properties": {
    "mat_path": {"type": "string", "description": "..."},
    ...
  },
  "required": ["mat_path"]
}
```

### reconstruct
...
```

Current size: **4,903 characters ≈ 1,225 tokens** for all six tools combined. Well under any LLM context budget — Qwen 3.6 has 128k tokens, Claude has 200k.

### Tests

| Test class | Count | Description |
|---|---|---|
| `TestValidator` | 9 | All schema features individually (defaults, type, enum, range, bool-vs-int, unknown key, missing required) |
| `TestSanitize` | 6 | numpy scalars, small/large arrays, NaN, nested, complex |
| `TestRegistry` | 4 | Unique names, all have object schemas, all have descriptions, prompt block renders |
| `TestDispatcher` | 3 | Unknown tool, missing arg, tool exception all return error dicts |
| `TestToolsWithSyntheticRecon` | 8 | End-to-end on a synthetic vessel phantom — covers each tool plus a load→suggest→segment→verify→analyze chain |

**30/30 passing.** Synthetic phantom design: 12×10×10×4 cylindrical vessel along Z with constant 0.8 m/s flow and bright PC-MRA signal. No MATLAB required, no real data required. Tests run in 0.5 seconds total.

---

## End-to-End Demonstration

The tool layer was sanity-checked against the real Stage 2a reconstruction (`/mnt/g/medict_tmp/recon_cs_5iter.mat`). This is the manual sequence a future orchestrator will execute one tool call at a time:

```python
from agents.tools import Workspace, call_tool

ws = Workspace()

# 1. Load existing recon (no MATLAB)
call_tool(ws, "load_reconstruction", {"mat_path": "/mnt/g/medict_tmp/recon_cs_5iter.mat"})
# → {"status": "loaded", "shape_ZYXT": [77, 96, 72, 20], ...}

# 2. Get vessel candidates
seeds = call_tool(ws, "suggest_seeds", {"n_candidates": 3})
# → {"candidates": [{"seed_zyx": [54, 32, 56], "size_voxels": 36356, ...}, ...]}

# 3. Segment from best seed
seg = call_tool(ws, "segment_from_seed", {
    "seed_z": 54, "seed_y": 32, "seed_x": 56,
    "mask_name": "top_vessel"
})
# → {"status": "ok", "size_voxels": 33618, "peak_speed_m_per_s": 2.588, ...}

# 4. Run physics verifier
verdict = call_tool(ws, "verify", {"mask_name": "top_vessel"})
# → verdict: "fail"
#    divergence: fail (43 s⁻¹ — under-converged recon)
#    net_flux:   fail (218% deviation — merged vessels)
#    peak_velocity: pass (2.59 m/s)
#    phase_unwrap:  pass (0.0% above threshold)

# 5. Run hemodynamic analysis
report = call_tool(ws, "analyze", {"mask_name": "top_vessel"})
# → summary:
#    mean_stroke_volume_mL:    93.945  ← physiologically plausible
#    mean_peak_Q_mL_per_s:    438.276
#    peak_velocity_m_per_s:     2.5882
```

**Cross-check:** All five tool outputs match the direct skill calls from earlier sessions exactly. The wrapper layer is pure forwarding — no behavior change.

---

## Design Decisions

### Why minimal in-house instead of LangChain / LangGraph

The course deliverable is auditability. LangChain adds retries, output parsers, callback chains, and chain-of-thought processing that are hard to reason about end-to-end. For a 200-line in-house implementation, the entire control flow is in one file, every prompt is grep-able, and the error contract is one assertion ("`call_tool` returns a dict, never raises").

LangChain/LangGraph would be the right call for a multi-vendor production system. They're the wrong call for a course project where "explain every step" is the grade.

### Why local Ollama as default

- Free during development (no rate limits, no surprise bills)
- Qwen 3.6 35B fits in the RTX 5090's 32 GB VRAM with a comfortable margin
- Identical message format to Claude — no rewriting prompts when switching
- Failure modes (daemon down, model not pulled) are local and easy to debug

### Why `Workspace` instead of passing state through prompts

Two reasons. (1) Big ndarrays can't fit into a prompt. (2) The agent should be able to refer back to "the mask I made three turns ago" without us having to summarize it into JSON and re-parse it. Names are stable references; serialized data is not.

### Why JSON Schema for tool args (not Pydantic, not raw Python)

JSON Schema is what every LLM is trained on. Both Ollama (function-calling mode) and Claude (tool-use mode) accept tools described as JSON Schema. Defining tools this way makes them portable to future frameworks at zero cost.

Pydantic would give us type hints in Python but the schemas would still need to be exported to JSON Schema for the LLM. Skipping Pydantic eliminates the extra dependency and the indirection.

### Why the validator rejects unknown keys

Suppose the LLM emits `{"mask_name": "aorta_v1", "mask": "aorta_v1"}` (typo). Lenient validators silently drop the typo, the tool runs with default, and the LLM is confused why its argument was ignored. Strict validation surfaces this immediately as a `ToolError`, and the LLM can re-read the schema in the system prompt to find the correct key name.

### Why every tool returns a dict on failure

The orchestrator loop is structurally simpler when there's only one branch: "feed the tool result back to the LLM." No try/except, no special crash paths. The LLM's natural reasoning loop handles errors as just another piece of context to react to.

---

## Test Inventory

| Suite | Count | File |
|---|---|---|
| Stage 3a — LLM | 17 | `tests/test_llm.py` |
| Stage 3b — Tools | 30 | `tests/test_tools.py` |
| Existing Stage 2 (verifier, segmentation, recon, hemodynamic) | 81 | various |
| **Total** | **128** | (all passing in 23 sec) |

Live-Ollama tests auto-skip when the daemon is down — CI without GPU still gets a clean test run from the 14 + 30 = 44 pure-unit tests.

---

## What This Unlocks for Stage 3c and 3d

### Stage 3c — Audit Log
Hooks into the data this layer already exposes:
- Every `LLMResponse` carries `latency_ms`, token counts, raw payload, and reasoning — all loggable as-is.
- Every `call_tool()` invocation produces a JSON-safe dict — already loggable as-is.
- Wrapping `call_tool` with a logger is a 20-line file.

### Stage 3d — Orchestrator
The Coordinator loop is approximately:

```python
def orchestrate(llm, ws, user_goal, max_iterations=10):
    history = [
        {"role": "system", "content": tools_prompt_block() + "\n\n" + ORCH_RULES},
        {"role": "user", "content": user_goal},
    ]
    for i in range(max_iterations):
        decision = llm.chat_json(history)
        if decision.get("done"):
            return decision
        result = call_tool(ws, decision["tool"], decision["args"])
        history.append({"role": "assistant", "content": json.dumps(decision)})
        history.append({"role": "user", "content": json.dumps(result)})
    return {"error": "max iterations reached"}
```

Everything else is prompt engineering (the `ORCH_RULES` block) and stopping-condition tuning. The substrate is done.

---

## How to Use This Today

### Talk to Qwen
```bash
cd ~/projects/medict
conda activate medict
python notebooks/llm_playground.py
```

### Inspect the tool catalog
```bash
python -c "from agents.tools import tools_prompt_block; print(tools_prompt_block())"
```

### Manually drive the pipeline through the tool layer
```bash
python -c "
from agents.tools import Workspace, call_tool
ws = Workspace()
call_tool(ws, 'load_reconstruction', {'mat_path': '/mnt/g/medict_tmp/recon_cs_5iter.mat'})
seeds = call_tool(ws, 'suggest_seeds', {'n_candidates': 3})
for c in seeds['candidates']:
    print(c)
"
```

### Run the test suites
```bash
python -m pytest tests/test_llm.py tests/test_tools.py -v
```

---

## Open Questions / Future Work

- **Function-calling mode vs JSON-output mode.** Ollama supports OpenAI-style function calling (`tools=[...]` parameter). We're currently planning to use JSON output mode in Stage 3d because it's vendor-agnostic, but function-calling mode might give cleaner argument validation. Decision deferred to 3d.
- **Cross-session persistence.** `Workspace` is in-memory. If the orchestrator dies mid-loop, all state is lost. For Stage 4 evaluation runs we may want disk snapshots — currently the `output_path` field in recon output is the only persistent reference.
- **Tool granularity for `reconstruct`.** Currently the tool accepts 7 arguments. The LLM may struggle to fill them all correctly. We may add a `reconstruct_with_defaults` convenience tool that takes only `kspace_path` if the LLM consistently messes this up in practice.
- **Concurrency.** All tools are synchronous. Once we add `reconstruct` to the agent loop (Stage 4), a 12-minute call blocks the whole agent. Either run it as a background subprocess (with polling) or accept the wait. Currently leaning toward "accept the wait, demo with 5-iter for live evaluation."

---

## Quick Reference

### Files added in Stages 3a + 3b
```
agents/
├── __init__.py              # (empty marker)
├── llm.py                   # Stage 3a — LLM, OllamaLLM, ClaudeLLM, MockLLM, LLMResponse
└── tools.py                 # Stage 3b — Workspace, ToolSpec, TOOLS, call_tool, validator, sanitizer

tests/
├── test_llm.py              # 17 tests
└── test_tools.py            # 30 tests

notebooks/
└── llm_playground.py        # Interactive REPL
```

### Memory references
- [`memory/project_stage3a_llm.md`](../memory/project_stage3a_llm.md) — LLM backend cheat-sheet + Qwen reasoning gotcha
- [`memory/project_stage3b_tools.md`](../memory/project_stage3b_tools.md) — Tool layer cheat-sheet

---

*End of Stage 3 (3a + 3b) report. Stage 3c (audit log) and Stage 3d (orchestrator loop) follow.*
