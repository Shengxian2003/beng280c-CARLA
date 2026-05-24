# MEDICT — Multi-Agent 4D Flow MRI Pipeline (V1)

> **An auditable multi-agent pipeline that turns 4D flow MRI k-space into hemodynamic metrics, with a deterministic Physics Verifier that quantifies its own detection performance and tells the user when its inputs aren't trustworthy.**

**Course deliverable:** BENG 280C, UCSD.
**Status:** V1 complete (Stages 1–4). 274 tests passing.

---

## TL;DR

* **Pipeline:** raw k-space → MATLAB reconstruction → PC-MRA segmentation → physics verification → hemodynamic metrics
* **Orchestration:** 7 LLM agents (Planner / Plan Critic / Coordinator / 4 specialists) running on local Qwen 3.6 35B via Ollama
* **Trust anchor:** deterministic 4-check Physics Verifier; LLMs interpret but **cannot override** its verdicts
* **Audit:** every prompt, decision, reasoning, and tool call is recorded in an append-only JSONL log
* **Quantified evaluation:** Physics Verifier achieves **0% false-positive rate** and **100% detection rate** above warn threshold (320 controlled trials); reference real-Qwen run scores **1.0** on every auditability metric

---

## Why this exists

4D flow MRI gives you a velocity field, but those fields are noisy. Computing clinical metrics (stroke volume, peak velocity, regurgitation) from a noisy field gives clinically misleading numbers. Manual quality assurance is slow, subjective, and non-reproducible.

MEDICT addresses this by:
1. **Automating the pipeline** end-to-end (k-space → metrics)
2. **Enforcing deterministic physics checks** (incompressibility, flux conservation, peak-velocity plausibility, phase-unwrap sanity)
3. **Documenting every decision** the LLM agents make in an audit log, so a reviewer can reconstruct why any number was produced
4. **Honestly flagging untrustworthy data** rather than producing clinically plausible-looking-but-wrong numbers

---

## Architecture

```
USER GOAL (natural language)
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  LLM-PICK-LLM AGENT SYSTEM                                    │
│                                                               │
│   PLANNER       (LLM)   emits structured plan                 │
│      ↓                                                        │
│   PLAN CRITIC   (LLM)   approves / revises                    │
│      ↓                                                        │
│   COORDINATOR   (LLM)   delegates by name                     │
│      ↓                                                        │
│   ┌──────────────────────────────────────────────────────┐    │
│   │ 4 SPECIALIST LLMs                                    │    │
│   │   - Reconstruction Operator   - Physics Verifier     │    │
│   │   - Segmentation Operator     - Hemodynamic Analyzer │    │
│   └────────────────────┬─────────────────────────────────┘    │
└────────────────────────┼──────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  4 DETERMINISTIC PYTHON SKILLS                                │
│   - reconstruction  (wraps MATLAB CS / CORe recon)            │
│   - segmentation    (PC-MRA + seed-based region growing)      │
│   - physics_verifier (4 checks — TRUST ANCHOR)                │
│   - hemodynamic     (Q(t), stroke volume, peak velocity)      │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  AUDIT LOG (append-only JSONL)                                │
│  Every prompt / decision / reasoning / tool call / verdict    │
└──────────────────────────────────────────────────────────────┘
```

**Key invariant:** the deterministic Physics Verifier is the trust anchor. LLM specialists *interpret* its verdicts but cannot override them. Architecturally enforced — verifier runs as a Python tool, its verdict is data, the LLM can only write commentary about it.

---

## Headline results

### Stage 4 — error-injection detection sweep

10 trials × 8 magnitudes × 4 error types = **320 controlled trials**, 13 seconds total:

| Check | False-positive rate | Detection rate (above warn threshold) |
|---|---|---|
| Divergence | **0.00** | **100%** |
| Net flux | **0.00** | **100%** |
| Peak velocity | **0.00** | **100%** |
| Phase unwrap | **0.00** | **100%** |

See `evaluation/results/detection_*.png` for per-check detection-vs-magnitude curves.

### Auditability metrics on a real-Qwen run

95-minute run with real MATLAB reconstruction + 5 segmentation attempts + 4 verifier failures (each correctly explained):

| Metric | Value |
|---|---|
| Traceability (tool calls trace to LLM decisions) | **1.0** |
| Reasoning coverage (LLM calls with chain-of-thought) | **1.0** |
| Verifier explanation coverage | **1.0** |
| Long-reasoning coverage (≥ 80 chars) | **1.0** |
| Decision density (LLM calls per tool call) | **6.93** |

### Test coverage

```
274 tests across:
  - physics_verifier:    13 (4 checks, error injection unit tests)
  - segmentation:        15 (PC-MRA + seed-based)
  - reconstruction:      34 (path translation, MATLAB wrapper)
  - hemodynamic:         19 (Q(t), SV, peak velocity)
  - llm backend:         17 (Ollama, Claude, Mock + Qwen live)
  - tools:               30 (JSON schema, sanitization, dispatcher)
  - audit log:           30 (JSONL writer/reader, summary)
  - agent_window:        22 (per-agent routing, rendering)
  - specialist + coordinator: 18 (delegation, project context injection)
  - planner + plan_critic + plan_policy: 33
  - eval_inject:         15 (clean phantom + 4 targeted injectors)
  - multi-pane TUI:      19
```

Run with `python -m pytest tests/`.

---

## Quick start

### Prerequisites
* WSL2 Ubuntu (this project assumes WSL because MATLAB lives on Windows)
* `conda` (Miniconda or Anaconda)
* Windows MATLAB 2019+ with Parallel Computing Toolbox (for reconstruction)
* Ollama with Qwen 3.6 pulled (`ollama pull qwen3.6`) — only needed for real-LLM runs
* NVIDIA RTX 5090 or any GPU with ≥ 16 GB VRAM (for MATLAB recon; CPU fallback works but slow)

### Setup
```bash
cd ~/projects/medict
conda env create -f environment.yml
conda activate medict
```

### Run the demo (MockLLM, ~5 seconds, deterministic)

```bash
python demos/single_window_demo.py --llm mock
```

You'll see:
1. **Planner** emit a 4-step plan
2. **Plan Critic** approve it
3. **Coordinator** delegate to each specialist
4. Specialists call their tools and report back
5. Final summary
6. **Thinking Process** section grouping each agent's conclusions and intermediate steps

### Run with real Qwen (~10–95 min depending on whether MATLAB triggers)

```bash
# Fast (load existing recon)
python demos/single_window_demo.py --llm ollama --no-fresh-recon

# Full pipeline including 50-iter MATLAB recon
python demos/single_window_demo.py --llm ollama
```

### Multi-window mode (7 separate terminal viewers)

```bash
# In 7 terminals, paste one command per terminal:
python demos/agent_window.py planner       logs/agent_demo.jsonl
python demos/agent_window.py plan_critic   logs/agent_demo.jsonl
python demos/agent_window.py coordinator   logs/agent_demo.jsonl
python demos/agent_window.py reconstruction logs/agent_demo.jsonl
python demos/agent_window.py segmentation   logs/agent_demo.jsonl
python demos/agent_window.py verifier       logs/agent_demo.jsonl
python demos/agent_window.py hemodynamic    logs/agent_demo.jsonl

# In an 8th terminal:
python demos/run_demo.py --llm mock     # or --llm ollama
```

### Run the Stage 4 evaluation

```bash
# Detection sweep (10 trials per cell, ~13 sec)
python evaluation/run_detection_eval.py --n-trials 10
ls evaluation/results/   # JSON + 4 PNG plots

# Paper-quality CIs (100 trials per cell, ~2 min)
python evaluation/run_detection_eval.py --n-trials 100

# Auditability metrics on any audit log
python evaluation/audit_metrics.py logs/agent_demo.jsonl
```

---

## Repository structure

```
medict/
├── agents/                   # Stage 3 — multi-agent system
│   ├── llm.py                  # OllamaLLM / ClaudeLLM / MockLLM
│   ├── tools.py                # 6 tools wrapping Stage 2 skills (Workspace pattern)
│   ├── audit.py                # JSONL audit log writer/reader
│   ├── planner.py              # Planner LLM agent
│   ├── plan_critic.py          # Plan Critic LLM (best-effort auditor)
│   ├── plan_policy.py          # Deterministic policy gate (NOT an LLM)
│   ├── specialist.py           # Specialist class + 4 default configs
│   ├── coordinator.py          # Top-level delegator
│   └── project_context.py      # Canonical paths injected into every LLM prompt
│
├── skills/                   # Stage 2 — deterministic Python skills
│   ├── reconstruction/         # MATLAB wrapper (CS / CORe via subprocess)
│   ├── segmentation/           # PC-MRA + seed-based region growing
│   ├── physics_verifier/       # 4 checks: divergence / flux / peak / phase unwrap
│   ├── hemodynamic/            # Q(t), stroke volume, peak velocity
│   └── eval_inject/            # Stage 4 — error injection harness
│
├── evaluation/               # Stage 4
│   ├── run_detection_eval.py   # Detection-rate sweep
│   ├── audit_metrics.py        # Auditability scoring
│   └── results/                # JSON + PNG outputs
│
├── demos/                    # Live + recorded demo tools
│   ├── single_window_demo.py   # One-terminal demo with thinking summary
│   ├── run_demo.py             # 7-window orchestrator demo
│   ├── agent_window.py         # Per-agent viewer (used by run_demo)
│   └── multi_pane.py           # Replay TUI for audit logs
│
├── tests/                    # 274 tests
├── notebooks/                # Stage reports (stage{1c,2,3,4}_report.md)
├── data/                     # gitignored: OSU-MR k-space (806 MB)
└── logs/                     # gitignored: session audit logs
```

---

## Stage reports (the long-form deliverables)

* [`notebooks/stage1_completion_report.md`](notebooks/stage1_completion_report.md) — environment, OSU-MR data, first MATLAB recon
* [`notebooks/stage2_report.md`](notebooks/stage2_report.md) — 4 deterministic skills (recon, segmentation, verifier, hemodynamic)
* [`notebooks/stage3_report.md`](notebooks/stage3_report.md) — multi-agent system (LLM-pick-LLM, audit log)
* [`notebooks/stage4_report.md`](notebooks/stage4_report.md) — error injection + detection sweep + auditability metrics (**the scientific contribution**)

---

## Honest limitations

V1 is **deliberately** not claiming things it can't prove:

| What V1 does NOT claim | Why |
|---|---|
| Our hemodynamic numbers are clinically valid on real data | They aren't — the PC-MRA segmentation merges the aorta with adjacent heart chambers, and the verifier correctly flags this. **The auditability story is that the system tells you when its data is bad.** |
| Modern segmentation models (TotalSegmentator, MedSAM2, base SAM2) solve our segmentation problem | They don't — PC-MRA contrast is outside their training distributions, AND the aorta/heart boundary genuinely isn't visible in PC-MRA at this resolution. See `memory/project_future_work.md` for full test results. |
| The LLM's interpretations are quantitatively correct | They look qualitatively right (we see Qwen call out "negative SV is impossible", "peak > VENC = aliasing"), but V1 doesn't measure interpretation correctness. |
| The system works with non-Qwen LLMs | Architecture supports it (`ClaudeLLM` and `MockLLM` both exist), but only Qwen was validated end-to-end. |

---

## What's coming in V2

See [`memory/project_future_work.md`](memory/project_future_work.md) for the full backlog. Tier-1 items (short + high-value):

1. **Energy/stamina budget for LLM agents** — replaces hard `max_delegations` cap with a visible-to-LLM budget so it concludes gracefully instead of timing out
2. **Natural-language CLI** — `medict "analyze the scan at X"` instead of running Python scripts
3. **Higher-N detection sweep** (`--n-trials 100`) for paper-quality confidence intervals
4. **Manual ROI tool on SAM2** — the one untried segmentation approach; should produce clean masks because user-specified bbox excludes anatomy by construction

---

## Citations

1. Arshad SM, Potter LC, Chen C, et al. **Motion-robust free-running volumetric
   cardiovascular MRI.** *Magn Reson Med.* 2024;92(3):1248–1262.
   doi:[10.1002/mrm.30123](https://doi.org/10.1002/mrm.30123). MATLAB
   reconstruction code: [github.com/OSU-MR/motion-robust-CMR](https://github.com/OSU-MR/motion-robust-CMR).
2. Arshad SM, Potter LC, Ahmad R. **3D cine, 4D flow, and exercise stress 4D
   flow CMR datasets.** Zenodo, 2024.
   doi:[10.5281/zenodo.12515230](https://doi.org/10.5281/zenodo.12515230).
3. Qwen team. **Qwen 3.6 35B** via [Ollama](https://ollama.com), 2025.

---

## License

MIT License. The OSU-MR reconstruction code under `skills/reconstruction/motion-robust-CMR-main/`
is used unmodified under its original Apache 2.0 license.
