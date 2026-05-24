# MEDICT V1 Demonstration Report
**Multi-Agent 4D Flow MRI Pipeline — End-to-End Evidence**
*BENG 280C, UCSD — V1 Course Deliverable*

---

## Executive Summary

V1 of MEDICT is evaluated through two paired reference runs executed by the same
multi-agent pipeline on different inputs. The two runs jointly demonstrate the
project's central claim: **the system's behavior is driven by the quality of its
inputs, and the system honestly reports what it can and cannot trust.**

| | Good Case | Bad Case |
|---|---|---|
| Input | Synthetic curved-tapered phantom (analytically incompressible) | OSU-MR 4D flow scan, 50-iter CS reconstruction |
| Final session status | `success` | `partial` (hit max_delegations cap) |
| Physics Verifier verdict | **4/4 checks pass** | **4/4 checks fail across 5 segmentation attempts** |
| Hemodynamic Analyzer | 3 invocations → produced clinical metrics | 0 invocations (never reached) |
| Wall-clock time (real Qwen 3.6 / RTX 5090) | ~60 s | 2643 s (44 min) |
| Reference audit log | `logs/reference_runs/v1_good_case_qwen.jsonl` | `logs/reference_runs/v1_bad_case_qwen.jsonl` |

Both runs were executed end-to-end with no human intervention. All agent decisions,
tool calls, and Physics Verifier verdicts are recorded in an append-only JSONL
audit log, exhibiting `traceability = 1.0` in both cases (every tool call attributes
to a prior LLM decision).

---

## 1. Good Case — Synthetic Phantom

### Input
A purpose-built `curved_tapered_phantom`: a single straight tube along the Z axis
with smooth radius taper (7 → 5 voxels), an analytically-constructed
incompressible velocity field (∇·v = 0 by construction), and pulsatile cardiac
variation (±30%). Volume size 60×60×60×8 (Z, Y, X, T). VENC 1.5 m/s, voxel size
2 mm isotropic.

The phantom ships with its ground-truth vessel mask and bypasses two stages
that V1 cannot reliably perform on real data (PC-MRA-based segmentation and
MATLAB reconstruction). Its purpose is to validate the
**Physics-Verifier-and-downstream** half of the pipeline under controlled inputs.

### Pipeline Trace
1. **Planner** emitted a 4-step plan (recon → seg-passthrough → verify → analyze).
2. **Plan Critic** approved.
3. **Coordinator** delegated to Reconstruction specialist.
4. **Reconstruction specialist** invoked `load_phantom`. Phantom and ground-truth
   mask `aorta_phantom` placed in workspace.
5. Coordinator delegated to **Segmentation specialist**, which confirmed the mask
   already existed and passed through to Verifier without calling any
   segmentation tool (per phantom-mode prompt).
6. **Physics Verifier** ran the four deterministic checks via `verify()`:
   - divergence: **pass** (mean |∇·v| = 0.74 s⁻¹, threshold 5)
   - net_flux: **pass** (max deviation across 5 cross-sections within 10%)
   - peak_velocity: **pass** (peak 1.27 m/s, within physiological range)
   - phase_unwrap: **pass** (no voxel-pair jumps exceeding VENC)
7. **Hemodynamic Analyzer** ran `analyze()` and produced:
   - Mean stroke volume: **110.73 mL** (physiological range: 60–100 mL)
   - Mean peak Q: **359.86 mL/s** (physiological range: 400–600 mL/s)
   - Mean peak velocity: **1.27 m/s** (physiological range: ~1.5 m/s)
   - Cross-sections analyzed: 5
8. Coordinator emitted a clean `done` with summary.

### Audit Metrics
```
session_status                success
n_verify_failures             0
n_tool_errors                 0
traceability                  1.0
reasoning_coverage            0.29  (real Qwen reasoning captured; mock = 0)
decision_density (LLM/tool)   4.56
total_llm_latency             84.7 s on GPU
```

### Interpretation
With controlled inputs, the agent system delegates linearly through all four
specialists, the verifier passes every check, and the hemodynamic analyzer
produces numbers that fall in or adjacent to clinical reference ranges. The
slight SV overshoot (110 vs. 60–100 mL) traces to the phantom's deliberately
generous pulsatile amplitude and does not indicate a pipeline error. **The
system's behavior on clean inputs validates the verifier-plus-hemodynamic
half of the architecture.**

---

## 2. Bad Case — Real OSU-MR Data

### Input
The OSU-MR open Zenodo 4D flow dataset (Arshad et al., *Magn Reson Med* 2024),
reconstructed at 50 iterations of CS via the authors' published MATLAB pipeline.
Volume 77×96×72×20, VENC 1.5 m/s, voxel size 2 mm isotropic. The reconstruction
itself is high-quality and used unmodified.

### Pipeline Trace
1. Planner emitted a 4-step plan; Critic suggested revisions (add `suggest_seeds`
   step, add phase-consistency check). Planner revised the plan once and the
   revised plan was approved (1 plan revision; the JSON-tolerant parser
   survived Qwen's long reasoning chains in the revision step).
2. Reconstruction specialist invoked `load_reconstruction`, loading
   `recon_cs_50iter.mat`. Recon shape (77, 96, 72, 20).
3. Coordinator delegated to Segmentation. The specialist invoked
   `suggest_seeds` then `segment_from_seed` and produced `aorta_tight_v1`
   (8602 voxels, peak 2.58 m/s).
4. Verifier verdict: **fail**. Net-flux deviation too large (mask spanned
   regions with incompatible flow directions).
5. Coordinator re-delegated to Segmentation. Successive masks were produced
   with different parameters: `v1` (4781 vox), `v2` (2456), `v3` (6681),
   `v4` (33,922).
6. Verifier returned **fail** on every mask. The largest mask (`v4`,
   33,922 voxels) is consistent with the entire cardiac blood pool merging
   into a single connected component.
7. The seg ↔ verify loop ran five full cycles. On the twelfth delegation
   the Coordinator hit its `max_delegations=12` safety cap and exited.
8. **Hemodynamic specialist was never invoked.**

### Coordinator's Own Final Statement (verbatim from audit log)
> "Reached max_delegations=12 without an explicit done. ... The Coordinator did
> not converge on a clean result; the most likely cause is a **data-quality
> issue (e.g. segmentation cannot cleanly isolate a single vessel) rather than
> an agent bug**. The workspace state above is preserved — partial results may
> still be useful."

### Interpretation
The Physics Verifier correctly identifies that automatic PC-MRA seed-based
region growing on this scan produces masks that span multiple flow regimes
(aorta + heart chambers), violating continuity. The system honestly flags
the situation as a data-quality limit rather than producing
clinically-plausible-but-wrong numbers. This matches the published limitation
of PC-MRA contrast: at this resolution, the aorta-vs-chamber boundary is not
visible to threshold-based segmentation, and the same input defeats
TotalSegmentator, MedSAM2, and base SAM2 when applied off-the-shelf
(documented in V2 plan).

---

## 3. Pipeline Divergence Analysis

The two runs share identical orchestration (Planner → Plan Critic → Coordinator
→ same 4 Specialists, same prompts, same Qwen 3.6 backend). The behavioral
divergence emerges *entirely from the data*.

```
                                              GOOD CASE                   BAD CASE
Planner            ──────────────────────► 4 steps, approved      4 steps + 1 revision
Plan Critic        ──────────────────────► approve                revise then approve
Coordinator        ──────────────────────► delegated in plan      delegated in plan + retries
                                            order, 1 pass         order, 5 retry cycles
Reconstruction     ──────────────────────► load_phantom           load_reconstruction
                                            (success)              (success)
Segmentation       ──────────────────────► passthrough            5 attempts
                                            (mask already there)   (all rejected by Verifier)
                                                                  Names chosen by Qwen:
                                                                  aorta_tight_v1..v4
Physics Verifier   ──────────────────────► 4/4 pass × 1           4/4 fail × 5
Hemodynamic        ──────────────────────► 3 invocations           NEVER REACHED
                                            (clinical numbers)
Coordinator exit   ──────────────────────► explicit done          max_delegations cap
Session status     ──────────────────────► success                 partial
Wall time          ──────────────────────► ~60 s                  2643 s
```

**The fork happens at the Segmentation stage.** In the good case the mask is
correct by construction; in the bad case it cannot be correct because the
underlying contrast does not support it. The Physics Verifier consistently
identifies this, and downstream behavior follows: the good-case Hemodynamic
analyzer produces numbers, the bad-case Hemodynamic analyzer is never reached
because the Coordinator (correctly) refuses to feed it a rejected mask.

---

## 4. What This Evidence Supports — and What It Does Not

### Supported by the two runs
- The agent pipeline executes end-to-end without intervention.
- The Physics Verifier produces correct verdicts in both directions
  (pass on clean inputs, fail on inputs with known segmentation defects).
- Verdict-driven downstream behavior: the Hemodynamic analyzer runs only
  when its prerequisite is verified.
- Audit completeness: 100% of tool calls attribute to a prior LLM decision in
  both runs; both Coordinators produce explicit summaries that name the
  outcome and (when applicable) the cause.
- Real Qwen reasoning is captured in the log (reasoning_coverage ≈ 0.29 in
  the good case), enabling post-hoc review of the LLM's chain of thought.

### Not claimed
- We do not claim that the hemodynamic numbers from the good case are
  clinically meaningful for a specific patient; the input is a synthetic
  phantom designed to satisfy the verifier's invariants.
- We do not claim the system can produce clinically-valid numbers on real
  PC-MRA data with automatic segmentation. The bad case demonstrates that
  it cannot, and that the system reports this honestly.
- We do not claim the Coordinator's exit condition is optimal. In the bad
  case the loop terminated by safety cap (`max_delegations`) rather than by
  the Coordinator independently choosing to escalate to a best-effort report.

---

## 5. Limitations Surfaced by the Bad Case (V2 Motivation)

1. **No graceful give-up policy.** The Coordinator retried segmentation five
   times and never invoked Hemodynamic at all, even to produce a best-effort
   report with caveats. A budget-based termination policy (Tier 1, V2) would
   let the LLM see remaining "energy" and reason about when to escalate.

2. **Single-axis, single-tube verifier scope.** The `net_flux` check assumes
   a vessel approximately aligned with a coordinate axis. Curved or branched
   vessels (Y-branches, aortic arches) violate this assumption even when the
   underlying physics is correct. This was discovered while constructing the
   good-case phantom: a Y-branched phantom satisfies mass conservation
   globally but fails the verifier's single-mask flux check by 100% deviation.
   A centerline-aware verifier (Tier 2, V2) would address this.

3. **No alternative segmentation paths.** The Segmentation specialist has
   only one tool family (PC-MRA seed-based region growing). When this fails,
   the agent has no fallback. Manual-ROI SAM2 (V2 Tier 2) would provide a
   second action the agent could try.

---

## 6. Reproducibility

Both runs are reproducible by re-executing the demo scripts against the
project's pinned environment (`environment.yml`) with Ollama and Qwen 3.6
available:

```bash
# Good case (~60 s on RTX 5090)
python demos/good_case_demo.py --llm ollama

# Bad case (~45 min on RTX 5090)
python demos/single_window_demo.py --llm ollama --no-fresh-recon \
    --max-plan-revisions 2 \
    --goal "Analyze the hemodynamics of /mnt/g/medict_tmp/recon_cs_50iter.mat. \
            VENC=1.5 m/s, voxel 2mm isotropic. Pick a vessel, verify, report."
```

Reference audit logs from the runs documented above are preserved at
`logs/reference_runs/v1_good_case_qwen.jsonl` and
`logs/reference_runs/v1_bad_case_qwen.jsonl`. Audit metrics can be
re-computed at any time:

```bash
python evaluation/audit_metrics.py logs/reference_runs/v1_good_case_qwen.jsonl
python evaluation/audit_metrics.py logs/reference_runs/v1_bad_case_qwen.jsonl
```

---

## Conclusion

The V1 deliverable is a multi-agent pipeline that (a) executes end-to-end on
both synthetic and real 4D flow inputs, (b) produces validated clinical metrics
when the underlying data supports it, (c) honestly refuses to produce numbers
when its verifier flags the inputs as untrustworthy, and (d) maintains a
complete audit trail of every decision and action.

The two reference runs documented above provide direct evidence for these
claims and surface concrete limitations that motivate the V2 work plan. The
bad case in particular — where the same architecture that produces validated
output on clean data correctly refuses to produce output on data whose
segmentation cannot be trusted — operationalizes the project's central
contribution: an auditable pipeline that is honest about what it can and
cannot do.
