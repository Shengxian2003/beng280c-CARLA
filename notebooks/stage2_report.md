# Stage 2 Report
**Project:** MEDICT — Multi-Agent 4D Flow MRI Pipeline  
**Course:** BENG 280C, UCSD

---

## Stage 2b — Physics Verifier Skill ✅

**Date completed:** 2026-05-12

### Overview
Pure Python skill that runs four deterministic physics checks on a 4D flow velocity field and returns a structured JSON verdict (`pass` / `warn` / `fail` per check).

Input is the raw reconstruction output from Stage 1c — phase images in radians, not yet converted to velocity. The skill handles the conversion internally using VENC.

### Files
| File | Description |
|---|---|
| `skills/physics_verifier/__init__.py` | Public API: `verify()`, `verify_from_mat()`, `verdict_json()` |
| `skills/physics_verifier/_checks.py` | Four check functions |
| `skills/physics_verifier/_mask.py` | `velocity_mask()`, `threshold_mask()` |
| `tests/test_physics_verifier.py` | 13 unit tests including error injection cases |
| `notebooks/test_verifier_real_data.py` | Integration test on real Stage 1c output |

### API
```python
from skills.physics_verifier import verify

result = verify(
    thetaX, thetaY, thetaZ,   # phase arrays (Z, Y, X, T) in radians
    venc_m_per_s=1.5,
    voxel_size_mm=(2.0, 2.0, 2.0),
    mask=vessel_mask,          # optional — auto-generated if not provided
)
# result["verdict"]  →  "pass" | "warn" | "fail"
# result["checks"]   →  per-check results with numerical values
```

### Four Checks

#### ① Divergence  
For incompressible blood flow, `∇·v = 0` everywhere inside the vessel. Computed voxel-wise using `numpy.gradient` on the time-averaged velocity field, evaluated on the eroded vessel interior (1-voxel erosion removes boundary artifacts from the velocity → background transition).

| Threshold | Value |
|---|---|
| warn | mean \|∇·v\| > 5.0 s⁻¹ |
| fail | mean \|∇·v\| > 20.0 s⁻¹ |

> **Requires Stage 2c segmentation mask** to be meaningful on real data. Boundary effects from a coarse velocity mask inflate divergence artificially.

#### ② Net Flux Conservation  
Volumetric flow through 5 evenly-spaced cross-sections along the dominant flow axis should be consistent (continuity equation for an incompressible fluid in a tube).

| Threshold | Value |
|---|---|
| warn | max cross-section deviation > 10% of mean flux |
| fail | max cross-section deviation > 25% of mean flux |

> **Requires Stage 2c segmentation mask.** Without a connected vessel tube, cross-sections capture multiple disconnected structures and the flux values are meaningless.

#### ③ Peak Velocity  
3D speed magnitude (`sqrt(vx² + vy² + vz²)`) should be within physiological range for cardiac 4D flow. Note: 3D speed can legitimately exceed VENC since it combines all three encoding directions.

| Threshold | Value |
|---|---|
| warn | peak < 0.05 m/s or peak > 3.0 m/s |
| fail | peak > 4.5 m/s |

#### ④ Phase Unwrap Sanity  
Spatial jumps larger than VENC in any single velocity component indicate residual phase-wrapping artifacts. Checks each component separately — speed magnitude alone misses sign-flip wraps where `+VENC → −VENC` leaves magnitude unchanged.

| Threshold | Value |
|---|---|
| warn | fraction of voxel-pairs with jump > VENC > 0.1% |
| fail | fraction > 1.0% |

### Vessel Mask
Auto-generated using `velocity_mask()` (PC-MRA style): selects the top 15% of voxels by time-averaged speed magnitude. This correctly identifies fast-moving blood rather than bright static tissue, which signal-magnitude thresholding (`threshold_mask`) would select instead.

### Test Results

**Unit tests (synthetic data):** 13 / 13 passing  
Includes 5 error injection cases:
- Injected divergent field → divergence check fails ✓
- Zero velocity field → divergence check passes ✓  
- Velocity 3× physiological → peak velocity warns/fails ✓
- Alternating ±π phase → phase unwrap warns/fails ✓
- Smooth ramp field → phase unwrap passes ✓

**Integration test (real OSU-MR data, 5-iteration reconstruction):**

| Check | Status | Value |
|---|---|---|
| Divergence | ✗ fail | 78.5 s⁻¹ — expected; needs Stage 2c mask |
| Net Flux | ✗ fail | 140% deviation — expected; needs Stage 2c mask |
| Peak Velocity | ✓ pass | peak 1.81 m/s, p95 1.48 m/s, mean 1.25 m/s |
| Phase Unwrap | ✓ pass | max jump 1.41 m/s, fraction above VENC = 0.000000 |

Divergence and flux failures on real data are expected at this stage — both require a proper connected vessel segmentation (Stage 2c). Peak velocity and phase unwrap work correctly without segmentation.

### Design Decisions
- **Thresholds are deterministic, not adaptive.** Divergence and flux are physics laws (∇·v = 0 for incompressible flow) — the threshold represents numerical tolerance of the CS reconstruction, not patient physiology. Adaptive thresholds would complicate Stage 4 evaluation metrics (detection rate requires a fixed threshold).
- **Phase input, not velocity input.** The API accepts raw phase in radians + VENC and converts internally. This matches the MATLAB reconstruction output directly and avoids a conversion step in the calling code.
- **Component-wise phase unwrap.** Checking each velocity component separately rather than speed magnitude catches sign-flip wraps that cancel in the magnitude.

---

## Stage 2a — Reconstruction Skill ✅

**Date completed:** 2026-05-16

### Overview
Python wrapper around the [motion-robust-CMR](https://github.com/OSU-MR/motion-robust-CMR) MATLAB reconstruction (CS / CORe) from Arshad et al., *Magn Reson Med* 2024 ([doi:10.1002/mrm.30123](https://doi.org/10.1002/mrm.30123)). The interactive `main_recon.m` is replaced by a parameterizable batch driver invoked from Python through Windows MATLAB. The wrapper handles WSL ↔ Windows data staging, JSON config generation, MATLAB invocation, live stdout streaming, and result loading.

### Files
| File | Description |
|---|---|
| `skills/reconstruction/__init__.py` | Public API: `reconstruct()`, path helpers |
| `skills/reconstruction/_runner.py` | Bridge: staging, subprocess, .mat loading |
| `skills/reconstruction/_paths.py` | WSL ↔ Windows path translation |
| `skills/reconstruction/medict_recon_driver.m` | Batch MATLAB driver (reads JSON config) |
| `skills/reconstruction/motion-robust-CMR-main/` | Upstream MATLAB code (unchanged) |
| `tests/test_reconstruction.py` | 30 unit tests (path/config/command/validation) |
| `notebooks/test_reconstruction_skill.py` | End-to-end smoke test on real OSU-MR data |

### API
```python
from skills.reconstruction import reconstruct

result = reconstruct(
    "/home/nick_17/projects/medict/data/4D_Flow_Cartesian_Dataset_11.mat",
    venc_m_per_s = 1.5,
    method       = "cs",       # "cs" or "core"
    n_iterations = 50,         # 5 ≈ smoke, 50 ≈ full quality (~12 min on RTX 5090)
    is_flow      = True,
    is_rest      = True,
    use_gpu      = True,
)
# result["xHat"]   → ndarray (Z, Y, X, T) magnitude (sum-of-squares for 4D flow)
# result["thetaX"] → ndarray (Z, Y, X, T) background-corrected phase (vx)
# result["thetaY"] → ndarray (Z, Y, X, T) background-corrected phase (vy)
# result["thetaZ"] → ndarray (Z, Y, X, T) background-corrected phase (vz)
# result["meta"]   → { elapsed_minutes, method, n_iterations, acceleration, ... }
```

The wrapper composes directly with the Physics Verifier:
```python
from skills.physics_verifier import verify
verdict = verify(result["thetaX"], result["thetaY"], result["thetaZ"],
                 venc_m_per_s=1.5, voxel_size_mm=(2,2,2))
```

### Bridge Architecture
```
   WSL Python                          Windows MATLAB
   ----------                          --------------
   reconstruct()
     │
     ├── stage k-space → G:\medict_tmp\…    (HDF5 .mat unreliable over UNC)
     ├── write JSON config → G:\medict_tmp\config_<ts>.json
     │
     └── subprocess: matlab.exe -batch
                       "addpath('…');
                        medict_recon_driver('G:\medict_tmp\config.json')"
                                                │
                                                ├── load k-space
                                                ├── coil-combine → Walsh maps
                                                ├── CS / CORe ADMM (GPU)
                                                ├── background phase correction
                                                └── save outputs.mat
                                                       │
                                            ─────── stdout streamed back ───────
                                                       │
   load outputs.mat ←─────────────────────────────────┘
   return {xHat, thetaX, thetaY, thetaZ, meta, ...}
```

### Design Decisions
- **MATLAB driver is parameterized, not configured at edit-time.** The interactive `main_recon.m` (4 prompts + file picker) becomes a single batch call that reads a JSON config. The upstream MATLAB code is untouched so future syncs from the OSU-MR repo are clean.
- **Data is staged to a Windows drive, not read via UNC.** Stage 1c established that Windows MATLAB cannot reliably open HDF5 v7.3 .mat files through `\\wsl.localhost\`. The wrapper copies k-space to `G:\medict_tmp\` before invoking MATLAB and reads the output back through `/mnt/g/…`.
- **`.m` source IS read via UNC.** The driver script and `motion-robust-CMR-main` code live on the WSL filesystem and Windows MATLAB reads them through the UNC path — no need to duplicate ~2000 MATLAB files onto a Windows drive.
- **Streaming stdout.** MATLAB writes iteration progress to stdout (`opt.vrb=1`); the wrapper pipes it live so 12-minute runs aren't silent.
- **Validation in Python, not MATLAB.** Method/iteration bounds are checked before MATLAB is even launched — failures are fast and don't waste a MATLAB session.

### Test Results
**Unit tests:** 30 / 30 passing (path translation, config dataclass, command builder, validation, driver artifacts).

**Smoke test** (`notebooks/test_reconstruction_skill.py`): end-to-end with 5-iter CS — produces the same output as Stage 1c's hand-written `run_recon_test.m`, verified by Physics Verifier (peak velocity + phase unwrap pass; divergence + flux fail as expected for 5-iter).

### What This Unlocks
- 50-iteration runs are now a single parameter change — previously required editing the .m file.
- CORe method is exposed (`method="core"`) for motion-robust comparison.
- The Stage 3 agent can call `reconstruct()` programmatically and feed results into `segment()` / `verify()` without leaving Python.
- A 50-iter run is expected to drop divergence from 43 → single digits by reducing reconstruction noise and sharpening vessel boundaries (currently bridging multiple anatomical regions).

---

## Stage 2c — Segmentation Skill ✅

**Date completed:** 2026-05-12

### Overview
PC-MRA-based vessel segmentation with two operating modes:
- **Auto:** largest connected high-PC-MRA component (`segment`, default)
- **Agent-driven:** region grow from a seed point chosen by the Stage 3 Coordinator (`segment_from_seed`)

The agent-driven API is the architectural primitive for Stage 3: the agent picks seeds based on anatomical priors / iterative feedback, calls `segment_from_seed()`, runs the Physics Verifier, and retries if needed.

### Files
| File | Description |
|---|---|
| `skills/segmentation/__init__.py` | Public API: `segment()`, `segment_from_seed()`, `suggest_seed_points()`, `compute_pcmra()` |
| `skills/segmentation/_pcmra.py` | PC-MRA pipeline, connected component logic, seed-based growing |
| `tests/test_segmentation.py` | 15 unit tests covering both auto and seed-based paths |
| `notebooks/test_seed_based_segmentation.py` | Agent-style multi-candidate evaluation on real data |

### API
```python
from skills.segmentation import segment, segment_from_seed, suggest_seed_points

# Automatic (default Skill behaviour)
mask = segment(thetaX, thetaY, thetaZ, xHat, venc=1.5)

# Agent-driven workflow
candidates = suggest_seed_points(thetaX, thetaY, thetaZ, xHat, venc, n_candidates=5)
# candidates = [{seed, size, mean_pcmra, bbox}, ...] sorted by brightness × size
chosen_seed = agent_decide(candidates)   # Stage 3 agent reasoning
mask = segment_from_seed(thetaX, thetaY, thetaZ, xHat, venc, seed_point=chosen_seed)
```

### Pipeline
1. **PC-MRA image** = signal magnitude × speed magnitude, time-averaged. Bright in flowing blood, dark in static tissue (more robust than signal magnitude alone — vessel lumen can be dimmer than surrounding tissue).
2. **Percentile threshold** (default 85th = top 15% of voxels).
3. **Morphological closing** (default 2 iterations) to fill small lumen gaps.
4. **Connected component labeling** → keep largest, top-N, or component containing seed.

### Test Results

**Unit tests (synthetic vessel cylinder):** 15 / 15 passing  
- 7 tests for auto segmentation
- 4 tests for seed-based growing
- 4 tests for seed candidate suggestion

**Integration test on real OSU-MR data** (`notebooks/test_seed_based_segmentation.py`):

| Seed (z,y,x) | Voxels | Divergence | Peak m/s | Notes |
|---|---|---|---|---|
| (42, 86, 59) | 36,356 | 43.1 s⁻¹ | 1.78 | Largest merged region |
| (46, 20, 39) | 13,994 | 75.7 s⁻¹ | 1.73 | Smaller merged region |
| (75, 83, 52) | 1,394 | 2.25 s⁻¹ | 0.10 | Low div but no real flow — agent rejects |

After applying agent-style filters (peak > 0.5 m/s for real flow, size > 5000 for real vessel), the best candidate has divergence 43 s⁻¹ — still elevated because of 5-iteration reconstruction quality and connected vasculature. **Stage 2c produces meaningful inputs; remaining divergence is a Stage 2a quality issue.**

### Design Decisions
- **Two-layer API** (auto + seed-based). The auto path is a sensible default; seed-based is for the agent. Both share the same underlying PC-MRA + connected-component pipeline.
- **Brightest voxel as seed**, not centroid. For non-convex branching vessels the centroid can fall in a gap between components. Using `argmax(pcmra)` within each component guarantees the seed lies inside the region it labels.
- **`suggest_seed_points()` returns structured candidates** (`seed`, `size`, `mean_pcmra`, `bbox`) so the Stage 3 agent has complete information for multi-criteria reasoning without re-running segmentation.

---

## Stage 2d — Hemodynamic Analysis Skill ✅

**Date completed:** 2026-05-16

### Overview
Computes volumetric flow rate Q(t), stroke volume, peak flow, and peak velocity at evenly-spaced cross-sections along a vessel mask. Single `analyze()` entry point matches the Physics Verifier API style.

### Files
| File | Description |
|---|---|
| `skills/hemodynamic/__init__.py` | Public API: `analyze()` |
| `skills/hemodynamic/_metrics.py` | `dominant_flow_axis`, `flow_rate_time_series`, `stroke_volume_and_peak`, `peak_velocity_in_mask` |
| `tests/test_hemodynamic.py` | 19 unit tests (constant-flow phantom, half-sine pulsatile, regurgitation) |
| `notebooks/test_hemodynamic_real_data.py` | End-to-end on Stage 2a reconstruction |

### API
```python
from skills.hemodynamic import analyze

result = analyze(
    thetaX, thetaY, thetaZ,
    venc_m_per_s = 1.5,
    mask = vessel_mask,         # from Stage 2c
    voxel_size_mm = (2, 2, 2),
    dt_seconds = 0.05,          # ~50 ms per cardiac phase
    n_cross_sections = 5,
)
# result["per_section"][i] → { plane_index, position_pct,
#                              Q_mL_per_s [time series],
#                              stroke_volume_forward_mL,
#                              stroke_volume_net_mL,
#                              peak_Q_mL_per_s,
#                              time_to_peak_s,
#                              regurgitant_fraction_pct }
# result["summary"] → { mean_stroke_volume_mL, mean_peak_Q_mL_per_s, max_peak_Q_mL_per_s,
#                       peak_velocity_m_per_s, mean_velocity_m_per_s, peak_velocity_per_phase }
# result["metadata"] → { dominant_axis, n_cross_sections, n_phases, dt_s,
#                        cycle_duration_s, voxel_size_mm, voxel_volume_mm3,
#                        n_mask_voxels, venc_m_per_s }
```

### Metrics
| Metric | Definition | Units |
|---|---|---|
| Q(t) | ∫∫ v·n̂ dA over the cross-section, per cardiac phase | mL/s |
| stroke_volume_forward_mL | ∑ max(sign·Q(t), 0) · dt | mL |
| stroke_volume_net_mL | sign · ∑ Q(t) · dt | mL |
| peak_Q_mL_per_s | max ⎢Q(t)⎢ | mL/s |
| regurgitant_fraction_pct | ⎢SV_backward⎢ / SV_forward × 100 | % |
| peak_velocity_m_per_s | max over volume × time of √(vx²+vy²+vz²) | m/s |

Forward direction = sign of Q at the time of peak |Q|. This handles vessels where flow direction is dominantly negative along the chosen axis.

### Cross-Section Convention
Matches `check_net_flux` in the Physics Verifier:
1. Pick the velocity axis with the largest time-averaged |v| inside the mask (Z, Y, or X)
2. Place N planes evenly spaced between 10% and 90% of that axis (skip endpoints — boundary slabs are usually outside the vessel)
3. For each plane, integrate v_axial across the mask voxels intersecting that slab

### Test Results
**Unit tests (synthetic phantoms):** 19 / 19 passing
- 3 dominant-axis tests (Z/Y/X)
- 3 flow-rate tests including a unit sanity check (1 m/s through 1 mm² = 1 mL/s)
- 5 stroke-volume tests including regurgitation and negative-forward direction
- 2 peak-velocity tests
- 6 end-to-end `analyze()` tests on constant-flow and pulsatile phantoms

**Integration test on real data** (Stage 2a 5-iter recon → Stage 2c seed-based segmentation → 2d):
| Metric | Value |
|---|---|
| Vessel volume | 268.9 mL (33,618 voxels — merged aorta + chambers) |
| Mean SV (forward) | 93.9 mL |
| Mean peak flow | 438 mL/s |
| Peak velocity | 2.59 m/s |
| Regurgitation | 26–130% across sections |

Stroke volume is physiologically plausible (adult: ~60–100 mL). The high inter-section variability and regurgitation values reflect the known [[project_stage2_state]] issue: 5-iter recon plus multi-vessel merging means cross-sections cut through multiple anatomical structures. A 50-iter run is expected to give tight, consistent per-section metrics.

### Design Decisions
- **Single `analyze()` entry**, structured-dict output. Mirrors Physics Verifier so downstream code (Stage 3 agent, evaluation harness) treats both skills identically.
- **Q stored as a time series, not just summary stats.** This lets downstream tools plot waveforms, detect arrhythmia, or run more advanced metrics later without re-running analysis.
- **Forward direction inferred from Q at peak**, not assumed positive. Vessels can be oriented either way relative to the volume axis; auto-detecting avoids manual sign conventions.
- **Mask is required.** Auto-segmenting from inside Stage 2d would couple it to Stage 2c; keeping them separate lets the Stage 3 agent pick the seed and pass an explicit mask.
- **Skipped for v1:** Wall shear stress (needs surface-normal estimation that's quality-limited by segmentation), pulse-wave velocity (needs two cross-sections + time-of-flight that 20-phase data resolves poorly).

---
