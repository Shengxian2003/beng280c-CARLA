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

## Stage 2a — Reconstruction Skill ⬜  
*Pending — Python wrapper around MATLAB reconstruction*

---

## Stage 2c — Segmentation Skill ⬜  
*Pending — will unblock divergence and net flux checks on real data*

---

## Stage 2d — Hemodynamic Analysis Skill ⬜  
*Pending — depends on Stage 2c vessel mask*
