"""Project-wide context shared with every LLM agent.

Why this file exists:
    Without explicit knowledge of where files live and what defaults apply,
    LLMs hallucinate plausible-looking paths (e.g. "data/kspace.mat" — does
    not exist). This file is the canonical "what every agent needs to know
    about MEDICT" string, injected into every specialist and the Coordinator's
    system prompt at construction time.

Update this file when paths or acquisition parameters change. The agents will
all pick up the new values on next run.
"""
from __future__ import annotations

from dataclasses import dataclass


# ============================================================================
# Canonical paths
# ============================================================================

# Raw k-space input — Stage 1c data, gitignored 806 MB .mat file on the WSL fs.
KSPACE_PATH = "/home/nick_17/projects/medict/data/4D_Flow_Cartesian_Dataset_11.mat"

# Existing reconstruction outputs on the Windows-mounted scratch drive.
# Stage 2a writes here; load_reconstruction reads from here.
EXISTING_RECON_5ITER = "/mnt/g/medict_tmp/recon_cs_5iter.mat"
EXISTING_RECON_50ITER = "/mnt/g/medict_tmp/recon_cs_50iter.mat"  # may not exist yet
STAGE_DIR = "/mnt/g/medict_tmp"


# ============================================================================
# Acquisition defaults (these match Stage 1c / Stage 2a / Stage 2d)
# ============================================================================

@dataclass(frozen=True)
class AcquisitionDefaults:
    venc_m_per_s:  float = 1.5
    voxel_size_mm: tuple = (2.0, 2.0, 2.0)     # (dz, dy, dx)
    n_cardiac_phases: int = 20
    dt_seconds:    float = 0.05                 # ≈ R-R / n_phases
    shape_ZYXT:    tuple = (77, 96, 72, 20)


ACQUISITION = AcquisitionDefaults()


# ============================================================================
# The string injected into every agent's system prompt
# ============================================================================

PROJECT_CONTEXT = f"""\
## Project: MEDICT — Multi-Agent 4D Flow MRI Pipeline (BENG 280C, UCSD)

Short goal: produce auditable hemodynamic metrics (volumetric flow Q(t),
stroke volume, peak velocity) from cardiac 4D flow MRI, with deterministic
physics verification flagging untrustworthy results.

### Standard data paths (use these — do NOT guess)

- **Raw k-space input** (input to `reconstruct`):
  `{KSPACE_PATH}`
  This is the gitignored ~800 MB .mat from the OSU-MR Zenodo dataset.

- **Existing reconstructions** (input to `load_reconstruction`):
  - 5-iteration CS (smoke-test quality, exists): `{EXISTING_RECON_5ITER}`
  - 50-iteration CS (full quality): `{EXISTING_RECON_50ITER}` — may not exist
    yet; if missing, you can produce it by calling `reconstruct` with
    `kspace_path={KSPACE_PATH}` and `n_iterations=50`.

- **Stage directory** (any new outputs go here): `{STAGE_DIR}`

### Acquisition parameters

- VENC: {ACQUISITION.venc_m_per_s} m/s
- Voxel size (dz, dy, dx): {ACQUISITION.voxel_size_mm} mm
- Cardiac phases: {ACQUISITION.n_cardiac_phases}  (dt = {ACQUISITION.dt_seconds} s)
- Expected reconstruction shape: {ACQUISITION.shape_ZYXT} (Z, Y, X, T)

### Pipeline conventions

The pipeline runs in this order:
  1. **reconstruct**  (or load_reconstruction)
  2. **suggest_seeds** → **segment_from_seed** (Segmentation Operator)
  3. **verify**          (Physics Verifier)
  4. **analyze**         (Hemodynamic Analyzer)

If unsure whether to load or re-run:
  - **Smoke test or fast iteration**: load `{EXISTING_RECON_5ITER}`
  - **Final quality / clinical metrics**: run `reconstruct` at 50 iterations
  - Re-running takes ~12 minutes on RTX 5090; loading is instant.

### Known quality issues to watch for

- 5-iteration recon is under-converged; divergence and net-flux checks
  routinely fail on it. Peak velocity and phase unwrap usually pass.
- Default segmentation (percentile=90) tends to merge the aorta with heart
  chambers via thin PC-MRA bridges. Tightening to percentile≥95 helps;
  closing_iter=1 is usually enough.

### What does NOT count as "the recon"

- `data/kspace.mat` (does not exist — likely a hallucination)
- `outputs/*` (a project subdirectory, not where recons live)
- Any path under `~` other than the canonical k-space path above
"""


__all__ = [
    "PROJECT_CONTEXT",
    "ACQUISITION",
    "AcquisitionDefaults",
    "KSPACE_PATH",
    "EXISTING_RECON_5ITER",
    "EXISTING_RECON_50ITER",
    "STAGE_DIR",
]
