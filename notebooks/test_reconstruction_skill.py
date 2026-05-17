"""Stage 2a smoke test: invoke the Reconstruction Skill end-to-end.

Runs the same 5-iteration CS reconstruction that Stage 1c did, but through
the new Python wrapper. This validates the full pipeline:
  1. Stage k-space from WSL → G:\
  2. Build JSON config
  3. Invoke Windows MATLAB
  4. Load outputs back into Python
  5. Verify shape and physical plausibility
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.reconstruction import reconstruct
from skills.physics_verifier import verify

INPUT_MAT  = "/home/nick_17/projects/medict/data/4D_Flow_Cartesian_Dataset_11.mat"
VENC       = 1.5
VOXEL_MM   = (2.0, 2.0, 2.0)

# Quick smoke test: 5 iterations (~1-2 min on RTX 5090).
# For full quality, pass n_iterations=50 (~12 min).
N_ITER = int(os.environ.get("N_ITER", "5"))
METHOD = os.environ.get("METHOD", "cs")

print(f"=== Stage 2a smoke test ===")
print(f"Input:      {INPUT_MAT}")
print(f"Method:     {METHOD}")
print(f"Iterations: {N_ITER}")
print()

result = reconstruct(
    INPUT_MAT,
    venc_m_per_s = VENC,
    method       = METHOD,
    n_iterations = N_ITER,
    is_flow      = True,
    is_rest      = True,
    use_gpu      = True,
)

print()
print("=== Reconstruction returned ===")
print(f"  xHat shape:   {result['xHat'].shape}")
print(f"  thetaX shape: {result['thetaX'].shape}")
print(f"  output:       {result['output_path']}")
print(f"  wall time:    {result['elapsed_wall_s']/60:.2f} min")
if "elapsed_minutes" in result.get("meta", {}):
    print(f"  matlab time:  {result['meta']['elapsed_minutes']:.2f} min")

assert result["xHat"].ndim == 4, "xHat must be 4D (Z, Y, X, T)"
assert result["thetaX"].shape == result["xHat"].shape
assert np.isfinite(result["xHat"]).all(), "xHat contains NaN/Inf"
assert np.isfinite(result["thetaX"]).all()

# Run Physics Verifier on the output
print()
print("=== Physics Verifier on reconstruction ===")
verdict = verify(
    result["thetaX"], result["thetaY"], result["thetaZ"],
    venc_m_per_s = VENC,
    voxel_size_mm = VOXEL_MM,
)
print(f"  Overall verdict: {verdict['verdict']}")
for name, check in verdict["checks"].items():
    print(f"  {name:14s} {check['status']:5s}")

print()
print("=== Stage 2a smoke test PASSED ===")
