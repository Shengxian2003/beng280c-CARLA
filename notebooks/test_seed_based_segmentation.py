"""
Stage 2c agent-driven workflow demo:
  1. Ask segmentation for candidate seed points
  2. For each candidate, run the Physics Verifier
  3. Report which seed produces the best (lowest divergence / flux deviation) result

This simulates what the Stage 3 Coordinator agent would do automatically.
"""
import sys
import os
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.segmentation import suggest_seed_points, segment_from_seed
from skills.physics_verifier import verify

MAT_PATH = "/mnt/g/medict_out/test_4dflow_cs.mat"
VENC = 1.5
VOXEL_MM = (2.0, 2.0, 2.0)

# Load reconstruction output
raw = sio.loadmat(MAT_PATH, squeeze_me=True)
o = raw["outputs"]
tx, ty, tz, xH = (np.array(o[k].item()) for k in ("thetaX", "thetaY", "thetaZ", "xHat"))
print(f"Loaded reconstruction: shape {tx.shape}\n")

# Step 1: get candidate seed points (what the agent inspects)
candidates = suggest_seed_points(tx, ty, tz, xH, VENC, n_candidates=5, percentile=90, closing_iter=1)
print(f"Found {len(candidates)} candidate vessel regions:\n")
print(f"{'#':>3} {'seed (z,y,x)':>16} {'size':>8} {'mean_pcmra':>11} {'bbox dims':>14}")
for i, c in enumerate(candidates):
    z0, y0, x0, z1, y1, x1 = c["bbox"]
    dims = f"{z1-z0}x{y1-y0}x{x1-x0}"
    print(f"{i:>3} {str(c['seed']):>16} {c['size']:>8,} {c['mean_pcmra']:>11.3f} {dims:>14}")

# Step 2: for each candidate, run the verifier and report results
print(f"\n{'='*70}")
print(f"{'Seed':>16} {'Voxels':>8} {'Div':>8} {'Flux %dev':>10} {'Peak m/s':>9} {'Verdict':>9}")
print(f"{'='*70}")

evaluations = []
for c in candidates:
    seed = c["seed"]
    mask = segment_from_seed(tx, ty, tz, xH, VENC, seed_point=seed,
                              percentile=90, closing_iter=1)
    if not mask.any():
        print(f"{str(seed):>16} {'EMPTY':>8}  — skip —")
        continue

    result = verify(tx, ty, tz, VENC, mask=mask, voxel_size_mm=VOXEL_MM)
    checks = result["checks"]
    div = checks["divergence"]["mean_abs_divergence_per_s"]
    flux_dev = checks["net_flux"]["max_deviation_pct"]
    peak = checks["peak_velocity"]["peak_m_per_s"]
    verdict = result["verdict"]
    size = int(mask.sum())

    print(f"{str(seed):>16} {size:>8,} {div:>8.2f} {flux_dev:>10.1f} {peak:>9.3f} {verdict:>9}")

    evaluations.append({
        "seed": seed, "size": size, "div": div,
        "flux_dev": flux_dev, "peak": peak, "verdict": verdict,
    })

# Agent-style multi-criteria selection
# A real vessel must have actual flow (peak > 0.5 m/s) and reasonable size (> 5000 voxels)
qualifying = [e for e in evaluations if e["peak"] > 0.5 and e["size"] > 5000]

print(f"\n{'='*70}")
print(f"Agent reasoning:")
print(f"  Total candidates: {len(evaluations)}")
print(f"  After 'peak > 0.5 m/s' filter (real flow): "
      f"{sum(1 for e in evaluations if e['peak'] > 0.5)}")
print(f"  After 'size > 5000 voxels' filter (real vessel): "
      f"{len(qualifying)}")

if qualifying:
    best = min(qualifying, key=lambda e: e["div"])
    print(f"\nBest vessel (lowest divergence among qualifying): {best['seed']}")
    print(f"  size:       {best['size']:,} voxels")
    print(f"  divergence: {best['div']:.2f} s⁻¹")
    print(f"  flux dev:   {best['flux_dev']:.1f}%")
    print(f"  peak vel:   {best['peak']:.3f} m/s")
    print(f"  verdict:    {best['verdict']}")
else:
    print(f"\nNo qualifying candidate — agent would either:")
    print(f"  - Retry with tighter percentile (95+) to break apart merged vessels")
    print(f"  - Request 50-iteration reconstruction for cleaner velocity field")
