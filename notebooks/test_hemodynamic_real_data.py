"""Stage 2d integration: hemodynamic analysis on the Stage 2a reconstruction.

Pipeline composed end-to-end:
    reconstruct() output → segment_from_seed() → analyze()

Loads the saved 5-iter reconstruction (no MATLAB needed), picks the same best
seed the agent-style segmentation test found, runs analyze(), and prints
flow / stroke-volume / peak metrics per cross-section.
"""
import json
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.segmentation import suggest_seed_points, segment_from_seed
from skills.hemodynamic import analyze

MAT_PATH  = "/mnt/g/medict_tmp/recon_cs_5iter.mat"  # Stage 2a output
VENC      = 1.5
VOXEL_MM  = (2.0, 2.0, 2.0)
DT_S      = 0.05  # ~50 ms per phase, 20 phases ≈ 1 s cardiac cycle

# ---- Load reconstruction ---------------------------------------------------
print(f"Loading reconstruction: {MAT_PATH}")
o = sio.loadmat(MAT_PATH, squeeze_me=True)["outputs"]
xHat = np.array(o["xHat"].item())
tx   = np.array(o["thetaX"].item())
ty   = np.array(o["thetaY"].item())
tz   = np.array(o["thetaZ"].item())
print(f"Shape: {tx.shape}")

# ---- Segment a vessel ------------------------------------------------------
print("\nPicking best vessel candidate (agent-style)...")
candidates = suggest_seed_points(tx, ty, tz, xHat, VENC,
                                 n_candidates=5, percentile=90, closing_iter=1)
# Same filter the Stage 3 agent uses: real flow + real vessel
viable = []
for c in candidates:
    mask = segment_from_seed(tx, ty, tz, xHat, VENC, seed_point=c["seed"],
                             percentile=90, closing_iter=1)
    if not mask.any():
        continue
    peak = float(np.sqrt((tx*VENC/np.pi)**2 + (ty*VENC/np.pi)**2 +
                         (tz*VENC/np.pi)**2)[mask].max())
    if peak > 0.5 and int(mask.sum()) > 5000:
        viable.append((c["seed"], mask, peak))
if not viable:
    sys.exit("No viable vessel — re-run reconstruct with more iterations")

seed, mask, peak_v = max(viable, key=lambda v: int(v[1].sum()))
print(f"Chosen seed: {seed}, vessel size: {int(mask.sum())} voxels, peak |v|: {peak_v:.2f} m/s")

# ---- Hemodynamic analysis --------------------------------------------------
print("\nRunning analyze()...")
result = analyze(tx, ty, tz, venc_m_per_s=VENC, mask=mask,
                 voxel_size_mm=VOXEL_MM, dt_seconds=DT_S, n_cross_sections=5)

# ---- Report ---------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Hemodynamic Analysis")
print(f"{'='*70}")
print(f"Dominant axis:      {result['metadata']['dominant_axis']}")
print(f"Cycle duration:     {result['metadata']['cycle_duration_s']:.2f} s "
      f"({result['metadata']['n_phases']} phases × {result['metadata']['dt_s']*1000:.0f} ms)")
print(f"Vessel volume:      {result['metadata']['n_mask_voxels']} voxels "
      f"× {result['metadata']['voxel_volume_mm3']:.1f} mm³ "
      f"= {result['metadata']['n_mask_voxels']*result['metadata']['voxel_volume_mm3']/1000:.1f} mL")

print(f"\nSummary:")
s = result["summary"]
print(f"  Mean SV (forward) across sections: {s['mean_stroke_volume_mL']:.2f} mL")
print(f"  Mean peak flow:                    {s['mean_peak_Q_mL_per_s']:.2f} mL/s")
print(f"  Max peak flow:                     {s['max_peak_Q_mL_per_s']:.2f} mL/s")
print(f"  Peak velocity in vessel:           {s['peak_velocity_m_per_s']:.2f} m/s")
print(f"  Mean velocity in vessel:           {s['mean_velocity_m_per_s']:.3f} m/s")

print(f"\nPer cross-section:")
print(f"  {'plane':>6} {'pos%':>6} {'SV_fwd':>9} {'SV_net':>9} {'Q_peak':>9} "
      f"{'t_peak':>8} {'reflux%':>9}")
for sec in result["per_section"]:
    print(f"  {sec['plane_index']:>6d} {sec['position_pct']:>6.1f} "
          f"{sec['stroke_volume_forward_mL']:>9.2f} {sec['stroke_volume_net_mL']:>9.2f} "
          f"{sec['peak_Q_mL_per_s']:>9.2f} {sec['time_to_peak_s']:>8.3f} "
          f"{sec['regurgitant_fraction_pct']:>9.1f}")

# Save full report next to the recon output
out_json = os.path.join(os.path.dirname(MAT_PATH), "hemodynamic_report.json")
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nFull JSON written: {out_json}")
