"""
Integration test: run Physics Verifier on the real Stage 1c reconstruction output.
Prints a human-readable verdict with actual numerical values.

Usage:
    conda activate medict
    python notebooks/test_verifier_real_data.py
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.physics_verifier import verify, verdict_json

MAT_PATH = "/mnt/g/medict_out/test_4dflow_cs.mat"

# VENC from OSU-MR dataset scan parameters
# Update if you find the exact value in data.D — typical aortic 4D flow is 1.5 m/s
VENC_M_PER_S = 1.5

# Isotropic voxel size from OSU-MR dataset (update if known exactly)
VOXEL_SIZE_MM = (2.0, 2.0, 2.0)

# Vessels occupy ~10-20% of cardiac volume; 75th percentile ≈ top 25% by magnitude
MASK_PERCENTILE = 75.0


def load_reconstruction(path: str) -> dict:
    print(f"Loading: {path}")
    try:
        import scipy.io as sio
        raw = sio.loadmat(path, squeeze_me=True)
        outputs = raw["outputs"]
        return {
            "thetaX": np.array(outputs["thetaX"].item()),
            "thetaY": np.array(outputs["thetaY"].item()),
            "thetaZ": np.array(outputs["thetaZ"].item()),
            "xHat":   np.array(outputs["xHat"].item()),
        }
    except NotImplementedError:
        print("  (v7.3 HDF5 format — using h5py)")
        import h5py
        with h5py.File(path, "r") as f:
            outputs = f["outputs"]
            return {
                "thetaX": np.array(outputs["thetaX"]).T,
                "thetaY": np.array(outputs["thetaY"]).T,
                "thetaZ": np.array(outputs["thetaZ"]).T,
                "xHat":   np.array(outputs["xHat"]).T,
            }


def print_verdict(result: dict) -> None:
    STATUS_ICONS = {"pass": "✓", "warn": "⚠", "fail": "✗"}
    overall = result["verdict"]
    icon = STATUS_ICONS[overall]
    meta = result["metadata"]

    print()
    print(f"{'='*55}")
    print(f"  Physics Verifier — {icon} {overall.upper()}")
    print(f"{'='*55}")
    print(f"  Shape (Z,Y,X,T) : {meta['shape_ZYXT']}")
    print(f"  VENC            : {meta['venc_m_per_s']} m/s")
    print(f"  Voxel size      : {meta['voxel_size_mm']} mm")
    print(f"  Mask voxels     : {meta['n_mask_voxels']}")
    print()

    checks = result["checks"]

    div = checks["divergence"]
    print(f"  {STATUS_ICONS[div['status']]} Divergence")
    print(f"      mean |∇·v| = {div['mean_abs_divergence_per_s']:.4f} s⁻¹"
          f"  (warn>{div['threshold_warn_per_s']}, fail>{div['threshold_fail_per_s']})")

    flux = checks["net_flux"]
    print(f"  {STATUS_ICONS[flux['status']]} Net Flux  [{flux['dominant_axis']} axis]")
    print(f"      flows     = {flux['flux_mL_per_s']} mL/s")
    print(f"      max dev   = {flux['max_deviation_pct']:.1f}%"
          f"  (warn>{flux['threshold_warn_pct']}%, fail>{flux['threshold_fail_pct']}%)")

    vel = checks["peak_velocity"]
    print(f"  {STATUS_ICONS[vel['status']]} Peak Velocity")
    print(f"      peak      = {vel['peak_m_per_s']:.4f} m/s")
    print(f"      p95       = {vel['p95_m_per_s']:.4f} m/s")
    print(f"      mean      = {vel['mean_m_per_s']:.4f} m/s")
    print(f"      range     = {vel['physiological_range_m_per_s']} m/s")

    pu = checks["phase_unwrap"]
    print(f"  {STATUS_ICONS[pu['status']]} Phase Unwrap")
    print(f"      max jump  = {pu['max_spatial_jump_m_per_s']:.4f} m/s  (VENC={pu['venc_m_per_s']})")
    print(f"      frac >thr = {pu['fraction_above_threshold']:.6f}"
          f"  (warn>{pu['threshold_warn_fraction']}, fail>{pu['threshold_fail_fraction']})")

    print(f"{'='*55}")
    print()


if __name__ == "__main__":
    data = load_reconstruction(MAT_PATH)

    print(f"  thetaX shape: {data['thetaX'].shape}")
    print(f"  thetaY shape: {data['thetaY'].shape}")
    print(f"  thetaZ shape: {data['thetaZ'].shape}")
    print(f"  xHat   shape: {data['xHat'].shape}")
    print(f"  Phase range:  [{data['thetaX'].min():.3f}, {data['thetaX'].max():.3f}] rad")

    from skills.physics_verifier._mask import velocity_mask
    mask = velocity_mask(data["thetaX"], data["thetaY"], data["thetaZ"],
                         venc_m_per_s=VENC_M_PER_S, percentile=MASK_PERCENTILE)
    print(f"  Mask voxels: {mask.sum():,} ({mask.sum()/mask.size*100:.1f}% of volume)")

    result = verify(
        data["thetaX"],
        data["thetaY"],
        data["thetaZ"],
        venc_m_per_s=VENC_M_PER_S,
        mask=mask,
        voxel_size_mm=VOXEL_SIZE_MM,
    )

    print_verdict(result)

    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "verifier_real_data.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Full JSON saved → outputs/verifier_real_data.json")
