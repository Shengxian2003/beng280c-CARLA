"""Base SAM2 (not MedSAM2) smoke test on the 50-iter recon.

Base SAM2 is trained on natural images (not medical). Hypothesis: it might
still segment our vessels because PC-MRA shows bright vessels on dark
background — visually similar to "bright blob on dark background" which
generic SAM2 should handle, regardless of training distribution.

If this works, we have a viable replacement for PC-MRA segmentation.
If it also returns -1024, then NO foundation model handles our data and we
should pivot to fine-tuning or stick with PC-MRA.
"""
from __future__ import annotations

import os, sys, time, tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.segmentation import suggest_seed_points, segment_from_seed
from skills.physics_verifier import verify
from sam2.build_sam import build_sam2_video_predictor

RECON_MAT = "/mnt/g/medict_tmp/recon_cs_50iter.mat"
SAM2_CKPT = "/tmp/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
VENC      = 1.5
VOXEL_MM  = (2.0, 2.0, 2.0)
OUT_DIR   = Path("/mnt/g/medict_tmp/sam2_base_smoketest")
OUT_DIR.mkdir(exist_ok=True)

# ---- 1. Load recon, compute PC-MRA -----------------------------------------
print("Loading 50-iter recon...")
o = sio.loadmat(RECON_MAT, squeeze_me=True)["outputs"]
xHat   = np.array(o["xHat"].item())
tx, ty, tz = (np.array(o[k].item()) for k in ("thetaX","thetaY","thetaZ"))
Z, Y, X, T = xHat.shape

speed = np.sqrt((tx*VENC/np.pi)**2 + (ty*VENC/np.pi)**2 + (tz*VENC/np.pi)**2)
pcmra = (np.abs(xHat) * speed).mean(axis=-1).astype(np.float32)
p99 = float(np.percentile(pcmra, 99))
mag_u8 = (np.clip(pcmra, 0, p99) / max(p99, 1e-9) * 255.0).astype(np.uint8)
print(f"PC-MRA uint8 stats: min={mag_u8.min()}, max={mag_u8.max()}, mean={mag_u8.mean():.1f}")

# ---- 2. Pick seed via PC-MRA ------------------------------------------------
cands = suggest_seed_points(tx, ty, tz, xHat, VENC, n_candidates=3,
                            percentile=90, closing_iter=1)
seed = cands[0]["seed"]; sz, sy, sx = [int(s) for s in seed]
print(f"seed (z,y,x) = ({sz},{sy},{sx})  PC-MRA at seed: {mag_u8[sz,sy,sx]}/255")

ref_mask = segment_from_seed(tx, ty, tz, xHat, VENC, seed_point=seed,
                             percentile=90, closing_iter=1)
print(f"reference PC-MRA mask: {int(ref_mask.sum()):,} voxels")

# ---- 3. Save Z slices as JPGs (base SAM2 wants a video folder) -------------
frames_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
print(f"\nWriting {Z} JPG frames to {frames_dir} ...")
for z in range(Z):
    Image.fromarray(mag_u8[z]).convert("RGB").save(frames_dir / f"{z:05d}.jpg")

# ---- 4. Build predictor + run inference ------------------------------------
print(f"\nBuilding base SAM2 predictor (config {SAM2_CFG})...")
predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
print("predictor ready")

# Tight 24x24 bbox in ORIGINAL (Y,X) frame coordinates — SAM2 expects xyxy
y0, y1 = max(0, sy-12), min(Y-1, sy+12)
x0, x1 = max(0, sx-12), min(X-1, sx+12)
bbox = np.array([x0, y0, x1, y1], dtype=np.float32)
point = np.array([[sx, sy]], dtype=np.float32)
labels = np.array([1], dtype=np.int32)
print(f"bbox xyxy: {bbox} (in {Y}x{X} frame)")
print(f"point xy:  {point[0]}")

t0 = time.time()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=str(frames_dir))
    print(f"  state initialized: {state['num_frames']} frames")

    # Try BOX prompt first
    _, out_ids, logits = predictor.add_new_points_or_box(
        inference_state=state, frame_idx=sz, obj_id=1, box=bbox)
    print(f"\n  BOX prompt: logits shape {logits.shape}, "
          f"min/max/mean = {float(logits.min()):.2f}/{float(logits.max()):.2f}/{float(logits.mean()):.2f}")
    print(f"  # voxels with logit > 0 on key slice: {int((logits > 0).sum())}")

    # Propagate forward + backward
    mask_3d = np.zeros((Z, Y, X), dtype=bool)
    for fidx, oids, lgs in predictor.propagate_in_video(state, start_frame_idx=sz, reverse=False):
        mask_3d[fidx] = (lgs[0] > 0).cpu().numpy()[0]
    for fidx, oids, lgs in predictor.propagate_in_video(state, start_frame_idx=sz, reverse=True):
        mask_3d[fidx] = (lgs[0] > 0).cpu().numpy()[0]

elapsed = time.time() - t0
print(f"\nbase SAM2 inference: {elapsed:.1f}s for {Z} slices")
print(f"base SAM2 mask: {int(mask_3d.sum()):,} voxels")

np.save(OUT_DIR / "sam2_base_mask.npy", mask_3d)

# Clean up frames
import shutil; shutil.rmtree(frames_dir)

# ---- 5. Verifier comparison -------------------------------------------------
print("\n--- Physics Verifier comparison ---\n")
for name, mask in [("PC-MRA (current)", ref_mask), ("base SAM2", mask_3d)]:
    if not mask.any():
        print(f"{name:20s}: EMPTY MASK")
        continue
    verdict = verify(tx, ty, tz, venc_m_per_s=VENC, mask=mask, voxel_size_mm=VOXEL_MM)
    print(f"{name:20s} {int(mask.sum()):>7,} vox  verdict={verdict['verdict']}")
    for ck, c in verdict["checks"].items():
        print(f"  {ck:14s} {c['status']}")
    print()
