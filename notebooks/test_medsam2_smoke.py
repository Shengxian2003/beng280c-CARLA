"""MedSAM2 smoke test on the 50-iter recon.

Workflow:
  1. Load the recon; compute time-averaged magnitude as the anatomy image
  2. Use our PC-MRA seed picker to find a vessel candidate
  3. Build a 2-D bounding box around the seed on its Z slice
  4. Hand it to MedSAM2 as a box prompt; let it propagate through all Z slices
  5. Compare the resulting mask to the existing PC-MRA segmentation via
     the Physics Verifier
"""
from __future__ import annotations

import os, sys, time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.segmentation import suggest_seed_points, segment_from_seed
from skills.physics_verifier import verify

MEDSAM2_ROOT = Path("/tmp/MedSAM2")
sys.path.insert(0, str(MEDSAM2_ROOT))
os.chdir(MEDSAM2_ROOT)  # SAM2 config loads relative to cwd
from sam2.build_sam import build_sam2_video_predictor_npz

RECON_MAT = "/mnt/g/medict_tmp/recon_cs_50iter.mat"
VENC      = 1.5
VOXEL_MM  = (2.0, 2.0, 2.0)
OUT_DIR   = Path("/mnt/g/medict_tmp/medsam2_smoketest")
OUT_DIR.mkdir(exist_ok=True)

# ---- 1. Load recon ----------------------------------------------------------
print("Loading 50-iter recon...")
o = sio.loadmat(RECON_MAT, squeeze_me=True)["outputs"]
xHat   = np.array(o["xHat"].item())
thetaX = np.array(o["thetaX"].item())
thetaY = np.array(o["thetaY"].item())
thetaZ = np.array(o["thetaZ"].item())
Z, Y, X, T = xHat.shape
print(f"shape (Z,Y,X,T) = {xHat.shape}")

# Use PC-MRA image instead of plain magnitude — PC-MRA = |signal| * |velocity|,
# which is bright wherever blood is flowing. This matches what the PC-MRA seed
# picker also uses, so the seed lands on a bright voxel that MedSAM2 can see.
# (Plain magnitude is too dark in vessel lumen; MedSAM2 returns all-background.)
signal_mag = np.abs(xHat)
speed_mag  = np.sqrt(
    (thetaX * VENC / np.pi) ** 2 + (thetaY * VENC / np.pi) ** 2 + (thetaZ * VENC / np.pi) ** 2
)
pcmra = (signal_mag * speed_mag).mean(axis=-1).astype(np.float32)
# Clip to 99th percentile so a few hot voxels don't compress everything else
p99 = float(np.percentile(pcmra, 99))
pcmra_n = (np.clip(pcmra, 0, p99) / max(p99, 1e-9)) * 255.0
mag_u8 = pcmra_n.astype(np.uint8)
print(f"PC-MRA uint8 range: {mag_u8.min()} - {mag_u8.max()} (99th-pct clipped at {p99:.3g})")

# ---- 2. Pick a seed via existing PC-MRA tool --------------------------------
print("\nFinding best PC-MRA vessel candidate...")
candidates = suggest_seed_points(thetaX, thetaY, thetaZ, xHat, VENC,
                                 n_candidates=3, percentile=90, closing_iter=1)
seed = candidates[0]["seed"]              # (z, y, x)
print(f"top seed: {seed}  size={candidates[0]['size']:,} voxels  "
      f"pcmra={candidates[0]['mean_pcmra']:.3f}")
sz, sy, sx = int(seed[0]), int(seed[1]), int(seed[2])

# Reference PC-MRA mask for comparison later
ref_mask = segment_from_seed(thetaX, thetaY, thetaZ, xHat, VENC,
                             seed_point=seed, percentile=90, closing_iter=1)
print(f"reference PC-MRA mask: {int(ref_mask.sum()):,} voxels")

# ---- 3. Build a SMALL bbox prompt centered on the seed ----------------------
# Use a small fixed-size window so we mimic "human points at a vessel".
# The PC-MRA mask is too inclusive (covers most of the slice on merged datasets).
key_slice = sz
BOX_HALF = 12   # 24x24 window — generous around an aorta cross-section
y0, y1 = max(0, sy - BOX_HALF), min(Y - 1, sy + BOX_HALF)
x0, x1 = max(0, sx - BOX_HALF), min(X - 1, sx + BOX_HALF)
print(f"key Z slice: {key_slice}, bbox (y0,x0,y1,x1) = ({y0},{x0},{y1},{x1})  "
      f"({y1-y0}x{x1-x0} in {Y}x{X})")

# ---- 4. Build SAM2 input volume (RGB, 512x512, ImageNet-normalized) --------
print("\nPreparing SAM2 input volume...")
from PIL import Image
def to_rgb_512(volume_u8):
    Z = volume_u8.shape[0]
    out = np.zeros((Z, 3, 512, 512), dtype=np.float32)
    for z in range(Z):
        img = Image.fromarray(volume_u8[z]).convert("RGB").resize((512, 512))
        out[z] = np.array(img, dtype=np.float32).transpose(2, 0, 1)
    return out

# Volume Z-axis = slice axis. mag_u8 is (Z,Y,X)
H_orig, W_orig = mag_u8.shape[1], mag_u8.shape[2]
img_resized = to_rgb_512(mag_u8) / 255.0
img_resized = torch.from_numpy(img_resized).cuda()
img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].cuda()
img_std  = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].cuda()
img_resized = (img_resized - img_mean) / img_std

# Rescale bbox to 512x512 coordinates (SAM2 expects xyxy)
scale_x = 512.0 / W_orig
scale_y = 512.0 / H_orig
bbox_xyxy = np.array([x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y], dtype=np.float32)
print(f"bbox in 512 space (xyxy): {bbox_xyxy}")

# ---- 5. Build predictor + run inference -------------------------------------
print("\nBuilding MedSAM2 predictor...")
predictor = build_sam2_video_predictor_npz(
    config_file="configs/sam2.1_hiera_t512.yaml",
    ckpt_path="checkpoints/MedSAM2_latest.pt",
)
print("predictor ready, running propagation...")

t0 = time.time()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(img_resized, H_orig, W_orig)
    _, _, mask_logits = predictor.add_new_points_or_box(
        inference_state=state, frame_idx=key_slice, obj_id=1, box=bbox_xyxy,
    )

    # Propagate forward (later Z) + backward (earlier Z)
    mask_3d = np.zeros((Z, H_orig, W_orig), dtype=bool)
    for frame_idx, obj_ids, logits in predictor.propagate_in_video(
            state, start_frame_idx=key_slice, reverse=False):
        mask_3d[frame_idx] = (logits[0] > 0).cpu().numpy()[0]
    for frame_idx, obj_ids, logits in predictor.propagate_in_video(
            state, start_frame_idx=key_slice, reverse=True):
        mask_3d[frame_idx] = (logits[0] > 0).cpu().numpy()[0]

elapsed = time.time() - t0
print(f"MedSAM2 inference: {elapsed:.1f}s for {Z} slices")
print(f"MedSAM2 mask: {int(mask_3d.sum()):,} voxels")

np.save(OUT_DIR / "medsam2_mask.npy", mask_3d)

# ---- 6. Verify both masks -----------------------------------------------
print("\n--- Physics Verifier comparison ---\n")
for name, mask in [("PC-MRA (current)", ref_mask), ("MedSAM2", mask_3d)]:
    if not mask.any():
        print(f"{name:20s}: EMPTY MASK, skipping verify")
        continue
    verdict = verify(thetaX, thetaY, thetaZ, venc_m_per_s=VENC,
                     mask=mask, voxel_size_mm=VOXEL_MM)
    print(f"{name:20s}  {int(mask.sum()):>7,} vox  verdict={verdict['verdict']}")
    for ck, cinfo in verdict["checks"].items():
        print(f"  {ck:14s} {cinfo['status']}")
    print()

print(f"\nMask saved: {OUT_DIR / 'medsam2_mask.npy'}")
