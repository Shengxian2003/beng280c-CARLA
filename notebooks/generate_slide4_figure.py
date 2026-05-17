"""
Generate the 3-panel figure for the Slide 4 Expected Outcomes:
  - Magnitude slice
  - Velocity slice (color)
  - Segmentation mask overlay
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "/home/nick_17/projects/medict")
from skills.segmentation import segment

RECON_PATH = "/mnt/g/medict_out/test_4dflow_cs.mat"
OUT_PATH   = "/home/nick_17/projects/medict/notebooks/slide4_figure.png"
VENC       = 1.5  # m/s, from the dataset scan parameters


# load reconstruction output
print("Loading reconstruction...")
try:
    import scipy.io as sio
    data = sio.loadmat(RECON_PATH)
    outputs = data["outputs"][0, 0]
    xHat   = np.asarray(outputs["xHat"])
    thetaX = np.asarray(outputs["thetaX"])
    thetaY = np.asarray(outputs["thetaY"])
    thetaZ = np.asarray(outputs["thetaZ"])
except Exception:
    import h5py
    with h5py.File(RECON_PATH, "r") as f:
        out = f["outputs"]
        xHat   = np.asarray(out["xHat"])
        thetaX = np.asarray(out["thetaX"])
        thetaY = np.asarray(out["thetaY"])
        thetaZ = np.asarray(out["thetaZ"])

# magnitude may be complex (recon output) — take abs
if np.iscomplexobj(xHat):
    xHat = np.abs(xHat)

print(f"  xHat shape:   {xHat.shape}")
print(f"  thetaX shape: {thetaX.shape}")

# convert phase to velocity (m/s)
vX = thetaX / np.pi * VENC
vY = thetaY / np.pi * VENC
vZ = thetaZ / np.pi * VENC
speed = np.sqrt(vX**2 + vY**2 + vZ**2)

print("Running segmentation...")
mask3d = segment(thetaX, thetaY, thetaZ, xHat, venc_m_per_s=VENC, percentile=93.0, n_vessels=2)
print(f"  mask shape: {mask3d.shape}, voxels: {mask3d.sum()}")

# pick the axial slice with the most mask voxels
slice_axis = 0
slice_counts = mask3d.sum(axis=(1, 2))
best_slice = int(np.argmax(slice_counts))
print(f"  best slice along axis 0: {best_slice} ({slice_counts[best_slice]} mask voxels)")

# time frame at peak flow (max speed inside mask)
masked_speed_time = np.array([
    speed[..., t][mask3d].mean() if mask3d.any() else 0
    for t in range(speed.shape[-1])
])
best_t = int(np.argmax(masked_speed_time))
print(f"  peak-flow time frame: {best_t}")

mag_slice   = xHat[best_slice, :, :, best_t]
speed_slice = speed[best_slice, :, :, best_t]
mask_slice  = mask3d[best_slice, :, :]

# plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# panel 1 — magnitude
axes[0].imshow(mag_slice, cmap="gray", aspect="equal")
axes[0].set_title("Reconstruction (Magnitude)", fontsize=13)
axes[0].axis("off")

# panel 2 — velocity speed
im = axes[1].imshow(speed_slice, cmap="jet", aspect="equal", vmin=0, vmax=VENC)
axes[1].set_title("Velocity Field (Speed, m/s)", fontsize=13)
axes[1].axis("off")
cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label("m/s")

# panel 3 — mask overlay on magnitude
axes[2].imshow(mag_slice, cmap="gray", aspect="equal")
mask_rgba = np.zeros((*mask_slice.shape, 4))
mask_rgba[mask_slice, 0] = 1.0    # red
mask_rgba[mask_slice, 3] = 0.5    # alpha
axes[2].imshow(mask_rgba, aspect="equal")
axes[2].set_title("Vessel Mask (PC-MRA Skill)", fontsize=13)
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT_PATH}")
