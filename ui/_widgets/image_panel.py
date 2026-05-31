"""
Image output panel — renders anatomy magnitude, PC-MRA speed map,
segmented mask overlay, and per-time-frame velocity preview.

Recon source is auto-detected from the audit log:
  - phantom run → regenerate phantom from skills.eval_inject._phantom_v2
  - real scan   → load the .mat file referenced in the load_reconstruction tool call
  - as4df       → re-read DICOM via the AS4DF loader

Mask discovery: when the recon dict does NOT carry a mask (real scan, or
AS4DF with STL toggle off), the panel falls back to the V2 session-store
directory at logs/runs/<session_id>/segmentation/masks/*.npy — the
segmentation specialist's actual output for this run. If multiple masks
were produced, the most recently written one wins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────
# Source discovery (from audit log)
# ─────────────────────────────────────────────────────────────────────

def _extract_recon_source(audit_path: Path) -> tuple[str, object]:
    """
    Walk the audit log and return ('phantom'|'real'|'as4df', detail).

    detail is:
      - None              for phantom
      - str (.mat path)   for real
      - dict (args)       for as4df

    Uses the LAST matching tool call so if the LLM accidentally calls multiple
    loaders the most recent one wins.
    """
    if not audit_path.exists():
        return ("unknown", None)
    src_type, detail = "unknown", None
    with open(audit_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") != "tool_call":
                continue
            d    = e["data"]
            name = d.get("name")
            if name == "load_phantom":
                src_type, detail = "phantom", None
            elif name == "load_reconstruction":
                src_type, detail = "real", d.get("args", {}).get("mat_path")
            elif name == "load_as4df":
                # Args may be empty (host env-anchor mode); prefer the tool
                # result which always records the resolved dataset_root.
                args   = d.get("args",   {}) or {}
                result = d.get("result", {}) or {}
                src_type = "as4df"
                detail = {
                    "dataset_root":  result.get("dataset_root") or args.get("dataset_root"),
                    "model":         result.get("model")        or args.get("model", "m_c1"),
                    "n_frames":      result.get("n_frames")     or args.get("n_frames", 50),
                    "load_stl_mask": result.get("load_stl_mask",
                                                  args.get("load_stl_mask", True)),
                }
    return (src_type, detail)


# ─────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_phantom() -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    from skills.eval_inject._phantom_v2 import curved_tapered_phantom
    p = curved_tapered_phantom(pulsatile=True)
    return _precompute_views(p, has_anatomy=False)


@st.cache_data(show_spinner="Loading AS4DF DICOMs (~30 s)...")
def _load_as4df_cached(dataset_root: str, model: str, n_frames: int,
                       load_stl: bool) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    from skills.dicom_loader import load_as4df, voxelize_stl_to_mask
    recon = load_as4df(dataset_root, model=model, n_frames=n_frames)
    if load_stl:
        stl_root = Path(dataset_root) / "stl"
        candidates = sorted(stl_root.glob("*.stl"),
                            key=lambda p: p.stat().st_size, reverse=True)
        if candidates:
            recon["mask"] = voxelize_stl_to_mask(
                candidates[0],
                shape_ZYX=recon["thetaX"].shape[:3],
                voxel_size_mm=recon["voxel_size_mm"],
                origin_mm=recon.get("origin_mm"),
            )
    # Real AS4DF magnitude IS available — treat as anatomy
    return _precompute_views(recon, has_anatomy=True)


@st.cache_data(show_spinner=False)
def _load_real_recon(mat_path: str) -> dict:
    import scipy.io as sio
    raw = sio.loadmat(mat_path, squeeze_me=True)
    if "outputs" not in raw:
        raise ValueError(f"{mat_path} has no 'outputs' struct")
    o = raw["outputs"]
    recon = {
        "xHat":   np.array(o["xHat"].item()),
        "thetaX": np.array(o["thetaX"].item()),
        "thetaY": np.array(o["thetaY"].item()),
        "thetaZ": np.array(o["thetaZ"].item()),
    }
    return _precompute_views(recon, has_anatomy=True)


def _precompute_views(recon: dict, *, has_anatomy: bool) -> dict:
    """Compute all derived images once at load time so re-renders are instant."""
    tX, tY, tZ = recon["thetaX"], recon["thetaY"], recon["thetaZ"]
    pcmra      = _pcmra_speed(tX, tY, tZ)
    out = dict(recon)
    out["pcmra"]    = pcmra
    out["mip_z"]    = pcmra.max(axis=0)
    out["mip_y"]    = pcmra.max(axis=1)
    out["mip_x"]    = pcmra.max(axis=2)
    if has_anatomy and "xHat" in recon:
        out["anatomy"] = _anatomy_image(recon["xHat"])
    else:
        out["anatomy"] = pcmra
    return out


# ─────────────────────────────────────────────────────────────────────
# Image computations
# ─────────────────────────────────────────────────────────────────────

def _anatomy_image(xHat: np.ndarray) -> np.ndarray:
    """Time-averaged magnitude — anatomical reference."""
    return np.abs(xHat).mean(axis=-1).astype(np.float32)


def _pcmra_speed(thetaX, thetaY, thetaZ, venc=1.5) -> np.ndarray:
    """Time-max speed magnitude — bright vessels."""
    speed = np.sqrt(thetaX**2 + thetaY**2 + thetaZ**2) * venc / np.pi
    return speed.max(axis=-1).astype(np.float32)


def _normalize(img: np.ndarray, lo_pct=1, hi_pct=99) -> np.ndarray:
    """Map to [0,1] using percentile clipping."""
    lo, hi = np.percentile(img, lo_pct), np.percentile(img, hi_pct)
    if hi - lo < 1e-9:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _overlay_mask(base: np.ndarray, mask: np.ndarray, color=(0, 1, 0)) -> np.ndarray:
    """RGB overlay: base in grayscale, mask tinted with `color`."""
    base_n = _normalize(base)
    rgb = np.stack([base_n, base_n, base_n], axis=-1)
    for c in range(3):
        rgb[..., c] = np.where(mask, 0.5 * rgb[..., c] + 0.5 * color[c], rgb[..., c])
    return rgb


def _session_id_from_audit(audit_path: Path) -> Optional[str]:
    """Pull the session_id off the first session_start entry."""
    if not audit_path.exists():
        return None
    with open(audit_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "session_start":
                return e.get("session_id") or e.get("data", {}).get("session_id")
    return None


def _latest_mask_from_session_store(audit_path: Path) -> tuple[Optional[np.ndarray], Optional[str]]:
    """If the V2 session store has a segmentation mask for this run, return
    (mask_array, mask_name). The most recently written .npy wins.
    Returns (None, None) when no store / no masks exist."""
    sid = _session_id_from_audit(audit_path)
    if not sid:
        return None, None
    masks_dir = PROJECT_ROOT / "logs" / "runs" / sid / "segmentation" / "masks"
    if not masks_dir.exists():
        return None, None
    npy_files = sorted(masks_dir.glob("*.npy"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not npy_files:
        return None, None
    latest = npy_files[0]
    return np.load(latest).astype(bool), latest.stem


# ─────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────

def render_images(audit_path: Path) -> None:
    st.subheader("Image output")

    src_type, detail = _extract_recon_source(audit_path)
    if src_type == "unknown":
        st.info("No reconstruction was loaded in this run — nothing to display.")
        return

    try:
        if src_type == "phantom":
            recon = _load_phantom()
            st.caption("Source: **synthetic curved-tapered phantom**")
        elif src_type == "as4df":
            args      = detail or {}
            root      = args.get("dataset_root")
            model     = args.get("model", "m_c1")
            n_frames  = int(args.get("n_frames", 50))
            load_stl  = bool(args.get("load_stl_mask", True))
            if not root or not Path(root).exists():
                st.warning(f"AS4DF dataset not found: `{root}`")
                return
            recon = _load_as4df_cached(root, model, n_frames, load_stl)
            st.caption(f"Source: **AS4DF** `{root}` · model={model} · {n_frames} frames"
                       + (" · STL ground-truth mask loaded" if load_stl else ""))
        else:  # real
            if not detail or not Path(detail).exists():
                st.warning(f"Reconstruction file not found: `{detail}`")
                return
            recon = _load_real_recon(detail)
            st.caption(f"Source: **`{detail}`**")
    except Exception as e:
        st.error(f"Failed to load reconstruction: {e}")
        return

    mask        = recon.get("mask")           # only set by phantom / AS4DF+STL loaders
    mask_source = "loader" if mask is not None else None
    # Fall back to the V2 session store: the segmentation specialist's
    # output mask is persisted to logs/runs/<session_id>/segmentation/masks/
    # regardless of input mode, so real-scan and AS4DF-no-STL runs can
    # still display the segmented mask.
    if mask is None:
        fb_mask, fb_name = _latest_mask_from_session_store(audit_path)
        if fb_mask is not None:
            mask        = fb_mask
            mask_source = f"session store ({fb_name}.npy)"
    anatomy_mip = recon["anatomy"]
    pcmra_mip   = recon["pcmra"]
    anatomy_label = (
        "Anatomy (\\|xHat\\| time-averaged)" if "xHat" in recon
        else "Anatomy not in phantom — showing PC-MRA instead"
    )

    # ── 3-axis MIP rendered FIRST so it's always visible on tab open ────
    # (Renders even before user touches the slider — independent of slice_idx.)
    st.markdown("**3-axis MIP**")
    mip_cols = st.columns(3)
    with mip_cols[0]:
        st.caption("MIP along Z (Y × X)")
        st.image(_normalize(recon["mip_z"]), use_container_width=True, clamp=True)
    with mip_cols[1]:
        st.caption("MIP along Y (Z × X)")
        st.image(_normalize(recon["mip_y"]), use_container_width=True, clamp=True)
    with mip_cols[2]:
        st.caption("MIP along X (Z × Y)")
        st.image(_normalize(recon["mip_x"]), use_container_width=True, clamp=True)

    st.divider()

    # ── Per-slice views (slider-driven) ────────────────────────────────
    Z = pcmra_mip.shape[0]
    slice_idx = st.slider("Z-slice", 0, Z - 1, Z // 2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**{anatomy_label}**")
        st.image(_normalize(anatomy_mip[slice_idx]), use_container_width=True, clamp=True)
    with col2:
        st.markdown("**PC-MRA speed (max over time)**")
        st.image(_normalize(pcmra_mip[slice_idx]), use_container_width=True, clamp=True)
    with col3:
        if mask is not None:
            st.markdown("**Mask overlay (green = vessel)**")
            st.caption(f"Source: {mask_source}")
            overlay = _overlay_mask(pcmra_mip[slice_idx], mask[slice_idx])
            st.image(overlay, use_container_width=True, clamp=True)
        else:
            st.markdown("**Mask** — no mask produced this session")
            st.caption("Neither the loader nor the segmentation specialist "
                       "wrote a mask to the session store yet.")
