"""
AS4DF DICOM loader implementation.

Parses Siemens .IMA files for a 4D-flow acquisition, builds (Z, Y, X, T)
magnitude + 3 phase volumes, and optionally voxelizes the STL ground-truth
mesh to a vessel mask aligned with the DICOM grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

AS4DF_MODELS          = ["m_c1", "m_c2", "m_r"]
AS4DF_TEMPORAL_FRAMES = [16, 25, 50]


# ─────────────────────────────────────────────────────────────────────
# Series discovery
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AS4DFSeries:
    magnitude_dir:   Path
    phase_x_dir:     Path
    phase_y_dir:     Path
    phase_z_dir:     Path
    venc_m_per_s:    float
    voxel_size_mm:   tuple[float, float, float]
    n_frames:        int
    use_corrected:   bool


def discover_as4df_series(dataset_root: Path, *,
                          model: str = "m_c1",
                          n_frames: int = 50,
                          use_corrected: bool = True) -> AS4DFSeries:
    """
    Locate the 4 DICOM series for one acquisition.

    AS4DF layout:
        <root>/dicoms/<model>/4DFLOWWIP_<n_frames>F/
            <prefix>_<series_id>/                  ← magnitude
            <prefix>_P_<series_id+1>/              ← phase x
            <prefix>_P_<series_id+2>/              ← phase y
            <prefix>_P_<series_id+3>/              ← phase z
    """
    root = Path(dataset_root)
    if model not in AS4DF_MODELS:
        raise ValueError(f"model must be one of {AS4DF_MODELS}, got {model!r}")
    if n_frames not in AS4DF_TEMPORAL_FRAMES:
        raise ValueError(f"n_frames must be one of {AS4DF_TEMPORAL_FRAMES}, got {n_frames}")

    base = root / "dicoms" / model / f"4DFLOWWIP_{n_frames}F"
    if not base.exists():
        raise FileNotFoundError(f"AS4DF series not found: {base}")

    subdirs = sorted(d for d in base.iterdir() if d.is_dir())
    # Pattern: one magnitude (no _P) + three _P phase series
    mag_dirs   = [d for d in subdirs if "_P_" not in d.name]
    phase_dirs = [d for d in subdirs if "_P_" in d.name]
    if len(mag_dirs) != 1 or len(phase_dirs) != 3:
        raise RuntimeError(
            f"Expected 1 magnitude + 3 phase series in {base}; "
            f"found {len(mag_dirs)} magnitude + {len(phase_dirs)} phase"
        )

    # AS4DF filenames encode VENC and voxel size: e.g. "V120_2_5_50FRAMES"
    venc_cm_per_s, voxel_mm = _parse_acquisition_params(mag_dirs[0].name)

    return AS4DFSeries(
        magnitude_dir = mag_dirs[0],
        phase_x_dir   = phase_dirs[0],
        phase_y_dir   = phase_dirs[1],
        phase_z_dir   = phase_dirs[2],
        venc_m_per_s  = venc_cm_per_s / 100.0,
        voxel_size_mm = (voxel_mm, voxel_mm, voxel_mm),
        n_frames      = n_frames,
        use_corrected = use_corrected,
    )


def _parse_acquisition_params(series_name: str) -> tuple[float, float]:
    """Extract VENC (cm/s) and voxel size (mm) from a series folder name like
    '4DFLOWWIP_AO_V120_2_5_50FRAMES_0004'."""
    parts = series_name.split("_")
    venc_cm = 120.0
    voxel   = 2.5
    for i, p in enumerate(parts):
        if p.startswith("V") and p[1:].isdigit():
            venc_cm = float(p[1:])
        if p.isdigit() and i + 1 < len(parts) and parts[i + 1].isdigit():
            try:
                voxel = float(f"{p}.{parts[i + 1]}")
            except ValueError:
                pass
    return venc_cm, voxel


# ─────────────────────────────────────────────────────────────────────
# DICOM stack reader
# ─────────────────────────────────────────────────────────────────────

def _read_dicom_stack(series_dir: Path) -> tuple[np.ndarray, dict]:
    """
    Read a Siemens .IMA series → (Z, Y, X, T) volume + metadata dict.

    Metadata also captures the spatial-registration info needed to align an
    external STL mesh to the same voxel grid:
      - origin_mm     : (z, y, x) world position of voxel [0,0,0]
      - voxel_size_mm : (dz, dy, dx)
    """
    import pydicom

    files = sorted(p for p in series_dir.iterdir()
                   if p.suffix.upper() == ".IMA" and ":Zone.Identifier" not in p.name)
    if not files:
        raise FileNotFoundError(f"No .IMA files in {series_dir}")

    entries = []
    for f in files:
        ds = pydicom.dcmread(str(f))
        entries.append({
            "instance":   int(getattr(ds, "InstanceNumber", 0)),
            "slice_loc":  float(getattr(ds, "SliceLocation", 0.0)),
            "n_card":     int(getattr(ds, "CardiacNumberOfImages", 0)) or None,
            "pixel":      ds.pixel_array,
            "ipp":        list(getattr(ds, "ImagePositionPatient", [0, 0, 0])),
            "iop":        list(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])),
            "ps":         list(getattr(ds, "PixelSpacing", [1.0, 1.0])),
            "st":         float(getattr(ds, "SliceThickness", 1.0)),
        })
    entries.sort(key=lambda e: e["instance"])

    n_files = len(entries)
    n_card  = entries[0]["n_card"] or _detect_n_phases(entries) or 1
    n_z     = n_files // n_card
    if n_z * n_card != n_files:
        raise RuntimeError(
            f"Cannot evenly split {n_files} DICOMs into Z × T (n_card={n_card})"
        )

    sample = entries[0]["pixel"]
    H, W = sample.shape
    vol = np.zeros((n_z, H, W, n_card), dtype=sample.dtype)
    time_major = entries[0]["slice_loc"] == entries[1]["slice_loc"]
    for k, e in enumerate(entries):
        if time_major:
            z = k // n_card
            t = k %  n_card
        else:
            t = k // n_z
            z = k %  n_z
        vol[z, :, :, t] = e["pixel"]

    e0 = entries[0]
    if time_major:
        last_z_entry = entries[(n_z - 1) * n_card]
    else:
        last_z_entry = entries[n_z - 1]
    dz = abs(float(last_z_entry["slice_loc"]) - float(e0["slice_loc"])) / max(1, n_z - 1) \
         if n_z > 1 else float(e0["st"])

    meta = {
        "n_files":         n_files,
        "n_z":             n_z,
        "n_frames":        n_card,
        "shape_YX":        [H, W],
        "pixel_dtype":     str(sample.dtype),
        "first_slice_loc": float(e0["slice_loc"]),
        "origin_mm":       e0["ipp"],
        "orientation":     e0["iop"],
        "voxel_size_mm":   [float(dz), float(e0["ps"][0]), float(e0["ps"][1])],
    }
    return vol, meta


def _detect_n_phases(entries: list[dict]) -> Optional[int]:
    if not entries:
        return None
    first_loc = entries[0]["slice_loc"]
    return sum(1 for e in entries if e["slice_loc"] == first_loc) or None


# ─────────────────────────────────────────────────────────────────────
# Phase → velocity conversion
# ─────────────────────────────────────────────────────────────────────

def _phase_pixel_to_velocity(phase_vol: np.ndarray,
                              venc_m_per_s: float) -> np.ndarray:
    """
    Siemens stores velocity-encoded phase as a 12-bit signed integer scaled
    to ±VENC. Returns velocity in m/s as float32.
    """
    arr = phase_vol.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_min >= 0 and arr_max > 2048:
        arr = arr - 2048.0
    return (arr / 2048.0) * venc_m_per_s


def _velocity_to_phase_rad(vel_m_per_s: np.ndarray,
                            venc_m_per_s: float) -> np.ndarray:
    """v / VENC * pi → phase in radians (verifier's input convention)."""
    return (vel_m_per_s / venc_m_per_s * np.pi).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def load_as4df(dataset_root: str | Path, *,
               model: str = "m_c1",
               n_frames: int = 50,
               use_corrected: bool = True) -> dict:
    """
    Load one AS4DF acquisition into the same dict shape as workspace.recon:
        {"xHat": (Z,Y,X,T) magnitude,
         "thetaX/Y/Z": (Z,Y,X,T) phase in radians,
         "venc_m_per_s", "voxel_size_mm", "dt_seconds", "origin_mm",
         "metadata"}
    """
    series = discover_as4df_series(Path(dataset_root),
                                   model=model, n_frames=n_frames,
                                   use_corrected=use_corrected)

    print(f"[AS4DF] loading model={model}, frames={n_frames}")
    print(f"[AS4DF]   VENC={series.venc_m_per_s} m/s  "
          f"voxel={series.voxel_size_mm} mm")

    mag_vol, mag_meta = _read_dicom_stack(series.magnitude_dir)
    vx_pix, _         = _read_dicom_stack(series.phase_x_dir)
    vy_pix, _         = _read_dicom_stack(series.phase_y_dir)
    vz_pix, _         = _read_dicom_stack(series.phase_z_dir)

    print(f"[AS4DF]   loaded shape (Z,Y,X,T)={mag_vol.shape}")

    vx = _phase_pixel_to_velocity(vx_pix, series.venc_m_per_s)
    vy = _phase_pixel_to_velocity(vy_pix, series.venc_m_per_s)
    vz = _phase_pixel_to_velocity(vz_pix, series.venc_m_per_s)

    # Assume RR-interval ≈ 1 s for the AS4DF phantom acquisition; dt = 1/n_frames.
    dt_seconds = 1.0 / float(series.n_frames)

    voxel_from_dicom = tuple(mag_meta.get("voxel_size_mm") or series.voxel_size_mm)
    origin_mm        = tuple(mag_meta.get("origin_mm")     or (0.0, 0.0, 0.0))

    return {
        "xHat":          mag_vol.astype(np.complex64),
        "thetaX":        _velocity_to_phase_rad(vx, series.venc_m_per_s),
        "thetaY":        _velocity_to_phase_rad(vy, series.venc_m_per_s),
        "thetaZ":        _velocity_to_phase_rad(vz, series.venc_m_per_s),
        "venc_m_per_s":  series.venc_m_per_s,
        "voxel_size_mm": voxel_from_dicom,
        "dt_seconds":    dt_seconds,
        "origin_mm":     origin_mm,
        "metadata": {
            "source":     "AS4DF",
            "model":      model,
            "n_frames":   series.n_frames,
            "magnitude":  mag_meta,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# STL → voxel mask
# ─────────────────────────────────────────────────────────────────────

def voxelize_stl_to_mask(stl_path: str | Path,
                          shape_ZYX: tuple[int, int, int],
                          voxel_size_mm: tuple[float, float, float],
                          *,
                          origin_mm: tuple[float, float, float] | None = None,
                          ) -> np.ndarray:
    """
    Voxelize an STL into a (Z, Y, X) binary mask aligned to a DICOM grid.

    When `origin_mm` (world position of voxel [0,0,0] from ImagePositionPatient)
    is provided, the mesh is rasterized directly into the DICOM grid:
        voxel[z,y,x] is occupied  iff  origin + (x*dx, y*dy, z*dz) is inside mesh

    When `origin_mm` is None, falls back to center-aligning a separately-voxelized
    mesh (rough — only useful for visualisation, not physics).
    """
    import trimesh

    mesh = trimesh.load_mesh(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"STL did not load as a single mesh: {stl_path}")

    sZ, sY, sX     = shape_ZYX
    dz, dy, dx     = voxel_size_mm

    if origin_mm is not None:
        # ── Direct DICOM-grid alignment ────────────────────────────
        ox, oy, oz = origin_mm
        zs = oz + np.arange(sZ) * dz
        ys = oy + np.arange(sY) * dy
        xs = ox + np.arange(sX) * dx
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        inside = mesh.contains(pts).reshape(shape_ZYX)
        print(f"[STL] DICOM-grid voxelized {Path(stl_path).name} → "
              f"{int(inside.sum())} mask voxels  "
              f"(origin={origin_mm}, voxel={voxel_size_mm})")
        return inside.astype(bool)

    # ── Fallback: center-align (legacy behavior) ──────────────────
    pitch = min(voxel_size_mm)
    vox   = mesh.voxelized(pitch=pitch).fill()
    binary = np.asarray(vox.matrix, dtype=bool)
    binary = np.transpose(binary, (2, 1, 0))
    out = np.zeros(shape_ZYX, dtype=bool)
    bZ, bY, bX = binary.shape
    z0 = max(0, (sZ - bZ) // 2);   bz0 = max(0, (bZ - sZ) // 2)
    y0 = max(0, (sY - bY) // 2);   by0 = max(0, (bY - sY) // 2)
    x0 = max(0, (sX - bX) // 2);   bx0 = max(0, (bX - sX) // 2)
    zN = min(sZ - z0, bZ - bz0)
    yN = min(sY - y0, bY - by0)
    xN = min(sX - x0, bX - bx0)
    out[z0:z0 + zN, y0:y0 + yN, x0:x0 + xN] = \
        binary[bz0:bz0 + zN, by0:by0 + yN, bx0:bx0 + xN]
    print(f"[STL] center-aligned (fallback) {Path(stl_path).name} → "
          f"{int(out.sum())} mask voxels in shape {shape_ZYX}")
    return out
