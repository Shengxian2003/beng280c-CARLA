r"""Python ↔ MATLAB bridge for the Stage 2a reconstruction skill.

Handles staging input data to a Windows drive (Windows MATLAB can't reliably
open HDF5 .mat over the \\wsl.localhost\ UNC bridge), generating a JSON config,
invoking Windows MATLAB in -batch mode, streaming its output, and loading the
result back into Python.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ._paths import (
    is_wsl_path,
    is_mnt_path,
    to_windows_for_matlab,
    to_wsl,
)


# ----- Defaults that match the existing project layout -----------------------

# Resolved from this file's location, so the wrapper still works if the repo moves.
_SKILL_DIR = Path(__file__).resolve().parent
_DRIVER_M  = _SKILL_DIR / "medict_recon_driver.m"
_RECON_DIR = _SKILL_DIR / "motion-robust-CMR-main" / \
             "3D cine-4D flow MRI Reconstruction (Study III IV V)"

DEFAULT_MATLAB_EXE = "/mnt/g/Application_Industry/Matlab/bin/matlab.exe"
DEFAULT_STAGE_DIR  = "/mnt/g/medict_tmp"


# ----- Config dataclass ------------------------------------------------------

@dataclass
class ReconConfig:
    input_mat:      str   # Windows-style path the MATLAB driver will open
    output_mat:     str   # Windows-style path
    recon_dir:      str   # Windows-style path to the motion-robust-CMR Study III/IV/V dir
    method:         str   # "cs" | "core"
    is_flow:        int   # 1 | 0
    is_rest:        int   # 1 | 0
    n_iterations:   int
    n_coils:        int
    use_gpu:        int
    data_field:     str   # top-level struct field in input .mat (e.g. "D")
    venc_m_per_s:   float


# ----- MATLAB invocation -----------------------------------------------------

def _build_matlab_cmd(matlab_exe: str, driver_dir_win: str, config_path_win: str) -> list[str]:
    """matlab.exe -batch "addpath('<driver_dir>'); medict_recon_driver('<config>')" """
    # The single-quoted MATLAB string uses '' to escape an embedded single quote.
    addpath_esc = driver_dir_win.replace("'", "''")
    cfg_esc     = config_path_win.replace("'", "''")
    batch_body = f"addpath('{addpath_esc}'); medict_recon_driver('{cfg_esc}')"
    return [matlab_exe, "-batch", batch_body]


def _stream_subprocess(cmd: list[str], verbose: bool) -> int:
    """Run a subprocess, optionally streaming its stdout/stderr live."""
    if verbose:
        print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if verbose:
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    return proc.returncode


# ----- Loading the .mat output ----------------------------------------------

def _load_outputs_mat(mat_path_wsl: str) -> dict:
    """Read the MATLAB-saved 'outputs' struct into a plain Python dict.

    Handles both v7 (scipy.io.loadmat) and v7.3 (HDF5 via h5py) formats.
    """
    import scipy.io as sio

    try:
        raw = sio.loadmat(mat_path_wsl, squeeze_me=True)
        o = raw["outputs"]
        get = lambda k: np.array(o[k].item())
        result = {
            "xHat":   get("xHat"),
            "thetaX": get("thetaX"),
            "thetaY": get("thetaY"),
            "thetaZ": get("thetaZ"),
        }
        # meta is a nested struct
        try:
            meta_raw = o["meta"].item()
            result["meta"] = {
                name: (np.array(meta_raw[name].item()).tolist()
                       if hasattr(meta_raw[name].item(), "__len__")
                       and not isinstance(meta_raw[name].item(), str)
                       else meta_raw[name].item())
                for name in meta_raw.dtype.names
            }
        except Exception:
            result["meta"] = {}
        return result

    except NotImplementedError:
        # v7.3 HDF5 fallback (MATLAB writes column-major, so transpose)
        import h5py
        result = {}
        with h5py.File(mat_path_wsl, "r") as f:
            o = f["outputs"]
            for k in ("xHat", "thetaX", "thetaY", "thetaZ"):
                result[k] = np.array(o[k]).T
            result["meta"] = {}  # meta struct decoding via h5py is messy; skip
        return result


# ----- Public entry point ----------------------------------------------------

def reconstruct(
    kspace_path: str | os.PathLike,
    *,
    venc_m_per_s: float = 1.5,
    method: str = "cs",
    n_iterations: int = 50,
    is_flow: bool = True,
    is_rest: bool = True,
    n_coils: int = 12,
    use_gpu: bool = True,
    output_path: Optional[str | os.PathLike] = None,
    data_field: str = "D",
    stage_dir: str = DEFAULT_STAGE_DIR,
    matlab_exe: str = DEFAULT_MATLAB_EXE,
    skip_stage: bool = False,
    keep_stage: bool = False,
    save_preview: bool = True,
    preview_dir: Optional[str | os.PathLike] = None,
    verbose: bool = True,
) -> dict:
    """Run a CS or CORe reconstruction by invoking Windows MATLAB.

    Parameters
    ----------
    kspace_path : path-like
        Input k-space .mat. Can be a WSL path (/home/...) or a Windows-mounted
        path (/mnt/g/...). HDF5 .mat files on the WSL filesystem are staged to
        ``stage_dir`` first because Windows MATLAB cannot reliably open them
        via the ``\\\\wsl.localhost\\`` UNC bridge.
    venc_m_per_s : float
        Velocity encoding (m/s). Recorded in output metadata for downstream
        tools; not used by the reconstruction itself.
    method : {"cs", "core"}
        Reconstruction algorithm. "core" adds outlier rejection.
    n_iterations : int
        ADMM outer iterations. 5 ≈ smoke test, 50 ≈ full quality.
    is_flow, is_rest : bool
        Selects regularization preset. 4D flow vs 3D cine, rest vs exercise.
    n_coils : int
        Coil compression target.
    use_gpu : bool
        Use CUDA via MATLAB Parallel Computing Toolbox.
    output_path : path-like, optional
        Destination .mat. Defaults to ``{stage_dir}/recon_<method>_<nit>iter.mat``.
    data_field : str
        Top-level struct field in the input .mat holding kb/kx/ky/kz.
    stage_dir : str
        Directory on a native Windows drive for staging k-space and output.
    matlab_exe : str
        Path to Windows MATLAB executable (accessible from WSL).
    skip_stage : bool
        If True, do not copy ``kspace_path`` even if it's on the WSL filesystem.
        Useful when the file is already on a Windows drive.
    keep_stage : bool
        If True, leave the staged input copy on the Windows drive after the
        run. Useful for debugging or re-running.
    save_preview : bool
        If True, write 4 preview PNGs (anatomy, speed overlay, velocity
        components, max-speed projection) alongside the output .mat.
    preview_dir : path-like, optional
        Where to save preview PNGs. Defaults to
        ``<output_path>.parent/preview_<output_stem>/``.
    verbose : bool
        Stream MATLAB stdout live.

    Returns
    -------
    dict
        {
          "xHat":   ndarray (Z, Y, X, T) — magnitude (sum-of-squares for 4D flow),
          "thetaX": ndarray (Z, Y, X, T) — background-corrected phase, vx,
          "thetaY": ndarray (Z, Y, X, T) — background-corrected phase, vy,
          "thetaZ": ndarray (Z, Y, X, T) — background-corrected phase, vz,
          "meta":   { elapsed_minutes, method, n_iterations, ... },
          "output_path": str — WSL-style path to the saved .mat,
          "elapsed_wall_s": float — measured by Python, includes MATLAB startup,
        }
    """
    if method not in ("cs", "core"):
        raise ValueError(f"method must be 'cs' or 'core', got {method!r}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

    kspace_path = Path(kspace_path)
    if not kspace_path.exists():
        raise FileNotFoundError(f"k-space input not found: {kspace_path}")

    stage_dir_p = Path(stage_dir)
    stage_dir_p.mkdir(parents=True, exist_ok=True)

    # ---- Stage input to Windows drive (if needed) --------------------------
    if skip_stage or is_mnt_path(kspace_path):
        staged_input = kspace_path
        copied = False
    else:
        staged_input = stage_dir_p / kspace_path.name
        if verbose:
            print(f"[stage] copying {kspace_path} → {staged_input}", flush=True)
        shutil.copy2(kspace_path, staged_input)
        copied = True

    # ---- Decide output path ------------------------------------------------
    if output_path is None:
        output_path = stage_dir_p / f"recon_{method}_{n_iterations}iter.mat"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Build config and write to stage dir -------------------------------
    cfg = ReconConfig(
        input_mat    = to_windows_for_matlab(staged_input),
        output_mat   = to_windows_for_matlab(output_path),
        recon_dir    = to_windows_for_matlab(_RECON_DIR),
        method       = method,
        is_flow      = int(bool(is_flow)),
        is_rest      = int(bool(is_rest)),
        n_iterations = int(n_iterations),
        n_coils      = int(n_coils),
        use_gpu      = int(bool(use_gpu)),
        data_field   = data_field,
        venc_m_per_s = float(venc_m_per_s),
    )
    config_path = stage_dir_p / f"recon_config_{int(time.time())}.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    if verbose:
        print(f"[config] {config_path}", flush=True)

    # ---- Invoke MATLAB -----------------------------------------------------
    cmd = _build_matlab_cmd(
        matlab_exe      = matlab_exe,
        driver_dir_win  = to_windows_for_matlab(_SKILL_DIR),
        config_path_win = to_windows_for_matlab(config_path),
    )

    t0 = time.time()
    rc = _stream_subprocess(cmd, verbose=verbose)
    elapsed_wall = time.time() - t0

    if rc != 0:
        raise RuntimeError(
            f"MATLAB reconstruction failed (exit {rc}). "
            f"Config: {config_path}, command: {' '.join(cmd)}"
        )
    if not output_path.exists():
        raise RuntimeError(
            f"MATLAB returned 0 but output file not found at {output_path}. "
            f"Check MATLAB stdout above for errors."
        )

    # ---- Cleanup staged input ---------------------------------------------
    if copied and not keep_stage:
        try:
            staged_input.unlink()
        except OSError:
            pass

    # ---- Load and return ---------------------------------------------------
    out = _load_outputs_mat(str(output_path))
    out["output_path"]    = str(output_path)
    out["elapsed_wall_s"] = elapsed_wall

    # ---- Optional preview PNGs --------------------------------------------
    if save_preview:
        from ._preview import save_preview as _save_preview
        if preview_dir is None:
            preview_dir = output_path.parent / f"preview_{output_path.stem}"
        # carry venc into meta if MATLAB metadata loading skipped it (v7.3 path)
        out.setdefault("meta", {}).setdefault("venc_m_per_s", float(venc_m_per_s))
        previews = _save_preview(out, preview_dir, venc_m_per_s=venc_m_per_s)
        out["preview_paths"] = [str(p) for p in previews]
        if verbose:
            print(f"[preview] saved {len(previews)} PNGs → {preview_dir}", flush=True)

    return out
