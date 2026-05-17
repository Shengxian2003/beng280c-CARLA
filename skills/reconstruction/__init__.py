"""Reconstruction Skill: Python wrapper around the motion-robust-CMR MATLAB code.

Public API:
    reconstruct(kspace_path, ...) → dict   # invokes Windows MATLAB, returns 4D fields
    save_preview(result, out_dir, ...)     # render preview PNGs from a result dict

Helpers (mostly for tests and advanced use):
    to_windows_for_matlab(path)
    to_wsl(path)
"""
from ._runner import reconstruct, ReconConfig, DEFAULT_MATLAB_EXE, DEFAULT_STAGE_DIR
from ._paths import to_windows_for_matlab, to_wsl, wsl_to_unc, mnt_to_windows
from ._preview import save_preview

__all__ = [
    "reconstruct",
    "save_preview",
    "ReconConfig",
    "DEFAULT_MATLAB_EXE",
    "DEFAULT_STAGE_DIR",
    "to_windows_for_matlab",
    "to_wsl",
    "wsl_to_unc",
    "mnt_to_windows",
]
