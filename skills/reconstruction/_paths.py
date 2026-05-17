r"""Path translation between WSL and Windows for the MATLAB bridge.

Windows MATLAB invoked from WSL can read MATLAB source code from a UNC path
(\\wsl.localhost\Ubuntu\...), but cannot reliably open HDF5-format .mat files
across that bridge. So input k-space and output reconstructions are staged on
a native Windows drive (default G:\).
"""
from __future__ import annotations

import os
from pathlib import PurePosixPath, PureWindowsPath


def is_wsl_path(p: str | os.PathLike) -> bool:
    s = str(p)
    return s.startswith("/") and not s.startswith("/mnt/")


def is_mnt_path(p: str | os.PathLike) -> bool:
    """/mnt/<drive>/... — WSL view of a Windows drive."""
    s = str(p)
    return len(s) >= 7 and s.startswith("/mnt/") and s[6] == "/"


def wsl_to_unc(wsl_path: str | os.PathLike, distro: str = "Ubuntu") -> str:
    """/home/x/y  →  \\\\wsl.localhost\\Ubuntu\\home\\x\\y"""
    p = PurePosixPath(wsl_path)
    parts = p.parts
    if not parts or parts[0] != "/":
        raise ValueError(f"Expected absolute POSIX path, got: {wsl_path!r}")
    rest = "\\".join(parts[1:])
    return f"\\\\wsl.localhost\\{distro}\\{rest}"


def mnt_to_windows(mnt_path: str | os.PathLike) -> str:
    """/mnt/g/foo/bar  →  G:\\foo\\bar"""
    s = str(mnt_path)
    if not is_mnt_path(s):
        raise ValueError(f"Not a /mnt/<drive>/... path: {mnt_path!r}")
    drive = s[5].upper()
    rest = s[7:].replace("/", "\\")
    return f"{drive}:\\{rest}" if rest else f"{drive}:\\"


def windows_to_mnt(win_path: str | os.PathLike) -> str:
    """G:\\foo\\bar  →  /mnt/g/foo/bar  (so WSL Python can read it back)"""
    p = PureWindowsPath(win_path)
    drive = p.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Not an absolute Windows drive path: {win_path!r}")
    rest = "/".join(p.parts[1:])
    return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}/"


def to_windows_for_matlab(p: str | os.PathLike, distro: str = "Ubuntu") -> str:
    """Return whatever string form Windows MATLAB can open.

    /mnt/x/...   → X:\\...           (native drive — preferred for data files)
    /home/...    → \\\\wsl.localhost\\<distro>\\...  (UNC — fine for .m source)
    X:\\...      → X:\\...           (already Windows)
    """
    s = str(p)
    if is_mnt_path(s):
        return mnt_to_windows(s)
    if is_wsl_path(s):
        return wsl_to_unc(s, distro=distro)
    # Already a Windows-style path
    return s


def to_wsl(p: str | os.PathLike) -> str:
    """Inverse of to_windows_for_matlab — returns a path usable from WSL Python."""
    s = str(p)
    if len(s) >= 2 and s[1] == ":":
        return windows_to_mnt(s)
    if s.startswith("\\\\wsl.localhost\\"):
        # \\wsl.localhost\Ubuntu\home\x  →  /home/x
        parts = s.split("\\")
        # parts = ['', '', 'wsl.localhost', '<distro>', 'home', 'x', ...]
        return "/" + "/".join(parts[4:])
    return s
