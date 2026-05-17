"""Generate preview PNGs from a reconstruction result.

Designed to be optional — matplotlib is imported lazily so headless users
(tests, the Stage 3 agent loop) don't pay for it.

Layout produced in ``out_dir``:
    1_anatomy.png                 magnitude |xHat| at mid-Z, peak-flow T
    2_speed.png                   anatomy + speed-magnitude overlay
    3_velocity_components.png     vx, vy, vz side-by-side at same slice/T
    4_max_speed_projection.png    max speed across the cardiac cycle
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def save_preview(
    result: dict,
    out_dir: str | os.PathLike,
    *,
    venc_m_per_s: float | None = None,
    z_slice: int | None = None,
    dpi: int = 120,
) -> list[Path]:
    """Save anatomy + velocity PNGs from a ``reconstruct()`` result dict.

    Parameters
    ----------
    result : dict
        Output of ``reconstruct()`` — must contain ``xHat``, ``thetaX/Y/Z``.
        ``meta["venc_m_per_s"]`` is used if ``venc_m_per_s`` is not passed.
    out_dir : path-like
        Destination directory (created if missing).
    venc_m_per_s : float, optional
        Velocity encoding (m/s). Overrides the value in result["meta"].
    z_slice : int, optional
        Z slice index to render. Defaults to the middle slice.
    dpi : int
        Output PNG resolution.

    Returns
    -------
    list[Path]
        Paths to the PNGs written, in display order.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless; safe for WSL / agent runs
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if venc_m_per_s is None:
        venc_m_per_s = float(result.get("meta", {}).get("venc_m_per_s", 1.5))

    # xHat comes back complex (SoS of complex velocity encodings); display
    # uses its magnitude, matching how the upstream MATLAB GIF code shows it.
    xHat = np.abs(result["xHat"])
    tx, ty, tz = result["thetaX"], result["thetaY"], result["thetaZ"]
    Z, Y, X, T = xHat.shape

    # Phase (radians) → velocity (m/s)
    s = venc_m_per_s / np.pi
    vx, vy, vz = tx * s, ty * s, tz * s
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    z = Z // 2 if z_slice is None else int(z_slice)
    peak_t = int(np.argmax(speed.sum(axis=(0, 1, 2))))

    saved: list[Path] = []

    # 1) Anatomy
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(xHat[z, :, :, peak_t], cmap="gray")
    ax.set_title(f"Anatomy (|xHat|) — Z={z}, T={peak_t}")
    ax.axis("off")
    p = out_dir / "1_anatomy.png"
    fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig); saved.append(p)

    # 2) Speed overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(xHat[z, :, :, peak_t], cmap="gray")
    im = ax.imshow(speed[z, :, :, peak_t], cmap="hot",
                   alpha=0.55, vmin=0.1, vmax=venc_m_per_s)
    ax.set_title(f"Speed |v| (m/s) — Z={z}, T={peak_t}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    p = out_dir / "2_speed.png"
    fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig); saved.append(p)

    # 3) Velocity components
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vol, name in zip(axes, [vx, vy, vz], ["vx", "vy", "vz"]):
        im = ax.imshow(vol[z, :, :, peak_t], cmap="RdBu_r",
                       vmin=-venc_m_per_s, vmax=venc_m_per_s)
        ax.set_title(f"{name} (m/s) — Z={z}, T={peak_t}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    p = out_dir / "3_velocity_components.png"
    fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig); saved.append(p)

    # 4) Max-speed projection over time
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(speed.max(axis=3)[z, :, :], cmap="hot",
                   vmin=0, vmax=venc_m_per_s)
    ax.set_title(f"Max-speed projection over time — Z={z}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    p = out_dir / "4_max_speed_projection.png"
    fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig); saved.append(p)

    return saved
