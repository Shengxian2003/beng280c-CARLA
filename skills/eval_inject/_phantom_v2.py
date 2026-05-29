"""
带粗细变化的不可压缩 4D flow phantom。

几何：直管沿 Z 方向，半径从 R_top 到 R_bot 平滑渐变。

物理：解析构造不可压缩流场
  vz(r, z) = Q / (π R(z)²)           [plug flow 假设]
  vr(r, z) = Q · r · dR/dz / (π R³)  [由质量守恒 ∇·v = 0 推出]

  解析上 ∇·v = 0 严格成立，离散化后只有数值误差
  → divergence ✓

  flux = v·A = Q = const 沿 z 不变 → net_flux ✓
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def curved_tapered_phantom(
    Z: int = 60,
    Y: int = 60,
    X: int = 60,
    T: int = 8,
    venc_m_per_s: float = 1.5,
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    dt_seconds: float = 0.05,
    base_velocity: float = 0.5,
    R_top: float = 7.0,
    R_bot: float = 5.0,
    pulsatile: bool = True,
) -> dict:
    """
    直管 + 平滑渐细，物理上正确的不可压缩流场。
    """
    cy, cx = Y // 2, X // 2
    dy_mm = voxel_size_mm[1]
    dx_mm = voxel_size_mm[2]

    # ── 1. 半径沿 z 平滑变化（用 tanh 让两端更平缓）─────────────
    z_arr = np.arange(Z, dtype=np.float64)
    z_n   = (z_arr - Z / 2) / (Z / 5)
    # 平滑过渡：tanh 从 R_top 到 R_bot
    R     = R_top + (R_bot - R_top) * (np.tanh(z_n) + 1) / 2   # (Z,)
    # dR/dz（数值梯度）
    dR_dz = np.gradient(R)                                       # (Z,)

    # ── 2. 构造 mask ──────────────────────────────────────────
    mask = np.zeros((Z, Y, X), dtype=bool)
    yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")
    for z in range(Z):
        mask[z] = ((yy - cy) ** 2 + (xx - cx) ** 2) <= R[z] ** 2

    # ── 3. 解析速度场（满足 ∇·v = 0）───────────────────────────
    # 流量守恒：Q = base_velocity * π * R_top²（以 voxel² 为单位）
    Q = base_velocity * np.pi * R_top ** 2

    vx_static = np.zeros((Z, Y, X), dtype=np.float64)
    vy_static = np.zeros((Z, Y, X), dtype=np.float64)
    vz_static = np.zeros((Z, Y, X), dtype=np.float64)

    for z in range(Z):
        R_z   = R[z]
        if R_z <= 0:
            continue
        vz_z  = Q / (np.pi * R_z ** 2)                          # 这层的轴向速度
        # vr / r = Q · dR/dz / (π R³)
        vr_over_r = Q * dR_dz[z] / (np.pi * R_z ** 3)

        # 只在 mask 内填
        m = mask[z]
        dy = (yy - cy)[m]
        dx = (xx - cx)[m]

        vz_static[z][m] = vz_z
        # vr 分解到 x 和 y（径向方向 = (dy, dx) 归一化后乘 vr）
        vy_static[z][m] = vr_over_r * dy
        vx_static[z][m] = vr_over_r * dx

    # 轻度平滑（消除离散化引起的高频噪声）
    sigma = 0.8
    vx_smooth = gaussian_filter(vx_static, sigma=sigma) * mask
    vy_smooth = gaussian_filter(vy_static, sigma=sigma) * mask
    vz_smooth = gaussian_filter(vz_static, sigma=sigma) * mask

    print(f"\n[Tapered Phantom (incompressible)]")
    print(f"  Radius:      {R_top:.1f} (top) → {R_bot:.1f} (bot)")
    print(f"  Velocity vz: {vz_smooth[mask].min():.3f} – {vz_smooth[mask].max():.3f} m/s")
    print(f"  Velocity vr: |max| = {max(abs(vx_smooth).max(), abs(vy_smooth).max()):.3f} m/s")

    # ── 4. 心动周期变化 ──────────────────────────────────────
    vx = np.zeros((Z, Y, X, T), dtype=np.float64)
    vy = np.zeros_like(vx)
    vz = np.zeros_like(vx)

    if pulsatile:
        phase   = np.linspace(0, 2 * np.pi, T, endpoint=False)
        cardiac = 1.0 + 0.3 * np.sin(phase)         # 振幅 ±30% 而不是 40%
    else:
        cardiac = np.ones(T)

    for t in range(T):
        vx[..., t] = vx_smooth * cardiac[t]
        vy[..., t] = vy_smooth * cardiac[t]
        vz[..., t] = vz_smooth * cardiac[t]

    peak = np.sqrt(vx**2 + vy**2 + vz**2).max()
    print(f"  Peak speed:  {peak:.3f} m/s  "
          f"({'OK' if peak < venc_m_per_s else 'near/above VENC'})")

    # ── 5. 速度 → 相位 ───────────────────────────────────────
    s = np.pi / venc_m_per_s
    return {
        "thetaX": (vx * s).astype(np.float64),
        "thetaY": (vy * s).astype(np.float64),
        "thetaZ": (vz * s).astype(np.float64),
        "vx": vx, "vy": vy, "vz": vz,
        "mask": mask,
        "venc_m_per_s":  venc_m_per_s,
        "voxel_size_mm": voxel_size_mm,
        "dt_seconds":    dt_seconds,
        "flow_axis":     "Z",
        "geometry":      "tapered_incompressible",
    }


# 别名，保持兼容
def branched_flow_phantom(*args, **kwargs):
    return curved_tapered_phantom(*args, **kwargs)
