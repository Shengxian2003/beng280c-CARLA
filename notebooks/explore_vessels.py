"""
血管探索工具 — 帮你在 napari 里找到降主动脉的位置。

不做分割，只显示数据 + 三种视图，让你用眼睛找到目标血管：
  1. 解剖图（看脊柱、心脏的形态轮廓）
  2. PC-MRA 速度图（血管亮起来）
  3. MIP（最大强度投影）—— 把所有 z 压缩成一张图，
     垂直血管会显示成一条直线，最容易识别降主动脉

用法：
    python notebooks/explore_vessels.py

找到降主动脉后，记下它的 (z 范围, Y, X) 坐标，告诉我们继续下一步。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

RECON_PATH = Path("/mnt/g/medict_tmp/recon_cs_50iter.mat")
VENC       = 1.5


def main() -> None:
    print(f"加载重建：{RECON_PATH}")
    raw  = sio.loadmat(str(RECON_PATH), squeeze_me=True)
    o    = raw["outputs"]
    xHat = np.array(o["xHat"].item())
    tX   = np.array(o["thetaX"].item())
    tY   = np.array(o["thetaY"].item())
    tZ   = np.array(o["thetaZ"].item())

    # 解剖图（时间平均）
    anatomy = np.abs(xHat).mean(axis=-1).astype(np.float32)
    # PC-MRA 速度图（时间最大）
    pcmra   = (np.sqrt(tX**2 + tY**2 + tZ**2)
               * VENC / np.pi).max(axis=-1).astype(np.float32)

    Z, Y, X = anatomy.shape
    print(f"  shape (Z,Y,X) = ({Z}, {Y}, {X})")
    print(f"  PC-MRA peak speed: {pcmra.max():.2f} m/s")

    # ── 关键：MIP（沿 z 投影）——血管会显示成"线" ────────
    # 沿 z 取最大值 = 把所有切片"压扁"
    pcmra_mip_z = pcmra.max(axis=0)          # (Y, X) — 看垂直血管最清楚
    # 沿 y 投影 — 看侧位
    pcmra_mip_y = pcmra.max(axis=1)          # (Z, X)
    # 沿 x 投影 — 看正位
    pcmra_mip_x = pcmra.max(axis=2)          # (Z, Y)

    print()
    print("=" * 70)
    print("📖 找降主动脉的步骤：")
    print()
    print(" 1) 先看 'MIP 沿 Z' 图层（关掉别的，只看这个）：")
    print("    所有沿 z 方向走的血管会显示成一条线/亮点")
    print("    降主动脉 = 一根孤立的、靠近脊柱（图像偏后/偏下）的圆形亮点")
    print()
    print(" 2) 看 'MIP 沿 Y' 或 'MIP 沿 X' 图层确认它是不是垂直管子")
    print("    降主动脉应该是一条**几乎垂直的直线**贯穿大半个 z 范围")
    print()
    print(" 3) 回到 '解剖图 + PC-MRA' 视图，拖 z 滑块找它：")
    print("    在每个 z 切片里，它都应该在**同一个 (Y, X) 位置**")
    print("    形状是**小圆形**，直径 ~5 体素")
    print()
    print(" 4) 找解剖标志确认：")
    print("    脊柱 = 解剖图里最暗的小凹陷（无信号的骨头）")
    print("    降主动脉 = 紧贴脊柱前方的小亮圆")
    print()
    print(" 5) 找到后，记下：")
    print("    - z 从 ___ 到 ___（开始和结束切片）")
    print("    - 大概的 (Y, X) 位置")
    print("    告诉 Claude 这些坐标，我们直接框选这一根血管。")
    print("=" * 70)
    print()

    # ── 开 napari ──
    import napari
    viewer = napari.Viewer(title="MEDICT — 找降主动脉")

    # 解剖图 contrast：用 1-99 百分位，避开极值，看清组织
    anat_lo, anat_hi = float(np.percentile(anatomy, 1)), float(np.percentile(anatomy, 99))
    pcmra_lo         = float(np.percentile(pcmra,    50))  # 中位以上才显示，背景全黑
    pcmra_hi         = float(pcmra.max())

    # 主 3D 视图
    viewer.add_image(anatomy, name="1. 解剖图 |xHat|",
                     colormap="gray", opacity=1.0,
                     contrast_limits=(anat_lo, anat_hi))
    viewer.add_image(pcmra, name="2. PC-MRA 血流速度",
                     colormap="hot", opacity=0.7,
                     contrast_limits=(pcmra_lo, pcmra_hi))

    # 三个 MIP（用单独的图层，不要混在 3D 体积里）
    viewer.add_image(pcmra_mip_z, name="MIP 沿 Z (Y, X) — 看垂直血管",
                     colormap="hot", visible=False,
                     contrast_limits=(0.3, float(pcmra_mip_z.max())))
    viewer.add_image(pcmra_mip_y, name="MIP 沿 Y (Z, X)",
                     colormap="hot", visible=False,
                     contrast_limits=(0.3, float(pcmra_mip_y.max())))
    viewer.add_image(pcmra_mip_x, name="MIP 沿 X (Z, Y)",
                     colormap="hot", visible=False,
                     contrast_limits=(0.3, float(pcmra_mip_x.max())))

    # 默认从中段切片开始
    viewer.dims.set_point(0, Z // 2)

    napari.run()


if __name__ == "__main__":
    main()
