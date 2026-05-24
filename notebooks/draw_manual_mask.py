"""
手动画降主动脉 mask 工具。

用法：
    python notebooks/draw_manual_mask.py

操作说明：
    - 左键拖动  : 在 aorta_mask 图层上画（画笔模式）
    - [  /  ]   : 缩小 / 放大画笔
    - E         : 切换橡皮擦
    - B         : 切换回画笔
    - S         : 保存当前 mask（不关窗口）
    - 关闭窗口  : 自动保存并退出

保存路径：/mnt/g/medict_tmp/aorta_manual_v1.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

RECON_PATH = Path("/mnt/g/medict_tmp/recon_cs_50iter.mat")
SAVE_PATH  = Path("/mnt/g/medict_tmp/aorta_manual_v1.npy")
VENC       = 1.5   # m/s — must match your acquisition


# ── 加载重建 ─────────────────────────────────────────────────────────────────

def load_recon(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[ERROR] 找不到重建文件：{path}\n"
                 f"       请先运行 50-iter 重建，或修改脚本里的 RECON_PATH。")
    print(f"正在加载重建：{path}")
    raw = sio.loadmat(str(path), squeeze_me=True)
    if "outputs" not in raw:
        sys.exit(f"[ERROR] {path} 里没有 'outputs' 字段，不是标准重建输出。")
    o = raw["outputs"]
    out = {k: np.array(o[k].item()) for k in ["xHat", "thetaX", "thetaY", "thetaZ"]}
    print(f"  数据 shape (Z, Y, X, T): {out['xHat'].shape}")
    return out


# ── 计算可视化图像 ────────────────────────────────────────────────────────────

def make_images(recon: dict, venc: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 (anatomy, pcmra)，shape 均为 (Z, Y, X)。"""
    # 解剖幅值图：看清心脏 / 血管的形态
    anatomy = np.abs(recon["xHat"]).mean(axis=-1).astype(np.float32)

    # PC-MRA 速度幅值：血流快的地方（血管腔）最亮
    speed = np.sqrt(
        recon["thetaX"] ** 2 +
        recon["thetaY"] ** 2 +
        recon["thetaZ"] ** 2
    ) * venc / np.pi                      # (Z, Y, X, T)  单位 m/s
    pcmra = speed.max(axis=-1).astype(np.float32)   # 时间最大值

    print(f"  anatomy 范围: {anatomy.min():.2f} – {anatomy.max():.2f}")
    print(f"  PC-MRA  范围: {pcmra.min():.3f} – {pcmra.max():.3f} m/s")
    return anatomy, pcmra


# ── 保存 ─────────────────────────────────────────────────────────────────────

def save_mask(labels_data: np.ndarray, path: Path) -> None:
    mask = labels_data.astype(bool)
    n = int(mask.sum())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), mask)
    if n == 0:
        print(f"\n[WARNING] mask 是空的！你还没画任何内容。保存到：{path}")
    elif n < 500:
        print(f"\n[WARNING] mask 只有 {n} 个体素，可能漏画了。保存到：{path}")
    elif n > 20_000:
        print(f"\n[WARNING] mask 有 {n} 个体素，可能画到心腔了。保存到：{path}")
    else:
        print(f"\n✓ 保存成功：{path}  ({n} 个体素，合理范围 1000–10000)")


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> None:
    recon           = load_recon(RECON_PATH)
    anatomy, pcmra  = make_images(recon, VENC)
    Z, Y, X         = anatomy.shape

    import napari  # 延迟导入，确保数据加载错误不被 napari 启动日志淹没
    from napari.qt import thread_worker

    viewer = napari.Viewer(title="MEDICT — 手动画降主动脉 mask")

    # 图层 1：解剖幅值（灰度，底图）
    viewer.add_image(
        anatomy,
        name="解剖图 |xHat|",
        colormap="gray",
        opacity=1.0,
    )

    # 图层 2：PC-MRA 速度幅值（热色，覆盖在上面，调低透明度对比）
    viewer.add_image(
        pcmra,
        name="PC-MRA 血流速度",
        colormap="hot",
        opacity=0.55,
        contrast_limits=(0.2, float(pcmra.max())),
    )

    # 图层 3：labels — 用户在这里画
    labels = viewer.add_labels(
        np.zeros((Z, Y, X), dtype=np.uint8),
        name="aorta_mask",
        opacity=0.6,
    )
    labels.mode = "paint"
    labels.brush_size = 4
    labels.selected_label = 1
    viewer.layers.selection.active = labels  # 默认激活 labels 图层

    # 从轴中段开始，降主动脉通常在这里最清楚
    viewer.dims.set_point(0, Z // 2)

    # ── 快捷键 S：随时保存 ──────────────────────────────────────────────────
    @viewer.bind_key("s")
    def _save_hotkey(v):
        save_mask(labels.data, SAVE_PATH)

    # ── 提示信息 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("napari 已打开。")
    print()
    print("  操作提示：")
    print("  1. 在左侧 Layers 面板确认 'aorta_mask' 图层是选中状态")
    print("  2. 左键拖动  → 画（画笔默认激活）")
    print("  3. [  /  ]   → 缩小 / 放大画笔")
    print("  4. E         → 橡皮擦   B → 回到画笔")
    print("  5. S         → 保存当前进度（不关窗口）")
    print("  6. 关闭窗口  → 保存并退出")
    print()
    print("  找降主动脉：拖下方 z 滑块到中段（z≈30-55），")
    print("  看 PC-MRA 热色图里的孤立亮圆管，靠近脊柱那根。")
    print("=" * 60)
    print()

    # ── 运行（阻塞到窗口关闭）────────────────────────────────────────────────
    napari.run()

    # 窗口关闭后自动保存
    save_mask(labels.data, SAVE_PATH)


if __name__ == "__main__":
    main()
