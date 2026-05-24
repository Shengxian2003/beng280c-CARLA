"""
PC-MRA 阈值分割 — Magnitude Mask + 自动找降主动脉。

流程：
  1. 幅值图 |xHat| 做体外剔除（Magnitude Mask）
  2. 只在体内 + z 范围内做 PC-MRA 速度阈值
  3. 连通分量分析，自动挑降主动脉
  4. napari 验证，关窗保存

用法：
    python notebooks/threshold_mask.py

结果不对时调这四个参数重跑。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi

# ══════════════════════════════════════════════════════════════
#  ★ 调这四个参数
PERCENTILE     = 80   # 体内 PC-MRA 速度阈值百分位（低→mask大，高→mask小）
Z_MIN          = 35   # 只在这个 z 范围内找降主动脉
Z_MAX          = 50
MAG_PERCENTILE = 50   # 幅值图体外剔除阈值（通常 15-25 即可）
# ══════════════════════════════════════════════════════════════

RECON_PATH = Path("/mnt/g/medict_tmp/recon_cs_50iter.mat")
SAVE_PATH  = Path("/mnt/g/medict_tmp/aorta_manual_v1.npy")
VENC       = 1.5   # m/s


# ── 1. 加载重建，返回幅值图 + PC-MRA ─────────────────────────

def load_images(path: Path, venc: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 (magnitude, pcmra)，shape 均为 (Z, Y, X)。"""
    print(f"加载重建：{path}")
    raw  = sio.loadmat(str(path), squeeze_me=True)
    o    = raw["outputs"]
    xHat = np.array(o["xHat"].item())
    tX   = np.array(o["thetaX"].item())
    tY   = np.array(o["thetaY"].item())
    tZ   = np.array(o["thetaZ"].item())

    magnitude = np.abs(xHat).mean(axis=-1).astype(np.float32)
    speed     = np.sqrt(tX**2 + tY**2 + tZ**2) * venc / np.pi
    pcmra     = speed.max(axis=-1).astype(np.float32)

    print(f"  shape (Z,Y,X): {magnitude.shape}")
    print(f"  magnitude 范围: {magnitude.min():.2f} – {magnitude.max():.2f}")
    print(f"  PC-MRA    范围: {pcmra.min():.3f} – {pcmra.max():.3f} m/s")
    return magnitude, pcmra


# ── 2. 幅值图 → 体内轮廓 Mask ────────────────────────────────

def make_body_mask(magnitude: np.ndarray, percentile: float) -> np.ndarray:
    """
    用幅值图剔除体外背景。
    幅值图在体内（组织/血液）远高于体外（空气），一个简单的低阈值就能分开。
    """
    thresh     = np.percentile(magnitude, percentile)
    body       = magnitude > thresh
    # 填洞（肺/气管里面的黑区也填成体内）
    # 逐切片 2D 填洞：胸壁在 z 端开口，3D 填洞会失败
    for z in range(body.shape[0]):
        body[z] = ndi.binary_fill_holes(body[z])

    # 稍微腐蚀一圈，避免边界伪影
    body       = ndi.binary_erosion(body, iterations=2)
    n_in       = int(body.sum())
    total      = body.size
    print(f"\n[Magnitude Mask]  阈值={thresh:.2f}  "
          f"体内体素={n_in} ({100*n_in/total:.1f}%)")
    return body.astype(bool)


# ── 3. 体内 PC-MRA 阈值 + 连通分量 ───────────────────────────

def find_candidates(pcmra: np.ndarray,
                    body_mask: np.ndarray,
                    percentile: float,
                    z_min: int,
                    z_max: int) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """沿 body_mask 外围做 bbox，在 bbox AND z 范围内做阈值。"""
    Z       = pcmra.shape[0]
    z_min   = max(0, z_min)
    z_max   = min(Z, z_max)

    # ★ 沿 body_mask 外围画长方体 bbox
    zz, yy, xx = np.where(body_mask)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    bbox_mask = np.zeros(pcmra.shape, dtype=bool)
    bbox_mask[z0:z1, y0:y1, x0:x1] = True
    print(f"\n[Body BBox]  z=[{z0}..{z1}]  y=[{y0}..{y1}]  x=[{x0}..{x1}]")

    # 构建 ROI：bbox AND z 范围
    z_band        = np.zeros(pcmra.shape, dtype=bool)
    z_band[z_min:z_max] = True
    roi_mask      = bbox_mask & z_band

    values_in_roi = pcmra[roi_mask]
    thresh        = np.percentile(values_in_roi, percentile)

    binary = (pcmra > thresh) & roi_mask
    print(f"\n[速度阈值]  {percentile}th 百分位 = {thresh:.3f} m/s  "
          f"(z={z_min}–{z_max}, bbox 内)")
    print(f"  阈值后体素数: {binary.sum()}")

    struct        = ndi.generate_binary_structure(3, 1)
    labeled, n    = ndi.label(binary, structure=struct)
    print(f"  连通分量数: {n}")

    candidates = []
    for lab in range(1, n + 1):
        cmask = labeled == lab
        size  = int(cmask.sum())
        if size < 200:
            continue
        mean_speed = float(pcmra[cmask].mean())
        zz, yy, xx = np.where(cmask)
        candidates.append({
            "label":      lab,
            "size":       size,
            "mean_speed": mean_speed,
            "z_range":    (int(zz.min()), int(zz.max())),
            "centroid":   (int(zz.mean()), int(yy.mean()), int(xx.mean())),
        })

    # 评分：速度 × 大小接近理想值（降主动脉约 2000–6000 voxels）
    SIZE_IDEAL = 4000
    for c in candidates:
        c["score"] = c["mean_speed"] * min(c["size"] / SIZE_IDEAL, 1.0)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return labeled, candidates, bbox_mask


# ── 4. 打印候选列表 ───────────────────────────────────────────

def print_candidates(candidates: list[dict]) -> None:
    print()
    print(f"{'排名':<4} {'label':<7} {'体素数':<9} "
          f"{'平均速度(m/s)':<16} {'z范围':<12} {'质心(z,y,x)'}")
    print("-" * 68)
    for i, c in enumerate(candidates[:8]):
        print(f"  {i+1:<4} {c['label']:<7} {c['size']:<9} "
              f"{c['mean_speed']:<16.3f} "
              f"{str(c['z_range']):<12} {c['centroid']}")
    if not candidates:
        print("  (没有候选，试试降低 PERCENTILE 或扩大 Z_MIN/Z_MAX)")


# ── 5. napari 验证 ────────────────────────────────────────────

def verify_in_napari(magnitude: np.ndarray,
                     pcmra: np.ndarray,
                     body_mask: np.ndarray,
                     bbox_mask: np.ndarray,
                     labeled: np.ndarray,
                     best_mask: np.ndarray,
                     best_label: int) -> None:
    import napari

    viewer = napari.Viewer(title="MEDICT — 阈值分割验证（绿色 = 选中的 mask）")

    # 底图：解剖幅值
    viewer.add_image(magnitude, name="解剖图 |xHat|",
                     colormap="gray", opacity=1.0)

    # PC-MRA 血流
    viewer.add_image(pcmra, name="PC-MRA 血流速度",
                     colormap="hot", opacity=0.5,
                     contrast_limits=(0.3, float(pcmra.max())))

    # 体内轮廓（蓝色半透明）
    viewer.add_image(body_mask.astype(np.float32),
                     name="体内轮廓 (Magnitude Mask)",
                     colormap="blue", opacity=1)

    # ★ Body BBox（青色，实际搜索 ROI）
    viewer.add_image(bbox_mask.astype(np.float32),
                     name="Body BBox（青，实际 ROI）",
                     colormap="cyan", opacity=0.25)

    # 所有候选连通分量
    viewer.add_labels(labeled.astype(np.uint8),
                      name="所有候选分量", opacity=0.3)

    # 选中的 mask（绿色）
    viewer.add_image(best_mask.astype(np.float32),
                     name=f"★ 选中 label={best_label}",
                     colormap="green", opacity=0.75)

    print()
    print("napari 已打开。")
    print("  绿色 = 自动选中的候选")
    print("  蓝色半透明 = 体内轮廓（Magnitude Mask 剔除的边界）")
    print()
    print("  ✓ 绿色就是降主动脉 → 直接关窗口，自动保存")
    print("  ✗ 选错了 → 关窗口，调参数重跑")
    print("    - 太大/包含心腔  : 提高 PERCENTILE（如 80）")
    print("    - 太小/只有几片  : 降低 PERCENTILE（如 70）")
    print("    - 仍有体外噪声   : 提高 MAG_PERCENTILE（如 25）")
    print("    - 选的不是最好的 : 把 main() 里 cands[0] 改成 cands[1] 等")
    print()

    napari.run()


# ── 主程序 ────────────────────────────────────────────────────

def main() -> None:
    if not RECON_PATH.exists():
        raise FileNotFoundError(f"找不到重建文件：{RECON_PATH}")

    magnitude, pcmra        = load_images(RECON_PATH, VENC)
    body_mask               = make_body_mask(magnitude, MAG_PERCENTILE)
    labeled, cands, bbox_mask = find_candidates(pcmra, body_mask,
                                                PERCENTILE, Z_MIN, Z_MAX)
    print_candidates(cands)

    if not cands:
        print("\n没有候选，请降低 PERCENTILE 或扩大 Z_MIN/Z_MAX 后重试。")
        return

    best      = cands[0]
    best_mask = (labeled == best["label"])
    print(f"\n自动选中：label={best['label']}  "
          f"大小={best['size']} 体素  "
          f"平均速度={best['mean_speed']:.3f} m/s  "
          f"z范围={best['z_range']}")

    verify_in_napari(magnitude, pcmra, body_mask, bbox_mask,
                     labeled, best_mask, best["label"])

    # 关窗后保存
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(SAVE_PATH), best_mask)
    n = int(best_mask.sum())
    print(f"\n✓ 保存：{SAVE_PATH}  ({n} 体素)")
    if n < 500:
        print("  ⚠ 太少，试试降低 PERCENTILE")
    elif n > 15_000:
        print("  ⚠ 太多，可能包含心腔，试试提高 PERCENTILE")
    else:
        print("  体素数合理，可以进下一步。")


if __name__ == "__main__":
    main()