"""
显示带分叉 + 粗细变化的 phantom，并跑 Physics Verifier 检查物理。

用法：
    python notebooks/show_phantom.py            # 显示分叉版（默认）
    python notebooks/show_phantom.py --simple   # 显示原来的圆柱版

跑 verifier 后会打印 4 个检查的结果，告诉你这个 phantom 是不是"good case"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills.eval_inject._phantom    import clean_flow_phantom
from skills.eval_inject._phantom_v2 import branched_flow_phantom
from skills.physics_verifier        import verify


def run_verifier(p: dict) -> dict:
    """跑 Physics Verifier，返回 verdict。"""
    return verify(
        p["thetaX"], p["thetaY"], p["thetaZ"],
        venc_m_per_s=p["venc_m_per_s"],
        mask=p["mask"],
        voxel_size_mm=p["voxel_size_mm"],
    )


def print_verdict(v: dict) -> None:
    print()
    print("=" * 70)
    print("Physics Verifier 结果")
    print("=" * 70)
    overall = v.get("verdict", "?")
    icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(overall, "?")
    print(f"  总体判定:  {icon} {overall.upper()}")
    print()
    checks = v.get("checks", {})
    for name in ["divergence", "net_flux", "peak_velocity", "phase_unwrap"]:
        c = checks.get(name, {})
        if not c:
            continue
        status = c.get("status", "?")
        icon   = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(status, "?")
        line = f"  {name:<15} {icon} {status.upper():<6}"
        if name == "divergence":
            line += (f" mean|∇·v|={c.get('mean_abs_divergence_per_s', '?')} s⁻¹  "
                     f"(thr warn/fail = {c.get('threshold_warn_per_s')}/"
                     f"{c.get('threshold_fail_per_s')})")
        elif name == "net_flux":
            line += (f" max deviation={c.get('max_deviation_pct', '?')}%  "
                     f"(thr warn/fail = {c.get('threshold_warn_pct')}/"
                     f"{c.get('threshold_fail_pct')}%)")
        elif name == "peak_velocity":
            line += (f" peak={c.get('peak_m_per_s', '?')} m/s  "
                     f"physio range = {c.get('physiological_range_m_per_s')}")
        elif name == "phase_unwrap":
            line += (f" wrap fraction={c.get('fraction_above_threshold', '?')}  "
                     f"(thr warn/fail = {c.get('threshold_warn_fraction')}/"
                     f"{c.get('threshold_fail_fraction')})")
        print(line)
    print("=" * 70)


def main(use_simple: bool) -> None:
    if use_simple:
        print("生成简单圆柱 phantom ...")
        p = clean_flow_phantom()
        title = "Simple Cylinder Phantom"
    else:
        print("生成 Y 分叉 + 粗细变化 phantom ...")
        p = branched_flow_phantom(pulsatile=True)
        title = "Branched + Tapered Phantom"

    mask  = p["mask"]
    speed = np.sqrt(p["vx"]**2 + p["vy"]**2 + p["vz"]**2)
    speed_mip = speed.max(axis=-1).astype(np.float32)

    print(f"  shape (Z,Y,X,T): {p['vx'].shape}")
    print(f"  mask 体素数:      {int(mask.sum())}")

    # ── 跑 Physics Verifier ──
    print("\n运行 Physics Verifier...")
    verdict = run_verifier(p)
    print_verdict(verdict)

    # ── napari 显示 ──
    import napari

    viewer = napari.Viewer(title=f"MEDICT — {title}")

    viewer.add_image(speed_mip,
                     name="速度幅值 (max-over-time)",
                     colormap="hot", opacity=1.0)
    viewer.add_image(p["vz"].astype(np.float32),
                     name="vz (4D - 沿 Z 速度分量)",
                     colormap="turbo", opacity=0.7,
                     contrast_limits=(-0.6, 0.6))
    viewer.add_image(p["vx"].astype(np.float32),
                     name="vx (4D - 分支侧向分量)",
                     colormap="turbo", opacity=0.7,
                     visible=False,
                     contrast_limits=(-0.6, 0.6))
    viewer.add_image(mask.astype(np.float32),
                     name="Ground Truth Mask",
                     colormap="green", opacity=0.4)

    viewer.dims.set_point(0, mask.shape[0] // 2)

    print("\nnapari 已打开。")
    print("  - 绿色 = ground truth mask（血管壁）")
    print("  - 热色 = 时间最大速度幅值（看血管的形状走向）")
    print("  - vz 4D 层 = 沿 Z 速度，拖时间滑块看心动周期")
    print("  - vx 4D 层（默认隐藏）= 侧向速度，分支处能看到流向偏转")
    print()

    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", action="store_true",
                        help="使用原来的简单圆柱 phantom（对照组）")
    args = parser.parse_args()
    main(use_simple=args.simple)
