"""Stage 4b — Detection-rate sweep.

For each of the four verifier checks, sweep the corresponding error
magnitude from zero (control) to large (clear failure) across N trials per
magnitude, and record the verifier's verdict. Output:

  evaluation/results/detection_eval.json   structured per-check results
  evaluation/results/detection_<check>.png  detection-vs-magnitude curve

This is the load-bearing scientific deliverable for Stage 4. It quantifies
HOW WELL the Physics Verifier detects each error class as a function of
magnitude — the auditability claim depends on these numbers.

Usage:
    python evaluation/run_detection_eval.py
    python evaluation/run_detection_eval.py --n-trials 20    # more trials
    python evaluation/run_detection_eval.py --no-plots       # skip matplotlib
"""
from __future__ import annotations

import argparse
import json
import time
import sys, os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.eval_inject import (
    clean_flow_phantom, velocity_to_phase,
    inject_divergence, inject_flux_imbalance,
    inject_peak_velocity, inject_phase_wrap,
)
from skills.physics_verifier import verify


# ============================================================================
# Per-check sweep configuration
#
# Magnitudes are picked to bracket the verifier's warn / fail thresholds so
# we can see the detection curve rise from 0% → 100% across the threshold.
# ============================================================================

SWEEP_CONFIG: dict[str, dict] = {
    "divergence": {
        "param_name":   "magnitude_per_s",
        "param_unit":   "s⁻¹",
        "magnitudes":   [0.0, 1.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0],
        "warn_thr":     5.0,
        "fail_thr":     20.0,
        "verifier_key": "divergence",
    },
    "net_flux": {
        "param_name":   "fractional_imbalance",
        "param_unit":   "× (start→end)",
        "magnitudes":   [0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.5, 2.0],
        "warn_thr":     0.4,      # corresponds roughly to 10% max-dev
        "fail_thr":     1.0,      # corresponds roughly to 25% max-dev
        "verifier_key": "net_flux",
    },
    "peak_velocity": {
        "param_name":   "target_peak_m_per_s",
        "param_unit":   "m/s",
        "magnitudes":   [0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.5, 6.0, 8.0],
        "warn_thr":     3.0,
        "fail_thr":     4.5,
        "verifier_key": "peak_velocity",
    },
    "phase_unwrap": {
        "param_name":   "fraction",
        "param_unit":   "(of mask voxels)",
        "magnitudes":   [0.0, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6],
        "warn_thr":     0.01,
        "fail_thr":     0.1,
        "verifier_key": "phase_unwrap",
    },
}


# ============================================================================
# Trial runner
# ============================================================================

def _run_trial(check: str, magnitude: float, seed: int, phantom: dict) -> dict:
    """One injection + one verify. Returns the per-check status from the verifier."""
    vx, vy, vz = phantom["vx"], phantom["vy"], phantom["vz"]
    tx, ty, tz = phantom["thetaX"], phantom["thetaY"], phantom["thetaZ"]
    mask = phantom["mask"]
    venc = phantom["venc_m_per_s"]
    vox  = phantom["voxel_size_mm"]

    if check == "divergence":
        vx2, vy2, vz2 = inject_divergence(vx, vy, vz, mask,
                                          voxel_size_mm=vox, magnitude_per_s=magnitude)
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, venc)
    elif check == "net_flux":
        vx2, vy2, vz2 = inject_flux_imbalance(vx, vy, vz, mask,
                                              flow_axis="Z", fractional_imbalance=magnitude)
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, venc)
    elif check == "peak_velocity":
        vx2, vy2, vz2 = inject_peak_velocity(vx, vy, vz, mask,
                                             target_peak_m_per_s=magnitude, seed=seed)
        tx, ty, tz = velocity_to_phase(vx2, vy2, vz2, venc)
    elif check == "phase_unwrap":
        tx, ty, tz = inject_phase_wrap(tx, ty, tz, mask, fraction=magnitude, seed=seed)
    else:
        raise ValueError(check)

    verdict = verify(tx, ty, tz, venc_m_per_s=venc, mask=mask, voxel_size_mm=vox)
    return verdict


def _sweep_one_check(check: str, n_trials: int, console_log=print) -> dict:
    """Run the full magnitude sweep for one check. Returns aggregate results."""
    cfg = SWEEP_CONFIG[check]
    target_key = cfg["verifier_key"]

    results = {
        "check":      check,
        "param_name": cfg["param_name"],
        "param_unit": cfg["param_unit"],
        "warn_thr":   cfg["warn_thr"],
        "fail_thr":   cfg["fail_thr"],
        "n_trials":   n_trials,
        "magnitudes": [],
    }

    for mag in cfg["magnitudes"]:
        pass_count = warn_count = fail_count = 0
        # We also track whether the FULL verdict flips (any check fails) so
        # we can plot "any flag raised" vs "the target flag raised" later
        any_failed = any_warned = 0

        # Per-magnitude trials with distinct seeds (some injectors are stochastic)
        for trial in range(n_trials):
            ph = clean_flow_phantom()  # fresh phantom each trial
            verdict = _run_trial(check, mag, seed=trial, phantom=ph)
            target = verdict["checks"][target_key]["status"]
            if target == "pass":   pass_count += 1
            elif target == "warn": warn_count += 1
            else:                   fail_count += 1
            if verdict["verdict"] == "fail":  any_failed += 1
            if verdict["verdict"] == "warn":  any_warned += 1

        results["magnitudes"].append({
            "magnitude":       mag,
            "target_pass":     pass_count,
            "target_warn":     warn_count,
            "target_fail":     fail_count,
            "any_check_fail":  any_failed,
            "any_check_warn":  any_warned,
            "detection_rate":  round((warn_count + fail_count) / n_trials, 4),
            "fail_rate":       round(fail_count / n_trials, 4),
        })
        console_log(f"  [{check}] mag={mag:>8.4f}  "
                    f"pass/warn/fail = {pass_count}/{warn_count}/{fail_count}  "
                    f"detect={results['magnitudes'][-1]['detection_rate']:.2f}")

    # Summary metrics: detection at "ground truth fail" magnitudes
    fail_mags = [m for m in results["magnitudes"] if m["magnitude"] >= cfg["fail_thr"]]
    pass_mags = [m for m in results["magnitudes"] if m["magnitude"] == 0]
    results["true_positive_rate"] = round(
        np.mean([m["fail_rate"] for m in fail_mags]) if fail_mags else 0.0, 4
    )
    results["false_positive_rate"] = round(
        np.mean([m["fail_rate"] for m in pass_mags]) if pass_mags else 0.0, 4
    )
    return results


# ============================================================================
# Plotting (matplotlib)
# ============================================================================

def _plot_check(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mags  = [m["magnitude"]      for m in results["magnitudes"]]
    det   = [m["detection_rate"] for m in results["magnitudes"]]
    fail  = [m["fail_rate"]      for m in results["magnitudes"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mags, det,  "o-", label="detected (warn or fail)", color="C0")
    ax.plot(mags, fail, "s--", label="fail verdict only",      color="C3")
    ax.axvline(results["warn_thr"], color="orange", ls=":", alpha=0.6, label=f"warn thr={results['warn_thr']}")
    ax.axvline(results["fail_thr"], color="red",    ls=":", alpha=0.6, label=f"fail thr={results['fail_thr']}")
    ax.set_xlabel(f"Injected magnitude  [{results['param_unit']}]")
    ax.set_ylabel("Rate (out of N trials)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Detection rate — {results['check']}\n"
                 f"TPR(at fail mags) = {results['true_positive_rate']:.2f}  "
                 f"FPR(at mag=0) = {results['false_positive_rate']:.2f}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Stage 4b detection-rate sweep")
    p.add_argument("--n-trials", type=int, default=10,
                   help="trials per (check × magnitude) cell (default 10)")
    p.add_argument("--checks", nargs="+",
                   default=list(SWEEP_CONFIG),
                   help="which checks to sweep (default all 4)")
    p.add_argument("--out-dir", default="evaluation/results",
                   help="output directory")
    p.add_argument("--no-plots", action="store_true",
                   help="skip matplotlib plots (JSON only)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage 4b: detection sweep  ({args.n_trials} trials per cell)")
    print(f"Checks: {args.checks}")
    print(f"Output: {out_dir.absolute()}")
    print()

    t0 = time.time()
    all_results = {}
    for check in args.checks:
        print(f"--- {check} ---")
        all_results[check] = _sweep_one_check(check, args.n_trials)
        if not args.no_plots:
            png = out_dir / f"detection_{check}.png"
            _plot_check(all_results[check], png)
            print(f"  → wrote {png}")
        print()

    # Aggregate report
    report = {
        "n_trials_per_cell": args.n_trials,
        "elapsed_seconds":   round(time.time() - t0, 1),
        "per_check":         all_results,
        "summary": {
            check: {
                "true_positive_rate":  r["true_positive_rate"],
                "false_positive_rate": r["false_positive_rate"],
            }
            for check, r in all_results.items()
        },
    }
    json_path = out_dir / "detection_eval.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"✓ JSON report: {json_path}")

    # Headline numbers
    print()
    print("Summary (averaged):")
    for check, r in all_results.items():
        tpr = r["true_positive_rate"]
        fpr = r["false_positive_rate"]
        print(f"  {check:<14s}  TPR={tpr:.2f}  FPR={fpr:.2f}")


if __name__ == "__main__":
    main()
