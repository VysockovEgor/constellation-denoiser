"""Summarize per-file results and create a conference-ready pilot figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import PROJECT_ROOT


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 20_000):
    n = len(values)
    samples = rng.integers(0, n, size=(reps, n))
    return np.percentile(values[samples].mean(axis=1), [2.5, 97.5])


def main():
    result_dir = PROJECT_ROOT / "reports" / "results"
    per_file = pd.read_csv(result_dir / "large_scale_pilot_per_file.csv")
    summary = pd.read_csv(result_dir / "large_scale_pilot_summary.csv")
    rng = np.random.default_rng(20261003)

    by_file = per_file.groupby(["model", "file"], as_index=False).agg(
        raw_ber=("raw_ber", "mean"), model_ber=("model_ber", "mean"),
    )
    wide = by_file.pivot(index="file", columns="model", values="model_ber")
    raw = by_file[by_file.model == "real"].set_index("file").loc[wide.index, "raw_ber"].to_numpy()
    tanh = wide["real"].to_numpy()
    complex_ber = wide["complex"].to_numpy()
    file_reductions = {
        "real": 100 * (raw - tanh) / raw,
        "complex": 100 * (raw - complex_ber) / raw,
        "complex_vs_real": 100 * (tanh - complex_ber) / tanh,
    }

    metrics = {}
    for name, vals in (("raw_ber", raw), ("real_ber", tanh), ("complex_ber", complex_ber)):
        ci = bootstrap_ci(vals, rng)
        metrics[name] = {"mean_ber": float(vals.mean()), "ber_ci95": ci.tolist()}
    for name, vals in file_reductions.items():
        ci = bootstrap_ci(vals, rng)
        metrics[name] = {
            "mean_reduction_pct": float(vals.mean()), "reduction_ci95_pct": ci.tolist(),
            "files_improved": int((vals > 0).sum()), "test_files": len(vals),
        }

    print("Per-seed BER summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nMean raw BER: {metrics['raw_ber']['mean_ber']:.6f}; 95% file-bootstrap CI={metrics['raw_ber']['ber_ci95']}")
    print(f"Mean tanh BER: {metrics['real_ber']['mean_ber']:.6f}; 95% file-bootstrap CI={metrics['real_ber']['ber_ci95']}")
    print(f"Mean complex BER: {metrics['complex_ber']['mean_ber']:.6f}; 95% file-bootstrap CI={metrics['complex_ber']['ber_ci95']}")
    print(f"Tanh reduction vs raw: {metrics['real']['mean_reduction_pct']:.2f}% (95% CI {metrics['real']['reduction_ci95_pct']}); improved files={metrics['real']['files_improved']}/{metrics['real']['test_files']}")
    print(f"Complex reduction vs raw: {metrics['complex']['mean_reduction_pct']:.2f}% (95% CI {metrics['complex']['reduction_ci95_pct']}); improved files={metrics['complex']['files_improved']}/{metrics['complex']['test_files']}")
    print(f"Complex reduction vs tanh: {metrics['complex_vs_real']['mean_reduction_pct']:.2f}% (95% CI {metrics['complex_vs_real']['reduction_ci95_pct']}); improved files={metrics['complex_vs_real']['files_improved']}/{metrics['complex_vs_real']['test_files']}")

    metrics_path = result_dir / "large_scale_pilot_bootstrap.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    labels = ["Raw nearest-point", "Real MLP (tanh)", "Complex MLP"]
    means = [metrics[k]["mean_ber"] for k in ("raw_ber", "real_ber", "complex_ber")]
    cis = [metrics[k]["ber_ci95"] for k in ("raw_ber", "real_ber", "complex_ber")]
    lower = [m - ci[0] for m, ci in zip(means, cis)]
    upper = [ci[1] - m for m, ci in zip(means, cis)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    colors = ["#6b7280", "#2878b5", "#d97706"]
    axes[0].bar(labels, means, color=colors, width=0.68)
    axes[0].errorbar(range(3), means, yerr=[lower, upper], fmt="none", color="#111827", capsize=4, lw=1.2)
    axes[0].set_ylabel("BER (ниже — лучше)")
    axes[0].set_title("Средний BER на 45 отложенных PR")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.22)
    for i, v in enumerate(means):
        axes[0].text(i, v + upper[i] + 0.00015, f"{v:.5f}", ha="center", va="bottom", fontsize=9)

    data = [file_reductions["real"], file_reductions["complex"], file_reductions["complex_vs_real"]]
    bp = axes[1].boxplot(data, tick_labels=["tanh vs raw", "complex vs raw", "complex vs tanh"],
                         patch_artist=True, showfliers=False, whis=(5, 95))
    for patch, color in zip(bp["boxes"], [colors[1], colors[2], "#7c3aed"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[1].axhline(0, color="#374151", lw=1)
    axes[1].axhline(40, color="#b91c1c", lw=1, ls="--", label="цель 40%")
    axes[1].set_ylabel("Снижение BER по каждому PR, %")
    axes[1].set_title("Распределение улучшения по файлам")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Pilot: контекст 31; train/validation/test = 210/45/45 уникальных PR", fontsize=12)
    fig.tight_layout()

    figure = PROJECT_ROOT / "reports" / "figures" / "large_scale_pilot.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    print(f"\nSaved {figure}")


if __name__ == "__main__":
    main()
