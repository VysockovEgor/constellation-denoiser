"""E014 summary: Kerr + KAN hybrids vs the Kerr base, with the added multiplies.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E014   (SMOKE=1 reads results_smoke/)

Writes results/kan_hybrid.csv, results/kan_hybrid_summary.csv,
results/kan_hybrid_verdicts.csv and figures/E014_kan_tradeoff.png.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .runner import exp_dir, results_root
from .summary_utils import compare

METRIC = "test_reduction_valbest_select"
ORDER = ["k0_kerr", "k0b_kerr_continued", "k1_kan_only", "k2_parallel_joint", "k3_parallel_staged", "k4_cascade_staged"]
BASELINE = {"k2_parallel_joint": "k0_kerr", "k3_parallel_staged": "k0b_kerr_continued",
            "k4_cascade_staged": "k0b_kerr_continued", "k1_kan_only": "k0_kerr"}


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E014")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E014", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    runs = runs[runs.stage == "main"] if not runs.empty else runs
    if runs.empty:
        print("нет прогонов main")
        return 1
    runs = runs.copy()
    runs["grid"] = runs.get("spec_grid", pd.Series(np.nan, index=runs.index))
    runs.to_csv(results / "kan_hybrid.csv", index=False)

    k0_macs = float(runs[runs.variant == "k0_kerr"].macs_dense_total.iloc[0]) if (runs.variant == "k0_kerr").any() \
        else 6480.0
    groups = []
    for (variant, grid), group in runs.groupby(["variant", runs.grid.fillna(-1)]):
        groups.append({
            "variant": variant, "grid": None if grid == -1 else int(grid), "n_seeds": len(group),
            "seeds": " ".join(str(s) for s in sorted(group.seed)), "n_diverged": int(group.diverged.sum()),
            "test_valbest_mean": group[METRIC].mean(), "test_valbest_min": group[METRIC].min(),
            "test_valbest_max": group[METRIC].max(), "test_final_mean": group.test_reduction.mean(),
            "val_best_mean": group.best_val_reduction.mean(), "n_params": int(group.n_params.iloc[0]),
            "kan_params": int(group.kan_params.iloc[0]), "kerr_macs": int(group.kerr_macs.iloc[0]),
            "kan_macs_dense": int(group.kan_macs_dense.iloc[0]), "kan_macs_sparse": int(group.kan_macs_sparse.iloc[0]),
            "macs_dense_total": int(group.macs_dense_total.iloc[0]),
            "macs_sparse_total": int(group.macs_sparse_total.iloc[0]),
            "macs_increase_dense_vs_k0_pct": 100 * (group.macs_dense_total.iloc[0] - k0_macs) / k0_macs,
            "macs_increase_sparse_vs_k0_pct": 100 * (group.macs_sparse_total.iloc[0] - k0_macs) / k0_macs,
            "steps": int(group.budget.iloc[0]),
        })
    summary = pd.DataFrame(groups)
    summary["order"] = summary.variant.map({v: i for i, v in enumerate(ORDER)})
    summary = summary.sort_values(["order", "grid"]).drop(columns="order")
    summary.to_csv(results / "kan_hybrid_summary.csv", index=False)

    verdicts = []
    for _, row in summary.iterrows():
        base = BASELINE.get(row.variant)
        if base is None:
            continue
        mine = runs[(runs.variant == row.variant) & (runs.grid == row.grid)][METRIC].tolist()
        ref = runs[runs.variant == base][METRIC].tolist()
        result = compare(mine, ref)
        if row.variant == "k1_kan_only":
            text = ("KAN без Керр-основы лучше k0" if result["beats"] else
                    "KAN без Керр-основы не лучше k0 (Керр-основа нужна или вопрос не решён)")
        else:
            text = "гибрид улучшает Керр-основу" if result["beats"] else "не доказано"
        verdicts.append({"variant": row.variant, "grid": row.grid, "baseline": base, "diff_pp": result["diff"],
                         "seed_spread_pp": result["spread"], "added_macs_dense": row.macs_dense_total - k0_macs,
                         "added_macs_sparse": row.macs_sparse_total - k0_macs,
                         "macs_increase_dense_vs_k0_pct": row.macs_increase_dense_vs_k0_pct,
                         "macs_increase_sparse_vs_k0_pct": row.macs_increase_sparse_vs_k0_pct,
                         "verdict": text})
    verdict_frame = pd.DataFrame(verdicts)
    verdict_frame["rule_status"] = config["rule_draft"]["status"]
    verdict_frame.to_csv(results / "kan_hybrid_verdicts.csv", index=False)
    plot(summary, figures)
    print(f"сводка: {results / 'kan_hybrid_summary.csv'}; рисунок: {figures / 'E014_kan_tradeoff.png'}")
    return 0


def plot(summary: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    variants = [v for v in ORDER if v in set(summary.variant)]
    for _, row in summary.iterrows():
        c = plot_style.color(ORDER.index(row.variant))
        marker = "s" if row.grid == 8 else "o"
        y = row.test_valbest_mean
        err = [[y - row.test_valbest_min], [row.test_valbest_max - y]]
        ax.errorbar(row.macs_dense_total, y, yerr=err, fmt=marker, color=c, markersize=7, capsize=2, linewidth=1)
        if row.macs_sparse_total != row.macs_dense_total:
            ax.plot([row.macs_sparse_total, row.macs_dense_total], [y, y], color=c, linewidth=0.8, alpha=0.5)
            ax.plot(row.macs_sparse_total, y, marker, color=c, markerfacecolor="none", markersize=7)
    ax.set_xlabel("real multiplies per symbol (filled: dense as implemented, hollow: sparse k+1 bases)")
    ax.set_ylabel("test BER reduction at best validation, %")
    ax.set_title("Kerr + KAN hybrids: gain vs added multiplies")
    handles = [Line2D([], [], color=plot_style.color(ORDER.index(v)), marker="o", linestyle="", label=v)
               for v in variants]
    handles += [Line2D([], [], color=plot_style.MUTED, marker="o", linestyle="", label="G=5 / no KAN"),
                Line2D([], [], color=plot_style.MUTED, marker="s", linestyle="", label="G=8")]
    ax.legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(figures / "E014_kan_tradeoff.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
