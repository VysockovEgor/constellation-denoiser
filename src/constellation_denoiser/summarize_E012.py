"""E012 summary: depth vs window table, merge check, draft-rule verdicts, figure.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E012   (SMOKE=1 reads results_smoke/)

Writes results/depth_window.csv, results/depth_window_summary.csv,
results/depth_window_verdicts.csv, results/merge_check.csv and
figures/E012_depth_window.png.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .runner import exp_dir, results_root
from .summary_utils import compare

METRIC = "test_reduction_valbest_select"
VARIANTS = ["v1_two_kerr_both", "v2_two_kerr_last", "v3_one_hidden", "v4_one_hidden_wide"]
MARKERS = {"v1_two_kerr_both": "o", "v2_two_kerr_last": "s", "v3_one_hidden": "^", "v4_one_hidden_wide": "D"}


def branch_of(lr_name: str) -> str:
    return "best_e010" if str(lr_name).startswith("best_e010") else str(lr_name)


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E012")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E012", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if runs.empty:
        print("нет прогонов")
        return 1
    runs["lr_branch"] = runs.lr_name.map(branch_of)
    runs.to_csv(results / "depth_window.csv", index=False)

    if "merge_max_abs_diff" in runs:
        merge = runs.dropna(subset=["merge_max_abs_diff"])[
            ["variant", "window", "budget", "lr_branch", "seed", "merge_max_abs_diff", "merge_windows_checked", "merge_ok"]]
        merge.to_csv(results / "merge_check.csv", index=False)

    keys = ["lr_branch", "budget", "window", "variant"]
    summary = runs.groupby(keys).agg(
        n_seeds=("seed", "count"), seeds=("seed", lambda s: " ".join(str(v) for v in sorted(s))),
        hidden=("hidden", "first"), n_params=("n_params", "first"),
        macs_per_symbol=("macs_per_symbol", "first"), macs_per_symbol_merged=("macs_per_symbol_merged", "first"),
        test_valbest_mean=(METRIC, "mean"), test_valbest_min=(METRIC, "min"), test_valbest_max=(METRIC, "max"),
        test_final_mean=("test_reduction", "mean"), val_best_mean=("best_val_reduction", "mean"),
        n_diverged=("diverged", "sum"),
    ).reset_index()
    summary["seed_spread"] = summary.test_valbest_max - summary.test_valbest_min
    summary.to_csv(results / "depth_window_summary.csv", index=False)

    verdicts = []
    for (branch, window), part in runs.groupby(["lr_branch", "window"]):
        gaps = {}
        for budget in sorted(part.budget.unique()):
            values = {v: part[(part.budget == budget) & (part.variant == v)][METRIC].tolist() for v in VARIANTS}
            deep = []
            for two in ("v1_two_kerr_both", "v2_two_kerr_last"):
                c3, c4 = compare(values[two], values["v3_one_hidden"]), compare(values[two], values["v4_one_hidden_wide"])
                deep.append(c3["beats"] and c4["beats"])
                verdicts.append({"lr_branch": branch, "window": window, "budget": budget, "deep": two,
                                 "diff_vs_v3": c3["diff"], "spread_vs_v3": c3["spread"],
                                 "diff_vs_v4": c4["diff"], "spread_vs_v4": c4["spread"],
                                 "beats_v3_and_v4": c3["beats"] and c4["beats"]})
            gaps[budget] = compare(values["v2_two_kerr_last"], values["v3_one_hidden"])
            verdicts[-1]["window_verdict"] = verdicts[-2]["window_verdict"] = (
                "второй слой возвращает преимущество" if any(deep) else "не доказано")
        budgets = sorted(gaps)
        if len(budgets) >= 2:
            short, long = gaps[budgets[0]], gaps[budgets[-1]]
            effect = ("эффект оптимизации" if (short["diff"] > short["spread"] and long["within_spread"])
                      else "не доказано")
            for item in verdicts:
                if item["lr_branch"] == branch and item["window"] == window:
                    item["v2_minus_v3_short"] = short["diff"]
                    item["v2_minus_v3_long"] = long["diff"]
                    item["optimization_effect"] = effect
    verdict_frame = pd.DataFrame(verdicts)
    verdict_frame["rule_status"] = config["rule_draft"]["status"]
    verdict_frame.to_csv(results / "depth_window_verdicts.csv", index=False)
    plot(summary, figures)
    print(f"сводка: {results / 'depth_window_summary.csv'}; рисунок: {figures / 'E012_depth_window.png'}")
    return 0


def plot(summary: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    budgets = sorted(summary.budget.unique())
    windows = sorted(summary.window.unique())
    branches = [b for b in ("const_1e-3", "best_e010") if b in set(summary.lr_branch)]
    fig, axes = plt.subplots(len(branches), len(budgets), figsize=(5.4 * len(budgets), 4.0 * len(branches)),
                             sharey=True, squeeze=False)
    for row, branch in enumerate(branches):
        for col, budget in enumerate(budgets):
            ax = axes[row][col]
            part = summary[(summary.lr_branch == branch) & (summary.budget == budget)]
            for w_index, window in enumerate(windows):
                for variant in VARIANTS:
                    item = part[(part.window == window) & (part.variant == variant)]
                    if item.empty:
                        continue
                    item = item.iloc[0]
                    y, low, high = item.test_valbest_mean, item.test_valbest_min, item.test_valbest_max
                    ax.errorbar(item.macs_per_symbol_merged, y, yerr=[[y - low], [high - y]], fmt=MARKERS[variant],
                                color=plot_style.color(w_index), markersize=7, capsize=2, linewidth=1)
                    if item.macs_per_symbol != item.macs_per_symbol_merged:
                        ax.plot(item.macs_per_symbol, y, MARKERS[variant], markersize=7, markerfacecolor="none",
                                color=plot_style.color(w_index))
            ax.set_xscale("log")
            ax.set_title(f"{branch}, {budget // 1000}k steps")
            ax.set_xlabel("real multiplies per symbol (filled: after merge, hollow: as trained)")
        axes[row][0].set_ylabel("test BER reduction at best validation, %")
    handles = [Line2D([], [], color=plot_style.color(i), marker="o", linestyle="", label=f"window {w}")
               for i, w in enumerate(windows)]
    handles += [Line2D([], [], color=plot_style.MUTED, marker=MARKERS[v], linestyle="", label=v) for v in VARIANTS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figures / "E012_depth_window.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
