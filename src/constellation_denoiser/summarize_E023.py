"""E023 summary: do the family order and the differences survive a change of the split?

    PYTHONPATH=src python -m constellation_denoiser.summarize_E023   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/split_summary.csv (family x split: mean/min/max test),
results/robustness_verdicts.csv and figures/E023_split_robustness.png.
"""

from __future__ import annotations

import itertools
import json
import os

import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .runner import exp_dir, results_root

TEST = "test_reduction_valbest_select"


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E023")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E023", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    runs = runs[runs.stage == "main"] if not runs.empty else runs
    if runs.empty:
        print("нет прогонов")
        return 1
    runs.to_csv(results / "runs_table.csv", index=False)
    summary = runs.groupby(["family", "variant", "split_seed"]).agg(
        n_seeds=("seed", "count"), test_mean=(TEST, "mean"), test_min=(TEST, "min"), test_max=(TEST, "max"),
        macs=("macs_per_symbol_merged", "first")).reset_index()
    splits = sorted(summary.split_seed.unique())
    summary["member"] = summary.family + ": " + summary.variant
    orders = {s: list(summary[summary.split_seed == s].sort_values("test_mean", ascending=False).member) for s in splits}
    same_order = len({tuple(o) for o in orders.values()}) == 1
    pivot = summary.pivot_table(index="member", columns="split_seed", values="test_mean")
    sign_ok = all(len({(pivot.loc[a, s] > pivot.loc[b, s]) for s in splits}) == 1
                  for a, b in itertools.combinations(pivot.index, 2))
    ranges = (pivot.max(axis=1) - pivot.min(axis=1))
    small = bool((ranges < 2.0).all())
    verdicts = [{"hypothesis": "порядок участников одинаков во всех разбиениях", "detail": " | ".join(
                     f"{s}: " + " > ".join(o) for s, o in orders.items()), "verdict": "подтверждено" if same_order else "не доказано"},
                {"hypothesis": "знак каждой попарной разницы сохраняется", "detail": "", "verdict": "подтверждено" if sign_ok else "не доказано"},
                {"hypothesis": "разброс средних между разбиениями < 2 п.п. у каждого участника",
                 "detail": "; ".join(f"{f}: {r:.2f}" for f, r in ranges.items()), "verdict": "подтверждено" if small else "не доказано"}]
    summary.to_csv(results / "split_summary.csv", index=False)
    robust = same_order and sign_ok and small
    verdicts.append({"hypothesis": "вывод устойчив к смене разбиения", "detail": "все три условия",
                     "verdict": "подтверждено" if robust else "не доказано"})
    frame = pd.DataFrame(verdicts)
    frame["rule_status"] = config["rule_draft"]["status"]
    frame.to_csv(results / "robustness_verdicts.csv", index=False)
    plot(summary, splits, figures)
    print(f"сводка: {results / 'split_summary.csv'}; вердикты: {results / 'robustness_verdicts.csv'}")
    return 0


def plot(summary: pd.DataFrame, splits: list, figures) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply()
    families = list(summary.groupby("member").test_mean.mean().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for index, split in enumerate(splits[:8]):
        part = summary[summary.split_seed == split].set_index("member").reindex(families)
        x = [i + (index - (len(splits) - 1) / 2) * 0.22 for i in range(len(families))]
        ax.errorbar(x, part.test_mean, yerr=[(part.test_mean - part.test_min).clip(lower=0), (part.test_max - part.test_mean).clip(lower=0)],
                    fmt="o", color=plot_style.color(index), capsize=2, label=f"split seed {split}")
    ax.set_xticks(range(len(families)), families, rotation=20, ha="right")
    ax.set_ylabel("test BER reduction, % (min-max over seeds)")
    ax.set_title("Family results for three file-level splits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E023_split_robustness.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
