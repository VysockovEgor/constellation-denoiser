"""Shared helpers of the E024-E033 summaries: per-run table, aggregation over seeds, interval checks."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .merge_runs import load_runs
from .q2 import q2_db
from .runner import exp_dir, results_root

TEST, VAL = "test_reduction_valbest_select", "best_val_reduction"


def setup(exp: str):
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir(exp)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root(exp, smoke)
    figures = results / "figures" if smoke else directory / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if not runs.empty:
        q2 = runs.get("valbest_select_q2_gain_db")
        if q2 is None or q2.isna().all():
            q2 = pd.Series([q2_db(r * (1 - t / 100)) - q2_db(r) if t == t else np.nan
                            for r, t in zip(runs.raw_ber, runs[TEST])], index=runs.index)
        runs["q2"] = q2
        from .rm_audit import rm_linear, spec_of_row

        runs["rm_total"] = runs.macs_per_symbol
        runs["rm_linear"] = [rm_linear(spec_of_row(r), r.macs_per_symbol) for _, r in runs.iterrows()]
        runs.to_csv(results / "runs_table.csv", index=False)
    return smoke, directory, config, results, figures, runs


def aggregate(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in runs.columns]
    out = runs.groupby(keys, dropna=False).agg(
        n_seeds=("seed", "nunique"), diverged=("diverged", "sum"), val_mean=(VAL, "mean"), val_min=(VAL, "min"),
        val_max=(VAL, "max"), test_mean=(TEST, "mean"), test_min=(TEST, "min"), test_max=(TEST, "max"),
        q2_mean=("q2", "mean"), q2_min=("q2", "min"), q2_max=("q2", "max"),
        rm=("macs_per_symbol_merged", "first"), rm_trained=("macs_per_symbol", "first")).reset_index()
    return out


def cyc(index: int) -> str:
    """Colour slot modulo the 8-colour palette; series beyond 8 are told apart by marker or hatch as well."""
    from . import plot_style

    return plot_style.SERIES[index % len(plot_style.SERIES)]


def disjoint(a_min, a_max, b_min, b_max) -> bool:
    return bool(a_min > b_max or b_min > a_max)


def better(a: pd.Series, b: pd.Series, threshold: float, metric: str = "test") -> tuple[float, bool]:
    """(difference a - b of the means, a better by >= threshold with non-overlapping min-max intervals)."""
    diff = float(a[f"{metric}_mean"] - b[f"{metric}_mean"])
    return diff, bool(diff >= threshold and disjoint(a[f"{metric}_min"], a[f"{metric}_max"], b[f"{metric}_min"],
                                                     b[f"{metric}_max"]))


def write_verdicts(rows: list[dict], config: dict, path) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["hypothesis", "value", "verdict", "detail"] if rows else
                         ["hypothesis", "value", "verdict", "detail"])
    frame["rule_status"] = config["rule_draft"]["status"]
    frame.to_csv(path, index=False)
    return frame


def verdict(ok: bool) -> str:
    return "подтверждено" if ok else "не доказано"


def bar_compare(frame: pd.DataFrame, label: str, groups: list[str], path, title: str, ylabel: str) -> None:
    """Grouped bars of test_mean with min-max whiskers: x = ``label`` values, one bar per group column value."""
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    labels = list(dict.fromkeys(frame[label]))
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(labels) * max(1, len(groups))), 4.6))
    width = 0.8 / max(1, len(groups))
    hatches = ["", "//", "..", "xx", "\\\\", "oo"]
    for gi, group in enumerate(groups):
        part = frame[frame.group == group].set_index(label).reindex(labels)
        x = np.arange(len(labels)) + (gi - (len(groups) - 1) / 2) * width
        ax.bar(x, part.test_mean, width, color=plot_style.color(gi), hatch=hatches[gi % len(hatches)],
               edgecolor=plot_style.TEXT, linewidth=0.4, label=group,
               yerr=[(part.test_mean - part.test_min).clip(lower=0), (part.test_max - part.test_mean).clip(lower=0)],
               capsize=2)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if len(groups) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
