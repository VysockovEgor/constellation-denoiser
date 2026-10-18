"""E010 summary: T90/T95, rule verdict against the const 1e-3 control, figures.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E010   (SMOKE=1 reads results_smoke/)

Writes results/lr_sweep.csv (runs + T90/T95), results/lr_sweep_summary.csv,
figures/E010_val_curves.png, figures/E010_t95_vs_peak_lr.png.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from . import plot_style
from .lr_schedules import LRConfig
from .merge_runs import load_runs
from .runner import exp_dir, results_root, steps_to_fraction


def curve_rows(results, run_id: str) -> list[dict]:
    frame = pd.read_csv(results / "curves" / f"{run_id}.csv")
    return frame.where(frame.notna(), None).to_dict("records")


def nan_mean(values) -> float:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E010")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E010", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if runs.empty:
        print("нет прогонов")
        return 1
    control_name = LRConfig.from_dict(config["base_lr"]).tag()
    full = int(runs.budget.max())

    targets, rows = {}, []
    for arch, group in runs[(runs.lr_name == control_name) & (runs.budget == full)].groupby("variant"):
        targets[arch] = float(group.val_reduction.mean())
    for _, run in runs.iterrows():
        curve = curve_rows(results, run.run_id)
        target = targets.get(run.variant)
        record = run.to_dict()
        for name, fraction in (("T90", 0.9), ("T95", 0.95)):
            record[name] = steps_to_fraction(curve, fraction * target) if target is not None else None
        record["control_target_val_reduction"] = target
        rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(results / "lr_sweep.csv", index=False)

    rule = config["rule_draft"]
    summary = []
    for (arch, lr_name, steps), group in table.groupby(["variant", "lr_name", "budget"]):
        t95 = [v for v in group.T95 if v is not None and not pd.isna(v)]
        summary.append({
            "variant": arch, "lr_name": lr_name, "budget": steps, "n_seeds": len(group),
            "seeds": " ".join(str(s) for s in sorted(group.seed)),
            "lr_peak_lr": group.lr_peak_lr.iloc[0], "lr_schedule": group.lr_schedule.iloc[0],
            "lr_warmup": group.lr_warmup.iloc[0], "lr_c_mult": group.lr_c_mult.iloc[0],
            "val_reduction_mean": group.val_reduction.mean(), "val_reduction_min": group.val_reduction.min(),
            "val_reduction_max": group.val_reduction.max(), "val_reduction_std": group.val_reduction.std(ddof=0),
            "test_reduction_mean": group.test_reduction.mean(), "test_reduction_min": group.test_reduction.min(),
            "test_reduction_max": group.test_reduction.max(), "test_reduction_std": group.test_reduction.std(ddof=0),
            "test_reduction_valbest_select_mean": group.test_reduction_valbest_select.mean(),
            "T90_mean": nan_mean(group.T90), "T95_mean": nan_mean(group.T95),
            "T95_reached_seeds": len(t95), "n_diverged": int(group.diverged.sum()),
            "is_control": lr_name == control_name and steps == full,
        })
    summary = pd.DataFrame(summary)

    verdicts = []
    for _, row in summary.iterrows():
        control = summary[(summary.variant == row.variant) & summary.is_control]
        if row.is_control:
            verdicts.append("control")
            continue
        if control.empty:
            verdicts.append("no control")
            continue
        control = control.iloc[0]
        members = table[(table.variant == row.variant) & (table.lr_name == row.lr_name)
                        & (table.budget == row.budget)]
        faster = row.T95_reached_seeds == row.n_seeds and row.T95_mean <= 0.7 * control.T95_mean
        accurate = row.test_reduction_mean >= control.test_reduction_mean - 0.5
        stable = row.n_diverged == 0 and bool((members.val_reduction >= control.val_reduction_mean - 2.0).all())
        verdicts.append("better than control" if (faster and accurate and stable) else "not proven")
    summary["verdict_draft_rule"] = verdicts
    summary["rule_status"] = rule["status"]
    summary.sort_values(["variant", "budget", "val_reduction_mean"], ascending=[True, True, False]).to_csv(
        results / "lr_sweep_summary.csv", index=False)

    plot_curves(table, summary, results, figures, full, control_name)
    plot_t95(summary, figures, full)
    print(f"сводка: {results / 'lr_sweep_summary.csv'}; рисунки: {figures}")
    return 0


def plot_curves(table, summary, results, figures, full, control_name) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply()
    archs = sorted(table.variant.unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(5.2 * len(archs), 3.6), sharey=True, squeeze=False)
    for ax, arch in zip(axes[0], archs):
        pool = summary[(summary.variant == arch) & (summary.budget == full)]
        best = pool[~pool.is_control].nlargest(3, "val_reduction_mean").lr_name.tolist()
        for index, lr_name in enumerate([control_name] + best):
            members = table[(table.variant == arch) & (table.lr_name == lr_name) & (table.budget == full)]
            curves = [pd.read_csv(results / "curves" / f"{rid}.csv").query("step > 0")[["step", "val_reduction"]].set_index("step")
                      for rid in members.run_id]
            if not curves:
                continue
            joined = pd.concat(curves, axis=1).dropna()
            label = ("control " if lr_name == control_name else "") + lr_name + f" (n={len(curves)})"
            plot_style.band(ax, joined.index, joined.mean(axis=1), joined.min(axis=1), joined.max(axis=1),
                            index, label)
        ax.set_title(f"{arch}: validation BER reduction")
        ax.set_xlabel("training step")
        ax.legend(fontsize=7, loc="lower right")
    axes[0][0].set_ylabel("validation reduction, %")
    fig.tight_layout()
    fig.savefig(figures / "E010_val_curves.png", dpi=160)
    plt.close(fig)


def plot_t95(summary, figures, full) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply()
    stage1 = summary[(summary.budget == full) & (summary.lr_warmup == 0) & (summary.lr_c_mult == 1)]
    archs = sorted(stage1.variant.unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(4.6 * len(archs), 3.4), sharey=True, squeeze=False)
    for ax, arch in zip(axes[0], archs):
        for index, schedule in enumerate(["const", "cosine_to_zero", "cosine_to_floor"]):
            part = stage1[(stage1.variant == arch) & (stage1.lr_schedule == schedule)].sort_values("lr_peak_lr")
            if part.empty:
                continue
            ax.plot(part.lr_peak_lr, part.T95_mean, marker="o", color=plot_style.color(index), label=schedule)
        ax.set_xscale("log")
        ax.set_title(f"{arch}: steps to 95% of control")
        ax.set_xlabel("peak lr")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("mean T95, steps (missing = not reached)")
    fig.tight_layout()
    fig.savefig(figures / "E010_t95_vs_peak_lr.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
