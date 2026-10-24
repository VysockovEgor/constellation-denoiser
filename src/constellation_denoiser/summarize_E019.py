"""E019 summary: long confirmation runs, plateau as in E011, 100k and 200k points.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E019   (SMOKE=1 reads results_smoke/)

For const_1e-3 only the 200k run exists; its 100k point is read from the curve
(identical to a 100k run with constant lr). Writes results/confirm.csv,
results/confirm_summary.csv and figures/E019_long_curves.png.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .q2 import q2_db
from .runner import exp_dir, results_root
from .summarize_E011 import curve_metrics


def horizon_point(curve: pd.DataFrame, horizon: int, select_every: int, raw_ber: float) -> dict:
    part = curve[(curve.step > 0) & (curve.step <= horizon)]
    selectable = part[(part.step % select_every == 0) & part.test_reduction.notna()]
    if selectable.empty:
        return {}
    best = selectable.loc[selectable.val_reduction.idxmax()]
    last = part.iloc[-1]
    return {"horizon": horizon, "best_val_step": int(best.step), "best_val_reduction": float(best.val_reduction),
            "test_reduction_valbest": float(best.test_reduction),
            "valbest_q2_gain_db": q2_db(raw_ber * (1 - best.test_reduction / 100)) - q2_db(raw_ber),
            "final_val_reduction": float(last.val_reduction), "final_test_reduction": float(last.test_reduction)}


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E019")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E019", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    runs = runs[runs.stage == "main"] if not runs.empty else runs
    if runs.empty:
        print("нет длинных прогонов")
        return 1
    rows, curves = [], []
    for _, run in runs.iterrows():
        curve = pd.read_csv(results / "curves" / f"{run.run_id}.csv")
        branch = "const_1e-3" if run.lr_name == "const_1e-3" else "best_e010"
        horizons = sorted(set(config["best_steps"]) | set(config["const_steps"])) if branch == "const_1e-3" \
            else [int(run.budget)]
        scale = int(run.planned_steps) / int(run.budget)            # 1 for real runs, < 1 in smoke
        for horizon in horizons:
            limit = int(round(horizon * scale))
            point = horizon_point(curve, limit, int(run.select_every), float(run.raw_ber))
            if not point:
                continue
            metrics = curve_metrics(curve[curve.step <= limit], int(run.eval_every),
                                    int(run.eval_every) if smoke else 10_000)
            rows.append({"run_id": run.run_id, "member": run.variant, "lr_branch": branch, "lr_name": run.lr_name,
                         "seed": run.seed, "budget": int(run.budget), **point, "nominal_horizon": horizon,
                         "derived_from_longer_run": horizon < int(run.budget), "macs_per_symbol_merged":
                         run.macs_per_symbol_merged, **{k: metrics.get(k) for k in
                         ("plateau", "gain_last_50k_pp", "y_inf", "t98", "falling")}})
        curves.append(curve.assign(member=run.variant, lr_branch=branch, seed=run.seed, budget=int(run.budget)))
    table = pd.DataFrame(rows)
    table.to_csv(results / "confirm.csv", index=False)

    summary = []
    stacked = pd.concat(curves, ignore_index=True)
    for (member, branch, horizon), group in table.groupby(["member", "lr_branch", "nominal_horizon"]):
        source = stacked[(stacked.member == member) & (stacked.lr_branch == branch)
                         & (stacked.budget == group.budget.iloc[0])]
        mean_curve = source.groupby("step", as_index=False)[["val_reduction", "test_reduction"]].mean()
        scale = horizon / group.budget.iloc[0]
        limit = int(round(mean_curve.step.max() * scale)) if len(mean_curve) else 0
        metrics = curve_metrics(mean_curve[mean_curve.step <= limit], int(config["eval_every"]) if not smoke else 100,
                                100 if smoke else 10_000)
        if metrics.get("plateau") is None:
            text = "меньше 50000 шагов: плато не оценивается"
        elif metrics["plateau"]:
            text = f"плато (прирост {metrics['gain_last_50k_pp']:.2f} п.п. за 50k)"
        else:
            text = f"плато не достигнуто за {horizon} шагов"
        summary.append({"member": member, "lr_branch": branch, "horizon": horizon, "n_seeds": len(group),
                        "derived_from_200k_curve": bool(group.derived_from_longer_run.all()),
                        "macs_per_symbol_merged": group.macs_per_symbol_merged.iloc[0],
                        "test_valbest_mean": group.test_reduction_valbest.mean(),
                        "test_valbest_min": group.test_reduction_valbest.min(),
                        "test_valbest_max": group.test_reduction_valbest.max(),
                        "valbest_q2_gain_db_mean": group.valbest_q2_gain_db.mean(),
                        "final_test_mean": group.final_test_reduction.mean(),
                        "y_inf_mean_curve": metrics.get("y_inf"), "plateau_text": text,
                        "rule_status": config["rule_draft"]["status"]})
    pd.DataFrame(summary).to_csv(results / "confirm_summary.csv", index=False)
    plot(stacked, figures)
    print(f"сводка: {results / 'confirm_summary.csv'}")
    return 0


def plot(stacked: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    longest = stacked[stacked.budget == stacked.budget.max()]
    members = list(dict.fromkeys(longest.member))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for index, member in enumerate(members[:8]):
        for style, branch in enumerate(("const_1e-3", "best_e010")):
            part = longest[(longest.member == member) & (longest.lr_branch == branch) & (longest.step > 0)]
            if part.empty:
                continue
            stats = part.groupby("step").val_reduction.agg(["mean", "min", "max"])
            plot_style.band(ax, stats.index, stats["mean"], stats["min"], stats["max"], index, None,
                            plot_style.LINESTYLES[style])
    ax.set_xlabel("training step")
    ax.set_ylabel("validation BER reduction, %")
    ax.set_title("Long confirmation runs (band: min-max over seeds)")
    handles = [Line2D([], [], color=plot_style.color(i), label=m) for i, m in enumerate(members[:8])]
    handles += [Line2D([], [], color=plot_style.MUTED, linestyle="-", label="const 1e-3"),
                Line2D([], [], color=plot_style.MUTED, linestyle="--", label="best_e010")]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "E019_long_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
