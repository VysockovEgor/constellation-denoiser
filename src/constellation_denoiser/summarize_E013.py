"""E013 summary: transformer vs complex MLP, BER reduction against multiplies per symbol.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E013   (SMOKE=1 reads results_smoke/)

Writes results/transformer_runs.csv, results/comparison_points.csv,
results/transformer_verdicts.csv and figures/E013_transformer_vs_macs.png.
Comparison inputs that do not exist yet (E011, E012) are skipped.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import model_zoo, plot_style
from .merge_runs import load_runs
from .paths import EXPERIMENTS_DIR
from .runner import exp_dir, results_root
from .summary_utils import compare

FAMILY = {"transformer": 0, "complex MLP": 1, "real tanh MLP": 2}


def point(source, family, model, window, n_train, steps, macs, values, comparable, note="") -> dict:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {"source": source, "family": family, "model": model, "window": window, "n_train": n_train,
            "steps": steps, "macs_per_symbol": macs, "n_seeds": len(values),
            "test_mean": float(np.mean(values)) if values else np.nan,
            "test_min": min(values) if values else np.nan, "test_max": max(values) if values else np.nan,
            "values": " ".join(f"{v:.4f}" for v in values), "comparable": comparable, "note": note}


def e011_points() -> tuple[list[dict], pd.DataFrame | None]:
    base = EXPERIMENTS_DIR / "E011_long_runs_convergence" / "results"
    curves_path, runs_path = base / "long_runs_curves.csv", base / "long_runs.csv"
    if not curves_path.is_file() or not runs_path.is_file():
        return [], None
    curves, runs = pd.read_csv(curves_path), pd.read_csv(runs_path)
    curves = curves[curves.lr_branch == "const_1e-3"].dropna(subset=["test_reduction"])
    macs = runs.groupby("variant").macs_per_symbol_merged.first().to_dict()
    points = []
    for (variant, n_train, step), group in curves[curves.step.isin([20000, 100000])].groupby(
            ["variant", "n_train_nominal", "step"]):
        family = "real tanh MLP" if variant == "real_tanh" else "complex MLP"
        points.append(point("E011", family, variant, 31, int(n_train), int(step), macs.get(variant),
                            group.sort_values("seed").test_reduction, True, "const 1e-3, test at this step"))
    return points, curves


def reference_points() -> list[dict]:
    points = []
    e009 = EXPERIMENTS_DIR / "E009_kerr_ablation" / "results" / "complex_mlp_ablation_summary.csv"
    for _, row in pd.read_csv(e009).iterrows():
        if row.variant == "c":
            continue
        points.append(point("E009", "complex MLP", f"E009 {row.variant}", 31, 30, 20000,
                            row.merged_total_real_multiplies, [row.min_improvement_pct, row.max_improvement_pct],
                            True, "min and max of 2 seeds; test at best validation"))
    e007 = EXPERIMENTS_DIR / "E007_window_width_sweep" / "results" / "complex_mlp_sweep_summary.csv"
    for _, row in pd.read_csv(e007).iterrows():
        points.append(point("E007", "complex MLP", f"E007 both h{row.hidden}", int(row.window), 30, 20000,
                            row.total_real_multiplies, [row.min_improvement_pct, row.max_improvement_pct],
                            True, "min and max of 2 seeds; test at best validation"))
    e012 = EXPERIMENTS_DIR / "E012_depth_vs_window" / "results" / "depth_window_summary.csv"
    if e012.is_file():
        for _, row in pd.read_csv(e012).query("lr_branch == 'const_1e-3'").iterrows():
            points.append(point("E012", "complex MLP", f"E012 {row.variant}", int(row.window), 30, int(row.budget),
                                row.macs_per_symbol_merged, [row.test_valbest_min, row.test_valbest_max], True,
                                "min and max over seeds; test at best validation"))
    e003 = pd.read_csv(EXPERIMENTS_DIR / "E003_large_scale_pilot" / "results" / "large_scale_pilot_summary.csv")
    for model, group in e003.groupby("model"):
        hidden = int(group.hidden_per_layer.iloc[0])
        family = "real tanh MLP" if model == "real" else "complex MLP"
        macs = (model_zoo.real_mlp_macs(31, hidden) if model == "real"
                else model_zoo.complex_two_hidden_macs(31, hidden))
        points.append(point("E003", family, f"E003 {model} h{hidden}", 31, 210, None, macs,
                            group.improvement_pct_vs_raw, False, "reference: 40 epochs (~134k steps), mean over files"))
    e001 = pd.read_csv(EXPERIMENTS_DIR / "E001_pilot_models" / "results" / "model_comparison_summary.csv")
    row = e001[e001.model == "Transformer"].iloc[0]
    macs = model_zoo.transformer_macs(7, 32, 2, 2)
    points.append(point("E001", "transformer", "E001 transformer d32 L2", 7, 7, None, macs,
                        [100 * (row.test_raw_ber - row.test_ber_mean) / row.test_raw_ber], False,
                        "reference: PR1-10 split, <=30 epochs, mean of 3 seeds"))
    return points


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E013")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E013", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if runs.empty:
        print("нет прогонов")
        return 1
    runs.to_csv(results / "transformer_runs.csv", index=False)
    own = runs[runs.model == "transformer"]
    control = runs[runs.stage == "control"]

    points = []
    for (variant, window, n_train, budget), group in own.groupby(["variant", "window", "n_train_nominal", "budget"]):
        diverged = bool(group.diverged.any())
        points.append(point("E013", "transformer", variant, int(window), int(n_train), int(budget),
                            int(group.macs_per_symbol.iloc[0]), group.sort_values("seed").test_reduction,
                            not diverged, "diverged seed" if diverged else "final step (cosine to 0)"))
    for (variant, budget), group in control.groupby(["variant", "budget"]):
        points.append(point("E013 control", "complex MLP", f"control {variant}", 31, 30, int(budget),
                            int(group.macs_per_symbol_merged.iloc[0]), group.test_reduction_valbest_select, True,
                            "test at best validation"))
    complex_points, _ = e011_points()
    points += complex_points + reference_points()
    table = pd.DataFrame(points)
    table.to_csv(results / "comparison_points.csv", index=False)

    verdicts = []
    for _, row in table[(table.source == "E013") & (table.window == 31)
                        & (table.n_train == max(config["stages"]["stageB"]["n_train_files"]))].iterrows():
        rivals = table[(table.source == "E011") & (table.family == "complex MLP") & (table.window == 31)
                       & (table.n_train == row.n_train) & (table.steps == row.steps)
                       & (table.macs_per_symbol <= row.macs_per_symbol)]
        if not row.comparable:
            verdicts.append({**row[["model", "window", "n_train", "steps"]].to_dict(), "verdict": "неравное сравнение (дивергенция)"})
            continue
        if rivals.empty:
            verdicts.append({**row[["model", "window", "n_train", "steps"]].to_dict(),
                             "verdict": "нет сопоставимых данных комплексной MLP (нужен E011 с тем же числом шагов)"})
            continue
        best = rivals.loc[rivals.test_mean.idxmax()]
        result = compare([float(v) for v in row["values"].split()], [float(v) for v in best["values"].split()])
        verdicts.append({**row[["model", "window", "n_train", "steps", "macs_per_symbol"]].to_dict(),
                         "rival": best.model, "rival_macs": best.macs_per_symbol, "diff_pp": result["diff"],
                         "seed_spread_pp": result["spread"],
                         "verdict": "трансформер превосходит" if result["beats"] else "не доказано"})
    verdict_frame = pd.DataFrame(verdicts)
    verdict_frame["rule_status"] = config["rule_draft"]["status"]
    verdict_frame.to_csv(results / "transformer_verdicts.csv", index=False)
    plot(table, figures)
    print(f"сравнение: {results / 'comparison_points.csv'}; рисунок: {figures / 'E013_transformer_vs_macs.png'}")
    return 0


def plot(table: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    budgets = [20000, 100000]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, budget in zip(axes, budgets):
        part = table[(table.steps == budget) | table.steps.isna()]
        for _, row in part.iterrows():
            if not np.isfinite(row.test_mean) or not row.macs_per_symbol:
                continue
            c = plot_style.color(FAMILY[row.family])
            marker = "s" if row.n_train >= 210 else "o"
            reference = row.source in ("E003", "E001")
            low, high = row.test_mean - row.test_min, row.test_max - row.test_mean
            ax.errorbar(row.macs_per_symbol, row.test_mean, yerr=[[low], [high]], fmt=marker, color=c,
                        markerfacecolor="none" if reference else c, markersize=6, capsize=2, linewidth=1)
            if row.source == "E013":
                ax.annotate(f"W{row.window} {row.model.replace('transformer_', '')}",
                            (row.macs_per_symbol, row.test_mean), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=plot_style.MUTED)
        ax.set_xscale("log")
        ax.set_title(f"{budget // 1000}k steps (hollow: reference, other protocol)")
        ax.set_xlabel("real multiplies per symbol (complex MLP after merge)")
    axes[0].set_ylabel("test BER reduction, %")
    handles = [Line2D([], [], color=plot_style.color(i), marker="o", linestyle="", label=f) for f, i in FAMILY.items()]
    handles += [Line2D([], [], color=plot_style.MUTED, marker="o", linestyle="", label="N=30 (E001: 7)"),
                Line2D([], [], color=plot_style.MUTED, marker="s", linestyle="", label="N=210")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(figures / "E013_transformer_vs_macs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
