"""E016 summary: PhysKerr quality vs multiplies, draft-rule verdicts, figures.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E016   (SMOKE=1 reads results_smoke/)

Writes results/phys_kerr.csv (one row per run), results/phys_kerr_summary.csv,
results/phys_kerr_verdicts.csv, figures/E016_quality_vs_macs.png and
figures/E016_quality_vs_R.png.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .paths import EXPERIMENTS_DIR
from .q2 import q2_db
from .runner import exp_dir, results_root
from .summary_utils import compare

METRIC = "test_reduction_valbest_select"
SELECT = "best_val_reduction"


def q2_from_reduction(raw_ber: float, reduction: float) -> float:
    """Q2 gain of the model whose BER is raw*(1 - reduction/100) (exact: reduction was computed from it)."""
    return q2_db(raw_ber * (1 - reduction / 100)) - q2_db(raw_ber)


def group_summary(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    runs = runs.copy()
    runs["q2_gain_valbest"] = [q2_from_reduction(r, v) for r, v in zip(runs.raw_ber, runs[METRIC])]
    out = runs.groupby(keys, dropna=False).agg(
        n_seeds=("seed", "count"), seeds=("seed", lambda s: " ".join(str(v) for v in sorted(s))),
        macs_per_symbol=("macs_per_symbol", "first"), macs_merged=("macs_per_symbol_merged", "first"),
        n_params=("n_params", "first"),
        val_select_mean=(SELECT, "mean"), test_mean=(METRIC, "mean"), test_min=(METRIC, "min"),
        test_max=(METRIC, "max"), test_final_mean=("test_reduction", "mean"),
        q2_gain_mean=("q2_gain_valbest", "mean"), q2_gain_min=("q2_gain_valbest", "min"),
        q2_gain_max=("q2_gain_valbest", "max"), n_diverged=("diverged", "sum"),
    ).reset_index()
    out["seed_spread"] = out.test_max - out.test_min
    return out


def e012_points() -> pd.DataFrame:
    path = EXPERIMENTS_DIR / "E012_depth_vs_window" / "results" / "depth_window.csv"
    if not path.is_file():
        return pd.DataFrame()
    runs = pd.read_csv(path)
    runs = runs[runs.lr_branch == "const_1e-3"]
    runs = runs.assign(q2_gain_valbest=[q2_from_reduction(r, v) for r, v in zip(runs.raw_ber, runs[METRIC])])
    return runs.groupby(["variant", "window", "budget"]).agg(
        macs=("macs_per_symbol_merged", "first"), test_mean=(METRIC, "mean"), test_min=(METRIC, "min"),
        test_max=(METRIC, "max"), q2_gain_mean=("q2_gain_valbest", "mean")).reset_index()


def runs_of(runs: pd.DataFrame, **match) -> list[float]:
    part = runs
    for key, value in match.items():
        part = part[part[key] == value]
    return part[METRIC].tolist()


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E016")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E016", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if runs.empty:
        print("нет прогонов")
        return 1
    runs.to_csv(results / "phys_kerr.csv", index=False)
    phys = runs[runs.model == "phys_kerr"].copy()
    control = runs[runs.stage == "control"]
    summary = group_summary(phys, ["variant", "lr_name", "spec_n_alpha", "spec_fir_len", "spec_sym"])
    summary.sort_values(["lr_name", "spec_fir_len", "spec_sym", "spec_n_alpha"]).to_csv(
        results / "phys_kerr_summary.csv", index=False)
    control_summary = group_summary(control, ["variant", "lr_name"]) if not control.empty else pd.DataFrame()

    verdicts = []
    rule = config["rule_draft"]
    # physics sufficient: best by validation among <= 10% of 6480 multiplies, vs complex_a on test
    nonlinear = summary[summary.spec_n_alpha > 0]
    cheap = nonlinear[nonlinear.macs_per_symbol <= 0.1 * 6480]
    if not cheap.empty and not control.empty:
        best = cheap.loc[cheap.val_select_mean.idxmax()]
        mine = runs_of(phys, variant=best.variant, lr_name=best.lr_name)
        ref = control[METRIC].tolist()
        diff = float(np.mean(mine) - np.mean(ref))
        verdicts.append({"hypothesis": "физика достаточна", "candidate": f"{best.variant} / {best.lr_name}",
                         "candidate_macs": best.macs_per_symbol, "reference": "complex_a const 1e-3 20k",
                         "reference_macs": 6480, "diff_pp": diff, "seed_spread_pp": compare(mine, ref)["spread"],
                         "n_seeds": len(mine), "threshold": "diff >= -2 p.p.",
                         "verdict": "подтверждено" if diff >= -2.0 else "не доказано"})
    # physics useful: one comparison - the R > 1 configuration with the best mean validation among those that
    # cost no more than FIR(31) + one Kerr coefficient of the same sym and lr branch, vs that reference
    pool = []
    for (lr_name, sym), part in nonlinear.groupby(["lr_name", "spec_sym"]):
        reference = part[(part.spec_n_alpha == 1) & (part.spec_fir_len == 31)]
        if reference.empty:
            continue
        candidates = part[(part.spec_n_alpha > 1) & (part.macs_per_symbol <= reference.iloc[0].macs_per_symbol)]
        pool += [(row, reference.iloc[0]) for _, row in candidates.iterrows()]
    if pool:
        best, reference = max(pool, key=lambda pair: pair[0].val_select_mean)
        result = compare(runs_of(phys, variant=best.variant, lr_name=best.lr_name),
                         runs_of(phys, variant=reference.variant, lr_name=reference.lr_name), min_pp=1.0)
        verdicts.append({"hypothesis": "физика полезна", "candidate": f"{best.variant} / {best.lr_name}",
                         "candidate_macs": best.macs_per_symbol, "reference": f"{reference.variant} / {reference.lr_name}",
                         "reference_macs": reference.macs_per_symbol, "diff_pp": result["diff"],
                         "seed_spread_pp": result["spread"], "n_seeds": result["n_a"],
                         "threshold": ">= 1 p.p. and > seed spread",
                         "verdict": "подтверждено" if result["beats"] else "не доказано"})
    verdict_frame = pd.DataFrame(verdicts)
    verdict_frame["rule_status"] = rule["status"]
    verdict_frame.to_csv(results / "phys_kerr_verdicts.csv", index=False)

    plot_vs_macs(summary, control_summary, e012_points(), figures)
    plot_vs_r(summary, figures)
    print(f"сводка: {results / 'phys_kerr_summary.csv'}; вердикты: {results / 'phys_kerr_verdicts.csv'}")
    return 0


def plot_vs_macs(summary: pd.DataFrame, control: pd.DataFrame, e012: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    branches = list(dict.fromkeys(summary.lr_name))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (column, low, high, label) in zip(axes, (("test_mean", "test_min", "test_max", "test BER reduction, %"),
                                                     ("q2_gain_mean", "q2_gain_min", "q2_gain_max", "Q2 gain, dB"))):
        for index, branch in enumerate(branches):
            part = summary[summary.lr_name == branch]
            marker = np.where(part.spec_n_alpha == 0, "x", "o")
            for m in ("o", "x"):
                sel = part[marker == m]
                ax.errorbar(sel.macs_per_symbol, sel[column], yerr=[np.clip(sel[column] - sel[low], 0, None),
                            np.clip(sel[high] - sel[column], 0, None)], fmt=m, color=plot_style.color(index),
                            markersize=5, capsize=1.5, linewidth=0.8, alpha=0.9)
        offset = len(branches)
        if not control.empty:
            for _, row in control.iterrows():
                ax.errorbar(row.macs_merged, row[column], yerr=[[max(row[column] - row[low], 0)], [max(row[high] - row[column], 0)]],
                            fmt="s", color=plot_style.color(offset), markersize=7, capsize=2)
        if not e012.empty and column == "test_mean":
            for _, row in e012[e012.budget == 20000].iterrows():
                ax.plot(row.macs, row.test_mean, "D", markerfacecolor="none", color=plot_style.color(offset + 1),
                        markersize=6)
        if not e012.empty and column == "q2_gain_mean":
            for _, row in e012[e012.budget == 20000].iterrows():
                ax.plot(row.macs, row.q2_gain_mean, "D", markerfacecolor="none", color=plot_style.color(offset + 1),
                        markersize=6)
        floor = -50.0 if column == "test_mean" else -3.0
        values = pd.concat([summary[column]] + ([control[column]] if not control.empty else [])
                           + ([e012[column]] if not e012.empty else []))
        clipped = summary[summary[column] < floor]
        if len(clipped):  # collapsed runs (e.g. BER ~ 0.5) would flatten the axis: draw them on the floor
            ax.plot(clipped.macs_per_symbol, np.full(len(clipped), floor), "v", color=plot_style.MUTED, markersize=5)
            ax.annotate(f"{len(clipped)} configs below {floor:g} (collapsed)", (0.01, 0.02), xycoords="axes fraction",
                        fontsize=7, color=plot_style.MUTED)
        top = float(np.nanmax(values)) if len(values) else 1.0
        ax.set_ylim(floor - 0.05 * abs(floor), top + 0.08 * (top - floor))
        ax.set_xscale("log")
        ax.set_xlabel("real multiplies per symbol (macs_phys; MLP after merge)")
        ax.set_ylabel(label)
    handles = [Line2D([], [], color=plot_style.color(i), marker="o", linestyle="", label=f"PhysKerr {b}")
               for i, b in enumerate(branches)]
    handles += [Line2D([], [], color=plot_style.MUTED, marker="x", linestyle="", label="FIR only (R = 0)"),
                Line2D([], [], color=plot_style.color(len(branches)), marker="s", linestyle="", label="complex_a (E016)"),
                Line2D([], [], color=plot_style.color(len(branches) + 1), marker="D", markerfacecolor="none",
                       linestyle="", label="E012 MLPs, 20k (both, a, one_hidden, wide; W 31/65/127)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("PhysKerr vs complex MLPs: test at best validation, 20k steps (bars: min-max over seeds)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(figures / "E016_quality_vs_macs.png", dpi=160)
    plt.close(fig)


def plot_vs_r(summary: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply()
    branches = list(dict.fromkeys(summary.lr_name))
    fig, axes = plt.subplots(1, len(branches), figsize=(4.8 * len(branches), 3.8), sharey=True, squeeze=False)
    for ax, branch in zip(axes[0], branches):
        part = summary[(summary.lr_name == branch) & (summary.spec_n_alpha > 0)]
        index = 0
        for fir_len in sorted(part.spec_fir_len.unique()):
            for sym in (False, True):
                line = part[(part.spec_fir_len == fir_len) & (part.spec_sym == sym)].sort_values("spec_n_alpha")
                if line.empty:
                    continue
                ax.plot(line.spec_n_alpha, line.test_mean, marker="o", color=plot_style.color(index),
                        linestyle="-" if sym else "--", label=f"L={fir_len}, {'sym' if sym else 'no sym'}")
                index += 1
        ax.set_xscale("log", base=2)
        ax.set_title(branch)
        ax.set_xlabel("number of alpha coefficients R")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("mean test BER reduction, %")
    low = summary[summary.spec_n_alpha > 0].test_mean
    if len(low) and low.min() < -50:
        for ax in axes[0]:
            ax.set_ylim(bottom=-50)
    fig.tight_layout()
    fig.savefig(figures / "E016_quality_vs_R.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
