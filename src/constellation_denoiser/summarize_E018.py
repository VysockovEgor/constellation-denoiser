"""E018 summary: DBP depth (Ns steps), layer-wise vs joint training, verdicts, figures.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E018   (SMOKE=1 reads results_smoke/)

Writes results/dbp_depth.csv, results/dbp_depth_summary.csv,
results/dbp_depth_verdicts.csv, figures/E018_quality_vs_ns.png and
figures/E018_quality_vs_macs.png.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .runner import exp_dir, results_root
from .summarize_E016 import METRIC, e012_points, group_summary
from .summary_utils import compare


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E018")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E018", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if runs.empty:
        print("нет прогонов")
        return 1
    runs.to_csv(results / "dbp_depth.csv", index=False)
    dbp = runs[runs.model == "phys_dbp"].copy()
    dbp["mode"] = np.where(dbp.variant.str.endswith("joint_from_scratch"), "joint_from_scratch", "layerwise")
    keys = ["variant", "mode", "spec_n_steps", "spec_fir_len", "spec_n_alpha"]
    summary = group_summary(dbp, keys)
    summary.sort_values(["mode", "spec_fir_len", "spec_n_alpha", "spec_n_steps"]).to_csv(
        results / "dbp_depth_summary.csv", index=False)
    controls = runs[runs.stage == "control"]
    control_summary = group_summary(controls, ["variant", "lr_name"]) if not controls.empty else pd.DataFrame()

    def values(mode, n_steps, fir_len, n_alpha):
        part = dbp[(dbp["mode"] == mode) & (dbp.spec_n_steps == n_steps) & (dbp.spec_fir_len == fir_len)
                   & (dbp.spec_n_alpha == n_alpha)]
        return part[METRIC].tolist()

    verdicts = []
    layer = summary[summary["mode"] == "layerwise"]
    deep = layer[layer.spec_n_steps.isin([4, 8]) & (layer.macs_per_symbol <= 16032)]
    if not deep.empty:
        best = deep.loc[deep.val_select_mean.idxmax()]
        result = compare(values("layerwise", best.spec_n_steps, best.spec_fir_len, best.spec_n_alpha),
                         values("layerwise", 1, best.spec_fir_len, best.spec_n_alpha), 1.0)
        verdicts.append({"hypothesis": "глубина помогает", "candidate": best.variant,
                         "candidate_macs": best.macs_per_symbol,
                         "reference": f"Ns=1, L={int(best.spec_fir_len)}, R={int(best.spec_n_alpha)}, layerwise",
                         "diff_pp": result["diff"], "seed_spread_pp": result["spread"], "n_seeds": result["n_a"],
                         "verdict": "подтверждено" if result["beats"] else "не доказано"})
    multi = layer[layer.spec_n_steps > 1]
    if not multi.empty:
        best = multi.loc[multi.val_select_mean.idxmax()]
        result = compare(values("layerwise", best.spec_n_steps, best.spec_fir_len, best.spec_n_alpha),
                         values("joint_from_scratch", best.spec_n_steps, best.spec_fir_len, best.spec_n_alpha), 1.0)
        verdicts.append({"hypothesis": "послойная инициализация нужна", "candidate": best.variant,
                         "candidate_macs": best.macs_per_symbol,
                         "reference": f"joint_from_scratch, Ns={int(best.spec_n_steps)}, "
                                      f"L={int(best.spec_fir_len)}, R={int(best.spec_n_alpha)}",
                         "diff_pp": result["diff"], "seed_spread_pp": result["spread"], "n_seeds": result["n_a"],
                         "verdict": "подтверждено" if result["beats"] else "не доказано"})
    verdict_frame = pd.DataFrame(verdicts)
    verdict_frame["rule_status"] = config["rule_draft"]["status"]
    verdict_frame.to_csv(results / "dbp_depth_verdicts.csv", index=False)
    plot_ns(summary, figures)
    plot_macs(summary, control_summary, e012_points(), figures)
    print(f"сводка: {results / 'dbp_depth_summary.csv'}; вердикты: {results / 'dbp_depth_verdicts.csv'}")
    return 0


def plot_ns(summary: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ("layerwise", "joint_from_scratch")):
        part = summary[summary["mode"] == mode]
        index = 0
        for fir_len in sorted(part.spec_fir_len.unique()):
            for n_alpha in sorted(part.spec_n_alpha.unique()):
                line = part[(part.spec_fir_len == fir_len) & (part.spec_n_alpha == n_alpha)].sort_values("spec_n_steps")
                if line.empty:
                    continue
                ax.errorbar(line.spec_n_steps, line.test_mean, yerr=[np.clip(line.test_mean - line.test_min, 0, None),
                            np.clip(line.test_max - line.test_mean, 0, None)], marker="o", capsize=2,
                            color=plot_style.color(index), label=f"L={int(fir_len)}, R={int(n_alpha)}")
                index += 1
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_title(mode)
        ax.set_xlabel("number of DBP steps Ns")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("test BER reduction at best validation, %")
    fig.tight_layout()
    fig.savefig(figures / "E018_quality_vs_ns.png", dpi=160)
    plt.close(fig)


def plot_macs(summary, controls, e012, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (column, label) in zip(axes, (("test_mean", "test BER reduction, %"), ("q2_gain_mean", "Q2 gain, dB"))):
        for index, mode in enumerate(("layerwise", "joint_from_scratch")):
            part = summary[summary["mode"] == mode]
            ax.plot(part.macs_per_symbol, part[column], "o", color=plot_style.color(index), markersize=6)
        for _, row in controls.iterrows():
            ax.plot(row.macs_merged, row[column], "s", color=plot_style.color(2), markersize=7)
        if not e012.empty:
            part = e012[e012.budget == 20000]
            ax.plot(part.macs, part[column], "D", markerfacecolor="none", color=plot_style.color(3), markersize=6,
                    linestyle="")
        ax.axvline(16032, color=plot_style.MUTED, linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("real multiplies per symbol")
        ax.set_ylabel(label)
    handles = [Line2D([], [], color=plot_style.color(0), marker="o", linestyle="", label="DBP layerwise"),
               Line2D([], [], color=plot_style.color(1), marker="o", linestyle="", label="DBP joint from scratch"),
               Line2D([], [], color=plot_style.color(2), marker="s", linestyle="", label="complex_a (E018)"),
               Line2D([], [], color=plot_style.color(3), marker="D", markerfacecolor="none", linestyle="",
                      label="E012 MLPs, 20k")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("DBP depth vs multiplies (dotted line: 16032 multiplies of complex_both)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figures / "E018_quality_vs_macs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
