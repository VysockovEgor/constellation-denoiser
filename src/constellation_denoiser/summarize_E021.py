"""E021 summary: Pareto 'quality vs multiplies per symbol' for all model families.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E021   (SMOKE=1 reads results_smoke/ of E020-E022)

Writes results/pareto_points.csv (all points), results/level_best.csv (best family point by validation
at each level), results/level_verdicts.csv (pairwise draft-rule verdicts), results/pareto_members.json
(best configuration of each family with <= 40k multiplies, read by E023), results/runs_table.csv and
figures/E021_pareto_ber.png, figures/E021_pareto_q2.png.
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd

from . import plot_style
from .grids import e014_phases, family_phases, load_config
from .merge_runs import load_runs
from .pareto_points import BASELINES, best_per_level, lr_from_point, points, spec_from_point
from .runner import atomic_write_json, exp_dir, results_root

FAMILY_ORDER = ["complex Kerr MLP", "Phys + MLP", "PhysKerr", "DBP", "real tanh MLP", "transformer", "KAN"]
BASELINE_MARKERS = {"линейный FIR": "x", "DBP (заданная нелинейность)": "+", "CNN (приближение)": "P"}


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E021")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E021", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    own = load_runs(results)
    if not own.empty:
        own.to_csv(results / "runs_table.csv", index=False)

    pts = points(smoke)
    pts.to_csv(results / "pareto_points.csv", index=False)
    levels, factor = config["levels"], config["level_window_factor"]
    best = best_per_level(pts, levels, factor)
    best.to_csv(results / "level_best.csv", index=False)

    verdicts = []
    for level, part in best.groupby("level"):
        part = part.sort_values("test_mean", ascending=False)
        for (_, x), (_, y) in itertools.combinations(part.iterrows(), 2):
            diff = x.test_mean - y.test_mean
            overlap = not (x.test_min > y.test_max or y.test_min > x.test_max)
            better = abs(diff) > 1.0 and not overlap
            hi, lo = (x, y) if diff >= 0 else (y, x)
            verdicts.append({"level": int(level), "hypothesis": f"{int(level)}: {hi.family} лучше {lo.family}",
                             "diff_pp": abs(diff), "intervals_overlap": overlap,
                             "verdict": "подтверждено" if better else "не различимы"})
    vframe = pd.DataFrame(verdicts, columns=["level", "hypothesis", "diff_pp", "intervals_overlap", "verdict"])
    vframe["rule_status"] = config["rule_draft"]["status"]
    vframe.to_csv(results / "level_verdicts.csv", index=False)

    configs = {e: load_config(e)[1] for e in ("E014", "E016", "E017", "E018")}
    members = []
    for family, part in pts[(pts.macs <= 40000) & ~pts.family.isin(BASELINES)].groupby("family"):
        part = part.sort_values("val_mean", ascending=False)
        for _, row in part.iterrows():
            spec, lr = spec_from_point(row), lr_from_point(row)
            if not spec.get("model") or lr is None:
                continue
            spec.pop("name", None)
            item = {"experiment": row.source, "variant": str(row.variant), "spec": spec}
            if row.source in ("E017", "E018"):
                phases = family_phases(item, configs)
            elif row.source == "E014" and str(row.variant) in configs["E014"]["models"]:
                phases = e014_phases(configs["E014"]["models"][str(row.variant)]["phases"], 2)   # structure only
            else:
                phases = [{"steps": 1}]
            members.append({"name": f"{family.split()[0].lower()}_{row.source}_{row.variant}".replace(" ", "_"),
                            "family": family, "spec": spec, "lr_config": lr, "lr": str(row.lr_name),
                            "phases": phases, "macs": int(row.macs), "source_point": row.label,
                            "val_mean": float(row.val_mean)})
            break
    atomic_write_json(results / "pareto_members.json", members)
    plot(pts, best, figures, levels, factor)
    print(f"точек {len(pts)}; лучшие по уровням: {results / 'level_best.csv'}; участники E023: {len(members)}")
    return 0


def plot(pts: pd.DataFrame, best: pd.DataFrame, figures, levels: list[int], factor: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    for metric, low, high, label, name in (("test_mean", "test_min", "test_max", "test BER reduction, %", "E021_pareto_ber.png"),
                                           ("q2_mean", "q2_min", "q2_max", "Q2 gain, dB", "E021_pareto_q2.png")):
        fig, ax = plt.subplots(figsize=(10, 5.4))
        for level in levels:
            ax.axvspan(level / factor, level * factor, color=plot_style.GRID, alpha=0.35, linewidth=0)
        shown = pts[pts[metric] > (-20 if metric == "test_mean" else -0.5)]
        handles = []
        for index, family in enumerate(FAMILY_ORDER):
            part = shown[shown.family == family]
            if part.empty:
                continue
            c = plot_style.color(index)
            ax.plot(part.macs, part[metric], "o", color=c, alpha=0.25, markersize=4)
            top = best[best.family == family]
            ax.errorbar(top.macs.to_numpy(float), top[metric].to_numpy(float),
                        yerr=[np.clip((top[metric] - top[low]).to_numpy(float), 0, None), np.clip((top[high] - top[metric]).to_numpy(float), 0, None)],
                        fmt="o", color=c, markersize=8, capsize=3, linewidth=1.2)
            handles.append(Line2D([], [], color=c, marker="o", linestyle="", label=family))
        for family, marker in BASELINE_MARKERS.items():
            part = shown[shown.family == family]
            if part.empty:
                continue
            ax.errorbar(part.macs.to_numpy(float), part[metric].to_numpy(float),
                        yerr=[np.clip((part[metric] - part[low]).to_numpy(float), 0, None), np.clip((part[high] - part[metric]).to_numpy(float), 0, None)],
                        fmt=marker, color=plot_style.TEXT, markersize=8, capsize=2, linewidth=1)
            handles.append(Line2D([], [], color=plot_style.TEXT, marker=marker, linestyle="", label=f"база: {family}"))
        ax.set_xscale("log")
        ax.set_xlabel("real multiplies per symbol (log; after merging, KAN sparse)")
        ax.set_ylabel(label)
        ax.set_title("Quality vs complexity: large markers = best by validation per family in each level band (min-max over seeds)")
        ax.legend(handles=handles, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(figures / name, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
