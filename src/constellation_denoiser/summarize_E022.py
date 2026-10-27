"""E022 summary: classical baselines vs our approach at equal multiplies per symbol.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E022   (SMOKE=1 reads results_smoke/)

Our point for a baseline = the best-by-validation point of the families complex Kerr MLP, Phys + MLP,
PhysKerr, DBP (E010-E021 per-run tables) whose MACs lie within a factor 1.5 of the baseline's MACs.
Writes results/runs_table.csv, results/baselines_summary.csv, results/baselines_verdicts.csv and
figures/E022_baselines_vs_macs.png.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .pareto_points import OURS, points
from .runner import exp_dir, results_root

FACTOR = 1.5


def non_overlap(a_min, a_max, b_min, b_max) -> bool:
    return a_min > b_max or b_min > a_max


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E022")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E022", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if runs.empty:
        print("нет прогонов")
        return 1
    runs.to_csv(results / "runs_table.csv", index=False)
    pts = points(smoke)
    own = pts[pts.source == "E022"].copy()
    own.to_csv(results / "baselines_summary.csv", index=False)
    ours = pts[pts.family.isin(OURS) & (pts.source != "E022")]

    verdicts = []
    for _, base in own[~own.variant.str.startswith("dbp_train")].iterrows():
        rivals = ours[(ours.macs >= base.macs / FACTOR) & (ours.macs <= base.macs * FACTOR)]
        if rivals.empty:
            verdicts.append({"hypothesis": f"наш подход лучше {base.variant} ({int(base.macs)} умн.)", "rival": "—",
                             "q2_diff_db": float("nan"), "verdict": "не доказано (нет нашей точки при равных умножениях)"})
            continue
        rival = rivals.loc[rivals.val_mean.idxmax()]
        diff = rival.q2_mean - base.q2_mean
        ok = diff > 0.1 and non_overlap(rival.q2_min, rival.q2_max, base.q2_min, base.q2_max)
        verdicts.append({"hypothesis": f"наш подход лучше {base.variant} ({int(base.macs)} умн.)",
                         "rival": f"{rival.label} ({int(rival.macs)} умн.)", "q2_diff_db": diff,
                         "verdict": "подтверждено" if ok else "не доказано"})
    for n_steps in config["dbp_fixed"]["n_steps"]:
        fixed = own[own.variant == f"dbp_fixed_Ns{n_steps}"]
        trained = own[own.variant == f"dbp_train_Ns{n_steps}"]
        if fixed.empty or trained.empty:
            continue
        f, t = fixed.iloc[0], trained.iloc[0]
        diff = t.q2_mean - f.q2_mean
        ok = diff > 0.1 and non_overlap(t.q2_min, t.q2_max, f.q2_min, f.q2_max)
        verdicts.append({"hypothesis": f"обучение нелинейности нужно (DBP Ns={n_steps})", "rival": t.variant,
                         "q2_diff_db": diff, "verdict": "подтверждено" if ok else "не доказано"})
    frame = pd.DataFrame(verdicts, columns=["hypothesis", "rival", "q2_diff_db", "verdict"])
    frame["rule_status"] = config["rule_draft"]["status"]
    frame.to_csv(results / "baselines_verdicts.csv", index=False)
    plot(own, ours, figures)
    print(f"базовые методы: {results / 'baselines_summary.csv'}; вердикты: {results / 'baselines_verdicts.csv'}")
    return 0


def plot(own: pd.DataFrame, ours: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(10, 5))
    handles = []
    for index, family in enumerate(("complex Kerr MLP", "Phys + MLP", "PhysKerr", "DBP")):
        part = ours[(ours.family == family) & (ours.q2_mean > -0.5)]
        if part.empty:
            continue
        ax.plot(part.macs.to_numpy(float), part.q2_mean.to_numpy(float), "o", color=plot_style.color(index), alpha=0.35, markersize=4)
        handles.append(Line2D([], [], color=plot_style.color(index), marker="o", linestyle="", label=f"наш: {family}"))
    markers = {"линейный FIR": "x", "DBP (заданная нелинейность)": "+", "DBP": "^", "CNN (приближение)": "P"}
    for family, part in own.groupby("family"):
        marker = markers.get(family, "s")
        ax.errorbar(part.macs.to_numpy(float), part.q2_mean.to_numpy(float),
                    yerr=[np.clip((part.q2_mean - part.q2_min).to_numpy(float), 0, None),
                          np.clip((part.q2_max - part.q2_mean).to_numpy(float), 0, None)],
                    fmt=marker, color=plot_style.TEXT, markersize=8, capsize=2, linewidth=1)
        handles.append(Line2D([], [], color=plot_style.TEXT, marker=marker, linestyle="", label=f"E022: {family}"))
    ax.set_xscale("log")
    ax.set_xlabel("real multiplies per symbol (log)")
    ax.set_ylabel("Q2 gain, dB (test at best validation)")
    ax.set_title("Baselines (black, min-max over seeds) vs our families (colour)")
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(figures / "E022_baselines_vs_macs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
