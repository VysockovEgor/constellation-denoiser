"""E041 summary: mixed stage types against the uniform cascade at the same RM levels (equal tuning budget).

    PYTHONPATH=src python -m constellation_denoiser.summarize_E041   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/mixed_summary.csv, results/mixed_verdicts.csv, results/type_by_position.csv and
figures E041_mixed_vs_uniform.png, E041_type_by_position.png.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .grids_e039 import e041_sample
from .summarize_E040 import control_spread
from .summary_e024 import aggregate, disjoint, setup, verdict, write_verdicts

TYPES = ["K", "T", "KT", "skip"]


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E041")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage.isin(["s1", "s2", "s3"])]
    table = aggregate(runs, ["arm", "level", "e041_config", "stage", "budget"])
    table.to_csv(results / "mixed_summary.csv", index=False)
    spread = control_spread(results)
    threshold = max(1.5, 2 * spread) if spread == spread else 1.5
    s3 = table[table.stage == ("s3" if (table.stage == "s3").any() else table.stage.max())]
    rows, wins = [], 0
    for level in config["levels"]:
        a, u = s3[(s3.level == level) & (s3.arm == "mixed")], s3[(s3.level == level) & (s3.arm == "uniform")]
        if a.empty or u.empty:
            continue
        a, u = a.loc[a.val_mean.idxmax()], u.loc[u.val_mean.idxmax()]
        diff = float(a.test_mean - u.test_mean)
        ok = diff >= threshold and disjoint(a.test_min, a.test_max, u.test_min, u.test_max)
        wins += ok
        rows.append({"hypothesis": f"{level} RM: смешанные ступени лучше равномерного каскада (≥ {threshold:.2f} п.п.)",
                     "value": diff, "verdict": verdict(ok), "detail": f"{a.e041_config} {a.test_mean:.2f} против {u.e041_config} {u.test_mean:.2f}"})
    rows.append({"hypothesis": "кандидат A лучше на 2 из 3 уровней", "value": wins, "verdict": verdict(wins >= 2), "detail": ""})
    write_verdicts(rows, config, results / "mixed_verdicts.csv")
    maps = []
    s1 = table[(table.stage == "s1") & (table.arm == "mixed")]
    for level in config["levels"]:
        specs = dict(e041_sample(config, "mixed", level))
        top = s1[s1.level == level].sort_values("val_mean", ascending=False).head(10)
        for name in top.e041_config:
            for pos, st in enumerate(specs.get(name, {}).get("stages", [])):
                maps.append({"level": level, "position": pos, "type": st["type"]})
    maps = pd.DataFrame(maps, columns=["level", "position", "type"])
    maps.to_csv(results / "type_by_position.csv", index=False)
    plot(s3, maps, config, figures)
    print(f"E041: {results / 'mixed_summary.csv'}; вердикты: {results / 'mixed_verdicts.csv'}")
    return 0


def plot(s3: pd.DataFrame, maps: pd.DataFrame, config: dict, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for index, arm in enumerate(("mixed", "uniform")):
        part = s3[s3.arm == arm].sort_values("level")
        best = part.loc[part.groupby("level").val_mean.idxmax()] if len(part) else part
        ax.errorbar(best.level.astype(str), best.test_mean, yerr=[best.test_mean - best.test_min, best.test_max - best.test_mean],
                    marker="os"[index], color=plot_style.color(index), capsize=2, linestyle="", label=arm)
    ax.set_xlabel("RM level")
    ax.set_ylabel("test BER reduction, % (best by validation, min-max)")
    ax.set_title("Mixed stage types vs uniform cascade (equal tuning budget)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E041_mixed_vs_uniform.png", dpi=160)
    plt.close(fig)
    if maps.empty:
        return
    levels = config["levels"]
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 3.2), squeeze=False)
    for ax, level in zip(axes[0], levels):
        part = maps[maps.level == level]
        npos = int(part.position.max()) + 1 if len(part) else 1
        grid = np.zeros((len(TYPES), npos))
        for _, r in part.iterrows():
            grid[TYPES.index(r.type), int(r.position)] += 1
        ax.imshow(grid, aspect="auto", cmap="Blues")
        for (i, j), v in np.ndenumerate(grid):
            if v:
                ax.text(j, i, int(v), ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(TYPES)), TYPES)
        ax.set_xlabel("stage position")
        ax.set_title(f"{level} RM: top-10 of S1")
    fig.tight_layout()
    fig.savefig(figures / "E041_type_by_position.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
