"""E040 summary: cascade of triplet stages against the one-step closed-form triplets at equal RM.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E040   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/cascade_summary.csv (with three RM conventions), results/cascade_verdicts.csv,
the gate ../../results/gate_e040.json and figures/E040_cascade_vs_one_step.png.
"""

from __future__ import annotations

import json

import pandas as pd

from .cascade_e040 import rm_variants
from .grids_e039 import e040_grid, in_level
from .one_step_triplets import one_step_points
from .paths import PROJECT_ROOT
from .runner import atomic_write_json
from .summary_e024 import aggregate, disjoint, setup, verdict, write_verdicts


def control_spread(results) -> float:
    path = results / "control_check.json"
    if not path.is_file():
        return float("nan")
    values = json.loads(path.read_text())["targets"][0]["values"]
    return float(max(values) - min(values)) if values else float("nan")


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E040")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage.isin(["main", "confirm"])]
    table = aggregate(runs, ["variant", "stage", "budget", "shared", "n_stages"])
    specs = dict(e040_grid(config))
    extra = pd.DataFrame([rm_variants(specs[v]["stages"]) if v in specs else {} for v in table.variant])
    table = pd.concat([table.reset_index(drop=True), extra], axis=1)
    table.to_csv(results / "cascade_summary.csv", index=False)
    spread = control_spread(results)
    threshold = max(1.5, 2 * spread) if spread == spread else 1.5
    ref = one_step_points()
    confirm = table[table.stage == "confirm"]
    rows, wins = [], 0
    for level in config["levels"]:
        cand = confirm[[in_level(r, level, config["level_tol"]) for r in confirm.rm]]
        base = ref[ref.rm_repo <= level * (1 + config["level_tol"])] if len(ref) else ref
        if cand.empty or base.empty:
            rows.append({"hypothesis": f"{level} RM: каскад лучше одношаговых триплетов", "value": float("nan"),
                         "verdict": "не доказано", "detail": "нет точки каскада или одношаговой в окне уровня"})
            continue
        c, b = cand.loc[cand.val_mean.idxmax()], base.loc[base.val_mean.idxmax()]
        diff = float(c.test_mean - b.test_mean)
        ok = diff >= threshold and disjoint(c.test_min, c.test_max, b.test_min, b.test_max)
        wins += ok
        rows.append({"hypothesis": f"{level} RM: каскад лучше одношаговых триплетов (≥ {threshold:.2f} п.п., интервалы не пересекаются)",
                     "value": diff, "verdict": verdict(ok),
                     "detail": f"{c.variant} ({int(c.rm)} RM, {c.test_mean:.2f}) против {b.label} ({int(b.rm_repo)} RM, {b.test_mean:.2f})"})
    better = wins >= 2
    rows.append({"hypothesis": "каскад лучше одношаговых на 2 из 3 уровней", "value": wins, "verdict": verdict(better),
                 "detail": f"порог max(1.5, 2 × разброс контроля {spread:.2f})"})
    for _, r in confirm[confirm.variant.str.endswith("_shared")].iterrows():
        twin = confirm[confirm.variant == r.variant[:-7]]
        if len(twin):
            rows.append({"hypothesis": f"{r.variant[:-7]}: общие C минус независимые (для отчёта)",
                         "value": float(r.test_mean - twin.test_mean.iloc[0]), "verdict": "справочно", "detail": ""})
    write_verdicts(rows, config, results / "cascade_verdicts.csv")
    gate = PROJECT_ROOT / ("results_smoke" if smoke else "results") / "gate_e040.json"
    atomic_write_json(gate, {"cascade_better": bool(better), "levels_won": int(wins), "threshold_pp": threshold,
                             "rule": config["rule_draft"]["cascade_better"],
                             "next": "E041 runs" if better else "E041 waits for the author's decision (results/gate_e040_decision.json)"})
    plot(table, ref, figures)
    print(f"E040: {results / 'cascade_summary.csv'}; каскад лучше: {better}")
    return 0


def plot(table: pd.DataFrame, ref: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (stage, marker) in enumerate((("main", "o"), ("confirm", "s"))):
        part = table[(table.stage == stage) & ~table.variant.str.endswith("_shared")]
        ax.errorbar(part.rm, part.test_mean, yerr=[part.test_mean - part.test_min, part.test_max - part.test_mean],
                    fmt=marker, color=plot_style.color(index), capsize=1.5, alpha=0.8, markersize=4,
                    label=f"cascade, {stage} ({'2 seeds, 20k' if stage == 'main' else '5 seeds, 200k'})")
    if len(ref):
        r = ref.sort_values("rm_repo")
        ax.plot(r.rm_repo, r.test_mean, "^", color=plot_style.color(2), label="one-step closed-form triplets (E031/E034/E036)")
    ax.set_xscale("log")
    ax.set_xlabel("real multiplies per symbol (repository convention)")
    ax.set_ylabel("test BER reduction, % (min-max over seeds)")
    ax.set_title("Cascade of triplet stages vs one-step triplets")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E040_cascade_vs_one_step.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
