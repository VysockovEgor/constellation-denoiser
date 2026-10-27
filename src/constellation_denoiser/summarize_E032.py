"""E032 summary: 1M-step runs, plateau and ranking stability.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E032   (SMOKE=1 reads results_smoke/)

For each run and horizon H (200k, 500k, 1M) the test at the validation-best evaluation among steps <= H is read
from the curve (evaluations every 20k). Writes results/runs_table.csv, results/horizons.csv,
results/convergence_verdicts.csv and figures/E032_valbest_test_vs_steps.png.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .summary_e024 import setup, verdict, write_verdicts

HORIZONS = (200_000, 500_000, 1_000_000)


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E032")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage == "main"]
    scale = (runs.planned_steps.max() / config["steps"]) if smoke else 1.0
    rows, curves = [], []
    for _, r in runs.iterrows():
        c = pd.read_csv(results / r.curve_csv).dropna(subset=["val_reduction", "test_reduction"])
        c = c[c.step > 0]
        curves.append(c.assign(variant=r.variant, seed=r.seed))
        for h in HORIZONS:
            part = c[c.step <= h * scale]
            if len(part):
                best = part.loc[part.val_reduction.idxmax()]
                rows.append({"variant": r.variant, "seed": r.seed, "horizon": h, "test": best.test_reduction,
                             "val": best.val_reduction})
    hz = pd.DataFrame(rows)
    agg = hz.groupby(["variant", "horizon"]).agg(test_mean=("test", "mean"), test_min=("test", "min"),
                                                 test_max=("test", "max"), val_mean=("val", "mean")).reset_index()
    agg.to_csv(results / "horizons.csv", index=False)
    verdicts = []
    for variant, part in agg.groupby("variant"):
        part = part.set_index("horizon")
        if 500_000 in part.index and 1_000_000 in part.index:
            gain = float(part.loc[1_000_000, "test_mean"] - part.loc[500_000, "test_mean"])
            verdicts.append({"hypothesis": f"{variant}: плато (500k → 1M < 0.3 п.п.)", "value": gain,
                             "verdict": verdict(gain < 0.3), "detail": "тест в точке лучшей валидации внутри горизонта"})
    first, last = agg[agg.horizon == HORIZONS[0]].set_index("variant"), agg[agg.horizon == HORIZONS[-1]].set_index("variant")
    common = sorted(set(first.index) & set(last.index))
    if len(common) >= 2:
        order_a = list(first.loc[common].test_mean.sort_values(ascending=False).index)
        order_b = list(last.loc[common].test_mean.sort_values(ascending=False).index)
        signs = all(np.sign(first.test_mean[a] - first.test_mean[b]) == np.sign(last.test_mean[a] - last.test_mean[b])
                    for a, b in itertools.combinations(common, 2))
        verdicts.append({"hypothesis": "ранжирование стабильно (порядок при 200k и 1M совпадает, знаки разностей те же)",
                         "value": float("nan"), "verdict": verdict(order_a == order_b and signs),
                         "detail": "200k: " + " > ".join(order_a) + " | 1M: " + " > ".join(order_b)})
    write_verdicts(verdicts, config, results / "convergence_verdicts.csv")
    plot(pd.concat(curves) if curves else pd.DataFrame(), figures)
    print(f"E032: {results / 'horizons.csv'}; вердикты: {results / 'convergence_verdicts.csv'}")
    return 0


def plot(curves: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    if curves.empty:
        return
    plot_style.apply()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, (variant, part) in enumerate(curves.groupby("variant")):
        part = part.sort_values("step")
        best = []
        for _, seed_part in part.groupby("seed"):
            s = seed_part.sort_values("step")
            idx = s.val_reduction.cummax()
            chosen = [s.test_reduction.iloc[int(np.argmax(s.val_reduction.iloc[: i + 1].to_numpy()))] for i in range(len(s))]
            best.append(pd.Series(chosen, index=s.step.to_numpy()))
        mean = pd.concat(best, axis=1).mean(axis=1)
        ax.plot(mean.index, mean.values, color=plot_style.color(index), label=variant)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("test at the validation-best point so far, % (mean of seeds)")
    ax.set_title("Convergence to 1M steps")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E032_valbest_test_vs_steps.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
