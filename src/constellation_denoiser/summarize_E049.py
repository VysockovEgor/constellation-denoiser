"""E049 summary: physical vs identity initialization of the DBP (10 seeds): BER at the same RM, seed spread, convergence.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E049   (SMOKE=1 reads results_smoke/)

Convergence: first evaluation step at which the validation reduction reaches 90 % of the mean final validation of the
delta arm of the same configuration (per run, mean over seeds). Writes results/init_summary.csv, results/init_verdicts.csv
and figures/E049_val_curves.png.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .summary_e024 import setup, verdict, write_verdicts


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E049")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage == "main"]
    curves = {r.run_id: pd.read_csv(results / r.curve_csv) for r in runs.itertuples()}
    rows = []
    for (ns, arm), part in runs.groupby(["n_steps", "arm"]):
        t = part.test_reduction_valbest_select
        rows.append({"n_steps": ns, "arm": arm, "rm": part.macs_per_symbol_merged.iloc[0], "n_seeds": len(part),
                     "test_mean": t.mean(), "test_std": t.std(), "test_se": t.std() / math.sqrt(len(t)),
                     "val_mean": part.best_val_reduction.mean()})
    table = pd.DataFrame(rows)
    for ns, part in runs.groupby("n_steps"):
        delta = part[part.arm == "delta"]
        if delta.empty:
            continue
        target = 0.9 * np.mean([curves[r].val_reduction.dropna().iloc[-1] for r in delta.run_id])
        for arm, group in part.groupby("arm"):
            steps = []
            for r in group.run_id:
                c = curves[r].dropna(subset=["val_reduction"])
                hit = c[c.val_reduction >= target]
                steps.append(float(hit.step.iloc[0]) if len(hit) else float("nan"))
            table.loc[(table.n_steps == ns) & (table.arm == arm), "steps_to_90pct_delta"] = np.nanmean(steps)
    table.to_csv(results / "init_summary.csv", index=False)
    verdicts = []
    for ns, part in table.groupby("n_steps"):
        d = part[part.arm == "delta"]
        if d.empty:
            continue
        d = d.iloc[0]
        for _, r in part[part.arm != "delta"].iterrows():
            diff = r.test_mean - d.test_mean
            se = math.sqrt(r.test_se ** 2 + d.test_se ** 2)
            faster = r.steps_to_90pct_delta <= 0.5 * d.steps_to_90pct_delta
            tighter = r.test_std <= 0.5 * d.test_std
            signal = diff > 2 * se or faster or tighter
            verdicts.append({"hypothesis": f"Ns={int(ns)}: {r.arm} даёт сигнал против delta (> 2 SE, или вдвое быстрее, или вдвое меньше std)",
                             "value": float(diff), "verdict": verdict(bool(signal)),
                             "detail": f"тест {r.test_mean:.2f} ± {r.test_std:.2f} против {d.test_mean:.2f} ± {d.test_std:.2f}; "
                                       f"2 SE = {2 * se:.2f}, > 2 std: {diff > 2 * max(r.test_std, d.test_std)}; шагов до 90 %: "
                                       f"{r.steps_to_90pct_delta:.0f} против {d.steps_to_90pct_delta:.0f}"})
    write_verdicts(verdicts, config, results / "init_verdicts.csv")
    plot(runs, curves, figures)
    print(table.round(2).to_string(index=False))
    return 0


def plot(runs: pd.DataFrame, curves: dict, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    groups = list(runs.groupby(["n_steps", "arm"]))
    fig, axes = plt.subplots(1, runs.n_steps.nunique(), figsize=(6 * runs.n_steps.nunique(), 4.2), squeeze=False)
    for ax, ns in zip(axes[0], sorted(runs.n_steps.unique())):
        for index, arm in enumerate(("delta", "phys_fir", "phys_fir_alpha")):
            ids = runs[(runs.n_steps == ns) & (runs.arm == arm)].run_id
            if not len(ids):
                continue
            frame = pd.concat([curves[r][["step", "val_reduction"]].dropna() for r in ids]).groupby("step").val_reduction
            m, lo, hi = frame.mean(), frame.min(), frame.max()
            plot_style.band(ax, m.index, m.values, lo.values, hi.values, index, arm)
        ax.set_title(f"DBP Ns = {int(ns)}")
        ax.set_xlabel("training step")
        ax.set_ylabel("validation BER reduction, % (mean, min-max over 10 seeds)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E049_val_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
