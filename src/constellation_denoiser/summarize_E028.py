"""E028 summary: DBP depth (A), shared steps, DBP + MLP and the layer-wise diagnosis (B).

    PYTHONPATH=src python -m constellation_denoiser.summarize_E028   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/a_summary.csv (read by E032), results/b_summary.csv,
results/b_per_phase.csv (validation after every added step), results/dbp_verdicts.csv and figures
E028_val_vs_ns.png, E028_layerwise_phases.png.
"""

from __future__ import annotations

import pandas as pd

from .summary_e024 import aggregate, better, setup, verdict, write_verdicts


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E028")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage != "control"]
    table = aggregate(runs, ["variant", "stage", "budget", "spec_model", "spec_n_steps", "spec_fir_len", "spec_n_alpha",
                             "spec_mlp_hidden"])
    table = table.rename(columns={"spec_n_steps": "n_steps", "spec_fir_len": "fir_len", "spec_n_alpha": "n_alpha"})
    table.to_csv(results / "a_summary.csv", index=False)
    rows = []
    a = table[table.stage == "a"]
    for (fir, r), part in a.groupby(["fir_len", "n_alpha"]):
        by_ns = part.set_index("n_steps")
        for ns in sorted(by_ns.index):
            if 2 * ns in by_ns.index:
                gain = float(by_ns.loc[2 * ns, "val_mean"] - by_ns.loc[ns, "val_mean"])
                rows.append({"hypothesis": f"L={int(fir)}, R={int(r)}: насыщение при удвоении Ns {int(ns)} → {int(2 * ns)} "
                             "(< 1 п.п.)", "value": gain, "verdict": verdict(gain < 1.0), "detail": "валидация, 100k"})
    shared = table[table.stage == "shared"]
    for _, s in shared.iterrows():
        ind = a[(a.n_steps == s.n_steps) & (a.fir_len == s.fir_len) & (a.n_alpha == s.n_alpha)]
        if len(ind):
            diff, ok = better(s, ind.iloc[0], 1.0)
            rows.append({"hypothesis": f"Ns={int(s.n_steps)}: shared лучше independent (≥ 1 п.п.)", "value": diff,
                         "verdict": verdict(ok), "detail": "тест, 3 seeds"})
    b = table[table.stage == "b"].copy()
    b.to_csv(results / "b_summary.csv", index=False)
    joint = b[b.variant.str.startswith("b_f_")]
    if len(joint):
        ref = float(joint.test_mean.iloc[0])
        for _, row in b[~b.variant.str.startswith("b_f_")].iterrows():
            share = row.test_mean / ref if ref else float("nan")
            rows.append({"hypothesis": f"диагноз: {row.variant} поднимает послойное до ≥ 90 % от joint", "value": 100 * share,
                         "verdict": verdict(share >= 0.9), "detail": f"тест {row.test_mean:.2f} против joint {ref:.2f}"})
    write_verdicts(rows, config, results / "dbp_verdicts.csv")
    phases = per_phase(runs[runs.stage == "b"], results)
    phases.to_csv(results / "b_per_phase.csv", index=False)
    plots(a, phases, figures)
    print(f"E028: {results / 'a_summary.csv'}; вердикты: {results / 'dbp_verdicts.csv'}")
    return 0


def per_phase(runs: pd.DataFrame, results) -> pd.DataFrame:
    rows = []
    for _, r in runs.iterrows():
        path = results / r.curve_csv
        if not path.is_file():
            continue
        curve = pd.read_csv(path).dropna(subset=["val_reduction"])
        for phase, part in curve.groupby("phase"):
            rows.append({"variant": r.variant, "seed": r.seed, "phase": int(phase), "step": int(part.step.iloc[-1]),
                         "val_reduction": float(part.val_reduction.iloc[-1])})
    frame = pd.DataFrame(rows, columns=["variant", "seed", "phase", "step", "val_reduction"])
    return frame.groupby(["variant", "phase"]).agg(step=("step", "max"), val_mean=("val_reduction", "mean"),
                                                   val_min=("val_reduction", "min"),
                                                   val_max=("val_reduction", "max")).reset_index()


def plots(a: pd.DataFrame, phases: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    if len(a):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for index, ((fir, r), part) in enumerate(a.groupby(["fir_len", "n_alpha"])):
            part = part.sort_values("n_steps")
            ax.errorbar(part.n_steps, part.val_mean, yerr=[part.val_mean - part.val_min, part.val_max - part.val_mean],
                        marker="osD^"[index % 4], color=plot_style.color(index), capsize=2, label=f"L={int(fir)}, R={int(r)}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("DBP steps Ns (joint from scratch, 100k)")
        ax.set_ylabel("validation BER reduction, % (min-max over 3 seeds)")
        ax.set_title("DBP depth")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "E028_val_vs_ns.png", dpi=160)
        plt.close(fig)
    if len(phases):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for index, (variant, part) in enumerate(phases.groupby("variant")):
            ax.plot(part.step, part.val_mean, marker="osD^vP*"[index % 7], color=plot_style.color(index), label=variant)
        ax.set_xlabel("training step (end of each phase = one more DBP step)")
        ax.set_ylabel("validation BER reduction, % (mean of 3 seeds)")
        ax.set_title("Layer-wise variants, Ns = 8, L = 31, R = 7")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "E028_layerwise_phases.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
