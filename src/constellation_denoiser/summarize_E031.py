"""E031: fit the perturbation triplets (closed form, CPU, minutes) and summarize.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E031 [fit|summarize|all]   (SMOKE=1: 2/1/1 files, M 3, 5)

Writes results/triplet_fits.csv (arm x M x replica), results/runs_table.csv (rows in the per-run format of
the aggregator: variant, seed = replica, best_val_reduction, test_reduction_valbest_select, RM), results/
triplet_summary.csv, results/triplet_verdicts.csv and figures/E031_triplets_vs_rm.png.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .q2 import q2_db
from .runner import atomic_write_json, default_files, exp_dir, git_commit, results_root

TEST = "test_reduction_valbest_select"


def fit(config: dict, smoke: bool, out) -> pd.DataFrame:
    from .triplets import fit_arm
    from .window_data import load_fileset, resolve_paths

    files = default_files(config["n_train_files"], smoke)
    data = {k: load_fileset(resolve_paths(v)) for k, v in files.items()}
    raw_val, raw_test = data["validation"].raw_ber(31), data["test"].raw_ber(31)
    reps = config["bootstrap"]["replicas"][:1] if smoke else config["bootstrap"]["replicas"]
    rows = []
    for M in (config["smoke_M"] if smoke else config["M"]):
        for arm, fir in [("T1", 0), ("T2", 0), ("T3", 0)] + [("T3+FIR", L) for L in config["fir_len"]]:
            name = arm if arm != "T3+FIR" else f"T3+FIR{fir}"
            base = "T3" if arm == "T3+FIR" else arm
            got = fit_arm(data["train"], data["validation"], data["test"], M, base, fir, config["lambda_grid"], reps,
                          config["max_block"])
            for r in got:
                r.update(arm=name, raw_val_ber=raw_val, raw_ber=raw_test)
            rows += got
            print(f"{name} M={M}: test {100 * (raw_test - np.mean([g['test_ber'] for g in got])) / raw_test:.2f} %", flush=True)
    frame = pd.DataFrame(rows)
    frame["val_reduction"] = 100 * (frame.raw_val_ber - frame.val_ber) / frame.raw_val_ber
    frame["test_reduction"] = 100 * (frame.raw_ber - frame.test_ber) / frame.raw_ber
    frame["q2_gain_db"] = [q2_db(t) - q2_db(r) for t, r in zip(frame.test_ber, frame.raw_ber)]
    frame["git_commit"] = git_commit()
    frame.to_csv(out / "triplet_fits.csv", index=False)
    return frame


def summarize(config: dict, out, figures) -> None:
    frame = pd.read_csv(out / "triplet_fits.csv")
    runs = pd.DataFrame({"experiment": "E031", "stage": "main", "model": "triplets",
                         "variant": [f"{a}_M{m}" for a, m in zip(frame.arm, frame.M)], "seed": frame.replica,
                         "budget": np.nan, "lr_name": "closed_form", "best_val_reduction": frame.val_reduction,
                         TEST: frame.test_reduction, "valbest_select_q2_gain_db": frame.q2_gain_db,
                         "macs_per_symbol": frame.rm_naive, "macs_per_symbol_merged": frame.rm_sym,
                         "rm_naive": frame.rm_naive, "rm_sym": frame.rm_sym, "raw_ber": frame.raw_ber,
                         "spec_arm": frame.arm, "spec_M": frame.M, "spec_fir_len": frame.fir_len, "diverged": False,
                         "recipe_source": "closed form"})
    runs.to_csv(out / "runs_table.csv", index=False)
    summary = frame.groupby(["arm", "M"]).agg(
        n_replicas=("replica", "count"), val_mean=("val_reduction", "mean"), test_mean=("test_reduction", "mean"),
        test_min=("test_reduction", "min"), test_max=("test_reduction", "max"), q2_mean=("q2_gain_db", "mean"),
        rm_naive=("rm_naive", "first"), rm_sym=("rm_sym", "first"), n_terms=("n_terms", "first"),
        lambda_median=("lambda", "median")).reset_index()
    summary.to_csv(out / "triplet_summary.csv", index=False)
    rule = config["rule_draft"]
    verdicts = []
    for M, part in summary.groupby("M"):
        t2, t3 = part[part.arm == "T2"], part[part.arm == "T3"]
        if len(t2) and len(t3):
            diff = float(t3.test_mean.iloc[0] - t2.test_mean.iloc[0])
            verdicts.append({"hypothesis": f"M={M}: диагональной (XPM) физики недостаточно (T3 − T2 ≥ 5 п.п.)",
                             "value": diff, "verdict": "подтверждено" if diff >= 5 else "не доказано"})
    t1 = summary[summary.arm == "T1"]
    if len(t1):
        best = t1.loc[t1.val_mean.idxmax()]
        verdicts.append({"hypothesis": f"T1 ≈ PhysKerr (|T1 − 9.3| < 3 п.п.; M = {int(best.M)} по валидации)",
                         "value": float(best.test_mean), "verdict": "подтверждено" if abs(best.test_mean - 9.3) < 3 else "не доказано"})
    verdicts.append({"hypothesis": "наш подход лучше PB-NLC при равных RM (≥ 0.1 дБ, интервалы не пересекаются)",
                     "value": float("nan"), "verdict": "проверяется в E033"})
    v = pd.DataFrame(verdicts)
    v["rule_status"] = rule["status"]
    v.to_csv(out / "triplet_verdicts.csv", index=False)
    plot(summary, figures)


def plot(summary: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, (arm, part) in enumerate(summary.groupby("arm")):
        part = part.sort_values("rm_sym")
        ax.errorbar(part.rm_sym, part.test_mean, yerr=[part.test_mean - part.test_min, part.test_max - part.test_mean],
                    marker="osD^v"[index % 5], color=plot_style.color(index), capsize=2, label=arm)
    ax.axhline(9.3, color=plot_style.TEXT, linestyle=":", linewidth=1, label="PhysKerr R31 L7 (E019) 9.3 %")
    ax.set_xscale("log")
    ax.set_xlabel("real multiplies per symbol (sym count; labels = M)")
    ax.set_ylabel("test BER reduction, % (min-max over 5 bootstrap replicas)")
    ax.set_title("First-order perturbation triplets, closed-form ridge fit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E031_triplets_vs_rm.png", dpi=160)
    plt.close(fig)


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E031")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    out = results_root("E031", smoke)
    figures = out / "figures" if smoke else directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    action = argv[0] if argv else "all"
    if action in ("fit", "all"):
        fit(config, smoke, out)
    if action in ("summarize", "all"):
        summarize(config, out, figures)
    atomic_write_json(out / "status.json", {"done": action})
    print(f"E031: {out / 'triplet_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
