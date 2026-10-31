"""E034: triplets with larger memory (M = 25, 31) and pruning to K terms (closed form, CPU).

    PYTHONPATH=src python -m constellation_denoiser.summarize_E034 [fit|summarize|all]   (SMOKE=1: 2/1/1 files, small M)

Writes results/triplet_ext_fits.csv, results/runs_table.csv (aggregator format), results/triplet_ext_summary.csv,
results/triplet_ext_verdicts.csv and figures/E034_triplets_pruned_vs_rm.png.
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
    from .triplets import FileGram, fit_arm, triplet_terms
    from .triplets_ext import prune_replica
    from .window_data import load_fileset, resolve_paths

    files = default_files(config["n_train_files"], smoke)
    data = {k: load_fileset(resolve_paths(v)) for k, v in files.items()}
    tr, va, te = data["train"], data["validation"], data["test"]
    raw_val, raw_test = va.raw_ber(31), te.raw_ber(31)
    reps = config["bootstrap"]["replicas"][:1] if smoke else config["bootstrap"]["replicas"]
    lambdas, block, fir = config["lambda_grid"], config["max_block"], config["fir_len"]
    rows = []
    base_m = config["smoke_prune_M"] if smoke else config["prune_M"]
    terms = triplet_terms(base_m, "T3")
    grams = [FileGram(tr.rx_c[f], tr.tx_c[f], terms, fir, False, block) for f in range(tr.n_files)]
    for replica in reps:
        rows += prune_replica(tr, va, te, base_m, fir, config["K"], lambdas, replica, grams, terms, block)
        print(f"pruning M={base_m}, replica {replica}: done", flush=True)
    del grams
    for M in (config["smoke_M"] if smoke else config["M"]):
        for r in fit_arm(tr, va, te, M, "T3", fir, lambdas, reps, block):
            rows.append({"variant": f"T3FIR{fir}_M{M}_full", "K": r["n_terms"], "replica": r["replica"],
                         "lambda": r["lambda"], "val_ber": r["val_ber"], "test_ber": r["test_ber"], "rm_sym": r["rm_sym"]})
        print(f"T3+FIR{fir} M={M}: done", flush=True)
    frame = pd.DataFrame(rows)
    frame["val_reduction"] = 100 * (raw_val - frame.val_ber) / raw_val
    frame["test_reduction"] = 100 * (raw_test - frame.test_ber) / raw_test
    frame["q2_gain_db"] = [q2_db(t) - q2_db(raw_test) for t in frame.test_ber]
    frame["raw_ber"] = raw_test
    frame["git_commit"] = git_commit()
    frame.to_csv(out / "triplet_ext_fits.csv", index=False)
    return frame


def summarize(config: dict, out, figures) -> None:
    frame = pd.read_csv(out / "triplet_ext_fits.csv")
    pd.DataFrame({"experiment": "E034", "stage": "main", "model": "triplets", "variant": frame.variant,
                  "seed": frame.replica, "budget": np.nan, "lr_name": "closed_form",
                  "best_val_reduction": frame.val_reduction, TEST: frame.test_reduction,
                  "valbest_select_q2_gain_db": frame.q2_gain_db, "macs_per_symbol": frame.rm_sym,
                  "macs_per_symbol_merged": frame.rm_sym, "raw_ber": frame.raw_ber, "spec_K": frame.K,
                  "diverged": False, "recipe_source": "closed form"}).to_csv(out / "runs_table.csv", index=False)
    summary = frame.groupby("variant").agg(
        K=("K", "first"), rm=("rm_sym", "first"), n_replicas=("replica", "count"), val_mean=("val_reduction", "mean"),
        test_mean=("test_reduction", "mean"), test_min=("test_reduction", "min"), test_max=("test_reduction", "max"),
        q2_mean=("q2_gain_db", "mean"), q2_min=("q2_gain_db", "min"), q2_max=("q2_gain_db", "max")).reset_index()
    summary = summary.sort_values("rm")
    summary.to_csv(out / "triplet_ext_summary.csv", index=False)
    get = summary.set_index("variant")
    fir = config["fir_len"]
    rows = []
    e031 = exp_dir("E031") / "results" / "triplet_summary.csv"
    m21 = None
    if e031.is_file():
        s = pd.read_csv(e031)
        hit = s[(s.arm == f"T3+FIR{fir}") & (s.M == 21)]
        m21 = float(hit.test_mean.iloc[0]) if len(hit) else None
    top_m = max(config["M"])
    if m21 is not None and f"T3FIR{fir}_M{top_m}_full" in get.index:
        gain = float(get.loc[f"T3FIR{fir}_M{top_m}_full", "test_mean"] - m21)
        rows.append({"hypothesis": f"рост с M продолжается (M={top_m} − M=21 ≥ 1 п.п.)", "value": gain,
                     "verdict": "подтверждено" if gain >= 1 else "не доказано", "detail": "E031 M=21 против E034"})
    full = f"T3FIR{fir}_M{config['prune_M']}_full"
    k = config["rule_k"]
    if full in get.index and f"T3FIR{fir}_M{config['prune_M']}_K{k}" in get.index:
        share = float(get.loc[f"T3FIR{fir}_M{config['prune_M']}_K{k}", "test_mean"] / get.loc[full, "test_mean"])
        rows.append({"hypothesis": f"K={k} членов сохраняют ≥ 90 % снижения полной модели M={config['prune_M']}",
                     "value": 100 * share, "verdict": "подтверждено" if share >= 0.9 else "не доказано", "detail": ""})
    rows.append({"hypothesis": "сжатые триплеты лучше Kerr-сети при равных RM (≥ 0.1 дБ, интервалы не пересекаются)",
                 "value": float("nan"), "verdict": "проверяется в E033", "detail": ""})
    v = pd.DataFrame(rows)
    v["rule_status"] = config["rule_draft"]["status"]
    v.to_csv(out / "triplet_ext_verdicts.csv", index=False)
    plot(summary, figures)


def plot(summary: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pruned = summary[summary.variant.str.contains("_K")]
    full = summary[summary.variant.str.endswith("_full")]
    for index, (frame, label, marker) in enumerate(((pruned, "pruned to K terms (M = 25 pool)", "o"),
                                                   (full, "all terms, M = 25 / 31", "s"))):
        ax.errorbar(frame.rm, frame.test_mean, yerr=[frame.test_mean - frame.test_min, frame.test_max - frame.test_mean],
                    marker=marker, color=plot_style.color(index), capsize=2, linestyle="-" if index == 0 else "",
                    label=label)
    for anchor, rm, text in ((66.1, 6480, "Kerr MLP h48 (E025 F2)"), (72.5, 16065, "Kerr MLP h119 (E025 F3)")):
        ax.plot([rm], [anchor], "D", color=plot_style.TEXT, markersize=6)
        ax.annotate(text, (rm, anchor), textcoords="offset points", xytext=(5, -12), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("real multiplies per symbol (sym count)")
    ax.set_ylabel("test BER reduction, % (min-max over 5 bootstrap replicas)")
    ax.set_title("Perturbation triplets + FIR7: memory and pruning (closed form)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E034_triplets_pruned_vs_rm.png", dpi=160)
    plt.close(fig)


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E034")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    out = results_root("E034", smoke)
    figures = out / "figures" if smoke else directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    action = argv[0] if argv else "all"
    if action in ("fit", "all"):
        fit(config, smoke, out)
    if action in ("summarize", "all"):
        summarize(config, out, figures)
    atomic_write_json(out / "status.json", {"done": action})
    print(f"E034: {out / 'triplet_ext_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
