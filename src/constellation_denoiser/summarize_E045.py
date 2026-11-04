"""E045 (phase 1): reference families on one RM scale from existing runs (no training).

    PYTHONPATH=src python -m constellation_denoiser.summarize_E045

Families: linear FIR (E022), DBP (E018/E025/E028), Kerr with memory = PhysKerr (E016/E025), full one-step triplets
(E031/E034/E036, closed form), DBP + selected triplets (E038), cascade of triplet stages (E040, if summarized).
Points: mean and std over training seeds (bootstrap replicas for closed form) on the E004 split; for each family the
Pareto points chosen by validation. Writes results/references.csv, results/references_pareto.csv and
figures/E045_references_vs_rm.png. Repository RM convention only (other conventions: E039/E043, not mixed here).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from .pareto_v2 import envelope, load_tables, points
from .q2 import q2_db
from .runner import exp_dir, results_root

FAMILIES = {"linear FIR": "1 linear (FIR)", "DBP": "2 DBP", "PhysKerr": "3 Kerr with memory (PhysKerr)",
            "PB-NLC (triplets)": "4 full triplets (one step)", "cascade": "5 cascade of triplet stages",
            "DBP + triplets": "DBP + selected triplets (E038)"}


def triplet_points() -> pd.DataFrame:
    rows = []
    path = exp_dir("E031") / "results" / "triplet_fits.csv"
    if path.is_file():
        f = pd.read_csv(path)
        f = f[f.arm.str.startswith("T3")]
        f["variant"] = f.arm + "_M" + f.M.astype(str)
        f["rm"] = f.rm_sym
        rows.append(f.assign(source="E031"))
    for exp in ("E034", "E036"):
        path = exp_dir(exp) / "results" / "triplet_ext_fits.csv"
        if path.is_file():
            f = pd.read_csv(path)
            rows.append(f.assign(source=exp, rm=f.rm_sym))
    if not rows:
        return pd.DataFrame()
    f = pd.concat(rows, ignore_index=True)
    g = f.groupby(["source", "variant"]).agg(rm=("rm", "first"), n_seeds=("replica", "count"), val_mean=("val_reduction", "mean"),
                                             test_mean=("test_reduction", "mean"), test_std=("test_reduction", "std"),
                                             test_min=("test_reduction", "min"), test_max=("test_reduction", "max"),
                                             q2_mean=("q2_gain_db", "mean"), q2_std=("q2_gain_db", "std")).reset_index()
    return g.assign(family="PB-NLC (triplets)", label=g.source + " · " + g.variant)


def table_points(exp: str, family: str, keys: list[str]) -> pd.DataFrame:
    path = exp_dir(exp) / "results" / "runs_table.csv"
    if not path.is_file():
        return pd.DataFrame()
    r = pd.read_csv(path)
    r = r[r.stage.isin(["main", "confirm"])] if "stage" in r else r
    if r.empty:
        return pd.DataFrame()
    raw = r.raw_ber.median()
    r["q2"] = [q2_db(raw * (1 - t / 100)) - q2_db(raw) for t in r.test_reduction_valbest_select]
    g = r.groupby(keys).agg(rm=("macs_per_symbol_merged", "first"), n_seeds=("seed", "nunique"),
                            val_mean=("best_val_reduction", "mean"), test_mean=("test_reduction_valbest_select", "mean"),
                            test_std=("test_reduction_valbest_select", "std"), test_min=("test_reduction_valbest_select", "min"),
                            test_max=("test_reduction_valbest_select", "max"), q2_mean=("q2", "mean"), q2_std=("q2", "std")).reset_index()
    g["label"] = exp + " · " + g[keys].astype(str).agg(" ".join, axis=1)
    return g.assign(family=family, source=exp)


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E045")
    out = results_root("E045", smoke)
    figures = out / "figures" if smoke else directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    pts = points(load_tables())
    pts = pts[pts.family.isin(["linear FIR", "DBP", "PhysKerr"])]
    frames = [pts.assign(label=pts.label), triplet_points(), table_points("E038", "DBP + triplets", ["variant"]),
              table_points("E040", "cascade", ["variant", "stage"])]
    refs = pd.concat([f for f in frames if len(f)], ignore_index=True, sort=False)
    refs["reference"] = refs.family.map(FAMILIES)
    cols = ["reference", "family", "label", "rm", "n_seeds", "val_mean", "test_mean", "test_std", "test_min", "test_max",
            "q2_mean", "q2_std"]
    refs[cols].to_csv(out / "references.csv", index=False)
    pareto = pd.concat([envelope(part.rename(columns={})) for _, part in refs.groupby("family")], ignore_index=True)
    pareto[cols].sort_values(["family", "rm"]).to_csv(out / "references_pareto.csv", index=False)
    plot(pareto, figures)
    print(pareto[cols].sort_values(["family", "rm"]).round(2).to_string(index=False))
    return 0


def plot(pareto: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for index, (family, part) in enumerate(pareto.groupby("family")):
        part = part.sort_values("rm")
        for ax, (metric, std, label) in zip(axes, (("test_mean", "test_std", "BER reduction, %"), ("q2_mean", "q2_std", "Delta Q2, dB"))):
            ax.errorbar(part.rm, part[metric], yerr=part[std].fillna(0), marker="osD^vP"[index % 6], capsize=2,
                        color=plot_style.SERIES[index % len(plot_style.SERIES)], label=FAMILIES.get(family, family))
            ax.set_xscale("log")
            ax.set_xlabel("real multiplies per symbol (repository convention)")
            ax.set_ylabel(label + " (mean +- std over seeds)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Phase 1: reference families, Pareto points by validation")
    fig.tight_layout()
    fig.savefig(figures / "E045_references_vs_rm.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
