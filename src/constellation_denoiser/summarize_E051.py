"""E051 summary: strict A-M (LK) against A1 / A3 and the pruning rules (cross, hyperbola, magnitude).

    TRIPLET_KERR_EXP=E051 PYTHONPATH=src python -m constellation_denoiser.summarize_E051

Reads results/fits/*.csv (5 bootstrap replicas) and results/fits_split*/ (3 E023 splits); writes a_summary.csv (also the
input of e046_fit splits), lk_verdicts.csv, pruning.csv and figures/E051_gain_vs_offaxis_terms.png,
figures/E051_lk_vs_a1_rm.png. Rule: config.json rule_draft.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .runner import exp_dir, results_root

KEYS = ["arch", "mode", "M", "n_c", "set"]


def load(folder) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby(KEYS)
    out = g.agg(n_terms=("n_terms", "first"), rm=("rm", "first"), n=("replica", "size"), val_mean=("val_reduction", "mean"),
                val_std=("val_reduction", "std"), test_mean=("test_reduction", "mean"),
                test_std=("test_reduction", "std"), q2_mean=("q2_gain_db", "mean")).reset_index()
    out["test_se"] = out.test_std / np.sqrt(out.n)
    out["pruning"] = out.set.str.extract(r"^(mag|geo|cross|full|none)")[0]
    return out


def compare(a, b) -> dict:
    diff = a.test_mean - b.test_mean
    se = float(np.sqrt(a.test_std ** 2 / a.n + b.test_std ** 2 / b.n))
    std = float(max(a.test_std, b.test_std))
    return {"diff_pp": diff, "two_se": 2 * se, "two_std": 2 * std, "signal": bool(diff > 2 * se and diff > 2 * std),
            "worse_signal": bool(-diff > 2 * se and -diff > 2 * std)}


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    res = results_root("E051", smoke)
    figs = exp_dir("E051") / ("results_smoke" if smoke else "figures")
    figs.mkdir(parents=True, exist_ok=True)
    s = summary(load(res / "fits"))
    s.to_csv(res / "a_summary.csv", index=False)
    rows = []
    lk = s[s.arch == "LK"]
    for _, a in lk.iterrows():
        for other in ("A1", "A3"):
            b = s[(s.arch == other) & (s.M == a.M) & (s.n_c == a.n_c) & (s.set == a.set)]
            if len(b):
                b = b.iloc[0]
                rows.append(dict(compare(a, b), question=f"LK {a['mode']} vs {other} (same M, N_c, set)", M=a.M, n_c=a.n_c,
                                 set=a.set, n_terms=a.n_terms, rm=a.rm, lk_test=a.test_mean, other_test=b.test_mean,
                                 other_mode=b["mode"]))
    best = s[s.set == "full"].loc[lambda f: f.groupby("arch").val_mean.idxmax()]
    best.to_csv(res / "best_by_validation.csv", index=False)
    splits = [load(p) for p in sorted(res.glob("fits_split*"))]
    splits = pd.concat([x for x in splits if len(x)], ignore_index=True) if any(len(x) for x in splits) else pd.DataFrame()
    if len(splits):
        sp = splits[splits.set == "full"].groupby(["split_seed", "arch"]).test_reduction.mean().unstack("arch")
        sp.to_csv(res / "splits_full.csv")
        for other in ("A1", "A3"):
            if "LK" in sp and other in sp:
                d = sp["LK"] - sp[other]
                rows.append({"question": f"LK (best by val) - {other} (best by val) on the 3 E023 splits", "diff_pp": d.mean(),
                             "per_split": json.dumps({str(k): round(v, 3) for k, v in d.items()}),
                             "signal": bool((d > 0).all()), "worse_signal": bool((d < 0).all())})
    pd.DataFrame(rows).to_csv(res / "lk_verdicts.csv", index=False)
    # pruning at a fixed Kerr: the best (mode, M, N_c) of LK and A1 by validation of the full set
    prune_rows = []
    for arch in ("LK", "A1"):
        b = best[best.arch == arch]
        if not len(b):
            continue
        b = b.iloc[0]
        part = s[(s.arch == arch) & (s["mode"] == b["mode"]) & (s.M == b.M) & (s.n_c == b.n_c)]
        prune_rows.append(part.assign(fixed=f"{arch} {b['mode']} M{b.M} Nc{b.n_c}"))
        cross, geo = part[part.pruning == "cross"], part[part.pruning == "geo"]
        for _, c in cross.iterrows():
            if len(geo):
                g = geo.loc[(geo.n_terms - c.n_terms).abs().idxmin()]
                rows.append(dict(compare(c, g), question=f"{arch}: cross vs hyperbola at the nearest number of terms",
                                 set=f"{c.set} vs {g.set}", n_terms=f"{c.n_terms} vs {g.n_terms}", lk_test=c.test_mean,
                                 other_test=g.test_mean))
    pd.DataFrame(rows).to_csv(res / "lk_verdicts.csv", index=False)
    prune = pd.concat(prune_rows) if prune_rows else pd.DataFrame()
    prune.to_csv(res / "pruning.csv", index=False)
    plot(s, prune, figs)
    print(pd.DataFrame(rows).to_string(index=False) if rows else "нет сравнений")
    return 0


def plot(s, prune, figs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fixed, part in prune.groupby("fixed") if len(prune) else []:
        for pr, mk in (("mag", "o"), ("geo", "s"), ("cross", "^")):
            p = part[part.pruning.isin([pr, "full", "none"])].sort_values("n_terms")
            ax.errorbar(p.n_terms, p.test_mean, yerr=p.test_std, marker=mk, label=f"{fixed}, {pr}", capsize=2)
    ax.set_xlabel("число внеосевых членов")
    ax.set_ylabel("снижение BER на test, %")
    ax.set_xscale("symlog")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "E051_gain_vs_offaxis_terms.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (arch, mode), part in s.groupby(["arch", "mode"]):
        ax.plot(part.rm, part.test_mean, ".", alpha=0.6, label=f"{arch} {mode}")
    ax.set_xscale("log")
    ax.set_xlabel("RM (конвенция проекта)")
    ax.set_ylabel("снижение BER на test, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "E051_lk_vs_a1_rm.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
