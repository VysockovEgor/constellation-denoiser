"""E056 summary: ablation of A1 (Kerr with memory + off-axis triplets) at equal RM.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E056

Arms (rows of results/fits/*.csv, 5 bootstrap replicas, E004 split):
  1 all triplets            T            E046 (M 3/5/8), E048 (M 11/15)
  2 Kerr N_c=1 + off-axis   A1, N_c = 1  E056
  3 Kerr memory, no off-ax  K,  N_c > 1  E046 (7/15/31), E048 (31/63)
  4 Kerr memory + off-axis  A1, N_c > 1  E046, E048
  5 Kerr N_c=1, no off-ax   K,  N_c = 1  E056
Per arm and RM level (config rm_levels, tolerance): the configuration (source, M, N_c, set) with rm <= level (1 + tol)
and the best mean validation; test mean, std, SE. Writes results/ablation.csv, results/ablation_verdicts.csv,
figures/E056_ablation_vs_rm.png. Rule: config.json rule_draft.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .runner import exp_dir, results_root

ARMS = {1: "1 все триплеты (T)", 2: "2 Керр без памяти + внеосевые", 3: "3 Керр с памятью, без внеосевых",
        4: "4 Керр с памятью + внеосевые (A1)", 5: "5 Керр без памяти, без внеосевых"}


def load(smoke: bool) -> pd.DataFrame:
    frames = []
    for exp in ("E046", "E048", "E056"):
        d = results_root(exp, smoke if exp == "E056" else False) / "fits"
        if d.is_dir():
            for f in d.glob("*.csv"):
                frames.append(pd.read_csv(f).assign(source=exp))
    t = pd.concat(frames, ignore_index=True)
    t = t[t.arch.isin(["T", "K", "A1"]) & ~((t.arch == "A1") & (t["mode"] != "varpro"))]

    def arm(r):
        if r.arch == "T":
            return 1
        if r.arch == "A1":
            return 2 if r.n_c == 1 else (4 if r.set != "none" else None)
        return 5 if r.n_c == 1 else 3

    t["arm"] = [arm(r) for r in t.itertuples()]
    t = t[t.arm.notna()]
    t.loc[(t.arch == "A1") & (t.n_c == 1) & (t.set == "none"), "arm"] = None
    return t[t.arm.notna()].astype({"arm": int})


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    cfg = json.loads((exp_dir("E056") / "config.json").read_text(encoding="utf-8"))
    res = results_root("E056", smoke)
    figs = exp_dir("E056") / ("results_smoke" if smoke else "figures")
    figs.mkdir(parents=True, exist_ok=True)
    t = load(smoke)
    keys = ["arm", "source", "M", "n_c", "set"]
    g = t.groupby(keys).agg(n=("replica", "nunique"), rm=("rm", "first"), n_terms=("n_terms", "first"),
                            val_mean=("val_reduction", "mean"), test_mean=("test_reduction", "mean"),
                            test_std=("test_reduction", "std"), q2_mean=("q2_gain_db", "mean")).reset_index()
    g["test_se"] = g.test_std / np.sqrt(g.n)
    g.to_csv(res / "ablation_all_configs.csv", index=False)
    rows = []
    for level in cfg["rm_levels"]:
        cap = level * (1 + cfg["rm_tolerance"])
        for a in ARMS:
            part = g[(g.arm == a) & (g.rm <= cap)]
            if not len(part):
                rows.append({"level": level, "arm": a, "name": ARMS[a], "note": "нет конфигурации с RM <= уровня"})
                continue
            b = part.loc[part.val_mean.idxmax()]
            rows.append({"level": level, "arm": a, "name": ARMS[a], "source": b.source, "M": b.M, "n_c": b.n_c,
                         "set": b.set, "n_terms": b.n_terms, "rm": b.rm, "n": b.n, "val_mean": b.val_mean,
                         "test_mean": b.test_mean, "test_std": b.test_std, "test_se": b.test_se,
                         "note": "RM фиксирован (только Керр)" if a in (3, 5) else ""})
    ab = pd.DataFrame(rows)
    ab.to_csv(res / "ablation.csv", index=False)
    ver = []
    for level, part in ab.groupby("level"):
        p = part.set_index("arm")

        def cmp(x, y, what):
            if x not in p.index or y not in p.index or pd.isna(p.loc[x, "test_mean"]) or pd.isna(p.loc[y, "test_mean"]):
                return
            a, b = p.loc[x], p.loc[y]
            diff = a.test_mean - b.test_mean
            se = float(np.sqrt(a.test_std ** 2 / a.n + b.test_std ** 2 / b.n))
            std = float(max(a.test_std, b.test_std))
            ver.append({"level": level, "question": what, "diff_pp": diff, "two_se": 2 * se, "two_std": 2 * std,
                        "signal": bool(diff > 2 * se and diff > 2 * std),
                        "reverse_signal": bool(-diff > 2 * se and -diff > 2 * std)})

        cmp(4, 2, "память Керра: (4) - (2)")
        cmp(4, 3, "внеосевые: (4) - (3)")
        cmp(2, 1, "форма (Керр N_c=1 + внеосевые) против свободных осевых: (2) - (1)")
        cmp(4, 1, "A1 против всех триплетов: (4) - (1)")
        cmp(3, 5, "память без внеосевых: (3) - (5)")
    pd.DataFrame(ver).to_csv(res / "ablation_verdicts.csv", index=False)
    plot(g, figs)
    print(ab.round(2).to_string(index=False))
    print(pd.DataFrame(ver).round(2).to_string(index=False))
    return 0


def plot(g, figs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for a, name in ARMS.items():
        part = g[g.arm == a].sort_values("rm")
        if not len(part):
            continue
        env, best = [], -1e9                          # upper envelope by validation over RM
        for _, r in part.iterrows():
            if r.val_mean > best:
                best = r.val_mean
                env.append(r)
        env = pd.DataFrame(env)
        ax.errorbar(env.rm, env.test_mean, yerr=env.test_std, marker="o", capsize=2, label=name)
    for lv in (430, 1200, 2200):
        ax.axvline(lv, color="gray", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("RM (конвенция проекта)")
    ax.set_ylabel("снижение BER на test, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "E056_ablation_vs_rm.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
