"""E053 summary: H1 (S chirps instead of the off-axis table, on the data) and H2 (recurrence check, RM versus M).

    PYTHONPATH=src python -m constellation_denoiser.summarize_E053

Inputs: results/runs/*.csv (e053_data), results/table_fit.csv (e053_table_fit). Writes results/summary.csv,
results/verdicts.csv, results/recurrence_check.csv, results/rm_vs_M.csv, figures/E053_reduction_vs_S.png,
figures/E053_rm_vs_M.png. Rule: config.json rule_draft.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .chirp_triplets import check, rm_direct, rm_fast_chirp, rm_oc_reuse, rm_xiao_offaxis
from .runner import exp_dir, results_root

KERR_RM = {"A1_M8_nc31": 4 * 7 + 2 + 16 + 4, "A1_M11_nc63": 4 * 7 + 2 + 32 + 4}
M_OF = {"A1_M8_nc31": 8, "A1_M11_nc63": 11}


def rm_of(row) -> dict:
    M = M_OF.get(row.base, 3)
    kerr = KERR_RM.get(row.base, 0)
    if row.variant == "full":
        return {"rm_project": kerr + rm_direct(M)["rm"], "rm_fast": np.nan}
    if row.variant == "none":
        return {"rm_project": kerr, "rm_fast": kerr}
    if row.variant.startswith("chirp") or row.variant.startswith("random"):
        Ws = json.loads(row.params)["W"] if isinstance(row.params, str) and row.params else []
        fast = kerr + sum(rm_fast_chirp(M, w)["rm"] for w in Ws)
        return {"rm_project": kerr + rm_direct(M)["rm"], "rm_fast": fast}
    return {"rm_project": kerr + rm_direct(M)["rm"], "rm_fast": np.nan}


def compare(a_mean, a_std, a_n, b_mean, b_std, b_n) -> dict:
    diff = a_mean - b_mean
    se = float(np.sqrt(a_std ** 2 / a_n + b_std ** 2 / b_n))
    std = float(max(a_std, b_std))
    return {"diff_pp": diff, "two_se": 2 * se, "two_std": 2 * std,
            "worse_by_both": bool(-diff > 2 * se and -diff > 2 * std), "better_by_both": bool(diff > 2 * se and diff > 2 * std)}


def main(smoke: bool | None = None) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1" if smoke is None else smoke
    res = results_root("E053", smoke)
    figs = exp_dir("E053") / ("results_smoke" if smoke else "figures")
    figs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(check()).to_csv(res / "recurrence_check.csv", index=False)
    rm_rows = []
    for M in range(3, 41):
        d, o = rm_direct(M), rm_oc_reuse(M)
        row = {"M": M, "terms": d["terms"], "rm_direct": d["rm"], "rm_oc_reuse": o["rm_oc"], "rm_grouped": o["rm_grouped"],
               "adds_direct": d["adds_real"]}
        for S in (1, 3, 5):
            f = rm_fast_chirp(M, 1e9, exp_computed=True)
            row[f"rm_chirp_S{S}"] = S * f["rm"]
            row[f"rm_chirp_S{S}_exp_computed"] = S * f["rm_exp_computed"]
            row[f"adds_chirp_S{S}"] = S * f["adds_real"]
        for K in (1, 2, 3):
            row[f"rm_xiao_K{K}"] = round(rm_xiao_offaxis(K, M), 1)
        row["rm_xiao_K=M"] = round(rm_xiao_offaxis(M, M), 1)
        row["rm_xiao_K=M_nf_M+1"] = round(rm_xiao_offaxis(M, M, "M+1"), 1)
        rm_rows.append(row)
    rm = pd.DataFrame(rm_rows)
    rm.to_csv(res / "rm_vs_M.csv", index=False)
    plot_rm(rm, figs)
    runs_dir = res / "runs"
    files = sorted(runs_dir.glob("*.csv")) if runs_dir.is_dir() else []
    if not files:
        print("нет прогонов")
        return 0
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    t = pd.concat([t, pd.DataFrame([rm_of(r) for r in t.itertuples()])], axis=1)
    main_t = t[t.split_seed.astype(str) == "E004"]
    g = main_t.groupby(["base", "variant", "size"])
    s = g.agg(n=("replica", "size"), n_real_params=("n_real_params", "first"), test_mean=("test_reduction", "mean"),
              test_std=("test_reduction", "std"), val_mean=("val_reduction", "mean"), q2_mean=("q2_gain_db", "mean"),
              tuning_evals=("tuning_evals", "mean"), fit_time_s=("fit_time_s", "mean"), rm_project=("rm_project", "mean"),
              rm_fast=("rm_fast", "mean")).reset_index()
    s["test_se"] = s.test_std / np.sqrt(s.n)
    s["test_range"] = g.test_reduction.agg(lambda x: x.max() - x.min()).values
    s.to_csv(res / "summary.csv", index=False)
    verdicts = []
    for base, part in s.groupby("base"):
        full = part[part.variant == "full"].iloc[0]
        none = part[part.variant == "none"].iloc[0]
        for _, r in part[part.variant.str.startswith(("chirp", "random", "radial", "lowrank"))].iterrows():
            c = compare(r.test_mean, r.test_std, r.n, full.test_mean, full.test_std, full.n)
            verdicts.append(dict(c, base=base, question="variant vs (i) full table", variant=r.variant, size=r["size"],
                                 n_real_params=r.n_real_params, test=r.test_mean, ref_test=full.test_mean,
                                 within_std_of_full=bool(full.test_mean - r.test_mean < full.test_std),
                                 share_of_ifwm_gain=(r.test_mean - none.test_mean) / (full.test_mean - none.test_mean)))
        ok = part[part.variant.str.startswith("chirp") & (part["size"] <= 5) & (part.n_real_params <= 20)]
        passed = ok[full.test_mean - ok.test_mean < full.test_std]
        s8 = part[part.variant.str.startswith("chirp") & (part["size"] == part["size"].max())]
        rejected = bool(len(s8) and all(compare(r.test_mean, r.test_std, r.n, full.test_mean, full.test_std, full.n)
                                        ["worse_by_both"] for _, r in s8.iterrows()))
        verdict = "подтверждено" if len(passed) else ("отвергнуто" if rejected else "не доказано")
        verdicts.append({"base": base, "question": "H1: S <= 5 chirps (<= 20 parameters) within std of (i)",
                         "verdict": verdict, "passing": ", ".join(f"{a} S{b}" for a, b in zip(passed.variant, passed["size"])),
                         "ref_test": full.test_mean, "ref_std": full.test_std, "ref_n": full.n})
        for _, r in part[part.variant == "chirp_imag"].iterrows():      # (iii) at equal number of parameters
            for ctrl in ("radial", "random_imag"):
                c_part = part[part.variant == ctrl]
                if len(c_part):
                    q = c_part.loc[(c_part.n_real_params - r.n_real_params).abs().idxmin()]
                    c = compare(r.test_mean, r.test_std, r.n, q.test_mean, q.test_std, q.n)
                    verdicts.append(dict(c, base=base, question=f"chirp_imag vs {ctrl} at the nearest number of parameters",
                                         variant=f"chirp_imag S{r['size']} ({r.n_real_params} p)",
                                         size=f"{ctrl} {q['size']} ({q.n_real_params} p)", test=r.test_mean,
                                         ref_test=q.test_mean))
    sp = t[t.split_seed.astype(str) != "E004"]
    if len(sp):
        spg = sp.groupby(["split_seed", "variant", "size"]).test_reduction.agg(mean="mean", std="std", n="count").reset_index()
        spg.to_csv(res / "splits_summary.csv", index=False)
        for seed, part in spg.groupby("split_seed"):
            full = part[part.variant == "full"].iloc[0]
            for _, r in part[part.variant.isin(["chirp_imag", "chirp_joint_imag"]) & part["size"].isin([3, 5, 8])].iterrows():
                c = compare(r["mean"], r["std"], r["n"], full["mean"], full["std"], full["n"])
                verdicts.append(dict(c, question=f"split {seed}: chirp vs (i)", variant=r.variant, size=r["size"],
                                     test=r["mean"], ref_test=full["mean"]))
    pd.DataFrame(verdicts).to_csv(res / "verdicts.csv", index=False)
    plot_s(s, figs)
    print(pd.DataFrame([v for v in verdicts if "verdict" in v]).to_string(index=False))
    return 0


def plot_s(s, figs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bases = list(s.base.unique())
    fig, axes = plt.subplots(1, len(bases), figsize=(6.5 * len(bases), 4.6), squeeze=False)
    for ax, base in zip(axes[0], bases):
        p = s[s.base == base]
        full, none = p[p.variant == "full"].iloc[0], p[p.variant == "none"].iloc[0]
        ax.axhline(full.test_mean, color="k", label=f"(i) полная таблица ({int(full.n_real_params)} пар.)")
        ax.axhspan(full.test_mean - full.test_std, full.test_mean + full.test_std, color="k", alpha=0.1)
        ax.axhline(none.test_mean, color="gray", ls="--", label="(ii) только Керр")
        for v, mk in (("chirp_complex", "o"), ("chirp_imag", "s"), ("chirp_joint_imag", "*"), ("random_imag", "x")):
            q = p[p.variant == v].sort_values("size")
            if len(q):
                ax.errorbar(q["size"], q.test_mean, yerr=q.test_std, marker=mk, capsize=2, label=v)
        q = p[p.variant == "radial"].sort_values("n_real_params")
        if len(q):     # radial plotted at the S of a chirp_imag with the same number of parameters (3S)
            ax.errorbar(q.n_real_params / 3, q.test_mean, yerr=q.test_std, marker="d", ls=":", capsize=2,
                        label="(iii) радиальный базис (по оси: пар./3)")
        ax.set_xlabel("S (число чирпов)")
        ax.set_ylabel("снижение BER на test, %")
        ax.set_title(base)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "E053_reduction_vs_S.png", dpi=150)
    plt.close(fig)


def plot_rm(rm, figs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(rm.M, rm.rm_direct, label="(1) таблица C, 12 RM на член")
    ax.plot(rm.M, rm.rm_oc_reuse, label="(2) повторное использование (OC 3.3.3, 1 пол.)")
    ax.plot(rm.M, rm.rm_grouped, ls="--", label="(2b) то же + группировка по m")
    for S in (1, 3, 5):
        ax.plot(rm.M, rm[f"rm_chirp_S{S}"], label=f"(3) рекурсия, S = {S} чирп(ов)")
    ax.plot(rm.M, rm["rm_xiao_K=M"], "k:", label="Xiao: K (2 C_conv(2M+1) + 12), K = M (весь домен)")
    ax.plot(rm.M, rm["rm_xiao_K1"], "k-.", label="Xiao: K = 1")
    ax.set_yscale("log")
    ax.set_xlabel("M (домен T3, внеосевые члены)")
    ax.set_ylabel("RM на символ (без Керра)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "E053_rm_vs_M.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
