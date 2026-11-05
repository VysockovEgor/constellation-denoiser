"""E050 summary: DBP-X (start without training and trained) against the windowed DBP of E028 on two RM scales.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E050

Inputs: results/start_best.csv, results/unit_test.json, results/dbpx_runs/*.json (dbpx train), results/runs/*.json
(stage ref, runner), E028 results/runs_table.csv (grid a, phys_dbp). Writes results/start_table.csv,
results/train_summary.csv, results/verdicts.csv, figures/E050_dbpx_vs_rm.png. The rule is config.json rule_draft.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

from .dbpx import c_conv
from .runner import exp_dir, results_root


def xiao_windowed(ns: int, L: int, R: int) -> float:
    """Xiao column for the windowed DBP of E028 (no D_total): Ns (C_conv(L) + C_conv(R)/4 + 6)."""
    return round(ns * (c_conv(L) + c_conv(R) / 4 + 6), 1)


def windowed_points(smoke: bool) -> pd.DataFrame:
    """Per-seed rows of the E028 DBP (grid a) and of the E050 reference (stage ref)."""
    rows = []
    t = pd.read_csv(exp_dir("E028") / "results" / "runs_table.csv")
    t = t[(t.stage == "a") & t.variant.str.match(r"dbp_Ns\d+_L\d+_R\d+$")]
    for _, r in t.iterrows():
        rows.append({"source": "E028 a", "variant": r.variant, "seed": r.seed,
                     "test": r.test_reduction_valbest_select, "val": r.val_reduction, "rm_project": r.macs_per_symbol})
    runs = results_root("E050", smoke) / "runs"
    for path in sorted(runs.glob("*.json")) if runs.is_dir() else []:
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("stage") != "ref":
            continue
        rows.append({"source": "E050 ref", "variant": r["variant"], "seed": r["seed"],
                     "test": r["test_reduction_valbest_select"], "val": r.get("val_reduction"),
                     "rm_project": r.get("macs_per_symbol")})
    out = pd.DataFrame(rows)
    if len(out):
        p = out.variant.str.extract(r"Ns(\d+)_L(\d+)_R(\d+)").astype(int)
        out["n_steps"], out["L"], out["R"] = p[0], p[1], p[2]
        out["rm_xiao"] = [xiao_windowed(a, b, c) for a, b, c in zip(out.n_steps, out.L, out.R)]
    return out


def group(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = frame.groupby(keys)
    out = g.agg(n=("test", "size"), test_mean=("test", "mean"), test_std=("test", "std"), val_mean=("val", "mean"),
                rm_project=("rm_project", "first"), rm_xiao=("rm_xiao", "first")).reset_index()
    out["test_se"] = out.test_std / np.sqrt(out.n)
    return out


def compare(a_mean, a_std, a_n, b_mean, b_std, b_n) -> dict:
    diff = a_mean - b_mean
    se = float(np.sqrt((a_std or 0) ** 2 / max(a_n, 1) + (b_std or 0) ** 2 / max(b_n, 1)))
    std = float(max(a_std or 0, b_std or 0))
    return {"diff_pp": diff, "two_se": 2 * se, "two_std": 2 * std, "signal": bool(diff > 2 * se and diff > 2 * std)}


def nearest(ref: pd.DataFrame, rm: float, column: str = "rm_xiao") -> pd.Series:
    """The reference group with the closest RM that is not higher (else the closest)."""
    lower = ref[ref[column] <= rm]
    pool = lower if len(lower) else ref
    return pool.loc[(pool[column] - rm).abs().idxmin()]


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    res = results_root("E050", smoke)
    fig_dir = exp_dir("E050") / ("results_smoke" if smoke else "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    verdicts = []
    win = windowed_points(smoke)
    win_g = group(win, ["source", "variant", "n_steps", "L", "R"]) if len(win) else pd.DataFrame()
    if len(win_g):
        win_g.to_csv(res / "windowed_dbp_summary.csv", index=False)

    start = pd.read_csv(res / "start_best.csv") if (res / "start_best.csv").is_file() else pd.DataFrame()
    if len(start):
        sign_of = {ns: p.loc[p.val_mse_trim.idxmin(), "sign"] for ns, p in start[start.alpha_kind == "classical"]
                   .groupby("n_steps")}
        st = start[start.sign == start.n_steps.map(sign_of)].copy()
        st.to_csv(res / "start_table.csv", index=False)
        sat = []                                   # saturation over Ns: empirical ceiling, not a measured limit
        for kind, part in st.groupby("alpha_kind"):
            part = part.sort_values("n_steps")
            col = "test_honest_reduction_trim" if "test_honest_reduction_trim" in part else "test_reduction_trim"
            for _, r in part[part.n_steps.isin([10, 16, 20, 40, 80])].iterrows():
                prev = part[part.n_steps < r.n_steps]
                sat.append({"alpha_kind": kind, "n_steps": int(r.n_steps), "test_exact": r.test_reduction_trim,
                            "test_honest": r.get("test_honest_reduction_trim"), "rm_project": r.get("rm_project"),
                            "rm_xiao": r.get("rm_xiao"),
                            "gain_over_previous_pp": r[col] - prev.iloc[-1][col] if len(prev) else np.nan})
            best = part.loc[part.val_mse_trim.idxmin()]
            sat.append({"alpha_kind": kind, "n_steps": "best by validation", "test_exact": best.test_reduction_trim,
                        "test_honest": best.get("test_honest_reduction_trim"),
                        "note": "эмпирический потолок по сетке Ns <= 80 (без обучения), не измеренный предел"})
        pd.DataFrame(sat).to_csv(res / "saturation.csv", index=False)
        for _, r in st.iterrows():
            if "rm_xiao" not in r or pd.isna(r.get("rm_xiao")) or not len(win_g):
                continue
            ref = nearest(win_g, r.rm_xiao)
            v = compare(r.test_honest_reduction_trim, 0.0, 1, ref.test_mean, ref.test_std, ref.n)
            verdicts.append(dict(v, question="start: DBP-X (no training, honest lengths) vs windowed DBP",
                                 arm=f"start Ns{int(r.n_steps)} {r.alpha_kind}", arm_test=r.test_honest_reduction_trim,
                                 arm_rm_xiao=r.rm_xiao, arm_rm_project=r.rm_project, ref=f"{ref.source} {ref.variant}",
                                 ref_test=ref.test_mean, ref_rm_xiao=ref.rm_xiao, ref_n=ref.n))

    runs_dir = res / "dbpx_runs"
    trained = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*.json"))] \
        if runs_dir.is_dir() else []
    tr_g = pd.DataFrame()
    if trained:
        t = pd.DataFrame([{k: r[k] for k in ("name", "ns", "L", "R", "alpha_init", "training", "seed", "plateau_iter_0p5pp",
                                             "best_iter", "diverged", "start_val_reduction_trim", "val_reduction_trim",
                                             "test_reduction_valbest_select", "test_reduction_c31", "test_q2_gain_trim",
                                             "rm_project", "rm_xiao", "rm_merged", "time_s")} for r in trained])
        t.to_csv(res / "dbpx_runs_table.csv", index=False)
        g = t.groupby(["ns", "L", "alpha_init", "training"])
        tr_g = g.agg(n=("seed", "size"), test_mean=("test_reduction_valbest_select", "mean"),
                     test_std=("test_reduction_valbest_select", "std"), val_mean=("val_reduction_trim", "mean"),
                     start_val=("start_val_reduction_trim", "mean"), plateau_mean=("plateau_iter_0p5pp", "mean"),
                     plateau_std=("plateau_iter_0p5pp", "std"), q2_mean=("test_q2_gain_trim", "mean"),
                     diverged=("diverged", "sum"), rm_project=("rm_project", "first"), rm_xiao=("rm_xiao", "first"),
                     rm_merged=("rm_merged", "first")).reset_index()
        tr_g["test_se"] = tr_g.test_std / np.sqrt(tr_g.n)
        tr_g["test_range"] = g["test_reduction_valbest_select"].agg(lambda s: s.max() - s.min()).values
        tr_g.to_csv(res / "train_summary.csv", index=False)
        for _, r in tr_g.iterrows():
            if len(win_g):
                ref = nearest(win_g, r.rm_xiao)
                v = compare(r.test_mean, r.test_std, r.n, ref.test_mean, ref.test_std, ref.n)
                verdicts.append(dict(v, question="trained DBP-X vs windowed DBP at the same or lower RM (xiao)",
                                     arm=f"Ns{r.ns} L{r.L} {r.alpha_init} {r.training}", arm_test=r.test_mean,
                                     arm_rm_xiao=r.rm_xiao, arm_rm_project=r.rm_project,
                                     ref=f"{ref.source} {ref.variant}", ref_test=ref.test_mean,
                                     ref_rm_xiao=ref.rm_xiao, ref_n=ref.n))
        for (ns, L, kind), part in tr_g.groupby(["ns", "L", "alpha_init"]):
            one, two = part[part.training == "one"], part[part.training == "two"]
            if len(one) and len(two):
                a, b = two.iloc[0], one.iloc[0]
                v = compare(a.test_mean, a.test_std, a.n, b.test_mean, b.test_std, b.n)
                verdicts.append(dict(v, question="two stages (Xiao) better than one stage", arm=f"Ns{ns} L{L} {kind} two",
                                     arm_test=a.test_mean, ref=f"Ns{ns} L{L} {kind} one", ref_test=b.test_mean,
                                     plateau_two=a.plateau_mean, plateau_one=b.plateau_mean))
    pd.DataFrame(verdicts).to_csv(res / "verdicts.csv", index=False)
    plot(win_g, start, tr_g, fig_dir)
    print(pd.DataFrame(verdicts).to_string(index=False) if verdicts else "нет сравнений")
    return 0


def plot(win_g, start, tr_g, fig_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, col, title in ((axes[0], "rm_xiao", "RM, формулы Xiao (overlap-save)"),
                           (axes[1], "rm_project", "RM, конвенция проекта (прямой КИХ)")):
        if len(win_g):
            for src, mk in (("E028 a", "o"), ("E050 ref", "s")):
                p = win_g[win_g.source == src]
                ax.errorbar(p[col], p.test_mean, yerr=p.test_std, fmt=mk, color="tab:gray", label=f"DBP в окне ({src})")
        if len(start) and col in start:
            for kind, mk in (("classical", "^"), ("ridge", "v")):
                p = start[(start.alpha_kind == kind) & start[col].notna()]
                if "test_honest_reduction_trim" in p:
                    ax.plot(p[col], p.test_honest_reduction_trim, mk, color="tab:blue", mfc="none",
                            label=f"DBP-X старт без обучения ({kind})")
        if len(tr_g):
            for kind, color in (("zero", "tab:orange"), ("ridge", "tab:green"), ("classical", "tab:red")):
                for training, mk in (("one", "D"), ("two", "P")):
                    p = tr_g[(tr_g.alpha_init == kind) & (tr_g.training == training)]
                    if len(p):
                        ax.errorbar(p[col], p.test_mean, yerr=p.test_std, fmt=mk, color=color,
                                    label=f"DBP-X обучен, α₀={kind}, {training}")
        ax.set_xscale("log")
        ax.set_xlabel(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("снижение BER на test, %")
    axes[0].set_ylim(bottom=-10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(fig_dir / "E050_dbpx_vs_rm.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
