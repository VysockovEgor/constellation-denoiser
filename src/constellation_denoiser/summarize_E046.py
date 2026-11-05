"""E046 summary: architecture A against triplets only and Kerr only at equal RM; gain vs kept off-axis terms.

Called by ``python -m constellation_denoiser.e046_fit summarize``. Writes results/fits_all.csv, results/a_summary.csv
(mean, std, min, max over replicas), results/a_verdicts.csv, figures E046_gain_vs_offaxis_terms.png and
E046_a_vs_controls_rm.png.
"""

from __future__ import annotations

import pandas as pd

from .runner import exp_dir, results_root
from .summary_e024 import write_verdicts


def summarize(config: dict, smoke: bool) -> int:
    exp = config["experiment_id"]
    out = results_root(exp, smoke)
    figures = out / "figures" if smoke else exp_dir(exp) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    files = sorted((out / "fits").glob("*.csv"))
    if not files:
        print("нет подгонок")
        return 1
    fits = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fits.to_csv(out / "fits_all.csv", index=False)
    keys = ["arch", "mode", "M", "n_c", "set"]
    s = fits.groupby(keys).agg(n_terms=("n_terms", "first"), rm=("rm", "first"), n_rep=("replica", "nunique"),
                               val_mean=("val_reduction", "mean"), test_mean=("test_reduction", "mean"),
                               test_std=("test_reduction", "std"), test_min=("test_reduction", "min"),
                               test_max=("test_reduction", "max"), q2_mean=("q2_gain_db", "mean"),
                               q2_std=("q2_gain_db", "std")).reset_index()
    s.to_csv(out / "a_summary.csv", index=False)
    rows = []
    A = s[s.arch.isin(["A1", "A2", "A3"])]
    T = s[s.arch == "T"]
    K = s[(s.arch == "K") & (s.set == "none")]
    k_best = K.loc[K.val_mean.idxmax()] if len(K) else None
    for arch, part in A.groupby("arch"):
        best = part.loc[part.val_mean.idxmax()]
        rivals = T[T.rm <= best.rm * 1.05]
        if rivals.empty or k_best is None:
            continue
        t = rivals.loc[rivals.val_mean.idxmax()]
        sd = max(best.test_std, t.test_std, k_best.test_std)
        ok = (best.test_mean - t.test_mean > 2 * sd) and (best.test_mean - k_best.test_mean > 2 * sd)
        rows.append({"hypothesis": f"{arch}: лучше только триплетов и только Керра при равных RM (> 2 std)",
                     "value": float(best.test_mean - t.test_mean), "verdict": "подтверждено" if ok else "не доказано",
                     "detail": f"{arch} {best['mode']} M{int(best.M)} Nc{int(best.n_c)} {best.set} ({int(best.rm)} RM, "
                               f"{best.test_mean:.2f} ± {best.test_std:.2f}) против T M{int(t.M)} {t.set} ({int(t.rm)} RM, "
                               f"{t.test_mean:.2f}) и K Nc{int(k_best.n_c)} ({k_best.test_mean:.2f}); 2 std = {2 * sd:.2f}"})
    full = s[(s.arch == "A3full") & (s.set == "full")]
    a3 = s[(s.arch == "A3") & (s.set == "full")]
    for _, r in full.iterrows():
        twin = a3[(a3.M == r.M) & (a3.n_c == r.n_c) & (a3["mode"] == r["mode"])]
        if len(twin):
            rows.append({"hypothesis": f"двойной счёт: Керр + T3 с осями − Керр + внеосевые (M{int(r.M)}, Nc{int(r.n_c)})",
                         "value": float(r.test_mean - twin.test_mean.iloc[0]), "verdict": "справочно",
                         "detail": f"оси добавляют {12 * (r.n_terms - twin.n_terms.iloc[0]):.0f} RM"})
    split_files = sorted(out.glob("fits_split*/*.csv"))
    if split_files:
        sp = pd.concat([pd.read_csv(f) for f in split_files], ignore_index=True)
        sp = sp[sp.set == "full"].groupby(["arch", "split_seed"]).agg(
            test_mean=("test_reduction", "mean"), test_std=("test_reduction", "std"), rm=("rm", "first")).reset_index()
        sp.to_csv(out / "a_splits.csv", index=False)
        for split, part in sp.groupby("split_seed"):
            t = part[part.arch == "T"]
            for _, r in part[part.arch.isin(["A1", "A2", "A3"])].iterrows():
                if len(t):
                    rows.append({"hypothesis": f"разбиение {split}: {r.arch} − T (лучшие конфигурации, полные наборы)",
                                 "value": float(r.test_mean - t.test_mean.iloc[0]), "verdict": "справочно",
                                 "detail": f"{r.test_mean:.2f} ± {r.test_std:.2f} при {int(r.rm)} RM против {t.test_mean.iloc[0]:.2f} при {int(t.rm.iloc[0])}"})
    write_verdicts(rows, config, out / "a_verdicts.csv")
    plot(s, figures, exp)
    print(f"E046: {out / 'a_summary.csv'}; вердикты: {out / 'a_verdicts.csv'}")
    return 0


def plot(s: pd.DataFrame, figures, exp: str = "E046") -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    mag = s[s.set.str.startswith("mag_") | s.set.isin(["none", "full"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    index = 0
    for (arch, mode, M, nc), part in mag[mag.arch.isin(["A1", "A3"])].groupby(["arch", "mode", "M", "n_c"]):
        if mode != "varpro" or nc != max(mag.n_c):
            continue
        part = part.sort_values("n_terms")
        ax.errorbar(part.n_terms, part.test_mean, yerr=part.test_std.fillna(0), marker="os"[index % 2], capsize=2,
                    color=plot_style.SERIES[index % len(plot_style.SERIES)], label=f"{arch} M={int(M)} Nc={int(nc)}")
        index += 1
    ax.set_xscale("symlog", linthresh=8)
    ax.set_xlabel("kept off-axis triplet terms K (magnitude pruning; 0 = Kerr with memory alone)")
    ax.set_ylabel("test BER reduction, % (mean +- std over replicas)")
    ax.set_title("Architecture A: gain vs number of off-axis (IFWM) terms at fixed Kerr")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / f"{exp}_gain_vs_offaxis_terms.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (arch, part) in enumerate(s.groupby("arch")):
        ax.errorbar(part.rm, part.test_mean, yerr=part.test_std.fillna(0), fmt="osD^vP"[index % 6], capsize=1.5, alpha=0.7,
                    color=plot_style.SERIES[index % len(plot_style.SERIES)], markersize=4, label=arch)
    ax.set_xscale("log")
    ax.set_xlabel("real multiplies per symbol (repository convention)")
    ax.set_ylabel("test BER reduction, % (mean +- std)")
    ax.set_title("Architecture A vs triplets only (T) and Kerr only (K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / f"{exp}_a_vs_controls_rm.png", dpi=160)
    plt.close(fig)
