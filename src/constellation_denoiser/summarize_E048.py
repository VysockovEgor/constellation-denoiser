"""E048 summary: (1) architecture A at larger memory (closed form, via e046_fit) and (2) DBP + off-axis triplets.

    TRIPLET_KERR_EXP=E048 PYTHONPATH=src python -m constellation_denoiser.e046_fit summarize   # part 1
    PYTHONPATH=src python -m constellation_denoiser.summarize_E048                          # part 2 + comparison

Writes results/dbp_offaxis_summary.csv, results/dbp_offaxis_verdicts.csv, results/a_vs_dbp.csv and
figures/E048_a_dbp_vs_rm.png.
"""

from __future__ import annotations

import pandas as pd

from .runner import exp_dir
from .summary_e024 import aggregate, better, setup, verdict, write_verdicts


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E048")
    rows, table = [], pd.DataFrame()
    if not runs.empty:
        runs = runs[runs.stage == "main"]
        table = aggregate(runs, ["n_steps", "k"])
        std = runs.groupby(["n_steps", "k"]).test_reduction_valbest_select.std().rename("test_std").reset_index()
        table = table.merge(std, on=["n_steps", "k"])
        table.to_csv(results / "dbp_offaxis_summary.csv", index=False)
        e038 = exp_dir("E038") / "results" / "dbp_triplets_summary.csv"
        axial = pd.read_csv(e038) if e038.is_file() else pd.DataFrame()
        for ns, part in table.groupby("n_steps"):
            base = part[part.k == 0]
            for _, r in part[part.k > 0].iterrows():
                if len(base):
                    diff, ok = better(r, base.iloc[0], 1.0)
                    rows.append({"hypothesis": f"Ns={int(ns)}, K={int(r.k)}: внеосевые триплеты добавляют к DBP ≥ 1 п.п.",
                                 "value": diff, "verdict": verdict(ok), "detail": f"RM {int(r.rm)}; {r.test_mean:.2f} ± {r.test_std:.2f}"})
                ax = axial[(axial.n_steps == ns) & (axial.k == r.k)] if len(axial) else axial
                if len(ax):
                    rows.append({"hypothesis": f"Ns={int(ns)}, K={int(r.k)}: внеосевые лучше осевых (E038) при том же K",
                                 "value": float(r.test_mean - ax.test_mean.iloc[0]), "verdict": "справочно",
                                 "detail": f"{r.test_mean:.2f} против {ax.test_mean.iloc[0]:.2f}"})
    a = results / "a_summary.csv"
    if a.is_file():
        s = pd.read_csv(a)
        best = s[s.arch.isin(["A1", "A2", "A3"])]
        if len(best) and len(table):
            for target in config["compare_rm"]:
                aa = best[best.rm <= target * 1.05]
                dd = table[table.rm <= target * 1.05]
                if len(aa) and len(dd):
                    ab, db = aa.loc[aa.val_mean.idxmax()], dd.loc[dd.val_mean.idxmax()]
                    sd = max(ab.test_std, db.test_std)
                    rows.append({"hypothesis": f"{target} RM: A лучше DBP + внеосевых (> 2 std)", "value": float(ab.test_mean - db.test_mean),
                                 "verdict": verdict(ab.test_mean - db.test_mean > 2 * sd),
                                 "detail": f"A {ab['arch']} M{int(ab.M)} Nc{int(ab.n_c)} {ab.set} ({int(ab.rm)} RM) против DBP Ns{int(db.n_steps)} K{int(db.k)} ({int(db.rm)} RM)"})
    write_verdicts(rows, config, results / "dbp_offaxis_verdicts.csv")
    print(f"E048: {results / 'dbp_offaxis_verdicts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
