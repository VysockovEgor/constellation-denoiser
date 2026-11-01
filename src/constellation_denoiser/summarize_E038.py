"""E038 summary: DBP + K selected triplet terms against DBP alone and against triplets alone at equal RM.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E038   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/dbp_triplets_summary.csv, results/dbp_triplets_verdicts.csv and
figures/E038_dbp_triplets.png.
"""

from __future__ import annotations

import pandas as pd

from .runner import exp_dir
from .summary_e024 import aggregate, better, disjoint, setup, verdict, write_verdicts


def triplet_points() -> pd.DataFrame:
    rows = []
    for exp, name in (("E031", "triplet_summary.csv"), ("E034", "triplet_ext_summary.csv")):
        path = exp_dir(exp) / "results" / name
        if path.is_file():
            s = pd.read_csv(path)
            label = s.arm + "_M" + s.M.astype(str) if "arm" in s else s.variant
            rm = s.rm_sym if "rm_sym" in s else s.rm
            rows.append(pd.DataFrame({"label": exp + " " + label, "rm": rm, "val_mean": s.val_mean, "q2_mean": s.q2_mean,
                                      "q2_min": s.get("q2_min", s.q2_mean), "q2_max": s.get("q2_max", s.q2_mean),
                                      "test_mean": s.test_mean}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E038")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage == "main"]
    table = aggregate(runs, ["n_steps", "k"])
    table.to_csv(results / "dbp_triplets_summary.csv", index=False)
    rows = []
    trip = triplet_points()
    for ns, part in table.groupby("n_steps"):
        base = part[part.k == 0]
        if base.empty:
            continue
        base = base.iloc[0]
        for _, r in part[part.k > 0].iterrows():
            diff, ok = better(r, base, 1.0)
            rows.append({"hypothesis": f"Ns={int(ns)}, K={int(r.k)}: триплеты добавляют к DBP ≥ 1 п.п. (интервалы не пересекаются)",
                         "value": diff, "verdict": verdict(ok), "detail": f"RM {int(r.rm)} против {int(base.rm)}"})
            if len(trip):
                rivals = trip[trip.rm <= r.rm]
                if len(rivals):
                    rival = rivals.loc[rivals.val_mean.idxmax()]
                    d = float(r.q2_mean - rival.q2_mean)
                    ok = d >= 0.1 and disjoint(r.q2_min, r.q2_max, rival.q2_min, rival.q2_max)
                    rows.append({"hypothesis": f"Ns={int(ns)}, K={int(r.k)}: лучше триплетов при равных RM (≥ 0.1 дБ)",
                                 "value": d, "verdict": verdict(ok), "detail": f"{rival.label} ({int(rival.rm)} RM)"})
    write_verdicts(rows, config, results / "dbp_triplets_verdicts.csv")
    plot(table, trip, figures)
    print(f"E038: {results / 'dbp_triplets_summary.csv'}; вердикты: {results / 'dbp_triplets_verdicts.csv'}")
    return 0


def plot(table: pd.DataFrame, trip: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for index, (ns, part) in enumerate(table.groupby("n_steps")):
        part = part.sort_values("rm")
        ax.errorbar(part.rm, part.q2_mean, yerr=[part.q2_mean - part.q2_min, part.q2_max - part.q2_mean],
                    marker="os"[index % 2], color=plot_style.color(index), capsize=2, label=f"DBP Ns={int(ns)} + K triplets")
    if len(trip):
        t = trip[(trip.rm < 4000)].sort_values("rm")
        ax.plot(t.rm, t.q2_mean, "^", color=plot_style.color(2), label="triplets alone (E031/E034)")
    ax.set_xscale("log")
    ax.set_xlabel("real multiplies per symbol")
    ax.set_ylabel("Q2 gain, dB (min-max over 5 seeds)")
    ax.set_title("DBP + selected triplet terms in the low-complexity range")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E038_dbp_triplets.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
