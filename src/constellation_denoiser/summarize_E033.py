"""E033 summary: final Pareto frontier 'RM vs Q2 gain' from confirmed runs (aggregator v2).

    PYTHONPATH=src python -m constellation_denoiser.summarize_E033   (SMOKE=1 reads results_smoke/ of E024+)

Confirmed = sources and limits in config.json["confirmed"]; points and envelope chosen by validation. Writes
results/runs_table.csv (fill runs), results/pareto_points.csv, results/level_best.csv, results/level_best.md,
results/dominance.csv, results/pareto_verdicts.csv and figures E033_pareto_q2.png, E033_pareto_ber.png,
E033_physics_ladder.png.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .pareto_v2 import LEVELS, best_per_level, confirmed, envelope, load_tables, plot_pareto, points
from .paths import EXPERIMENTS_DIR
from .q2 import q2_db
from .summary_e024 import disjoint, setup, verdict, write_verdicts

OURS = ("Kerr MLP", "Phys + MLP", "DBP", "DBP + MLP", "PhysKerr")


def hd_fec_lines(config: dict) -> tuple[float, float]:
    path = EXPERIMENTS_DIR / "E024_data_audit" / "results" / "data_summary.json"
    raw = json.loads(path.read_text(encoding="utf-8"))["splits"]["test"]["raw_ber_window31_centres"] if path.is_file() \
        else 0.009551859551859552
    hd = float(config["hd_fec_ber"])
    return 100 * (raw - hd) / raw, q2_db(hd) - q2_db(raw)


def dominates(a: pd.Series, b: pd.Series) -> bool:
    return bool(a.rm <= b.rm and a.q2_mean - b.q2_mean >= 0.1 and disjoint(a.q2_min, a.q2_max, b.q2_min, b.q2_max))


def main() -> int:
    smoke, directory, config, results, figures, _ = setup("E033")
    runs = load_tables(smoke)
    runs = confirmed(runs, config) if not runs.empty else runs
    pts = points(runs, min_seeds=config["confirmed"]["min_seeds"]) if not runs.empty else pd.DataFrame()
    if pts.empty and smoke:                 # smoke: let the plots run on whatever exists
        pts = points(load_tables(smoke), min_seeds=1)
    if pts.empty:
        print("нет подтверждённых точек")
        return 1
    pts.to_csv(results / "pareto_points.csv", index=False)
    best = best_per_level(pts, config["levels"], config["level_factor"])
    best.to_csv(results / "level_best.csv", index=False)
    lines = ["| уровень RM | семейство | точка | RM | Q², дБ (min … max) | снижение BER, % | seeds |", "|---|---|---|---|---|---|---|"]
    for _, r in best.iterrows():
        lines.append(f"| {int(r.level)} | {r.family} | {r.label} | {int(r.rm)} | {r.q2_mean:.3f} ({r.q2_min:.3f} … {r.q2_max:.3f}) | "
                     f"{r.test_mean:.2f} | {int(r.n_seeds)} |")
    (results / "level_best.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dom = []
    for _, a in best.iterrows():
        for _, b in best.iterrows():
            if a.family != b.family and dominates(a, b):
                dom.append({"a": a.label, "a_family": a.family, "a_rm": a.rm, "b": b.label, "b_family": b.family,
                            "b_rm": b.rm, "q2_diff_db": a.q2_mean - b.q2_mean})
    dom = pd.DataFrame(dom, columns=["a", "a_family", "a_rm", "b", "b_family", "b_rm", "q2_diff_db"])
    dom.to_csv(results / "dominance.csv", index=False)
    rows = []
    pb = best[best.family == "PB-NLC (triplets)"]
    for _, b in pb.iterrows():
        rivals = best[best.family.isin(OURS) & (best.rm <= b.rm)]
        hit = [r for _, r in rivals.iterrows() if dominates(r, b)]
        rows.append({"hypothesis": f"наш подход лучше PB-NLC при равных RM (уровень {int(b.level)}, {b.label})",
                     "value": max((h.q2_mean - b.q2_mean for h in hit), default=float("nan")),
                     "verdict": verdict(bool(hit)), "detail": "; ".join(h.label for h in hit)})
    for family in sorted(set(best.family)):
        n = int((dom.a_family == family).sum())
        rows.append({"hypothesis": f"{family} доминирует другие семейства (число пар A ≻ B)", "value": n,
                     "verdict": "подтверждено" if n else "не доказано", "detail": ""})
    write_verdicts(rows, config, results / "pareto_verdicts.csv")

    hd_pct, hd_db = hd_fec_lines(config)
    env = envelope(pts)
    plot_pareto(pts, env, figures / "E033_pareto_q2.png", "q2", "Final Pareto frontier (confirmed: >= 5 seeds, >= 200k, "
                "recipes of E025; envelope by validation)", hd_fec=hd_db)
    plot_pareto(pts, env, figures / "E033_pareto_ber.png", "test", "Final Pareto frontier, BER reduction", hd_fec=hd_pct)
    ladder(pts, figures, hd_db)
    print(f"E033: точек {len(pts)}; лучшие по уровням: {results / 'level_best.md'}")
    return 0


LADDER = [("linear FIR", None), ("PB-NLC (triplets)", "T1_"), ("PB-NLC (triplets)", "T2_"), ("PB-NLC (triplets)", "T3_"),
          ("PhysKerr", None), ("DBP", None), ("Kerr MLP", None)]


def ladder(pts: pd.DataFrame, figures, hd_db: float) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    steps = []
    for family, prefix in LADDER:
        part = pts[pts.family == family]
        if prefix:
            part = part[part.variant.astype(str).str.startswith(prefix)]
        if len(part):
            top = part.loc[part.val_mean.idxmax()]
            steps.append((f"{family}{' ' + prefix.rstrip('_') if prefix else ''}", top))
    if not steps:
        return
    plot_style.apply()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(steps))
    q = np.array([s[1].q2_mean for s in steps], float)
    lo = np.array([s[1].q2_mean - s[1].q2_min for s in steps], float)
    hi = np.array([s[1].q2_max - s[1].q2_mean for s in steps], float)
    ax.bar(x, q, color=[plot_style.SERIES[i % len(plot_style.SERIES)] for i in range(len(steps))], edgecolor=plot_style.TEXT,
           yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], capsize=3)
    for i, (_, top) in enumerate(steps):
        ax.text(i, max(q[i], 0), f"{int(top.rm)} RM", ha="center", va="bottom", fontsize=7)
    ax.axhline(hd_db, color=plot_style.TEXT, linestyle="--", linewidth=1, label="HD-FEC 3.8e-3")
    ax.set_xticks(x, [s[0] for s in steps], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Q2 gain, dB (best by validation in each step)")
    ax.set_title("Physical ladder: each step adds physical structure or flexibility")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E033_physics_ladder.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
