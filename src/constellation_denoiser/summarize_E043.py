"""E043: summary and fair comparison - Q2 gain and BER reduction vs RM under three counting conventions.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E043   (no training; SMOKE=1 writes results_smoke/)

Families: one-step triplets (E031/E034/E036), cascade (E040 confirm), mixed stages (E041 S3), Kerr MLP and DBP (best
points of the E033 aggregator, which already uses the corrected aggregator v2 of E024). For the network and DBP families
the three RM axes coincide (the conventions differ only for triplet terms). Writes results/points.csv,
results/level_best.csv, results/level_best.md and figures/E043_pareto_three_rm.png.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from .one_step_triplets import one_step_points
from .runner import exp_dir, results_root

AXES = ("rm_repo", "rm_buffered_my_estimate", "rm_arxiv_2409_20023")
LEVELS = [3000, 8000, 16000, 25000]
FACTOR = 1.25


def gather() -> pd.DataFrame:
    frames = []
    one = one_step_points()
    if len(one):
        frames.append(one.assign(family="one-step triplets"))
    path = exp_dir("E040") / "results" / "cascade_summary.csv"
    if path.is_file():
        c = pd.read_csv(path)
        c = c[c.stage == "confirm"]
        if len(c):
            frames.append(c.rename(columns={"variant": "label"}).assign(
                family=np.where(c.variant.str.endswith("_shared"), "cascade (shared C)", "cascade"), source="E040"))
    path = exp_dir("E041") / "results" / "mixed_summary.csv"
    if path.is_file():
        m = pd.read_csv(path)
        m = m[m.stage == "s3"]
        if len(m):
            from .cascade_e040 import rm_variants
            from .grids import load_config
            from .grids_e039 import e041_sample

            _, cfg = load_config("E041")
            specs = {n: s for lev in cfg["levels"] for arm in cfg["arms"] for n, s in e041_sample(cfg, arm, lev)}
            extra = pd.DataFrame([rm_variants(specs[n]["stages"]) for n in m.e041_config])
            m = pd.concat([m.reset_index(drop=True), extra], axis=1)
            frames.append(m.rename(columns={"e041_config": "label"}).assign(
                family=np.where(m.arm == "mixed", "mixed stages", "uniform cascade (E041)"), source="E041"))
    path = exp_dir("E033") / "results" / "pareto_points.csv"
    if path.is_file():
        p = pd.read_csv(path)
        p = p[p.family.isin(["Kerr MLP", "DBP"])]
        frames.append(pd.DataFrame({"family": p.family, "source": p.source, "label": p.label, "rm_repo": p.rm,
                                    "rm_buffered_my_estimate": p.rm, "rm_arxiv_2409_20023": p.rm, "val_mean": p.val_mean,
                                    "test_mean": p.test_mean, "test_min": p.test_min, "test_max": p.test_max,
                                    "q2_mean": p.q2_mean, "q2_min": p.q2_min, "q2_max": p.q2_max, "n_seeds": p.n_seeds}))
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if "rm" in out and "rm_repo" in out:
        out["rm_repo"] = out.rm_repo.fillna(out.rm)
    return out


def level_table(points: pd.DataFrame, axis: str = "rm_repo") -> pd.DataFrame:
    rows = []
    for level in LEVELS:
        part = points[[abs(math.log(r / level)) <= math.log(FACTOR) if r > 0 else False for r in points[axis]]]
        for family, fam in part.groupby("family"):
            best = fam.loc[fam.val_mean.idxmax()]
            rows.append({"level": level, "family": family, "label": best.label, "rm_repo": best.rm_repo,
                         "rm_buffered_my_estimate": best.rm_buffered_my_estimate, "rm_arxiv_2409_20023": best.rm_arxiv_2409_20023,
                         "test_mean": best.test_mean, "test_min": best.test_min, "test_max": best.test_max,
                         "q2_mean": best.q2_mean, "n_seeds": best.n_seeds})
    return pd.DataFrame(rows).sort_values(["level", "test_mean"], ascending=[True, False]) if rows else pd.DataFrame()


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E043")
    out = results_root("E043", smoke)
    figures = out / "figures" if smoke else directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    points = gather()
    if points.empty:
        print("нет точек")
        return 1
    points.to_csv(out / "points.csv", index=False)
    table = level_table(points)
    table.to_csv(out / "level_best.csv", index=False)
    lines = ["| уровень RM | семейство | точка | RM (репозиторий / буфер / arXiv) | тест, % (min … max) | Q², дБ | seeds |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {int(r.level)} | {r.family} | {r.label} | {int(r.rm_repo)} / {int(r.rm_buffered_my_estimate)} / "
                     f"{int(r.rm_arxiv_2409_20023)} | {r.test_mean:.2f} ({r.test_min:.2f} … {r.test_max:.2f}) | {r.q2_mean:.3f} | "
                     f"{int(r.n_seeds)} |")
    (out / "level_best.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    rows = []
    for level in LEVELS:
        leaders = []
        for axis in AXES:
            t = level_table(points, axis)
            t = t[t.level == level].sort_values("test_mean", ascending=False) if len(t) else t
            if len(t) == 0:
                leaders.append(None)
                continue
            first = t.iloc[0]
            gap = len(t) < 2 or first.test_min > t.iloc[1].test_max
            leaders.append((first.family, gap))
        same = all(l is not None for l in leaders) and len({l[0] for l in leaders}) == 1 and all(l[1] for l in leaders)
        rows.append({"hypothesis": f"{level} RM: устойчивый лидер при трёх конвенциях RM", "value": float("nan"),
                     "verdict": "подтверждено" if same else "не доказано",
                     "detail": " | ".join(f"{a}: {l[0] if l else '—'}" for a, l in zip(AXES, leaders))})
    pd.DataFrame(rows).assign(rule_status=config["rule_draft"]["status"]).to_csv(out / "summary_verdicts.csv", index=False)
    plot(points, figures)
    print(f"E043: {len(points)} точек; {out / 'level_best.md'}")
    return 0


def plot(points: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from . import plot_style

    plot_style.apply()
    families = list(dict.fromkeys(points.family))
    markers = "o^sDvP*Xh"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey="row")
    titles = {"rm_repo": "RM: repository convention", "rm_buffered_my_estimate": "RM: buffered, single pol. (my estimate)",
              "rm_arxiv_2409_20023": "RM: arXiv 2409.20023 formula"}
    for col, axis in enumerate(AXES):
        for row, (metric, low, high, label) in enumerate((("q2_mean", "q2_min", "q2_max", "Delta Q2, dB"),
                                                          ("test_mean", "test_min", "test_max", "BER reduction, %"))):
            ax = axes[row][col]
            for index, family in enumerate(families):
                part = points[(points.family == family) & (points[metric] > (-0.3 if row == 0 else -10))]
                if part.empty:
                    continue
                c = plot_style.SERIES[index % len(plot_style.SERIES)]
                ax.errorbar(part[axis].to_numpy(float), part[metric].to_numpy(float),
                            yerr=[np.clip((part[metric] - part[low]).to_numpy(float), 0, None),
                                  np.clip((part[high] - part[metric]).to_numpy(float), 0, None)],
                            fmt=markers[index % len(markers)], color=c, markersize=4, capsize=1.5, alpha=0.8)
            ax.set_xscale("log")
            if row == 1:
                ax.set_xlabel(titles[axis])
            if col == 0:
                ax.set_ylabel(label + " (min-max over seeds)")
    handles = [Line2D([], [], color=plot_style.SERIES[i % len(plot_style.SERIES)], marker=markers[i % len(markers)],
                      linestyle="", label=f) for i, f in enumerate(families)]
    fig.legend(handles=handles, loc="upper center", ncol=len(families), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures / "E043_pareto_three_rm.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
