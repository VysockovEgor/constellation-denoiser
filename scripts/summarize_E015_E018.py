"""Collect the E015-E018 (+ E019 confirm) verdicts and draw one quality-vs-multiplies figure for E011-E019.

    PYTHONPATH=src python scripts/summarize_E015_E018.py      (SMOKE=1 reads results_smoke/ everywhere)

Writes results/E015_E018_verdicts.csv (one row per hypothesis; "не запущено"
when an experiment has no results) and results/E015_E018_quality_vs_macs.png
(results_smoke/ in smoke mode). Each experiment's own summarize stage must run first.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from constellation_denoiser import plot_style  # noqa: E402
from constellation_denoiser.q2 import q2_db  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SUB = "results_smoke" if SMOKE else "results"
EXP = {"E011": "E011_long_runs_convergence", "E012": "E012_depth_vs_window", "E013": "E013_transformer_scaling",
       "E014": "E014_kerr_kan_hybrid", "E015": "E015_physics_diagnostics", "E016": "E016_phys_kerr_only",
       "E017": "E017_phys_plus_mlp", "E018": "E018_dbp_depth", "E019": "E019_confirm_long_runs"}


def path(exp: str, name: str, old: bool = False) -> Path:
    return ROOT / "experiments" / EXP[exp] / ("results" if old else SUB) / name


def read(exp: str, name: str, old: bool = False) -> pd.DataFrame | None:
    p = path(exp, name, old)
    return pd.read_csv(p) if p.is_file() else None


def verdict_rows() -> list[dict]:
    rows = []
    verdict = path("E015", "verdict.json")
    if verdict.is_file():
        v = json.loads(verdict.read_text(encoding="utf-8"))
        rows.append({"experiment": "E015", "hypothesis": "память есть (фаза зависит от мощности соседей)",
                     "verdict": "подтверждено" if v["verdict"] == "память есть" else "не доказано",
                     "numbers": (f"rho_own={v['rho_own_validation']:.3f}; rho_nb={json.dumps(v['rho_nb_validation'])}; "
                                 f"R7-R1 val={v['fit_gap_R7_minus_R1_validation_pp']:.2f} p.p., "
                                 f"test={v['fit_gap_R7_minus_R1_test_pp']:.2f} p.p."),
                     "spread": "95% file bootstrap in diagnostics_summary.csv", "source": str(verdict.relative_to(ROOT))})
    else:
        rows.append({"experiment": "E015", "hypothesis": "память есть", "verdict": "не запущено"})
    for exp, name, hypotheses in (("E016", "phys_kerr_verdicts.csv", ["физика достаточна", "физика полезна"]),
                                  ("E017", "phys_plus_mlp_verdicts.csv", ["гибрид лучше при равной стоимости",
                                                                         "нейросеть на остаток не нужна (h = 4)"]),
                                  ("E018", "dbp_depth_verdicts.csv", ["глубина помогает",
                                                                      "послойная инициализация нужна"])):
        frame = read(exp, name)
        for hypothesis in hypotheses:
            match = frame[frame.hypothesis == hypothesis] if frame is not None else pd.DataFrame()
            if match.empty:
                rows.append({"experiment": exp, "hypothesis": hypothesis, "verdict": "не запущено"})
                continue
            item = match.iloc[0]
            numbers = "; ".join(f"{k}={item[k]}" for k in item.index if k.startswith(("diff", "candidate")) and
                                str(item[k]) not in ("nan", ""))
            spread = "; ".join(f"{k}={item[k]}" for k in item.index if "spread" in k)
            verdict = str(item.verdict)
            rows.append({"experiment": exp, "hypothesis": hypothesis,
                         "verdict": "подтверждено" if verdict == "подтверждено" else
                         ("не доказано" if verdict.startswith("не доказано") else verdict),
                         "numbers": numbers, "spread": spread, "source": str(path(exp, name).relative_to(ROOT))})
    confirm = read("E019", "confirm_summary.csv")
    if confirm is None:
        rows.append({"experiment": "E019", "hypothesis": "плато за 200k шагов", "verdict": "не запущено"})
    else:
        for _, item in confirm[confirm.horizon == confirm.horizon.max()].iterrows():
            rows.append({"experiment": "E019", "hypothesis": f"плато за 200k шагов: {item.member} ({item.lr_branch})",
                         "verdict": "подтверждено" if str(item.plateau_text).startswith("плато (") else "не доказано",
                         "numbers": f"test_valbest_mean={item.test_valbest_mean:.3f}; q2_gain={item.valbest_q2_gain_db_mean:.3f}; "
                                    f"{item.plateau_text}",
                         "spread": f"min={item.test_valbest_min:.3f}, max={item.test_valbest_max:.3f}",
                         "source": str(path("E019", "confirm_summary.csv").relative_to(ROOT))})
    return rows


def points() -> pd.DataFrame:
    """(family, label, macs, steps, test mean/min/max, Q2 gain) at the test-at-best-validation metric."""
    out = []

    def add(family, frame, label_col, macs_col, test_col, q2_col, steps):
        for _, row in frame.iterrows():
            out.append({"family": family, "label": row[label_col], "macs": row[macs_col], "steps": steps,
                        "test_mean": row[test_col], "q2_gain": row[q2_col]})

    def by_group(frame, keys, family, steps, macs="macs_per_symbol_merged"):
        frame = frame.copy()
        frame["valbest_q2"] = [q2_db(r * (1 - t / 100)) - q2_db(r) for r, t in
                               zip(frame.raw_ber, frame.test_reduction_valbest_select)]
        grouped = frame.groupby(keys).agg(macs=(macs, "first"), test_mean=("test_reduction_valbest_select", "mean"),
                                          q2_gain=("valbest_q2", "mean")).reset_index()
        grouped["label"] = grouped[keys].astype(str).agg(" ".join, axis=1)
        add(family, grouped, "label", "macs", "test_mean", "q2_gain", steps)

    e012 = read("E012", "depth_window.csv", old=True)
    if e012 is not None:
        e012 = e012[e012.lr_branch == "const_1e-3"]
        for budget in (20000, 100000):
            by_group(e012[e012.budget == budget], ["variant", "window"], "complex MLP (E012)", budget)
    e011 = read("E011", "long_runs.csv", old=True)
    if e011 is not None:
        by_group(e011[(e011.variant == "real_tanh") & (e011.lr_name == "const_1e-3")], ["variant", "n_train_nominal"],
                 "real tanh MLP (E011)", 200000)
    e013 = read("E013", "transformer_runs.csv", old=True)
    if e013 is not None:
        by_group(e013[e013.model == "transformer"], ["variant", "window", "n_train_nominal", "budget"],
                 "transformer (E013)", None)
    e014 = read("E014", "kan_hybrid.csv", old=True)
    if e014 is not None:
        by_group(e014, ["variant", "spec_grid"] if "spec_grid" in e014 else ["variant"], "Kerr + KAN (E014)", 40000)
    for exp, name, family in (("E016", "phys_kerr_summary.csv", "PhysKerr (E016)"),
                              ("E017", "phys_plus_mlp_summary.csv", "Phys + MLP_res (E017)"),
                              ("E018", "dbp_depth_summary.csv", "DBP depth (E018)")):
        frame = read(exp, name)
        if frame is not None:
            frame = frame[frame.test_mean > -50]
            add(family, frame.assign(label=frame.variant), "label", "macs_per_symbol", "test_mean", "q2_gain_mean", None)
    confirm = read("E019", "confirm_summary.csv")
    if confirm is not None:
        part = confirm[confirm.horizon == confirm.horizon.max()]
        add("200k confirm (E019)", part.assign(label=part.member + " " + part.lr_branch), "label",
            "macs_per_symbol_merged", "test_valbest_mean", "valbest_q2_gain_db_mean", 200000)
    return pd.DataFrame(out)


def plot(frame: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    families = list(dict.fromkeys(frame.family))[:8]
    markers = "osD^vPX*"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (column, label) in zip(axes, (("test_mean", "test BER reduction at best validation, %"),
                                          ("q2_gain", "Q2 gain, dB"))):
        for index, family in enumerate(families):
            part = frame[(frame.family == family) & np.isfinite(frame[column])]
            ax.plot(part.macs, part[column], markers[index], color=plot_style.color(index), markersize=5, alpha=0.85,
                    linestyle="")
        ax.set_xscale("log")
        ax.set_xlabel("real multiplies per symbol (MLPs after merge; KAN sparse; physics macs_phys)")
        ax.set_ylabel(label)
    handles = [Line2D([], [], color=plot_style.color(i), marker=markers[i], linestyle="", label=f)
               for i, f in enumerate(families)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("All families E011-E019: quality vs cost")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    out_dir = ROOT / SUB
    out_dir.mkdir(exist_ok=True)
    verdicts = pd.DataFrame(verdict_rows())
    verdicts.to_csv(out_dir / "E015_E018_verdicts.csv", index=False)
    frame = points()
    frame.to_csv(out_dir / "E015_E018_quality_points.csv", index=False)
    if not frame.empty:
        plot(frame, out_dir / "E015_E018_quality_vs_macs.png")
    print(verdicts[["experiment", "hypothesis", "verdict"]].to_string(index=False))
    print(f"-> {out_dir / 'E015_E018_verdicts.csv'}, {out_dir / 'E015_E018_quality_vs_macs.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
