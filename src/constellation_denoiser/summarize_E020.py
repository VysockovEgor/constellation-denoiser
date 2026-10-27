"""E020 summary: why the E017 hybrid falls to ~51 % with the best_e010 branch in E019.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E020   (SMOKE=1 reads results_smoke/)

Without training: compares the anomalous and normal E019 runs of the hybrid (configuration, curves,
training loss, validation/test gap, final vs validation-best). With the reruns of this experiment:
mean/min/max per arm and step budget, verdict by the rule in config.json.
Writes results/differences.csv, results/hypotheses.csv, results/arms.csv, results/arms_verdicts.csv,
results/runs_table.csv and figures/E020_val_test_vs_steps.png.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .paths import EXPERIMENTS_DIR
from .runner import exp_dir, results_root

E019 = EXPERIMENTS_DIR / "E019_confirm_long_runs" / "results"
MEMBER = "E017_hyb_R31_L7_sym_h48_joint"


def e019_curves() -> pd.DataFrame:
    frames = []
    for path in sorted((E019 / "curves").glob(f"{MEMBER}_*_200k_seed*.csv")):
        branch = "anomalous: best_e010_cm1" if "best_e010" in path.name else "normal: const_1e-3"
        frames.append(pd.read_csv(path).assign(branch=branch, seed=int(path.stem.rsplit("seed", 1)[1])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def at(curves: pd.DataFrame, step: int, column: str) -> pd.Series:
    return curves[curves.step == step].groupby("branch")[column].mean()


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E020")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E020", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    curves = e019_curves()
    confirm = pd.read_csv(E019 / "confirm.csv")
    confirm = confirm[confirm.member == MEMBER]
    rows = []
    if not curves.empty:
        for branch, part in confirm.groupby("lr_branch"):
            label = "anomalous: best_e010_cm1" if branch == "best_e010" else "normal: const_1e-3"
            p200 = part[part.nominal_horizon == 200000]
            p100 = part[part.nominal_horizon == 100000]
            last = curves[(curves.branch == label)].groupby("seed").tail(1)
            rows.append({
                "branch": label, "lr": "cosine_to_zero, peak 3e-3, c_mult 1" if branch == "best_e010" else "const 1e-3, c_mult 1",
                "init": "same (PhysKerr identity + MLP_res output zero)", "mode": "joint", "seeds": len(p200),
                "val_5k": at(curves, 5000, "val_reduction").get(label), "val_40k": at(curves, 40000, "val_reduction").get(label),
                "test_valbest_100k": p100.test_reduction_valbest.mean(), "test_valbest_200k": p200.test_reduction_valbest.mean(),
                "test_final_200k": p200.final_test_reduction.mean(), "val_test_gap_200k": (p200.test_reduction_valbest - p200.best_val_reduction).mean(),
                "train_loss_final": last.train_loss.mean(), "c1_final": last.c1.mean()})
    differences = pd.DataFrame(rows)
    differences.to_csv(results / "differences.csv", index=False)

    hyp = config["hypotheses"]
    evidence = []
    if len(differences) == 2:
        an = differences[differences.branch.str.startswith("anomalous")].iloc[0]
        no = differences[differences.branch.str.startswith("normal")].iloc[0]
        gap_100_200 = an.test_valbest_200k - an.test_valbest_100k
        overfit = "отвергнута" if abs(gap_100_200) < 1.0 and an.train_loss_final >= no.train_loss_final else "не отвергнута"
        evidence.append(("c", hyp["c"], overfit,
                         f"100k → 200k: {gap_100_200:+.2f} п.п.; train loss {an.train_loss_final:.4f} против {no.train_loss_final:.4f} "
                         "(у аномального выше — недообучение, не переобучение)"))
        sel = abs(an.test_valbest_200k - an.test_final_200k)
        evidence.append(("d", hyp["d"], "выбор чекпоинта отвергнут" if sel < 1.0 else "не отвергнута",
                         f"тест в точке лучшей валидации {an.test_valbest_200k:.2f} %, на последнем шаге {an.test_final_200k:.2f} %; "
                         f"разрыв валидация−тест {an.val_test_gap_200k:+.2f} п.п. (норма {no.val_test_gap_200k:+.2f})"))
        evidence.append(("a/b/e", "c_mult, расписание или пиковый lr", "проверяется перезапусками A1–A3",
                         f"валидация на 5k: {an.val_5k:.1f} % против {no.val_5k:.1f} %, на 40k: {an.val_40k:.1f} % против "
                         f"{no.val_40k:.1f} % — разрыв нарастает, пока lr высокий (3e-3), и лишь частично сокращается на спаде"))
    pd.DataFrame(evidence, columns=["hypothesis", "text", "status", "evidence"]).to_csv(results / "hypotheses.csv", index=False)

    runs = load_runs(results)
    arms_rows, verdicts = [], []
    threshold = 61.039
    if not runs.empty:
        runs.to_csv(results / "runs_table.csv", index=False)
        own = runs[runs.stage == "main"]
        for (variant, lr_name, budget), group in own.groupby(["variant", "lr_name", "budget"]):
            arms_rows.append({"arm": variant, "lr": lr_name, "steps": budget, "n_seeds": len(group),
                              "test_mean": group.test_reduction_valbest_select.mean(),
                              "test_min": group.test_reduction_valbest_select.min(),
                              "test_max": group.test_reduction_valbest_select.max(),
                              "val_mean": group.best_val_reduction.mean(), "derived": False})
            if lr_name.startswith("const"):          # 40k point of the constant-lr arm from its 200k curve
                vals = []
                for run_id in group.run_id:
                    c = pd.read_csv(results / "curves" / f"{run_id}.csv")
                    limit = int(round(40000 * group.planned_steps.iloc[0] / budget))
                    part = c[(c.step > 0) & (c.step <= limit) & c.test_reduction.notna()]
                    if len(part):
                        vals.append(part.loc[part.val_reduction.idxmax(), "test_reduction"])
                if vals:
                    arms_rows.append({"arm": variant, "lr": lr_name, "steps": 40000, "n_seeds": len(vals),
                                      "test_mean": float(np.mean(vals)), "test_min": min(vals), "test_max": max(vals),
                                      "val_mean": float("nan"), "derived": True})
        arms = pd.DataFrame(arms_rows)
        for _, row in arms.iterrows():
            restored = row.test_mean >= threshold
            verdicts.append({"hypothesis": f"{row.arm} ({row.lr}, {int(row.steps) // 1000}k) восстанавливает уровень ≥ {threshold} %",
                             "test_mean": row.test_mean, "verdict": "подтверждено" if restored else "не доказано"})
        arms.to_csv(results / "arms.csv", index=False)
    vframe = pd.DataFrame(verdicts, columns=["hypothesis", "test_mean", "verdict"])
    vframe["rule_status"] = config["rule_draft"]["status"]
    vframe.to_csv(results / "arms_verdicts.csv", index=False)
    plot(curves, results, runs, figures)
    print(f"различия: {results / 'differences.csv'}; гипотезы: {results / 'hypotheses.csv'}")
    return 0


def plot(curves: pd.DataFrame, results, runs: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, metric, title in ((axes[0], "val_reduction", "validation"), (axes[1], "test_reduction", "test (report only)")):
        index = 0
        for branch, part in curves.groupby("branch"):
            stats = part[part.step > 0].dropna(subset=[metric]).groupby("step")[metric].agg(["mean", "min", "max"])
            plot_style.band(ax, stats.index, stats["mean"], stats["min"], stats["max"], index, f"E019 {branch}")
            index += 1
        if not runs.empty:
            for (variant, lr_name, budget), group in runs[runs.stage == "main"].groupby(["variant", "lr_name", "budget"]):
                if budget != runs.budget.max() or index >= 8:
                    continue
                frames = [pd.read_csv(results / "curves" / f"{r}.csv") for r in group.run_id]
                c = pd.concat(frames).query("step > 0").dropna(subset=[metric]).groupby("step")[metric].mean()
                ax.plot(c.index, c.values, color=plot_style.color(index), linestyle="--", label=f"E020 {variant} {lr_name}")
                index += 1
        ax.set_title(f"Hybrid h48 joint: {title}")
        ax.set_xlabel("training step")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("BER reduction, % (band: min-max over seeds)")
    fig.tight_layout()
    fig.savefig(figures / "E020_val_test_vs_steps.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
