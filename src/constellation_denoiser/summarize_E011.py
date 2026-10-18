"""E011 summary: plateau, y_inf fit, t98, N=30 vs N=210 overfitting check, figure.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E011   (SMOKE=1 reads results_smoke/)

Writes results/long_runs.csv (one row per run), results/long_runs_curves.csv,
results/long_runs_summary.csv and figures/E011_long_runs.png.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from . import plot_style
from .merge_runs import load_runs
from .runner import exp_dir, results_root

PLATEAU_WINDOW = 50_000
PLATEAU_PP = 0.2
FALL_PP = 0.2
SMOOTH_STEPS = 10_000
FIT_FROM = 10_000
MODEL_ORDER = ["complex_both", "complex_a", "complex_one_hidden", "real_tanh"]


def branch_of(lr_name: str) -> str:
    return "best_e010" if str(lr_name).startswith("best_e010") else str(lr_name)


def smooth(curve: pd.DataFrame, every: int) -> pd.Series:
    points = max(1, SMOOTH_STEPS // max(1, every) + 1)
    return curve.set_index("step")["val_reduction"].rolling(points, center=True, min_periods=1).mean()


def fit_limit(steps: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Fit y = y_inf - a * steps^(-b) with a >= 0 (rising curve); NaN triple if the fit fails or < 4 points."""
    from scipy.optimize import curve_fit

    if len(steps) < 4 or not np.all(np.isfinite(values)):
        return float("nan"), float("nan"), float("nan")

    def model(s, y_inf, a, b):
        return y_inf - a * np.power(s, -b)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            span = float(values[-1] - values[0])
            p0 = (float(values[-1]) + 0.1 * abs(span), abs(span) * steps[0] ** 0.5 + 1e-3, 0.5)
            params, _ = curve_fit(model, steps, values, p0=p0, bounds=([-1e3, 0.0, 1e-2], [1e3, 1e9, 5.0]),
                                  maxfev=20_000)
        return tuple(float(p) for p in params)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def curve_metrics(curve: pd.DataFrame, every: int, fit_from: int) -> dict:
    curve = curve[curve.step > 0].sort_values("step")
    if curve.empty:
        return {}
    smoothed = smooth(curve, every)
    last = int(curve.step.iloc[-1])
    final_val = float(curve.val_reduction.iloc[-1])
    reach = curve[curve.val_reduction >= 0.98 * final_val] if final_val > 0 else curve.iloc[0:0]
    earlier = smoothed[smoothed.index <= last - PLATEAU_WINDOW]
    gain = float(smoothed.iloc[-1] - earlier.iloc[-1]) if len(earlier) else float("nan")
    fit_part = curve[curve.step >= fit_from]
    y_inf, a, b = fit_limit(fit_part.step.to_numpy(float), fit_part.val_reduction.to_numpy(float))
    drop = float(smoothed.max() - smoothed.iloc[-1])
    return {
        "last_step": last, "t98": int(reach.step.iloc[0]) if len(reach) else None,
        "smoothed_final_val": float(smoothed.iloc[-1]), "smoothed_max_val": float(smoothed.max()),
        "gain_last_50k_pp": gain,
        "plateau": None if np.isnan(gain) else bool(gain < PLATEAU_PP),
        "falling": bool(drop > FALL_PP), "drop_from_max_pp": drop,
        "y_inf": y_inf, "fit_a": a, "fit_b": b,
    }


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E011")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E011", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    runs = runs[runs.stage != "control"] if not runs.empty else runs
    if not runs.empty:
        runs["n_train_nominal"] = runs.n_train_nominal.fillna(runs.n_train_files).astype(int)
    if runs.empty:
        print("нет длинных прогонов")
        return 1
    every = int(runs.eval_every.iloc[0])
    fit_from = every if smoke else FIT_FROM

    rows, curves = [], []
    for _, run in runs.iterrows():
        curve = pd.read_csv(results / "curves" / f"{run.run_id}.csv")
        curve = curve.assign(run_id=run.run_id, variant=run.variant, n_train_nominal=run.n_train_nominal,
                             seed=run.seed, lr_branch=branch_of(run.lr_name))
        curves.append(curve)
        rows.append({**run.to_dict(), "lr_branch": branch_of(run.lr_name), **curve_metrics(curve, every, fit_from)})
    table = pd.DataFrame(rows)
    table.to_csv(results / "long_runs.csv", index=False)
    stacked = pd.concat(curves, ignore_index=True)
    stacked.to_csv(results / "long_runs_curves.csv", index=False)

    summary = []
    for (variant, branch, n_files), group in table.groupby(["variant", "lr_branch", "n_train_nominal"]):
        mean_curve = (stacked[(stacked.variant == variant) & (stacked.lr_branch == branch)
                              & (stacked.n_train_nominal == n_files)]
                      .groupby("step", as_index=False)[["val_reduction", "test_reduction"]].mean())
        mean_metrics = curve_metrics(mean_curve, every, fit_from)
        last = mean_metrics.get("last_step")
        if mean_metrics.get("plateau") is None:
            plateau_text = f"меньше {PLATEAU_WINDOW} шагов: плато не оценивается"
        elif mean_metrics["plateau"]:
            plateau_text = f"плато (прирост {mean_metrics['gain_last_50k_pp']:.2f} п.п. за 50k)"
        else:
            plateau_text = f"плато не достигнуто за {last} шагов"
        summary.append({
            "variant": variant, "lr_branch": branch, "n_train_nominal": n_files, "n_seeds": len(group),
            "seeds": " ".join(str(s) for s in sorted(group.seed)), "n_diverged": int(group.diverged.sum()),
            "final_val_mean": group.val_reduction.mean(), "final_val_std": group.val_reduction.std(ddof=0),
            "final_test_mean": group.test_reduction.mean(), "final_test_std": group.test_reduction.std(ddof=0),
            "final_test_min": group.test_reduction.min(), "final_test_max": group.test_reduction.max(),
            "best_val_mean": group.best_val_reduction.mean(),
            "test_at_best_val_mean": group.test_reduction_at_best_val.mean(),
            "best_val_step_mean": group.best_val_step.mean(),
            "t98_mean": pd.to_numeric(group.t98, errors="coerce").mean(),
            "y_inf_mean_of_runs": group.y_inf.mean(), "y_inf_fit_failures": int(group.y_inf.isna().sum()),
            "y_inf_mean_curve": mean_metrics.get("y_inf"),
            "gain_last_50k_pp_mean_curve": mean_metrics.get("gain_last_50k_pp"),
            "plateau_runs": int((group.plateau == True).sum()),  # noqa: E712
            "plateau_mean_curve": mean_metrics.get("plateau"), "plateau_text": plateau_text,
            "falling_mean_curve": mean_metrics.get("falling"),
            "drop_from_max_pp_mean_curve": mean_metrics.get("drop_from_max_pp"),
            "macs_per_symbol": group.macs_per_symbol.iloc[0],
            "macs_per_symbol_merged": group.macs_per_symbol_merged.iloc[0],
        })
    summary = pd.DataFrame(summary)

    verdicts = []
    for _, row in summary.iterrows():
        pair = summary[(summary.variant == row.variant) & (summary.lr_branch == row.lr_branch)]
        small, large = pair[pair.n_train_nominal == min(config["n_train_files"])], pair[
            pair.n_train_nominal == max(config["n_train_files"])]
        if small.empty or large.empty:
            verdicts.append("нет пары N=30/N=210")
        elif bool(small.falling_mean_curve.iloc[0]) and not bool(large.falling_mean_curve.iloc[0]):
            verdicts.append("разрыв N=30/N=210 трактуется как переобучение (по черновику правила)")
        else:
            verdicts.append("переобучение не показано")
    summary["overfitting_verdict_draft"] = verdicts
    summary["rule_status"] = config["rule_draft"]["status"]
    summary.to_csv(results / "long_runs_summary.csv", index=False)
    plot(stacked, figures)
    print(f"сводка: {results / 'long_runs_summary.csv'}; рисунок: {figures / 'E011_long_runs.png'}")
    return 0


def plot(stacked: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    data = stacked[stacked.step > 0]
    branches = [b for b in ("const_1e-3", "best_e010") if b in set(data.lr_branch)]
    n_values = sorted(data.n_train_nominal.unique())
    models = [m for m in MODEL_ORDER if m in set(data.variant)]
    fig, axes = plt.subplots(2, len(branches), figsize=(6.0 * len(branches), 7.0), sharex=True, squeeze=False)
    for col, branch in enumerate(branches):
        for row, metric in enumerate(("val_reduction", "test_reduction")):
            ax = axes[row][col]
            for index, model in enumerate(models):
                for style_index, n_files in enumerate(n_values):
                    part = data[(data.lr_branch == branch) & (data.variant == model) & (data.n_train_nominal == n_files)]
                    part = part.dropna(subset=[metric])
                    if part.empty:
                        continue
                    stats = part.groupby("step")[metric].agg(["mean", "min", "max"])
                    plot_style.band(ax, stats.index, stats["mean"], stats["min"], stats["max"], index,
                                    f"{model}, N={n_files}", plot_style.LINESTYLES[style_index])
            ax.set_title(f"{branch}: {'validation' if row == 0 else 'test (report only)'}")
            ax.set_ylabel("BER reduction, %")
            if row == 1:
                ax.set_xlabel("training step")
    handles = [Line2D([], [], color=plot_style.color(i), label=m) for i, m in enumerate(models)]
    handles += [Line2D([], [], color=plot_style.MUTED, linestyle=plot_style.LINESTYLES[i], label=f"N={n}")
                for i, n in enumerate(n_values)]
    handles.append(Line2D([], [], color=plot_style.MUTED, alpha=0.3, linewidth=6, label="min-max over seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figures / "E011_long_runs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
