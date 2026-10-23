"""E017 summary: PhysKerr + MLP_res vs complex_a and the chosen E016 PhysKerr.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E017   (SMOKE=1 reads results_smoke/)

Writes results/phys_plus_mlp.csv, results/phys_plus_mlp_summary.csv,
results/phys_plus_mlp_verdicts.csv, results/frozen_vs_joint.csv and
figures/E017_quality_vs_macs.png.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import plot_style
from .grids import e016_results_csv
from .merge_runs import load_runs
from .runner import exp_dir, results_root
from .summarize_E016 import METRIC, e012_points, group_summary
from .summary_utils import compare


def e016_reference(choice: dict, smoke: bool) -> tuple[list[float], pd.DataFrame]:
    """Per-seed test values of the chosen E016 configuration and the E016 summary for plotting."""
    path = e016_results_csv(smoke)
    if path is None or choice.get("lr_name") is None:
        return [], pd.DataFrame()
    runs = pd.read_csv(path)
    phys = runs[runs.model == "phys_kerr"]
    mine = phys[(phys.spec_n_alpha == choice["n_alpha"]) & (phys.spec_fir_len == choice["fir_len"])
                & (phys.spec_sym.astype(bool) == bool(choice["sym"])) & (phys.lr_name == choice["lr_name"])]
    summary = group_summary(phys, ["variant", "lr_name", "spec_n_alpha", "spec_fir_len", "spec_sym"])
    return mine[METRIC].tolist(), summary


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E017")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E017", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if runs.empty:
        print("нет прогонов")
        return 1
    runs.to_csv(results / "phys_plus_mlp.csv", index=False)
    choice_path = results / "e016_choice.json"
    choice = json.loads(choice_path.read_text()) if choice_path.is_file() else {}
    hybrids = runs[runs.model == "phys_plus_mlp"].copy()
    hybrids["mode"] = hybrids.variant.str.rsplit("_", n=1).str[-1]
    controls = runs[runs.stage == "control"]
    summary = group_summary(hybrids, ["variant", "lr_name", "spec_mlp_hidden", "mode"])
    summary.sort_values(["mode", "spec_mlp_hidden"]).to_csv(results / "phys_plus_mlp_summary.csv", index=False)
    control_summary = group_summary(controls, ["variant", "lr_name"]) if not controls.empty else pd.DataFrame()
    e016_values, e016_summary = e016_reference(choice, smoke)

    verdicts = []
    complex_a = controls[controls.variant == "complex_a"][METRIC].tolist()
    eligible = summary[summary.macs_per_symbol <= 6480]
    if not eligible.empty:
        best = eligible.loc[eligible.val_select_mean.idxmax()]
        mine = hybrids[hybrids.variant == best.variant][METRIC].tolist()
        vs_a, vs_phys = compare(mine, complex_a, 1.0), compare(mine, e016_values, 1.0)
        ok = vs_a["beats"] and vs_phys["beats"]
        verdicts.append({"hypothesis": "гибрид лучше при равной стоимости", "candidate": best.variant,
                         "candidate_macs": best.macs_per_symbol, "diff_vs_complex_a_pp": vs_a["diff"],
                         "spread_vs_complex_a_pp": vs_a["spread"], "diff_vs_e016_pp": vs_phys["diff"],
                         "spread_vs_e016_pp": vs_phys["spread"], "n_seeds": vs_a["n_a"],
                         "verdict": ("подтверждено" if ok else "не доказано") if e016_values else
                         "не доказано (нет данных E016)"})
    small = []
    for mode in config["modes"]:
        values = hybrids[(hybrids.spec_mlp_hidden == 4) & (hybrids["mode"] == mode)][METRIC].tolist()
        small.append((mode, compare(values, e016_values)))
    if small:
        within = all(r["within_spread"] for _, r in small) and bool(e016_values)
        verdicts.append({"hypothesis": "нейросеть на остаток не нужна (h = 4)",
                         "candidate": ", ".join(m for m, _ in small),
                         "diff_vs_e016_pp": "; ".join(f"{m}: {r['diff']:.3f}" for m, r in small),
                         "spread_vs_e016_pp": "; ".join(f"{m}: {r['spread']:.3f}" for m, r in small),
                         "verdict": ("подтверждено" if within else "не доказано") if e016_values else
                         "не доказано (нет данных E016)"})
    verdict_frame = pd.DataFrame(verdicts)
    verdict_frame["rule_status"] = config["rule_draft"]["status"]
    verdict_frame["e016_choice"] = json.dumps({k: choice.get(k) for k in ("n_alpha", "fir_len", "sym", "lr_name",
                                                                          "source")}, ensure_ascii=False)
    verdict_frame.to_csv(results / "phys_plus_mlp_verdicts.csv", index=False)

    observation = []
    for hidden in sorted(hybrids.spec_mlp_hidden.unique()):
        frozen = hybrids[(hybrids.spec_mlp_hidden == hidden) & (hybrids["mode"] == "frozen")][METRIC].tolist()
        joint = hybrids[(hybrids.spec_mlp_hidden == hidden) & (hybrids["mode"] == "joint")][METRIC].tolist()
        result = compare(joint, frozen)
        observation.append({"mlp_hidden": hidden, "joint_minus_frozen_pp": result["diff"],
                            "seed_spread_pp": result["spread"], "note": "observation only (different optimization paths)"})
    pd.DataFrame(observation).to_csv(results / "frozen_vs_joint.csv", index=False)
    plot(summary, control_summary, e016_summary, e012_points(), figures)
    print(f"сводка: {results / 'phys_plus_mlp_summary.csv'}; вердикты: {results / 'phys_plus_mlp_verdicts.csv'}")
    return 0


def plot(summary, controls, e016, e012, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (column, low, high, label) in zip(axes, (("test_mean", "test_min", "test_max", "test BER reduction, %"),
                                                     ("q2_gain_mean", "q2_gain_min", "q2_gain_max", "Q2 gain, dB"))):
        if not e016.empty:
            part = e016[e016.spec_n_alpha > 0]
            part = part[part[column] > (-50 if column == "test_mean" else -3)]
            ax.plot(part.macs_per_symbol, part[column], ".", color=plot_style.color(3), alpha=0.5, markersize=5)
        if not e012.empty:
            for _, row in e012[e012.budget == 20000].iterrows():
                ax.plot(row.macs, row[column if column != "test_mean" else "test_mean"], "D", markerfacecolor="none",
                        color=plot_style.color(4), markersize=6)
        for index, mode in enumerate(("frozen", "joint")):
            part = summary[summary["mode"] == mode].sort_values("macs_per_symbol")
            ax.errorbar(part.macs_per_symbol, part[column], yerr=[np.clip(part[column] - part[low], 0, None),
                        np.clip(part[high] - part[column], 0, None)], fmt="-o", color=plot_style.color(index),
                        markersize=6, capsize=2, linewidth=1.2)
        for _, row in controls.iterrows():
            ax.errorbar(row.macs_merged, row[column], yerr=[[max(row[column] - row[low], 0)],
                        [max(row[high] - row[column], 0)]], fmt="s", color=plot_style.color(2), markersize=7, capsize=2)
        ax.axvline(6480, color=plot_style.MUTED, linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("real multiplies per symbol")
        ax.set_ylabel(label)
    handles = [Line2D([], [], color=plot_style.color(0), marker="o", label="Phys + MLP_res, frozen (h = 4, 8, 16, 48)"),
               Line2D([], [], color=plot_style.color(1), marker="o", label="Phys + MLP_res, joint"),
               Line2D([], [], color=plot_style.color(2), marker="s", linestyle="", label="complex_a / one_hidden (E017)"),
               Line2D([], [], color=plot_style.color(3), marker=".", linestyle="", label="PhysKerr (E016)"),
               Line2D([], [], color=plot_style.color(4), marker="D", markerfacecolor="none", linestyle="",
                      label="E012 MLPs, 20k")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("PhysKerr + residual MLP: test at best validation (dotted line: 6480 multiplies of complex_a)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(figures / "E017_quality_vs_macs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
