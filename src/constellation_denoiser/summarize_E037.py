"""E037: are the triplets robust to the file split? Closed-form T3 + FIR7 on the three E023 splits.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E037 [fit|summarize|all]   (SMOKE=1: 2/1/1 files)

For every split seed of E023 (same 300 files permuted; train 30, validation 45, test 45) the closed-form fit of
E031/E034 (5 bootstrap replicas). The networks of the same splits come from E023 (results/runs_table.csv):
the best member of each split by mean validation. Writes results/split_fits.csv, split_summary.csv,
split_verdicts.csv and figures/E037_triplets_by_split.png.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

from .grids import e023_split
from .runner import atomic_write_json, exp_dir, results_root
from .summarize_E034 import fit


def networks_by_split() -> pd.DataFrame:
    path = exp_dir("E023") / "results" / "runs_table.csv"
    if not path.is_file():
        return pd.DataFrame()
    r = pd.read_csv(path)
    r = r[r.stage == "main"]
    agg = r.groupby(["split_seed", "family", "variant"]).agg(
        val_mean=("best_val_reduction", "mean"), test_mean=("test_reduction_valbest_select", "mean"),
        test_min=("test_reduction_valbest_select", "min"), test_max=("test_reduction_valbest_select", "max"),
        rm=("macs_per_symbol_merged", "first")).reset_index()
    return agg.loc[agg.groupby("split_seed").val_mean.idxmax()]


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E037")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    out = results_root("E037", smoke)
    figures = out / "figures" if smoke else directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    action = argv[0] if argv else "all"
    if action in ("fit", "all"):
        frames = []
        for seed in config["split_seeds"]:
            part = fit(config, smoke, out, files=e023_split(seed, smoke), name=f"fits_split{seed}.csv", prune=False)
            frames.append(part.assign(split_seed=seed))
        pd.concat(frames).to_csv(out / "split_fits.csv", index=False)
    if action in ("summarize", "all"):
        f = pd.read_csv(out / "split_fits.csv")
        s = f.groupby(["split_seed", "variant"]).agg(
            rm=("rm_sym", "first"), val_mean=("val_reduction", "mean"), test_mean=("test_reduction", "mean"),
            test_min=("test_reduction", "min"), test_max=("test_reduction", "max")).reset_index()
        nets = networks_by_split()
        s.to_csv(out / "split_summary.csv", index=False)
        nets.to_csv(out / "networks_by_split.csv", index=False)
        rows = []
        key = f"T3FIR{config['fir_len']}_M{config['rule_M']}_full"
        tri = s[s.variant == key].set_index("split_seed")
        if len(tri) and len(nets):
            net = nets.set_index("split_seed")
            common = sorted(set(tri.index) & set(net.index))
            wins = [bool(tri.loc[k, "test_min"] > net.loc[k, "test_max"]) for k in common]
            rows.append({"hypothesis": f"триплеты {key} лучше лучшей сети E023 на каждом разбиении (интервалы не пересекаются)",
                         "value": sum(wins), "verdict": "подтверждено" if wins and all(wins) else "не доказано",
                         "detail": "; ".join(f"{k}: {tri.loc[k, 'test_mean']:.2f} против {net.loc[k, 'test_mean']:.2f} "
                                             f"({net.loc[k, 'variant']})" for k in common)})
        if len(tri):
            spread = float(tri.test_mean.max() - tri.test_mean.min())
            rows.append({"hypothesis": f"разброс средних {key} между разбиениями < 2 п.п.", "value": spread,
                         "verdict": "подтверждено" if spread < 2 else "не доказано", "detail": ""})
        v = pd.DataFrame(rows, columns=["hypothesis", "value", "verdict", "detail"])
        v["rule_status"] = config["rule_draft"]["status"]
        v.to_csv(out / "split_verdicts.csv", index=False)
        plot(s, nets, figures)
    atomic_write_json(out / "status.json", {"done": action})
    print(f"E037: {out / 'split_summary.csv'}")
    return 0


def plot(s: pd.DataFrame, nets: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for index, (variant, part) in enumerate(s.groupby("variant")):
        ax.errorbar(part.split_seed.astype(str), part.test_mean, yerr=[part.test_mean - part.test_min,
                    part.test_max - part.test_mean], marker="osD^"[index % 4], color=plot_style.color(index), capsize=2,
                    linestyle="", label=f"{variant} ({int(part.rm.iloc[0])} RM)")
    if len(nets):
        ax.errorbar(nets.split_seed.astype(int).astype(str), nets.test_mean, yerr=[nets.test_mean - nets.test_min,
                    nets.test_max - nets.test_mean], marker="x", color=plot_style.TEXT, capsize=2, linestyle="",
                    label="best network of E023 (by validation)")
    ax.set_xlabel("split seed (E023)")
    ax.set_ylabel("test BER reduction, % (min-max)")
    ax.set_title("Triplets vs networks on three file splits")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "E037_triplets_by_split.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
