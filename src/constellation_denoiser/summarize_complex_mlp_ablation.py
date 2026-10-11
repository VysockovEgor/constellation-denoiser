"""Summarize Kerr ablation runs: table, merge checks, improvement-vs-cost plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .complex_mlp_ablation import complexity, verify_merge
from .experiment import PROJECT_ROOT
WINDOW = 31
HIDDEN = 48
N_FILES = 30
SEEDS = (17, 29)
VARIANT_LABELS = {
    "baseline": "baseline (2× Kerr)",
    "a": "(a) Kerr after L2",
    "b": "(b) Kerr after L1",
    "c": "(c) linear",
    "d": "(d) 1 hidden layer",
}


def load_baseline_rows(sweep_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(sweep_csv)
    frame = frame[(frame.window == WINDOW) & (frame.hidden == HIDDEN) & (frame.N == N_FILES)]
    frame = frame[frame.seed.isin(SEEDS)]
    best = frame.loc[frame.groupby("seed")["val_ber"].idxmin()].copy()
    cx = complexity(WINDOW, HIDDEN, "baseline")
    best["variant"] = "baseline"
    best["test_improvement_pct"] = 100 * (best.raw_ber - best.test_ber) / best.raw_ber
    best["train_total_real_multiplies"] = cx.train_total_real_multiplies
    best["merged_total_real_multiplies"] = cx.merged_total_real_multiplies
    return best


def load_ablation_rows(ablation_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(ablation_csv)
    frame = frame[frame.seed.isin(SEEDS)]
    best = frame.loc[frame.groupby(["variant", "seed"])["val_ber"].idxmin()].copy()
    best["test_improvement_pct"] = 100 * (best.raw_ber - best.test_ber) / best.raw_ber
    return best


def summarize_table(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("variant", as_index=False).agg(
        mean_improvement_pct=("test_improvement_pct", "mean"),
        min_improvement_pct=("test_improvement_pct", "min"),
        max_improvement_pct=("test_improvement_pct", "max"),
        train_total_real_multiplies=("train_total_real_multiplies", "first"),
        merged_total_real_multiplies=("merged_total_real_multiplies", "first"),
    )
    order = ["baseline", "a", "b", "c", "d"]
    grouped["variant"] = pd.Categorical(grouped["variant"], categories=order, ordered=True)
    grouped = grouped.sort_values("variant")
    grouped["label"] = grouped["variant"].map(VARIANT_LABELS)
    return grouped


def verify_best_checkpoints(ablation_csv: Path, train_windows) -> pd.DataFrame:
    from .complex_mlp_ablation import AblationComplexMLP

    rows = []
    frame = pd.read_csv(ablation_csv)
    for variant in ("a", "c"):
        for seed in SEEDS:
            subset = frame[(frame.variant == variant) & (frame.seed == seed)]
            if subset.empty:
                continue
            best_row = subset.sort_values(["val_ber", "step"]).iloc[0]
            ckpt = (
                PROJECT_ROOT
                / "models/complex_mlp_ablation"
                / f"ablation_{variant}_w{WINDOW}_h{HIDDEN}_N{N_FILES}_seed{seed}_best.pt"
            )
            if not ckpt.is_file():
                rows.append({"variant": variant, "seed": seed, "merge_max_abs_diff": float("nan"), "ok": False})
                continue
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            model = AblationComplexMLP(window=WINDOW, hidden=HIDDEN, variant=variant)
            model.load_state_dict(payload["model_state"])
            diff, count = verify_merge(model, train_windows)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "merge_max_abs_diff": diff,
                    "merge_windows_checked": count,
                    "ok": diff <= 1e-5,
                }
            )
    return pd.DataFrame(rows)


def plot_tradeoff(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in summary.iterrows():
        ax.scatter(
            row.merged_total_real_multiplies,
            row.mean_improvement_pct,
            s=80,
        )
        ax.annotate(
            row.label,
            (row.merged_total_real_multiplies, row.mean_improvement_pct),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xlabel("Real multiplies per symbol after linear merge")
    ax.set_ylabel("Test BER improvement vs raw (%)")
    ax.set_title("Kerr ablation: accuracy vs merged inference cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def interpret(summary: pd.DataFrame) -> list[str]:
    by_variant = summary.set_index("variant")
    baseline = float(by_variant.loc["baseline", "mean_improvement_pct"])
    variant_a = float(by_variant.loc["a", "mean_improvement_pct"])
    variant_c = float(by_variant.loc["c", "mean_improvement_pct"])
    variant_d = float(by_variant.loc["d", "mean_improvement_pct"])
    notes = []
    if abs(variant_a - baseline) < 2.0:
        ratio = (
            by_variant.loc["baseline", "train_total_real_multiplies"]
            / by_variant.loc["a", "merged_total_real_multiplies"]
        )
        notes.append(
            f"(a) vs baseline within 2 p.p.: first Kerr likely redundant; merged cost ratio ≈ {ratio:.2f}×."
        )
    if variant_d + 2.0 < variant_a:
        notes.append("(d) worse than (a) by >2 p.p.: two-layer parameterization helps training.")
    if variant_c < 20.0:
        notes.append("(c) <20% improvement: nonlinearity is decisive.")
    elif variant_c >= 40.0:
        notes.append("(c) ≥40% improvement: most gain comes from the linear part.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-csv", type=Path, default=PROJECT_ROOT / "reports/results/complex_mlp_sweep.csv")
    parser.add_argument("--ablation-csv", type=Path, default=PROJECT_ROOT / "reports/results/complex_mlp_ablation.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/results")
    args = parser.parse_args()

    baseline = load_baseline_rows(args.sweep_csv)
    ablation = load_ablation_rows(args.ablation_csv)
    combined = pd.concat([baseline, ablation], ignore_index=True, sort=False)
    summary = summarize_table(combined)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "complex_mlp_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    config = json.loads((PROJECT_ROOT / "reports/results/learning_curve_config.json").read_text(encoding="utf-8"))
    from .fixed_step_curve import collect
    from .large_scale_pilot import get_unique_files

    pilot = json.loads((PROJECT_ROOT / "reports/results/large_scale_pilot_config.json").read_text(encoding="utf-8"))
    candidates = get_unique_files(Path(pilot["dataset_root"]), PROJECT_ROOT / "reports/results/data_audit_all.csv")
    by_name = {path.name: path for path in candidates.path}
    train_data = collect([by_name[name] for name in config["train_pool_files"][:N_FILES]], WINDOW)
    merge_report = verify_best_checkpoints(args.ablation_csv, train_data[0])
    merge_report.to_csv(out_dir / "complex_mlp_ablation_merge_check.csv", index=False)

    plot_tradeoff(summary, out_dir / "complex_mlp_ablation_tradeoff.png")

    print("Summary table (variant × BER improvement, multiplies):")
    printable = summary[
        [
            "label",
            "mean_improvement_pct",
            "min_improvement_pct",
            "max_improvement_pct",
            "train_total_real_multiplies",
            "merged_total_real_multiplies",
        ]
    ]
    print(printable.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if not merge_report.empty:
        print("\nMerge verification (a,c on best checkpoints):")
        print(merge_report.to_string(index=False))

    notes = interpret(summary)
    if notes:
        print("\nInterpretation:")
        for note in notes:
            print(f"- {note}")

    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
