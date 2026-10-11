"""Create a homogeneous seed-17 learning-curve table and figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .experiment import PROJECT_ROOT


def main():
    results = PROJECT_ROOT / "reports/results"
    runs = pd.read_csv(results / "learning_curve_runs.csv")
    # Keep one identical initialization seed at every sample size. The run log
    # also contains completed exploratory seeds for the smallest sizes.
    curve = runs[runs.seed == 17].copy().sort_values(["model", "train_files"])
    expected = {7, 30, 60, 120, 210}
    for model in ("real", "complex"):
        observed = set(curve.loc[curve.model == model, "train_files"].astype(int))
        if observed != expected:
            raise ValueError(f"Incomplete seed-17 curve for {model}: {sorted(observed)}")

    table = curve[[
        "train_files", "model", "best_epoch", "epochs_run",
        "test_raw_ber_mean_over_files", "test_model_ber_mean_over_files",
        "improvement_pct_vs_raw", "train_seconds",
    ]].sort_values(["train_files", "model"])
    table.to_csv(results / "learning_curve_seed17.csv", index=False)

    labels = {"real": "Real MLP (tanh)", "complex": "Complex phase MLP"}
    colors = {"real": "#2878B5", "complex": "#D55E00"}
    fig, (ax_ber, ax_gain) = plt.subplots(1, 2, figsize=(12.5, 5.1), constrained_layout=True)
    for model in ("real", "complex"):
        rows = curve[curve.model == model].sort_values("train_files")
        x = rows.train_files.to_numpy()
        ax_ber.plot(x, rows.test_model_ber_mean_over_files, marker="o", linewidth=2.2,
                    markersize=6, color=colors[model], label=labels[model])
        ax_gain.plot(x, rows.improvement_pct_vs_raw, marker="o", linewidth=2.2,
                     markersize=6, color=colors[model], label=labels[model])
    raw = float(curve.test_raw_ber_mean_over_files.iloc[0])
    ax_ber.axhline(raw, color="#555555", linestyle="--", linewidth=1.5,
                   label=f"Raw BER ({raw:.4f})")
    ax_ber.set_xscale("log")
    ax_ber.set_xticks([7, 30, 60, 120, 210], labels=["7", "30", "60", "120", "210"])
    ax_ber.set_xlabel("Training files")
    ax_ber.set_ylabel("Mean BER on the same 45 test files")
    ax_ber.set_title("Test BER")
    ax_ber.grid(True, alpha=0.25)
    ax_ber.legend(frameon=False)

    ax_gain.axhline(0, color="#555555", linestyle="--", linewidth=1.3)
    ax_gain.axhline(40, color="#777777", linestyle=":", linewidth=1.3, label="40% reference")
    ax_gain.set_xscale("log")
    ax_gain.set_xticks([7, 30, 60, 120, 210], labels=["7", "30", "60", "120", "210"])
    ax_gain.set_xlabel("Training files")
    ax_gain.set_ylabel("BER reduction vs raw (%)")
    ax_gain.set_title("Relative improvement")
    ax_gain.grid(True, alpha=0.25)
    ax_gain.legend(frameon=False)
    fig.suptitle("Learning curve on a fixed test set · seed 17", fontsize=14)
    fig.savefig(PROJECT_ROOT / "reports/figures/learning_curve_seed17.png", dpi=180)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"Raw BER={raw:.6f}")
    print(f"Wrote {results / 'learning_curve_seed17.csv'}")
    print(f"Wrote {PROJECT_ROOT / 'reports/figures/learning_curve_seed17.png'}")


if __name__ == "__main__":
    main()
