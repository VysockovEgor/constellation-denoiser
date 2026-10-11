"""Validation-only sweep for a tanh MLP with different symbol memories."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from .experiment import evaluate, evaluate_raw, load_split, parse_args, train_one
from .paths import E002_CHECKPOINTS, E002_RESULTS


CONFIGS = [(7, 64), (15, 64), (31, 64), (31, 128)]


def main():
    base = parse_args()
    seeds = base.seeds
    device = torch.device("cpu")
    torch.set_num_threads(base.threads)
    runs_by_config = {}
    rows = []

    for window, hidden in CONFIGS:
        train_data, validation_data, test_data = load_split(window)
        config_runs = []
        for seed in seeds:
            args = SimpleNamespace(
                window=window,
                hidden=hidden,
                learning_rate=base.learning_rate,
                weight_decay=base.weight_decay,
                batch_size=base.batch_size,
                epochs=base.epochs,
                patience=base.patience,
            )
            print(f"Training tanh MLP window={window}, hidden={hidden}, seed={seed} ...", flush=True)
            run = train_one("MLP-Tanh", seed, train_data, validation_data, None, args, device)
            config_runs.append(run)
            rows.append(
                {
                    "window": window,
                    "hidden_per_layer": hidden,
                    "seed": seed,
                    "parameters": run["params"],
                    "best_epoch": run["best_epoch"],
                    "epochs_run": run["epochs_run"],
                    "train_seconds": run["seconds"],
                    "validation_ber": run["val"]["ber"],
                    "validation_mse": run["val"]["mse"],
                }
            )
            print(
                f"  best epoch {run['best_epoch']}/{run['epochs_run']}; "
                f"validation BER={run['val']['ber']:.6f}; "
                f"raw validation BER={evaluate_raw(validation_data):.6f}; "
                f"time={run['seconds']:.1f}s",
                flush=True,
            )
        runs_by_config[(window, hidden)] = (config_runs, train_data, validation_data, test_data)

    sweep = pd.DataFrame(rows)
    summary = sweep.groupby(["window", "hidden_per_layer"]).agg(
        seeds=("seed", "count"),
        parameters=("parameters", "first"),
        val_ber_mean=("validation_ber", "mean"),
        val_ber_std=("validation_ber", "std"),
        val_mse_mean=("validation_mse", "mean"),
        train_seconds_mean=("train_seconds", "mean"),
    ).reset_index()
    chosen_row = summary.sort_values(["val_ber_mean", "val_mse_mean"]).iloc[0]
    chosen_config = (int(chosen_row["window"]), int(chosen_row["hidden_per_layer"]))
    chosen_runs, train_data, validation_data, test_data = runs_by_config[chosen_config]
    print(
        f"\nSelected by mean validation BER: window={chosen_config[0]}, "
        f"hidden={chosen_config[1]}; raw test BER={evaluate_raw(test_data):.6f}. "
        "Evaluating this selected configuration on PR9–PR10 now.",
        flush=True,
    )

    test_rows = []
    for run in chosen_runs:
        metrics = evaluate(run["model"], test_data, device)
        test_rows.append(
            {
                "window": chosen_config[0],
                "hidden_per_layer": chosen_config[1],
                "seed": run["seed"],
                "parameters": run["params"],
                "validation_ber": run["val"]["ber"],
                "test_ber": metrics["ber"],
                "test_raw_ber": metrics["raw_ber"],
                "test_improvement_pct_vs_raw": 100 * (metrics["raw_ber"] - metrics["ber"]) / metrics["raw_ber"],
                "test_ber_PR_9": metrics["per_run"]["Symbols_1m_1ch_PR_9"],
                "test_ber_PR_10": metrics["per_run"]["Symbols_1m_1ch_PR_10"],
            }
        )

    output_dir = E002_RESULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = E002_CHECKPOINTS
    model_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_dir / "tanh_mlp_sweep.csv", index=False)
    summary.to_csv(output_dir / "tanh_mlp_sweep_summary.csv", index=False)
    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(output_dir / "tanh_mlp_selected_test.csv", index=False)
    selected_seed = min(chosen_runs, key=lambda r: (r["val"]["ber"], r["val"]["mse"]))
    torch.save(
        {
            "model": "MLP-Tanh",
            "window": chosen_config[0],
            "hidden_per_layer": chosen_config[1],
            "seed": selected_seed["seed"],
            "state_dict": selected_seed["model"].state_dict(),
            "validation_ber": selected_seed["val"]["ber"],
            "train_files": train_data[5],
            "validation_files": validation_data[5],
            "test_files": test_data[5],
        },
        model_dir / "tanh_mlp_selected.pt",
    )
    config = {
        "configs": CONFIGS,
        "seeds": seeds,
        "selected_by": "lowest mean validation BER across seeds; MSE breaks ties",
        "validation_files": validation_data[5],
        "test_files": test_data[5],
        "activation": "tanh",
        "target": "I/Q of the central transmitted symbol",
        "loss": "MSE",
    }
    (output_dir / "tanh_mlp_sweep_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nValidation sweep:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSelected-configuration test by seed:")
    print(test_frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nWrote tanh sweep, selected checkpoint, and test scores.")


if __name__ == "__main__":
    main()
