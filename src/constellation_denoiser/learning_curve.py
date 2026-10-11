"""Train tanh and complex MLPs on nested file subsets with fixed holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .core import load_channel_data
from .experiment import ber_from_iq, make_windows, train_one
from .large_scale_pilot import get_unique_files
from .paths import E003_DIR, E004_RESULTS, MANIFEST_DIR, project_path
from .paper_complex_mlp import predict_complex, train as train_complex


def collect(paths: list[Path], window: int):
    xs, ys, rx_centers, tx_centers, names, files = [], [], [], [], [], []
    edge = window // 2
    for path in paths:
        tx, rx = load_channel_data(path)
        x, y = make_windows(rx, tx, window)
        xs.append(x)
        ys.append(y)
        rx_centers.append(rx[edge : len(rx) - edge])
        tx_centers.append(tx[edge : len(tx) - edge])
        names.extend([path.stem] * len(y))
        files.append(path.stem)
    return (
        np.concatenate(xs), np.concatenate(ys), np.concatenate(rx_centers),
        np.concatenate(tx_centers), np.asarray(names), files,
    )


def per_file_metrics(pred: np.ndarray, data) -> list[dict]:
    _, _, rx_center, tx_center, names, files = data
    raw = np.column_stack((rx_center.real, rx_center.imag))
    rows = []
    for name in files:
        mask = names == name
        rows.append({
            "file": name,
            "raw_ber": ber_from_iq(raw[mask], tx_center[mask]),
            "model_ber": ber_from_iq(pred[mask], tx_center[mask]),
            "symbols": int(mask.sum()),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[7, 30, 60, 120, 210])
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--subset-seed", type=int, default=20261004)
    p.add_argument("--window", type=int, default=31)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--resume", action="store_true", help="continue from completed runs in the results CSVs")
    p.add_argument("--audit-csv", type=Path, default=MANIFEST_DIR / "data_audit_all.csv")
    p.add_argument("--pilot-config", type=Path, default=E003_DIR / "config.json")
    a = p.parse_args()

    config = json.loads(a.pilot_config.read_text(encoding="utf-8"))
    data_dir = project_path(config["dataset_root"])
    candidates = get_unique_files(data_dir, a.audit_csv)
    by_name = {path.name: path for path in candidates.path}
    train_pool = [by_name[name] for name in config["train_files"]]
    val_paths = [by_name[name] for name in config["validation_files"]]
    test_paths = [by_name[name] for name in config["test_files"]]
    if max(a.sizes) > len(train_pool):
        raise ValueError(f"Largest requested size {max(a.sizes)} exceeds fixed training pool {len(train_pool)}")
    if len(test_paths) != 45:
        raise ValueError(f"Expected the existing 45-file holdout, got {len(test_paths)}")

    # One seeded ordering makes every smaller training set a subset of every larger set.
    rng = np.random.default_rng(a.subset_seed)
    ordered_train = [train_pool[i] for i in rng.permutation(len(train_pool))]
    torch.set_num_threads(a.threads)
    val_data = collect(val_paths, a.window)
    test_data = collect(test_paths, a.window)
    raw_per_file = per_file_metrics(
        np.column_stack((test_data[2].real, test_data[2].imag)), test_data
    )
    raw_ber = float(np.mean([r["raw_ber"] for r in raw_per_file]))

    out = E004_RESULTS
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "learning_curve_runs.csv"
    per_file_path = out / "learning_curve_per_file.csv"
    run_rows: list[dict] = []
    file_rows: list[dict] = []
    if a.resume and rows_path.is_file():
        run_rows = pd.read_csv(rows_path).to_dict("records")
    if a.resume and per_file_path.is_file():
        file_rows = pd.read_csv(per_file_path).to_dict("records")
    completed = {(int(r["train_files"]), str(r["model"]), int(r["seed"])) for r in run_rows}
    base = SimpleNamespace(
        window=a.window, hidden=64, learning_rate=a.learning_rate,
        weight_decay=0.0, batch_size=a.batch_size, epochs=a.epochs,
        patience=a.patience, optimizer="adam", initializer="xavier_normal",
    )
    print(
        f"Fixed holdouts: validation={len(val_paths)} files, test={len(test_paths)} files; "
        f"raw mean test BER={raw_ber:.6f}; training sizes={a.sizes}; seeds={a.seeds}",
        flush=True,
    )

    for size in a.sizes:
        train_paths = ordered_train[:size]
        train_data = collect(train_paths, a.window)
        print(f"\n=== train_files={size}; nested subset ===", flush=True)
        for kind in ("real", "complex"):
            for seed in a.seeds:
                if (size, kind, seed) in completed:
                    print(f"Skipping completed {kind} size={size} seed={seed}", flush=True)
                    continue
                print(f"Training {kind} size={size} seed={seed} ...", flush=True)
                if kind == "real":
                    result = train_one("MLP-Tanh", seed, train_data, val_data, None, base, torch.device("cpu"))
                    model = result["model"]
                    best_epoch, epochs_run = result["best_epoch"], result["epochs_run"]
                    val_ber, val_mse, seconds = result["val"]["ber"], result["val"]["mse"], result["seconds"]
                    from .experiment import predict
                    pred = predict(model, test_data[0], torch.device("cpu"))
                else:
                    model, best_epoch, epochs_run, val_ber, val_mse, seconds = train_complex(
                        seed, train_data, val_data, window=a.window, hidden=32,
                        epochs=a.epochs, patience=a.patience, batch_size=a.batch_size,
                        learning_rate=a.learning_rate, threads=a.threads,
                    )
                    pred = predict_complex(model, test_data[0])
                file_metrics = per_file_metrics(pred, test_data)
                mean_model = float(np.mean([r["model_ber"] for r in file_metrics]))
                improvement = 100 * (raw_ber - mean_model) / raw_ber
                run_rows.append({
                    "train_files": size, "model": kind, "seed": seed,
                    "window": a.window, "epochs_limit": a.epochs,
                    "best_epoch": best_epoch, "epochs_run": epochs_run,
                    "validation_ber": val_ber, "validation_mse": val_mse,
                    "test_raw_ber_mean_over_files": raw_ber,
                    "test_model_ber_mean_over_files": mean_model,
                    "improvement_pct_vs_raw": improvement,
                    "train_seconds": seconds,
                    "train_file_names": "|".join(p.name for p in train_paths),
                })
                file_rows.extend({
                    "train_files": size, "model": kind, "seed": seed, **r,
                    "improvement_pct_vs_raw": 100 * (r["raw_ber"] - r["model_ber"]) / r["raw_ber"],
                } for r in file_metrics)
                pd.DataFrame(run_rows).to_csv(rows_path, index=False)
                pd.DataFrame(file_rows).to_csv(per_file_path, index=False)
                print(
                    f"  best epoch={best_epoch}/{epochs_run}; test BER={mean_model:.6f}; "
                    f"improvement={improvement:.2f}%; elapsed={seconds/60:.1f} min",
                    flush=True,
                )

    result_config = {
        "training_sizes": a.sizes, "seeds": a.seeds,
        "subset_seed": a.subset_seed,
        "train_pool_files": [p.name for p in ordered_train],
        "validation_files": [p.name for p in val_paths],
        "test_files": [p.name for p in test_paths],
        "window": a.window, "epochs_limit": a.epochs, "early_stopping_patience": a.patience,
        "batch_size": a.batch_size, "learning_rate": a.learning_rate,
        "real_model": {"hidden": 64, "activation": "tanh", "optimizer": "Adam", "loss": "MSE"},
        "complex_model": {"hidden": 32, "activation": "exp(i*c*|z|^2)*z", "optimizer": "Adam", "loss": "complex MSE"},
        "test_mean_raw_ber": raw_ber,
    }
    (out / "learning_curve_config.json").write_text(json.dumps(result_config, indent=2), encoding="utf-8")
    print("\nCompleted learning curve.", flush=True)
    print(pd.DataFrame(run_rows).groupby(["train_files", "model"])[["test_model_ber_mean_over_files", "improvement_pct_vs_raw", "train_seconds"]].agg(["mean", "std"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
