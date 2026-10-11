"""Train and compare small equalizers on file-level train/validation/test splits."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .core import BIT_TABLE, RAW_DATA_DIR, calculate_ber, get_indexes, load_channel_data
from .models import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("MLP", "1D-CNN", "LSTM", "Transformer")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_windows(rx: np.ndarray, tx: np.ndarray, window: int):
    if window % 2 != 1:
        raise ValueError("window must be odd so the target is its center symbol")
    if len(rx) < window:
        raise ValueError("sequence is shorter than the selected window")
    x = np.lib.stride_tricks.sliding_window_view(rx, window_shape=window)
    x = np.stack((x.real, x.imag), axis=-1).astype(np.float32)
    center = window // 2
    target = np.stack((tx.real[center : len(tx) - center], tx.imag[center : len(tx) - center]), axis=1)
    return x, target.astype(np.float32)


def load_split(window: int):
    files = sorted(RAW_DATA_DIR.glob("Symbols_1m_1ch_PR_*.csv"), key=lambda p: int(re.search(r"PR_(\d+)", p.name).group(1)))
    expected = {f"Symbols_1m_1ch_PR_{i}.csv" for i in range(1, 11)}
    if {p.name for p in files} != expected:
        raise FileNotFoundError("Expected the ten unique input files PR_1 through PR_10 in data/raw/")

    # Hold out complete pseudo-random sequences: this prevents overlapping
    # windows from the same sequence crossing a split boundary.
    train_files = files[:7]
    validation_files = files[7:8]
    test_files = files[8:]

    def collect(selected):
        xs, ys, rx_centers, tx_centers, run_names = [], [], [], [], []
        center = window // 2
        for path in selected:
            tx, rx = load_channel_data(path)
            x, y = make_windows(rx, tx, window)
            xs.append(x)
            ys.append(y)
            rx_centers.append(rx[center : len(rx) - center])
            tx_centers.append(tx[center : len(tx) - center])
            run_names.extend([path.stem] * len(y))
        return (
            np.concatenate(xs),
            np.concatenate(ys),
            np.concatenate(rx_centers),
            np.concatenate(tx_centers),
            np.asarray(run_names),
            [p.stem for p in selected],
        )

    return collect(train_files), collect(validation_files), collect(test_files)


def ber_from_iq(pred_iq: np.ndarray, tx_complex: np.ndarray) -> float:
    pred_complex = pred_iq[:, 0] + 1j * pred_iq[:, 1]
    pred_bits = BIT_TABLE[get_indexes(pred_complex)].reshape(-1)
    target_bits = BIT_TABLE[get_indexes(tx_complex)].reshape(-1)
    return calculate_ber(pred_bits, target_bits)


@torch.inference_mode()
def predict(model, x: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        chunks.append(model(xb).cpu().numpy())
    return np.concatenate(chunks)


def evaluate(model, data, device):
    x, y, rx_center, tx_center, run_names, run_files = data
    pred = predict(model, x, device)
    mse = float(np.mean((pred - y) ** 2))
    return {
        "ber": ber_from_iq(pred, tx_center),
        "mse": mse,
        "per_run": {
            name: ber_from_iq(pred[run_names == name], tx_center[run_names == name])
            for name in run_files
        },
        "raw_ber": ber_from_iq(np.column_stack((rx_center.real, rx_center.imag)), tx_center),
    }


def train_one(name: str, seed: int, train_data, validation_data, test_data, args, device):
    set_seed(seed)
    x_train, y_train = train_data[:2]
    x_val, y_val = validation_data[:2]
    model = build_model(name, args.window, hidden=getattr(args, "hidden", 64)).to(device)
    if getattr(args, "initializer", "") == "xavier_normal":
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_name = getattr(args, "optimizer", "adamw").lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError("optimizer must be 'adam' or 'adamw'")
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=False,
    )

    best_val_mse = float("inf")
    best_val_ber = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_pred = predict(model, x_val, device)
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        val_ber = ber_from_iq(val_pred, validation_data[3])
        # Select by BER, breaking ties by continuous-symbol MSE.
        if (val_ber, val_mse) < (best_val_ber, best_val_mse):
            best_val_ber, best_val_mse = val_ber, val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_mse < best_val_mse + 1e-12:
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, validation_data, device)
    test_metrics = evaluate(model, test_data, device) if test_data is not None else None
    return {
        "model": model,
        "seed": seed,
        "params": params,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "seconds": time.perf_counter() - started,
        "val": val_metrics,
        "test": test_metrics,
    }


def run_experiment(args):
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    train_data, validation_data, test_data = load_split(args.window)
    print(
        f"Device={device}; window={args.window}; train={len(train_data[1]):,} "
        f"({train_data[5]}); val={len(validation_data[1]):,} ({validation_data[5]}); "
        f"test={len(test_data[1]):,} ({test_data[5]})"
    )
    print(f"Raw BER: val={evaluate_raw(validation_data):.6f}, test={evaluate_raw(test_data):.6f}")

    results = []
    chosen = {}
    for name in MODEL_NAMES:
        runs = []
        for seed in args.seeds:
            print(f"Training {name} seed={seed} ...", flush=True)
            result = train_one(name, seed, train_data, validation_data, test_data, args, device)
            runs.append(result)
            print(
                f"  best epoch {result['best_epoch']}/{result['epochs_run']}; "
                f"val BER={result['val']['ber']:.6f}; test BER={result['test']['ber']:.6f}; "
                f"time={result['seconds']:.1f}s",
                flush=True,
            )
            results.append(
                {
                    "model": name,
                    "seed": seed,
                    "parameters": result["params"],
                    "best_epoch": result["best_epoch"],
                    "epochs_run": result["epochs_run"],
                    "train_seconds": result["seconds"],
                    "val_ber": result["val"]["ber"],
                    "val_mse": result["val"]["mse"],
                    "test_ber": result["test"]["ber"],
                    "test_mse": result["test"]["mse"],
                    "test_raw_ber": result["test"]["raw_ber"],
                    "test_ber_improvement_pct": 100 * (result["test"]["raw_ber"] - result["test"]["ber"]) / result["test"]["raw_ber"],
                    "test_ber_PR_9": result["test"]["per_run"]["Symbols_1m_1ch_PR_9"],
                    "test_ber_PR_10": result["test"]["per_run"]["Symbols_1m_1ch_PR_10"],
                }
            )
        # Final stored weights are picked using validation BER, never test BER.
        chosen[name] = min(runs, key=lambda r: (r["val"]["ber"], r["val"]["mse"]))

    output_dir = PROJECT_ROOT / "reports" / "results"
    model_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "model_comparison.csv", index=False)
    summary = frame.groupby("model").agg(
        seeds=("seed", "count"),
        parameters=("parameters", "first"),
        val_ber_mean=("val_ber", "mean"),
        val_ber_std=("val_ber", "std"),
        test_ber_mean=("test_ber", "mean"),
        test_ber_std=("test_ber", "std"),
        test_mse_mean=("test_mse", "mean"),
        train_seconds_mean=("train_seconds", "mean"),
        test_raw_ber=("test_raw_ber", "first"),
    ).reset_index()
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    for name, result in chosen.items():
        torch.save(
            {
                "model": name,
                "window": args.window,
                "seed": result["seed"],
                "state_dict": result["model"].state_dict(),
                "validation_ber": result["val"]["ber"],
                "train_files": train_data[5],
                "validation_files": validation_data[5],
                "test_files": test_data[5],
            },
            model_dir / f"{name.lower().replace('-', '')}.pt",
        )
    config = vars(args) | {
        "device": str(device),
        "train_files": train_data[5],
        "validation_files": validation_data[5],
        "test_files": test_data[5],
        "selection": "lowest validation BER, then lowest validation MSE",
        "training_target": "complex transmitted symbol represented as two real-valued outputs; MSE loss",
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nSummary (mean ± std over seeds):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote metrics to {output_dir} and selected checkpoints to {model_dir}.")


def evaluate_raw(data):
    rx_center, tx_center = data[2], data[3]
    return ber_from_iq(np.column_stack((rx_center.real, rx_center.imag)), tx_center)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_experiment(parse_args())
