"""Fixed-update-count learning curves for the paper complex MLP and tanh MLP."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .core import load_channel_data
from .experiment import PROJECT_ROOT, ber_from_iq, make_windows, predict, set_seed
from .large_scale_pilot import get_unique_files
from .models import build_model
from .paper_complex_mlp import PaperComplexMLP, predict_complex


CSV_FIELDS = ["model", "N", "seed", "step", "val_ber", "test_ber", "raw_ber"]
DEFAULT_SIZES = (7, 30, 105, 210)
SEED = 17
WINDOW = 31
BATCH_SIZE = 1024
MAX_STEPS = 20_000
CHECKPOINT_EVERY = 1_000


def collect(paths: list[Path], window: int):
    if not paths:
        raise ValueError("Cannot collect an empty file list")
    rows_per_file = 16_384 - window + 1
    total = rows_per_file * len(paths)
    xs = np.empty((total, window, 2), dtype=np.float32)
    ys = np.empty((total, 2), dtype=np.float32)
    rx_centers = np.empty(total, dtype=np.complex128)
    tx_centers = np.empty(total, dtype=np.complex128)
    edge = window // 2
    offset = 0
    for path in paths:
        tx, rx = load_channel_data(path)
        x, y = make_windows(rx, tx, window)
        if len(y) != rows_per_file:
            raise ValueError(f"Unexpected row count in {path}: expected {rows_per_file + window - 1}, got {len(y) + window - 1}")
        stop = offset + len(y)
        xs[offset:stop] = x
        ys[offset:stop] = y
        rx_centers[offset:stop] = rx[edge : len(rx) - edge]
        tx_centers[offset:stop] = tx[edge : len(tx) - edge]
        offset = stop
    return xs, ys, rx_centers, tx_centers


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_metric(path: Path, row: dict) -> None:
    """Append under a process lock, suppressing a row replayed on resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        existing = csv.DictReader(f)
        existing_fields = existing.fieldnames
        duplicate = any(
            r.get("model") == row["model"]
            and r.get("N") == str(row["N"])
            and r.get("seed", str(SEED)) == str(row["seed"])
            and r.get("step") == str(row["step"])
            for r in existing
        )
        f.seek(0, os.SEEK_END)
        empty = f.tell() == 0
        fields = CSV_FIELDS if empty else existing_fields
        writer = csv.DictWriter(f, fieldnames=fields)
        if empty:
            writer.writeheader()
        if not duplicate:
            writer.writerow({key: row.get(key) for key in fields})
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def make_model(model_name: str, seed: int = SEED):
    set_seed(seed)
    if model_name == "tanh":
        model = build_model("MLP-Tanh", WINDOW, hidden=64)
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        return model
    if model_name == "complex":
        return PaperComplexMLP(window=WINDOW, hidden=32)
    raise ValueError(model_name)


@torch.inference_mode()
def evaluate_model(model, model_name: str, data) -> float:
    x, _, _, tx_centers = data
    prediction = predict(model, x, torch.device("cpu")) if model_name == "tanh" else predict_complex(model, x)
    return ber_from_iq(prediction, tx_centers)


def training_batch(x: torch.Tensor, y: torch.Tensor, permutation: torch.Tensor,
                   cursor: int, sampler: torch.Generator):
    if cursor >= len(permutation):
        permutation = torch.randperm(len(x), generator=sampler)
        cursor = 0
    stop = min(cursor + BATCH_SIZE, len(permutation))
    ix = permutation[cursor:stop]
    return x.index_select(0, ix), y.index_select(0, ix), permutation, stop


def train_one(model_name: str, n_files: int, train_data, val_data, test_data,
              raw_ber: float, steps: int, interval: int, threads: int, resume: bool,
              csv_path: Path, seed: int = SEED):
    torch.set_num_threads(threads)
    checkpoint_dir = PROJECT_ROOT / "models/step_curve"
    seed_tag = "" if seed == SEED else f"_seed{seed}"
    last_path = checkpoint_dir / f"{model_name}_N{n_files}{seed_tag}_last.pt"
    best_path = checkpoint_dir / f"{model_name}_N{n_files}{seed_tag}_best.pt"
    x = torch.from_numpy(train_data[0])
    y_real = torch.from_numpy(train_data[1])
    y_complex = torch.complex(y_real[:, 0], y_real[:, 1])
    model = make_model(model_name, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(x), generator=sampler)
    cursor, step = 0, 0
    best_val = float("inf")
    best_step = 0
    best_state = None
    last_metrics = None

    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        sampler.set_state(state["sampler_state"])
        permutation = state["permutation"]
        cursor = int(state["cursor"])
        step = int(state["step"])
        best_val = float(state["best_val_ber"])
        best_step = int(state["best_step"])
        best_state = state["best_model_state"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        last_metrics = state.get("last_metrics")
        print(f"{model_name} N={n_files}: resumed at step {step}", flush=True)
    else:
        print(f"{model_name} N={n_files}: starting at step 0", flush=True)

    if last_metrics is not None:
        last_metrics.setdefault("seed", seed)
        append_metric(csv_path, last_metrics)

    x_val, tx_val = val_data[0], val_data[3]
    x_test, tx_test = test_data[0], test_data[3]
    mse = nn.MSELoss()
    started = time.perf_counter()
    while step < steps:
        model.train()
        xb, yb, permutation, cursor = training_batch(x, y_real, permutation, cursor, sampler)
        optimizer.zero_grad(set_to_none=True)
        if model_name == "tanh":
            loss = mse(model(xb), yb)
        else:
            yb_c = torch.complex(yb[:, 0], yb[:, 1])
            pred = model(xb)
            loss = (pred - yb_c).abs().square().mean()
        loss.backward()
        optimizer.step()
        step += 1

        if step % interval != 0 and step != steps:
            continue

        val_ber = evaluate_model(model, model_name, val_data)
        test_ber = evaluate_model(model, model_name, test_data)
        if val_ber < best_val:
            best_val = val_ber
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_torch_save({
                "model": model_name, "N": n_files, "seed": seed, "window": WINDOW,
                "step": step, "best_val_ber": best_val, "model_state": best_state,
            }, best_path)
        last_metrics = {
            "model": model_name, "N": n_files, "seed": seed, "step": step,
            "val_ber": val_ber, "test_ber": test_ber, "raw_ber": raw_ber,
        }
        atomic_torch_save({
            "model": model_name, "N": n_files, "seed": seed, "window": WINDOW,
            "step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "sampler_state": sampler.get_state(), "permutation": permutation,
            "cursor": cursor, "best_val_ber": best_val, "best_step": best_step,
            "best_model_state": best_state,
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(), "last_metrics": last_metrics,
        }, last_path)
        append_metric(csv_path, last_metrics)
        print(
            f"{model_name} N={n_files}: step {step}/{steps}; val BER={val_ber:.6f}; "
            f"test BER={test_ber:.6f}; best val={best_val:.6f}@{best_step}; "
            f"elapsed={(time.perf_counter()-started)/60:.1f} min",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["complex", "tanh"], required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--csv-path", type=Path, default=PROJECT_ROOT / "reports/results/fixed_step_learning_curve.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "reports/results/learning_curve_config.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    train_pool = config["train_pool_files"]
    if max(args.sizes) > len(train_pool):
        raise ValueError("A requested N exceeds the fixed 210-file training pool")
    pilot_config = json.loads((PROJECT_ROOT / "reports/results/large_scale_pilot_config.json").read_text())
    data_dir = Path(pilot_config["dataset_root"])
    candidates = get_unique_files(data_dir, PROJECT_ROOT / "reports/results/data_audit_all.csv")
    by_name = {p.name: p for p in candidates.path}
    val_data = collect([by_name[name] for name in config["validation_files"]], WINDOW)
    test_data = collect([by_name[name] for name in config["test_files"]], WINDOW)
    raw_ber = ber_from_iq(
        np.column_stack((test_data[2].real, test_data[2].imag)), test_data[3]
    )
    csv_path = args.csv_path
    for n_files in args.sizes:
        train_data = collect([by_name[name] for name in train_pool[:n_files]], WINDOW)
        train_one(
            args.model, n_files, train_data, val_data, test_data, raw_ber,
            args.steps, args.checkpoint_every, args.threads, args.resume, csv_path,
            args.seed,
        )


if __name__ == "__main__":
    main()
