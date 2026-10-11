"""Resumable fixed-step sweep of the paper complex MLP over context and width."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .experiment import PROJECT_ROOT, ber_from_iq, make_windows, set_seed
from .fixed_step_curve import collect
from .large_scale_pilot import get_unique_files
from .paper_complex_mlp import PaperComplexMLP, predict_complex


CONFIGS = [(65, 48), (31, 32), (127, 48), (65, 32), (31, 48), (127, 32)]
SEEDS = (17, 29)
TRAIN_FILES = 30
STEPS = 20_000
INTERVAL = 1_000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
FIELDS = [
    "model", "window", "hidden", "N", "seed", "step", "val_ber", "test_ber",
    "raw_ber", "parameters", "dense_real_multiplies", "phase_real_multiplies",
    "total_real_multiplies",
]


def complexity(window: int, hidden: int) -> dict[str, int]:
    # Three complex dense maps: window->hidden, hidden->hidden, hidden->1.
    dense_complex_products = window * hidden + hidden * hidden + hidden
    dense_real_multiplies = 4 * dense_complex_products
    # Each of 2*hidden phase units: |z|^2 (2), c*|z|^2 (1), exp(i*theta)*z (4).
    phase_real_multiplies = 7 * (2 * hidden)
    # Includes the two trainable scalar c parameters in NonlinearPhase.
    parameters = 2 * dense_complex_products + 2 * (2 * hidden + 1) + 2
    return {
        "parameters": parameters,
        "dense_real_multiplies": dense_real_multiplies,
        "phase_real_multiplies": phase_real_multiplies,
        "total_real_multiplies": dense_real_multiplies + phase_real_multiplies,
    }


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as f:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        reader = csv.DictReader(f)
        duplicate = any(
            r.get("window") == str(row["window"])
            and r.get("hidden") == str(row["hidden"])
            and r.get("seed") == str(row["seed"])
            and r.get("step") == str(row["step"])
            for r in reader
        )
        f.seek(0, os.SEEK_END)
        empty = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if empty:
            writer.writeheader()
        if not duplicate:
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def batch_from_permutation(x, y, permutation, cursor, sampler):
    if cursor >= len(permutation):
        permutation = torch.randperm(len(x), generator=sampler)
        cursor = 0
    stop = min(cursor + BATCH_SIZE, len(permutation))
    ix = permutation[cursor:stop]
    return x.index_select(0, ix), y.index_select(0, ix), permutation, stop


@torch.inference_mode()
def evaluate(model, data) -> float:
    prediction = predict_complex(model, data[0])
    return ber_from_iq(prediction, data[3])


def train_run(window, hidden, seed, train_data, val_data, test_data, raw_ber,
              steps, interval, threads, resume, csv_path):
    torch.set_num_threads(threads)
    run_id = f"complex_w{window}_h{hidden}_N{TRAIN_FILES}_seed{seed}"
    checkpoint_dir = PROJECT_ROOT / "models/complex_mlp_sweep"
    last_path = checkpoint_dir / f"{run_id}_last.pt"
    best_path = checkpoint_dir / f"{run_id}_best.pt"
    model = PaperComplexMLP(window=window, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x = torch.from_numpy(train_data[0])
    y_np = train_data[1]
    y = torch.complex(torch.from_numpy(y_np[:, 0]), torch.from_numpy(y_np[:, 1]))
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(x), generator=sampler)
    step, cursor = 0, 0
    best_val, best_step, best_state = float("inf"), 0, None
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        sampler.set_state(state["sampler_state"])
        permutation, cursor, step = state["permutation"], int(state["cursor"]), int(state["step"])
        best_val, best_step, best_state = float(state["best_val"]), int(state["best_step"]), state["best_model_state"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if state.get("last_metrics") is not None:
            append_row(csv_path, state["last_metrics"])
        print(f"{run_id}: resumed at step {step}", flush=True)
    else:
        set_seed(seed)
        # Recreate the seeded model after setting all RNGs; use the initial model already made only
        # when no checkpoint exists, so reset its weights deterministically here.
        model = PaperComplexMLP(window=window, hidden=hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print(f"{run_id}: starting at step 0", flush=True)

    if step == 0:
        permutation = torch.randperm(len(x), generator=sampler)
    mse = nn.MSELoss()
    started = time.perf_counter()
    while step < steps:
        model.train()
        xb, yb, permutation, cursor = batch_from_permutation(x, y, permutation, cursor, sampler)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        loss = (prediction - yb).abs().square().mean()
        loss.backward()
        optimizer.step()
        step += 1
        if step % interval and step != steps:
            continue
        val_ber = evaluate(model, val_data)
        test_ber = evaluate(model, test_data)
        if val_ber < best_val:
            best_val, best_step = val_ber, step
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            atomic_save({
                "run_id": run_id, "window": window, "hidden": hidden, "N": TRAIN_FILES,
                "seed": seed, "step": step, "best_val_ber": best_val,
                "model_state": best_state, "complexity": complexity(window, hidden),
            }, best_path)
        row = {
            "model": "complex_mlp", "window": window, "hidden": hidden,
            "N": TRAIN_FILES, "seed": seed, "step": step,
            "val_ber": val_ber, "test_ber": test_ber, "raw_ber": raw_ber,
            **complexity(window, hidden),
        }
        atomic_save({
            "run_id": run_id, "window": window, "hidden": hidden, "N": TRAIN_FILES,
            "seed": seed, "step": step, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "sampler_state": sampler.get_state(),
            "permutation": permutation, "cursor": cursor, "best_val": best_val,
            "best_step": best_step, "best_model_state": best_state,
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(), "last_metrics": row,
            "complexity": complexity(window, hidden),
        }, last_path)
        append_row(csv_path, row)
        print(
            f"{run_id}: step {step}/{steps}; val BER={val_ber:.6f}; test BER={test_ber:.6f}; "
            f"best val={best_val:.6f}@{best_step}; elapsed={(time.perf_counter()-started)/60:.1f} min",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--config-index", type=int, choices=range(len(CONFIGS)),
                        help="Run only the zero-based configuration from the prescribed order")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=INTERVAL)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--csv-path", type=Path, default=PROJECT_ROOT / "reports/results/complex_mlp_sweep.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "reports/results/learning_curve_config.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pilot = json.loads((PROJECT_ROOT / "reports/results/large_scale_pilot_config.json").read_text())
    candidates = get_unique_files(Path(pilot["dataset_root"]), PROJECT_ROOT / "reports/results/data_audit_all.csv")
    by_name = {path.name: path for path in candidates.path}
    configs = [CONFIGS[args.config_index]] if args.config_index is not None else CONFIGS
    for window, hidden in configs:
        val_data = collect([by_name[name] for name in config["validation_files"]], window)
        test_data = collect([by_name[name] for name in config["test_files"]], window)
        raw_ber = ber_from_iq(np.column_stack((test_data[2].real, test_data[2].imag)), test_data[3])
        train_data = collect([by_name[name] for name in config["train_pool_files"][:TRAIN_FILES]], window)
        model = PaperComplexMLP(window=window, hidden=hidden)
        measured_parameters = sum(p.numel() for p in model.parameters())
        expected = complexity(window, hidden)["parameters"]
        if measured_parameters != expected:
            raise RuntimeError(f"Parameter formula disagrees with implementation: {measured_parameters} != {expected}")
        print(
            f"Configuration (window={window}, hidden={hidden}): parameters={measured_parameters}; "
            f"dense real multiplies={complexity(window, hidden)['dense_real_multiplies']}; "
            f"phase real multiplies={complexity(window, hidden)['phase_real_multiplies']}; "
            f"total={complexity(window, hidden)['total_real_multiplies']}; raw BER={raw_ber:.6f}",
            flush=True,
        )
        train_run(
            window, hidden, args.seed, train_data, val_data, test_data, raw_ber,
            args.steps, args.checkpoint_every, args.threads, args.resume, args.csv_path,
        )
        del train_data, val_data, test_data


if __name__ == "__main__":
    main()
