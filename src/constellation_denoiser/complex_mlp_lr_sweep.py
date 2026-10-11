"""Cosine-schedule comparison and phase-activation diagnostics for complex MLPs."""

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

from .experiment import ber_from_iq, set_seed
from .fixed_step_curve import collect
from .large_scale_pilot import get_unique_files
from .paper_complex_mlp import PaperComplexMLP, predict_complex
from .paths import E003_DIR, E004_DIR, E008_CHECKPOINTS, E008_RESULTS, MANIFEST_DIR, project_path


CONFIGS = [(31, 32), (31, 48), (65, 48)]
SEEDS = (17, 29)
N_FILES = 30
MAX_STEPS = 20_000
CHECKPOINT_EVERY = 1_000
BATCH_SIZE = 1024
LR = 1e-3
DIAGNOSTIC_STEPS = {0, 1_000, 5_000, 20_000}
METRIC_FIELDS = ["model", "schedule", "window", "hidden", "N", "seed", "step", "val_ber", "test_ber", "raw_ber", "lr"]
DIAG_FIELDS = ["model", "schedule", "window", "hidden", "N", "seed", "step", "lr", "c1", "c2", "std_abs_z1_sq", "std_abs_z2_sq"]


def append_row(path: Path, fields: list[str], row: dict, keys: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as f:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        reader = csv.DictReader(f)
        duplicate = any(all(r.get(k) == str(row[k]) for k in keys) for r in reader)
        f.seek(0, os.SEEK_END)
        empty = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=fields)
        if empty:
            writer.writeheader()
        if not duplicate:
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def get_batch(x, y, permutation, cursor, sampler):
    if cursor >= len(permutation):
        permutation = torch.randperm(len(x), generator=sampler)
        cursor = 0
    stop = min(cursor + BATCH_SIZE, len(permutation))
    indices = permutation[cursor:stop]
    return x.index_select(0, indices), y.index_select(0, indices), permutation, stop


@torch.inference_mode()
def evaluate(model, data) -> float:
    return ber_from_iq(predict_complex(model, data[0]), data[3])


@torch.inference_mode()
def diagnostics(model, x_probe, window, hidden, seed, step, lr) -> dict:
    z = torch.complex(x_probe[..., 0], x_probe[..., 1]).reshape(x_probe.shape[0], window)
    z1 = model.fc1(z)
    z2 = model.fc2(model.act1(z1))
    return {
        "model": "complex_mlp", "schedule": "cosine", "window": window,
        "hidden": hidden, "N": N_FILES, "seed": seed, "step": step, "lr": lr,
        "c1": float(model.act1.c.detach().item()),
        "c2": float(model.act2.c.detach().item()),
        "std_abs_z1_sq": float(z1.abs().square().std(unbiased=False).item()),
        "std_abs_z2_sq": float(z2.abs().square().std(unbiased=False).item()),
    }


def train_run(window, hidden, seed, train_data, val_data, test_data, raw_ber,
              steps, interval, threads, resume, metrics_path, diagnostics_path):
    torch.set_num_threads(threads)
    run_id = f"complex_cosine_w{window}_h{hidden}_N{N_FILES}_seed{seed}"
    ckpt_dir = E008_CHECKPOINTS
    last_path = ckpt_dir / f"{run_id}_last.pt"
    best_path = ckpt_dir / f"{run_id}_best.pt"

    set_seed(seed)
    model = PaperComplexMLP(window=window, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.0)
    x = torch.from_numpy(train_data[0])
    y_real = torch.from_numpy(train_data[1])
    y = torch.complex(y_real[:, 0], y_real[:, 1])
    x_probe = x[:min(BATCH_SIZE, len(x))]
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(x), generator=sampler)
    cursor, step = 0, 0
    best_val, best_step, best_state = float("inf"), 0, None
    last_metric, last_diag = None, None

    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        sampler.set_state(state["sampler_state"])
        permutation, cursor, step = state["permutation"], int(state["cursor"]), int(state["step"])
        best_val, best_step, best_state = float(state["best_val"]), int(state["best_step"]), state["best_model_state"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        last_metric, last_diag = state.get("last_metric"), state.get("last_diag")
        if last_metric:
            append_row(metrics_path, METRIC_FIELDS, last_metric, ("window", "hidden", "seed", "step"))
        if last_diag:
            append_row(diagnostics_path, DIAG_FIELDS, last_diag, ("window", "hidden", "seed", "step"))
        print(f"{run_id}: resumed at step {step}", flush=True)
    else:
        print(f"{run_id}: starting at step 0", flush=True)

    if step == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        last_diag = diagnostics(model, x_probe, window, hidden, seed, 0, lr_now)
        append_row(diagnostics_path, DIAG_FIELDS, last_diag, ("window", "hidden", "seed", "step"))
        atomic_save({
            "run_id": run_id, "window": window, "hidden": hidden, "N": N_FILES,
            "seed": seed, "step": 0, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "sampler_state": sampler.get_state(), "permutation": permutation, "cursor": cursor,
            "best_val": best_val, "best_step": best_step, "best_model_state": best_state,
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(), "last_metric": None, "last_diag": last_diag,
        }, last_path)

    started = time.perf_counter()
    while step < steps:
        model.train()
        xb, yb, permutation, cursor = get_batch(x, y, permutation, cursor, sampler)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        loss = (prediction - yb).abs().square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        if step % interval and step != steps:
            continue

        val_ber = evaluate(model, val_data)
        test_ber = evaluate(model, test_data)
        if val_ber < best_val:
            best_val, best_step = val_ber, step
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            atomic_save({
                "run_id": run_id, "window": window, "hidden": hidden, "N": N_FILES,
                "seed": seed, "step": step, "best_val_ber": best_val,
                "model_state": best_state,
            }, best_path)
        lr_now = optimizer.param_groups[0]["lr"]
        last_metric = {
            "model": "complex_mlp", "schedule": "cosine", "window": window,
            "hidden": hidden, "N": N_FILES, "seed": seed, "step": step,
            "val_ber": val_ber, "test_ber": test_ber, "raw_ber": raw_ber, "lr": lr_now,
        }
        append_row(metrics_path, METRIC_FIELDS, last_metric, ("window", "hidden", "seed", "step"))
        if step in DIAGNOSTIC_STEPS:
            last_diag = diagnostics(model, x_probe, window, hidden, seed, step, lr_now)
            append_row(diagnostics_path, DIAG_FIELDS, last_diag, ("window", "hidden", "seed", "step"))
        atomic_save({
            "run_id": run_id, "window": window, "hidden": hidden, "N": N_FILES,
            "seed": seed, "step": step, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "sampler_state": sampler.get_state(), "permutation": permutation, "cursor": cursor,
            "best_val": best_val, "best_step": best_step, "best_model_state": best_state,
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(), "last_metric": last_metric,
            "last_diag": last_diag,
        }, last_path)
        print(
            f"{run_id}: step {step}/{steps}; lr={lr_now:.8g}; val BER={val_ber:.6f}; "
            f"test BER={test_ber:.6f}; best val={best_val:.6f}@{best_step}; "
            f"elapsed={(time.perf_counter()-started)/60:.1f} min",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--config-index", type=int, choices=range(len(CONFIGS)), required=True)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=E004_DIR / "config.json")
    parser.add_argument("--metrics-csv", type=Path, default=E008_RESULTS / "complex_mlp_lr_sweep.csv")
    parser.add_argument("--diagnostics-csv", type=Path, default=E008_RESULTS / "complex_mlp_lr_diagnostics.csv")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pilot = json.loads((E003_DIR / "config.json").read_text())
    candidates = get_unique_files(project_path(pilot["dataset_root"]), MANIFEST_DIR / "data_audit_all.csv")
    by_name = {path.name: path for path in candidates.path}
    window, hidden = CONFIGS[args.config_index]
    train_data = collect([by_name[name] for name in config["train_pool_files"][:N_FILES]], window)
    val_data = collect([by_name[name] for name in config["validation_files"]], window)
    test_data = collect([by_name[name] for name in config["test_files"]], window)
    raw_ber = ber_from_iq(np.column_stack((test_data[2].real, test_data[2].imag)), test_data[3])
    print(
        f"N={N_FILES}, window={window}, hidden={hidden}, seed={args.seed}, "
        f"Adam cosine lr={LR}->0, steps={args.steps}, batch={BATCH_SIZE}; raw BER={raw_ber:.6f}",
        flush=True,
    )
    train_run(window, hidden, args.seed, train_data, val_data, test_data, raw_ber,
              args.steps, args.checkpoint_every, args.threads, args.resume,
              args.metrics_csv, args.diagnostics_csv)


if __name__ == "__main__":
    main()
