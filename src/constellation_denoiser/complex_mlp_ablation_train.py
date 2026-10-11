"""Resumable Kerr ablation training for (window=31, hidden=48), N=30."""

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

from .complex_mlp_ablation import (
    ABLATION_VARIANTS,
    activation_diagnostics,
    build_model,
    complexity,
    verify_merge,
)
from .experiment import PROJECT_ROOT, ber_from_iq, set_seed
from .fixed_step_curve import collect
from .large_scale_pilot import get_unique_files
from .paper_complex_mlp import predict_complex

WINDOW = 31
HIDDEN = 48
N_FILES = 30
SEEDS = (17, 29)
STEPS = 20_000
INTERVAL = 1_000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
DIAGNOSTIC_STEPS = {0, 5_000, 20_000}

METRIC_FIELDS = [
    "variant", "window", "hidden", "N", "seed", "step", "val_ber", "test_ber", "raw_ber",
    "parameters", "train_dense_real_multiplies", "train_phase_real_multiplies",
    "train_total_real_multiplies", "merged_dense_real_multiplies", "merged_phase_real_multiplies",
    "merged_total_real_multiplies", "merge_max_abs_diff", "merge_windows_checked",
]
DIAG_FIELDS = [
    "variant", "window", "hidden", "N", "seed", "step", "c1", "c2",
    "mean_abs_z_act1_sq", "mean_abs_z_act2_sq",
]


def append_row(path: Path, fields: list[str], row: dict, keys: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        duplicate = any(all(record.get(key) == str(row[key]) for key in keys) for record in reader)
        handle.seek(0, os.SEEK_END)
        empty = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=fields)
        if empty:
            writer.writeheader()
        if not duplicate:
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def batch_from_permutation(x, y, permutation, cursor, sampler):
    if cursor >= len(permutation):
        permutation = torch.randperm(len(x), generator=sampler)
        cursor = 0
    stop = min(cursor + BATCH_SIZE, len(permutation))
    indices = permutation[cursor:stop]
    return x.index_select(0, indices), y.index_select(0, indices), permutation, stop


@torch.inference_mode()
def evaluate(model, data) -> float:
    return ber_from_iq(predict_complex(model, data[0]), data[3])


def train_run(
    variant: str,
    seed: int,
    train_data,
    val_data,
    test_data,
    raw_ber: float,
    *,
    steps: int,
    interval: int,
    threads: int,
    resume: bool,
    metrics_path: Path,
    diagnostics_path: Path,
) -> None:
    torch.set_num_threads(threads)
    run_id = f"ablation_{variant}_w{WINDOW}_h{HIDDEN}_N{N_FILES}_seed{seed}"
    checkpoint_dir = PROJECT_ROOT / "models/complex_mlp_ablation"
    last_path = checkpoint_dir / f"{run_id}_last.pt"
    best_path = checkpoint_dir / f"{run_id}_best.pt"

    model = build_model(WINDOW, HIDDEN, variant)
    cx = complexity(WINDOW, HIDDEN, variant)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x = torch.from_numpy(train_data[0])
    y = torch.complex(
        torch.from_numpy(train_data[1][:, 0]),
        torch.from_numpy(train_data[1][:, 1]),
    )
    x_probe = x[: min(BATCH_SIZE, len(x))]
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(x), generator=sampler)
    step, cursor = 0, 0
    best_val, best_step, best_state = float("inf"), 0, None
    last_metric, last_diag = None, None

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
        last_metric, last_diag = state.get("last_metric"), state.get("last_diag")
        if last_metric:
            append_row(metrics_path, METRIC_FIELDS, last_metric, ("variant", "seed", "step"))
        if last_diag:
            append_row(diagnostics_path, DIAG_FIELDS, last_diag, ("variant", "seed", "step"))
        print(f"{run_id}: resumed at step {step}", flush=True)
    else:
        set_seed(seed)
        model = build_model(WINDOW, HIDDEN, variant)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print(f"{run_id}: starting at step 0", flush=True)

    if step == 0:
        diag = activation_diagnostics(model, x_probe, WINDOW)
        last_diag = {
            "variant": variant, "window": WINDOW, "hidden": HIDDEN, "N": N_FILES,
            "seed": seed, "step": 0, **diag,
        }
        append_row(diagnostics_path, DIAG_FIELDS, last_diag, ("variant", "seed", "step"))

    started = time.perf_counter()
    while step < steps:
        model.train()
        x_batch, y_batch, permutation, cursor = batch_from_permutation(x, y, permutation, cursor, sampler)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_batch)
        loss = (prediction - y_batch).abs().square().mean()
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
            atomic_save(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "window": WINDOW,
                    "hidden": HIDDEN,
                    "N": N_FILES,
                    "seed": seed,
                    "step": step,
                    "best_val_ber": best_val,
                    "model_state": best_state,
                    "complexity": cx,
                },
                best_path,
            )

        merge_diff, merge_n = float("nan"), 0
        if variant in ("a", "c"):
            merge_diff, merge_n = verify_merge(model, train_data[0])

        last_metric = {
            "variant": variant,
            "window": WINDOW,
            "hidden": HIDDEN,
            "N": N_FILES,
            "seed": seed,
            "step": step,
            "val_ber": val_ber,
            "test_ber": test_ber,
            "raw_ber": raw_ber,
            "parameters": parameters,
            "train_dense_real_multiplies": cx.train_dense_real_multiplies,
            "train_phase_real_multiplies": cx.train_phase_real_multiplies,
            "train_total_real_multiplies": cx.train_total_real_multiplies,
            "merged_dense_real_multiplies": cx.merged_dense_real_multiplies,
            "merged_phase_real_multiplies": cx.merged_phase_real_multiplies,
            "merged_total_real_multiplies": cx.merged_total_real_multiplies,
            "merge_max_abs_diff": merge_diff,
            "merge_windows_checked": merge_n,
        }
        append_row(metrics_path, METRIC_FIELDS, last_metric, ("variant", "seed", "step"))

        if step in DIAGNOSTIC_STEPS:
            diag = activation_diagnostics(model, x_probe, WINDOW)
            last_diag = {
                "variant": variant, "window": WINDOW, "hidden": HIDDEN, "N": N_FILES,
                "seed": seed, "step": step, **diag,
            }
            append_row(diagnostics_path, DIAG_FIELDS, last_diag, ("variant", "seed", "step"))

        atomic_save(
            {
                "run_id": run_id,
                "variant": variant,
                "window": WINDOW,
                "hidden": HIDDEN,
                "N": N_FILES,
                "seed": seed,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sampler_state": sampler.get_state(),
                "permutation": permutation,
                "cursor": cursor,
                "best_val": best_val,
                "best_step": best_step,
                "best_model_state": best_state,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "last_metric": last_metric,
                "last_diag": last_diag,
            },
            last_path,
        )
        print(
            f"{run_id}: step {step}/{steps}; val BER={val_ber:.6f}; test BER={test_ber:.6f}; "
            f"best val={best_val:.6f}@{best_step}; merge_diff={merge_diff:.2e}; "
            f"elapsed={(time.perf_counter() - started) / 60:.1f} min",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ABLATION_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=INTERVAL)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "reports/results/learning_curve_config.json")
    parser.add_argument("--metrics-csv", type=Path, default=PROJECT_ROOT / "reports/results/complex_mlp_ablation.csv")
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        default=PROJECT_ROOT / "reports/results/complex_mlp_ablation_diagnostics.csv",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    pilot = json.loads((PROJECT_ROOT / "reports/results/large_scale_pilot_config.json").read_text(encoding="utf-8"))
    candidates = get_unique_files(Path(pilot["dataset_root"]), PROJECT_ROOT / "reports/results/data_audit_all.csv")
    by_name = {path.name: path for path in candidates.path}

    train_data = collect([by_name[name] for name in config["train_pool_files"][:N_FILES]], WINDOW)
    val_data = collect([by_name[name] for name in config["validation_files"]], WINDOW)
    test_data = collect([by_name[name] for name in config["test_files"]], WINDOW)
    raw_ber = ber_from_iq(np.column_stack((test_data[2].real, test_data[2].imag)), test_data[3])

    cx = complexity(WINDOW, HIDDEN, args.variant)
    print(
        f"variant={args.variant}, window={WINDOW}, hidden={HIDDEN}, N={N_FILES}, seed={args.seed}, "
        f"Adam lr={LEARNING_RATE}, steps={args.steps}, batch={BATCH_SIZE}; "
        f"train mult={cx.train_total_real_multiplies}; merged mult={cx.merged_total_real_multiplies}; "
        f"raw BER={raw_ber:.6f}",
        flush=True,
    )
    train_run(
        args.variant,
        args.seed,
        train_data,
        val_data,
        test_data,
        raw_ber,
        steps=args.steps,
        interval=args.checkpoint_every,
        threads=args.threads,
        resume=args.resume,
        metrics_path=args.metrics_csv,
        diagnostics_path=args.diagnostics_csv,
    )


if __name__ == "__main__":
    main()
