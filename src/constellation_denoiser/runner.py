"""Shared resumable task runner for E010-E014.

Usage (from the repository root)::

    PYTHONPATH=src python -m constellation_denoiser.runner --exp E010 --stage stage1 [--plan|--list]

Every task writes ``<exp>/results/runs/<run_id>.json`` atomically and its
curve to ``<exp>/results/curves/<run_id>.csv``. An existing JSON is skipped
unless ``FORCE=1``. Runs longer than ``LONG_RUN_STEPS`` checkpoint every
``ckpt_every`` updates to ``checkpoints/<exp>/`` and resume from there.

Environment: JOBS, THREADS (``auto`` = from resources.json), DEVICE
(auto|cpu|cuda), SEEDS (e.g. ``17,29``), SMOKE=1, FORCE=1, SYNTHETIC=1
(smoke only), POLITE_MAX_WAIT_S, POLITE=0 to disable load checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import model_zoo
from .lr_schedules import LRConfig, apply_lr, build_optimizer
from .paths import CHECKPOINTS_DIR, EXPERIMENTS_DIR, PROJECT_ROOT
from .window_data import FileSet, load_fileset, model_ber, resolve_paths, split_files, synthetic_fileset

BATCH_SIZE = 1024
LONG_RUN_STEPS = 20_000
RESOURCES_PATH = PROJECT_ROOT / "resources.json"
SMOKE_MAX_STEPS = 300
SMOKE_MAX_FILES = 2
CURVE_FIELDS = ["step", "phase", "lr", "train_loss", "val_ber", "val_reduction", "test_ber",
                "test_reduction", "c1", "c2"]


# --------------------------------------------------------------------------- utilities

def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


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


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "experiments", "scripts"],
                               cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def reduction(raw: float, value: float) -> float:
    return 100.0 * (raw - value) / raw if raw > 0 else float("nan")


def env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


def exp_dir(exp: str) -> Path:
    matches = sorted(EXPERIMENTS_DIR.glob(f"{exp}_*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one experiments/{exp}_* folder, found {matches}")
    return matches[0]


def results_root(exp: str, smoke: bool) -> Path:
    return exp_dir(exp) / ("results_smoke" if smoke else "results")


def load_resources() -> dict | None:
    if RESOURCES_PATH.is_file():
        return json.loads(RESOURCES_PATH.read_text(encoding="utf-8"))
    return None


def resolve_parallelism(resources: dict | None) -> tuple[int, int, str]:
    jobs_env, threads_env = os.environ.get("JOBS", "auto"), os.environ.get("THREADS", "auto")
    device_env = os.environ.get("DEVICE", "auto")
    rec = (resources or {}).get("recommendation", {})
    jobs = int(rec.get("jobs", 1)) if jobs_env == "auto" else int(jobs_env)
    threads = int(rec.get("threads", 1)) if threads_env == "auto" else int(threads_env)
    if device_env == "auto":
        device_env = rec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_env == "cuda" and not torch.cuda.is_available():
        log("DEVICE=cuda requested but CUDA is not available; using cpu")
        device_env = "cpu"
    return max(1, jobs), max(1, threads), device_env


# --------------------------------------------------------------------------- data cache (per process)

_DATA_CACHE: dict[tuple, FileSet] = {}


def get_fileset(names: tuple[str, ...], synthetic_seed: int | None, device: torch.device) -> FileSet:
    key = (names, synthetic_seed, str(device))
    if key not in _DATA_CACHE:
        if synthetic_seed is not None:
            data = synthetic_fileset(len(names), synthetic_seed)
        else:
            data = load_fileset(resolve_paths(list(names)))
        _DATA_CACHE[key] = data.to(device)
    return _DATA_CACHE[key]


def task_data(task: dict, device: torch.device) -> tuple[FileSet, FileSet, FileSet]:
    files = task["files"]
    synthetic = task.get("synthetic", False)
    return (
        get_fileset(tuple(files["train"]), 1 if synthetic else None, device),
        get_fileset(tuple(files["validation"]), 2 if synthetic else None, device),
        get_fileset(tuple(files["test"]), 3 if synthetic else None, device),
    )


def default_files(n_train: int, smoke: bool) -> dict[str, list[str]]:
    split = split_files()
    if smoke:
        n_train = min(n_train, SMOKE_MAX_FILES)
        return {"train": split["train_pool"][:n_train], "validation": split["validation"][:1],
                "test": split["test"][:1]}
    if n_train > len(split["train_pool"]):
        raise ValueError(f"N={n_train} exceeds the {len(split['train_pool'])}-file train pool")
    return {"train": split["train_pool"][:n_train], "validation": split["validation"], "test": split["test"]}


def data_available(files: dict[str, list[str]]) -> bool:
    try:
        resolve_paths(files["train"] + files["validation"] + files["test"])
        return True
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- training

def kerr_coefficients(model: torch.nn.Module) -> tuple[float, float]:
    values = []
    for name in ("act1", "act2"):
        module = getattr(model, name, None)
        if module is None:
            for sub in model.modules():
                if hasattr(sub, "act1") and sub is not model:
                    module = getattr(sub, name, None)
                    break
        try:
            values.append(float(module.c.detach().item()))
        except (AttributeError, RuntimeError, ValueError):
            values.append(float("nan"))
    return values[0], values[1]


def set_trainable(model: torch.nn.Module, prefixes: list[str] | str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(prefixes == "all" or any(name.startswith(p) for p in prefixes))


def steps_to_fraction(curve: list[dict], target: float, consecutive: int = 3) -> int | None:
    """First step after which val_reduction stays >= target for ``consecutive`` evaluations."""
    points = [(row["step"], row["val_reduction"]) for row in curve if row.get("val_reduction") is not None]
    run = 0
    for index, (_, value) in enumerate(points):
        run = run + 1 if (value is not None and not math.isnan(value) and value >= target) else 0
        if run >= consecutive:
            return points[index - consecutive + 1][0]
    return None


def write_curve(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in CURVE_FIELDS})
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_task(task: dict, threads: int, device_name: str) -> dict:
    """Train one task (all phases), evaluate, write JSON + curve; return the JSON payload."""
    torch.set_num_threads(threads)
    started_wall = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        identity = mp.current_process()._identity
        device = torch.device(f"cuda:{(identity[0] - 1) % torch.cuda.device_count() if identity else 0}")

    spec, seed = task["spec"], int(task["seed"])
    window = int(spec["window"])
    out_json = Path(task["out_json"])
    out_curve = Path(task["out_curve"])
    ckpt_dir = Path(task["ckpt_dir"])
    last_path = ckpt_dir / f"{task['run_id']}_last.pt"
    phases = task["phases"]
    total_steps = sum(int(p["steps"]) for p in phases)
    eval_every, test_every = int(task["eval_every"]), int(task.get("test_every") or task["eval_every"])
    select_every = int(task.get("select_every", 1000))
    ckpt_every = int(task.get("ckpt_every", 10_000))
    use_ckpt = total_steps > LONG_RUN_STEPS or task.get("force_checkpoints", False)

    train, val, test = task_data(task, device)
    raw_val, raw_test = val.raw_ber(window), test.raw_ber(window)
    n_train_windows = train.n_windows(window)

    model = model_zoo.build(spec, seed=seed).to(device)
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(n_train_windows, generator=sampler)
    cursor, step, phase_index, phase_step = 0, 0, 0, 0
    curve: list[dict] = []
    best = {"all": (float("inf"), 0, None), "select": (float("inf"), 0, None)}
    diverged, loss_sum, loss_count = False, 0.0, 0

    def make_optimizer(index: int):
        phase = phases[index]
        set_trainable(model, phase.get("trainable", "all"))
        lr_cfg = LRConfig.from_dict(phase.get("lr") or task["lr"])
        optimizer, base = build_optimizer(model, lr_cfg)
        return optimizer, base, lr_cfg

    optimizer, base_lrs, lr_cfg = make_optimizer(0)
    if use_ckpt and last_path.is_file() and not task.get("force", False):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        step, phase_index, phase_step = int(state["step"]), int(state["phase_index"]), int(state["phase_step"])
        optimizer, base_lrs, lr_cfg = make_optimizer(phase_index)
        optimizer.load_state_dict(state["optimizer_state"])
        sampler.set_state(state["sampler_state"])
        permutation, cursor = state["permutation"], int(state["cursor"])
        curve, best = state["curve"], state["best"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        log(f"{task['run_id']}: resumed at step {step}")

    def evaluate_point(with_test: bool) -> dict:
        val_ber = model_ber(model, val, window)
        row = {"step": step, "phase": phase_index, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum / loss_count if loss_count else None,
               "val_ber": val_ber, "val_reduction": reduction(raw_val, val_ber)}
        if with_test:
            test_ber = model_ber(model, test, window)
            row.update(test_ber=test_ber, test_reduction=reduction(raw_test, test_ber))
        row["c1"], row["c2"] = kerr_coefficients(model)
        return row

    def snapshot() -> dict:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def save_last() -> None:
        atomic_torch_save({
            "run_id": task["run_id"], "step": step, "phase_index": phase_index, "phase_step": phase_step,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "sampler_state": sampler.get_state(), "permutation": permutation, "cursor": cursor,
            "curve": curve, "best": best, "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
        }, last_path)

    if step == 0 and not curve:
        curve.append(evaluate_point(with_test=True))

    check_every = min(eval_every, 250)
    while step < total_steps and not diverged:
        if phase_step >= int(phases[phase_index]["steps"]):
            phase_index += 1
            phase_step = 0
            if phases[phase_index].get("reset_optimizer", True):
                optimizer, base_lrs, lr_cfg = make_optimizer(phase_index)
            else:
                set_trainable(model, phases[phase_index].get("trainable", "all"))
        apply_lr(optimizer, base_lrs, phase_step, int(phases[phase_index]["steps"]), lr_cfg)
        model.train()
        if cursor >= len(permutation):
            permutation = torch.randperm(n_train_windows, generator=sampler)
            cursor = 0
        stop = min(cursor + BATCH_SIZE, len(permutation))
        xb, yb = train.gather(permutation[cursor:stop], window)
        cursor = stop
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        if torch.is_complex(prediction):
            loss = (prediction - torch.complex(yb[:, 0], yb[:, 1])).abs().square().mean()
        else:
            loss = torch.nn.functional.mse_loss(prediction, yb)
        loss.backward()
        optimizer.step()
        step += 1
        phase_step += 1
        if device.type == "cpu" or step % 50 == 0:  # avoid a GPU sync on every step
            loss_sum += float(loss.detach())
            loss_count += 1

        if step % check_every == 0 or step == total_steps:
            if not math.isfinite(loss_sum) or not all(torch.isfinite(p).all() for p in model.parameters()):
                diverged = True
        if diverged or step % eval_every == 0 or step == total_steps:
            row = evaluate_point(with_test=(step % test_every == 0 or step == total_steps or diverged))
            curve.append(row)
            loss_sum, loss_count = 0.0, 0
            if row["val_ber"] < best["all"][0]:
                best["all"] = (row["val_ber"], step, row.get("test_ber"))
                if task.get("save_best", True):
                    atomic_torch_save({"run_id": task["run_id"], "step": step, "spec": spec,
                                       "model_state": snapshot()}, ckpt_dir / f"{task['run_id']}_best.pt")
            if step % select_every == 0 and row.get("test_ber") is not None and row["val_ber"] < best["select"][0]:
                best["select"] = (row["val_ber"], step, row["test_ber"])
        if use_ckpt and step % ckpt_every == 0 and not diverged:
            save_last()

    final = curve[-1]
    if task.get("save_final", True):
        atomic_torch_save({"run_id": task["run_id"], "step": step, "spec": spec, "model_state": snapshot()},
                          ckpt_dir / f"{task['run_id']}_final.pt")

    extra = {}
    hook = task.get("post_hook")
    if hook and not diverged:
        from . import grids

        extra = grids.POST_HOOKS[hook](model, task, train, val, test)

    cost = model_zoo.cost(spec)
    target = task.get("target_val_reduction")
    payload = {
        "experiment": task["experiment"], "run_id": task["run_id"], "stage": task["stage"],
        "model": spec["model"], "variant": task.get("variant", spec["model"]), "window": window,
        "hidden": spec.get("hidden"), "spec": spec, "n_train_files": len(task["files"]["train"]),
        "seed": seed, "steps": step, "planned_steps": total_steps, "budget": task.get("budget", total_steps), "phases": phases,
        "lr_config": task["lr"], "lr_name": task.get("lr_name"),
        "val_reduction": final.get("val_reduction"), "test_reduction": final.get("test_reduction"),
        "val_ber": final.get("val_ber"), "test_ber": final.get("test_ber"),
        "raw_ber": raw_test, "raw_val_ber": raw_val,
        "best_val_step": best["all"][1], "best_val_reduction": reduction(raw_val, best["all"][0]),
        "test_reduction_at_best_val": None if best["all"][2] is None else reduction(raw_test, best["all"][2]),
        "select_every": select_every, "best_val_step_select": best["select"][1],
        "test_reduction_valbest_select": None if best["select"][2] is None else reduction(raw_test, best["select"][2]),
        "macs_per_symbol": cost.macs, "macs_per_symbol_merged": cost.macs_merged,
        "n_params": model_zoo.n_params(model),
        "steps_to_target": None if target is None else {
            name: steps_to_fraction(curve, fraction * target) for name, fraction in (("T90", 0.9), ("T95", 0.95))
        },
        "target_val_reduction": target, "eval_every": eval_every, "test_every": test_every,
        "wall_time_s": time.perf_counter() - started_wall, "device": str(device), "threads": threads,
        "git_commit": task.get("git_commit", "unknown"), "torch_version": torch.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "diverged": diverged, "smoke": task.get("smoke", False), "synthetic": task.get("synthetic", False),
        "control_key": task.get("control_key"), "n_val_files": len(task["files"]["validation"]),
        "n_test_files": len(task["files"]["test"]), **extra,
    }
    write_curve(out_curve, curve)
    atomic_write_json(out_json, payload)
    if last_path.is_file():
        last_path.unlink()
    return payload


def _worker_init(threads: int) -> None:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_task_safe(task: dict, threads: int, device: str) -> tuple[str, str | None, dict | None]:
    try:
        payload = run_task(task, threads, device)
        return task["run_id"], None, {k: payload.get(k) for k in ("val_reduction", "test_reduction", "diverged",
                                                                   "wall_time_s", "steps")}
    except Exception:  # keep the queue alive; the error is written next to the results
        return task["run_id"], traceback.format_exc(), None


# --------------------------------------------------------------------------- polite scheduling

def loadavg() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def polite_wait(running: int, jobs: int, threads: int, resources: dict | None) -> None:
    """Block while foreign load leaves no room for one more of our processes."""
    if not env_flag("POLITE") and os.environ.get("POLITE") is not None:
        return
    rec = (resources or {}).get("recommendation", {})
    threshold = rec.get("foreign_load_threshold")
    if threshold is None:
        return
    max_wait = float(os.environ.get("POLITE_MAX_WAIT_S", rec.get("polite_max_wait_s", 7200)))
    waited = 0.0
    while True:
        foreign = loadavg() - running * threads
        if foreign <= threshold:
            return
        if waited >= max_wait:
            log(f"polite mode: foreign load {foreign:.1f} > {threshold:.1f} for {waited:.0f}s; starting anyway")
            return
        if waited % 600 == 0:
            log(f"polite mode: foreign load {foreign:.1f} > {threshold:.1f}; waiting (checked every 30 s)")
        time.sleep(30)
        waited += 30


def execute(tasks: list[dict], jobs: int, threads: int, device: str, resources: dict | None) -> int:
    if not tasks:
        log("nothing to run: all results exist (use FORCE=1 to recompute)")
        return 0
    log(f"running {len(tasks)} tasks with JOBS={jobs}, THREADS={threads}, DEVICE={device}")
    failures = 0
    pending = list(tasks)
    started = time.perf_counter()
    done_count = 0
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=context, initializer=_worker_init,
                                initargs=(threads,)) as pool:
        running: dict[cf.Future, dict] = {}
        try:
            while pending or running:
                while pending and len(running) < jobs:
                    polite_wait(len(running), jobs, threads, resources)
                    task = pending.pop(0)
                    running[pool.submit(_run_task_safe, task, threads, device)] = task
                done, _ = cf.wait(running, timeout=30, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    task = running.pop(future)
                    run_id, error, summary = future.result()
                    done_count += 1
                    error_path = Path(task["out_json"]).with_suffix(".error.txt")
                    if error:
                        failures += 1
                        atomic_write_text(error_path, error)
                        log(f"[fail] {run_id}: see {error_path}")
                    else:
                        if error_path.exists():
                            error_path.unlink()
                        elapsed = time.perf_counter() - started
                        eta = elapsed / done_count * (len(tasks) - done_count)
                        log(f"[ok {done_count}/{len(tasks)}] {run_id}: val={summary['val_reduction']:.2f}% "
                            f"test={summary['test_reduction']:.2f}% diverged={summary['diverged']} "
                            f"time={summary['wall_time_s']:.0f}s; ETA ~{eta / 60:.0f} min")
        except KeyboardInterrupt:
            log("interrupted: cancelling pending tasks; long runs resume from their last checkpoint")
            for future in running:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    log(f"finished: {len(tasks) - failures} ok, {failures} failed")
    return 1 if failures else 0


# --------------------------------------------------------------------------- planning

def estimate_task_seconds(task: dict, resources: dict) -> tuple[float, float]:
    """Rough (train, evaluation) wall seconds of one task from probe throughput."""
    from .resources import scaled_rates

    spec = task["spec"]
    rates = scaled_rates(spec, resources)
    steps = sum(int(p["steps"]) for p in task["phases"])
    windows_val = len(task["files"]["validation"]) * (16_384 - int(spec["window"]) + 1)
    windows_test = len(task["files"]["test"]) * (16_384 - int(spec["window"]) + 1)
    n_val = steps // int(task["eval_every"]) + 1
    n_test = steps // int(task.get("test_every") or task["eval_every"]) + 1
    return steps / rates["steps_per_s"], (n_val * windows_val + n_test * windows_test) / rates["eval_windows_per_s"]


def plan(tasks_all: list[dict], tasks_todo: list[dict], resources: dict | None, stage: str) -> None:
    print(f"Стадия {stage}: всего прогонов {len(tasks_all)}, уже готово {len(tasks_all) - len(tasks_todo)}, "
          f"осталось {len(tasks_todo)}.")
    if not tasks_todo:
        return
    total_steps = sum(sum(int(p["steps"]) for p in t["phases"]) for t in tasks_todo)
    print(f"Шагов обучения осталось: {total_steps:,}".replace(",", " "))
    if resources is None:
        print("Нет resources.json: сначала выполните стадию probe "
              "(bash experiments/<папка>/run.sh probe), затем повторите plan.")
        return
    jobs, threads, device = resolve_parallelism(resources)
    parts = [estimate_task_seconds(t, resources) for t in tasks_todo]
    seconds = [a + b for a, b in parts]
    serial = sum(seconds)
    eval_share = sum(b for _, b in parts) / serial if serial else 0.0
    longest = max(seconds)

    def wall(j: int) -> float:
        return max(serial / j, longest)

    half = max(1, jobs // 2)
    from .resources import memory_per_process_gb

    mem = max(memory_per_process_gb(t, resources) for t in tasks_todo)
    disk_mb = sum(_disk_mb(t) for t in tasks_todo)
    print(f"Рекомендовано: JOBS={jobs}, THREADS={threads}, DEVICE={device} (из resources.json).")
    print(f"Оценка времени: ~{wall(jobs) / 3600:.1f} ч при JOBS={jobs}; ~{wall(half) / 3600:.1f} ч при "
          f"JOBS={half} (если сервер занят). Самый длинный прогон ~{longest / 3600:.1f} ч.")
    print(f"Из них валидация и тест: ~{100 * eval_share:.0f}% времени.")
    print(f"Память: ~{mem:.2f} ГБ на процесс, ~{mem * jobs:.1f} ГБ всего; диск: ~{disk_mb:.0f} МБ "
          f"(results + checkpoints).")
    print("Оценка грубая: скорость измерена probe на эталонной модели и масштабирована по числу умножений.")


def _disk_mb(task: dict) -> float:
    try:
        n = model_zoo.n_params(model_zoo.build(task["spec"]))
    except Exception:
        n = 50_000
    steps = sum(int(p["steps"]) for p in task["phases"])
    curve_rows = steps // int(task["eval_every"]) + 1
    return (n * 4 * 6 + curve_rows * 200 + 4000) / 1e6


# --------------------------------------------------------------------------- control check

def check_control(exp: str, smoke: bool) -> int:
    """Compare control runs with the known numbers in config.json["control"]; write control_check.json."""
    directory = exp_dir(exp)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    control = config.get("control", {})
    runs_dir = results_root(exp, smoke) / "runs"
    records = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]
    report, passed = [], True
    for key, rule in control.get("targets", {}).items():
        metric = rule.get("metric", "test_reduction_valbest_select")
        seeds = set(rule.get("seeds", [17, 29]))
        values = [r[metric] for r in records if r.get("control_key") == key and r["seed"] in seeds
                  and r.get(metric) is not None and not r.get("diverged")]
        mean = float(np.mean(values)) if values else float("nan")
        ok = len(values) == len(seeds) and abs(mean - rule["target"]) <= rule["tol"]
        passed &= ok
        report.append({"key": key, "metric": metric, "target": rule["target"], "tol": rule["tol"],
                       "seeds": sorted(seeds), "values": values, "mean": mean, "ok": ok,
                       "source": rule.get("source")})
        status = "OK" if ok else "НЕ ПРОШЁЛ"
        log(f"control {key}: mean {metric} = {mean:.2f}% по {len(values)} seed; цель {rule['target']}±{rule['tol']} "
            f"п.п. -> {status}")
    out = results_root(exp, smoke) / "control_check.json"
    atomic_write_json(out, {"experiment": exp, "passed": bool(passed and report), "smoke": smoke,
                            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "targets": report})
    if smoke:
        log("SMOKE: контроль посчитан только для проверки пайплайна, стадии не блокируются")
        return 0
    if not (passed and report):
        log(f"КОНТРОЛЬ НЕ ПРОШЁЛ ДОПУСК: остальные стадии {exp} заблокированы. Пришлите {out} и логи разработчику.")
        return 3
    return 0


def require_control(exp: str, smoke: bool) -> bool:
    if smoke or env_flag("SKIP_CONTROL_CHECK"):
        return True
    path = results_root(exp, smoke) / "control_check.json"
    if not path.is_file():
        log(f"нет {path}: сначала выполните контрольную стадию этого эксперимента")
        return False
    if not json.loads(path.read_text()).get("passed"):
        log(f"{path}: контроль не прошёл допуск; стадия остановлена")
        return False
    return True


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    from . import grids

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, help="E010 ... E014")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--plan", action="store_true", help="print run count and time estimate; no training")
    parser.add_argument("--list", action="store_true", help="print run ids")
    parser.add_argument("--check-control", action="store_true", help="evaluate the control tolerance only")
    parser.add_argument("--no-control-gate", action="store_true", help="stage does not need a passed control")
    args = parser.parse_args(argv)

    smoke = env_flag("SMOKE")
    if args.check_control:
        return check_control(args.exp, smoke)
    options = grids.Options.from_env()
    special = grids.SPECIAL_STAGES.get((args.exp, args.stage))
    if special is not None:
        if args.plan or args.list:
            print(f"Стадия {args.stage}: без обучения (обработка готовых результатов).")
            return 0
        return special(options)
    try:
        tasks = grids.build_tasks(args.exp, args.stage, options)
    except SystemExit as exc:
        if args.plan and not isinstance(exc.code, int):
            print(f"Стадия {args.stage}: список прогонов пока не определён. {exc.code}")
            return 0
        raise
    commit = git_commit()
    for task in tasks:
        task["git_commit"] = commit
    todo = [t for t in tasks if options.force or not Path(t["out_json"]).is_file()]
    resources = load_resources()
    if args.list:
        for task in tasks:
            mark = "todo" if task in todo else "done"
            print(f"{mark}\t{task['run_id']}")
        return 0
    if args.plan:
        plan(tasks, todo, resources, args.stage)
        return 0
    if not args.no_control_gate and not require_control(args.exp, smoke):
        return 3
    if not smoke and not options.synthetic and todo and not data_available(todo[0]["files"]):
        log("данные split не найдены в data/raw/archive: см. RUNBOOK, раздел B")
        return 4
    if options.synthetic and not smoke:
        log("SYNTHETIC разрешён только вместе с SMOKE=1")
        return 2
    jobs, threads, device = resolve_parallelism(resources)
    if resources is None and os.environ.get("JOBS", "auto") == "auto":
        log("нет resources.json: JOBS=1, THREADS=1 (запустите стадию probe для рекомендации)")
    for task in todo:
        task["force"] = options.force
    return execute(todo, jobs, threads, device, resources)


if __name__ == "__main__":
    sys.exit(main())
