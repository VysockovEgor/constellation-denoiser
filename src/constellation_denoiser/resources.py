"""Resource probe for shared servers: detect limits, benchmark (JOBS, THREADS), recommend.

Run on the server via ``PYTHONPATH=src python scripts/probe_resources.py``; it
writes ``resources.json`` in the repository root (not tracked by Git).
Overrides: SHARE (default 0.5), MAX_CORES, MAX_MEM_GB, MAX_JOBS, PROBE_SECONDS.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .paths import PROJECT_ROOT

REFERENCE_SPECS = {
    "complex": {"model": "complex_both", "window": 31, "hidden": 48},
    "real": {"model": "real_tanh", "window": 31, "hidden": 96},
    "transformer": {"model": "transformer", "window": 31, "d_model": 64, "layers": 2, "heads": 4},
    "kan": {"model": "kan", "window": 31, "kan_hidden": 16, "grid": 5},
}
BENCH_STEPS = 200
WARMUP_STEPS = 20


# --------------------------------------------------------------------------- detection

def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_cpu_limit() -> float | None:
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        quota, period = v2.split()[:2]
        if quota != "max":
            return int(quota) / int(period)
        return None
    quota = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") or _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
    period = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us") or _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period)
    return None


def physical_cores() -> int | None:
    text = _read("/proc/cpuinfo")
    if not text:
        return None
    pairs, physical, core = set(), None, None
    for line in text.splitlines():
        if line.startswith("physical id"):
            physical = line.split(":")[1].strip()
        elif line.startswith("core id"):
            core = line.split(":")[1].strip()
            pairs.add((physical, core))
    return len(pairs) or None


def meminfo_gb() -> dict[str, float]:
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            out[key] = int(value.split()[0]) / 1024 ** 2
    return out


def cgroup_memory_gb() -> float | None:
    value = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if value and value != "max" and int(value) < 1 << 60:
        return int(value) / 1024 ** 3
    return None


def gpu_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "count": 0, "devices": []}
    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "free_gb": free / 1024 ** 3, "total_gb": total / 1024 ** 3})
    return {"available": True, "count": len(devices), "devices": devices}


def detect() -> dict:
    affinity = len(os.sched_getaffinity(0))
    quota = cgroup_cpu_limit()
    available = min(affinity, int(quota)) if quota else affinity
    load1, load5, load15 = (float(x) for x in (_read("/proc/loadavg") or "0 0 0").split()[:3])
    memory = meminfo_gb()
    cg_mem = cgroup_memory_gb()
    mem_available = min(memory.get("MemAvailable", 0.0), cg_mem) if cg_mem else memory.get("MemAvailable", 0.0)
    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "cpu": {"affinity": affinity, "cgroup_quota": quota, "available": max(1, available),
                "physical_cores": physical_cores(), "os_cpu_count": os.cpu_count()},
        "load": {"load1": load1, "load5": load5, "load15": load15,
                 "busy_cores_estimate": round(max(load1, load5), 2),
                 "free_cores_estimate": round(max(0.0, available - max(load1, load5)), 2)},
        "memory": {"total_gb": memory.get("MemTotal"), "mem_available_gb": memory.get("MemAvailable"),
                   "cgroup_limit_gb": cg_mem, "usable_gb": mem_available},
        "gpu": gpu_info(),
        "disk": {"free_gb": disk.free / 1024 ** 3, "path": str(PROJECT_ROOT)},
        "python": sys.version.split()[0],
    }


# --------------------------------------------------------------------------- benchmark worker

def bench_worker(spec: dict, threads: int, device: str, steps: int, start_at: float, n_files: int) -> dict:
    """One benchmark process: real training steps on synthetic windows; returns rates and peak RSS."""
    import torch

    from . import model_zoo
    from .window_data import model_ber, synthetic_fileset

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    dev = torch.device(device)
    window = int(spec["window"])
    data = synthetic_fileset(n_files, 7).to(dev)
    model = model_zoo.build(spec, seed=17).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(17)
    n = data.n_windows(window)

    def one_step():
        indices = torch.randint(0, n, (1024,), generator=generator)
        xb, yb = data.gather(indices, window)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        if torch.is_complex(prediction):
            loss = (prediction - torch.complex(yb[:, 0], yb[:, 1])).abs().square().mean()
        else:
            loss = torch.nn.functional.mse_loss(prediction, yb)
        loss.backward()
        optimizer.step()

    for _ in range(WARMUP_STEPS):
        one_step()
    while time.time() < start_at:
        time.sleep(0.01)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    eval_set = synthetic_fileset(2, 8).to(dev)
    started = time.perf_counter()
    model_ber(model, eval_set, window)
    eval_seconds = time.perf_counter() - started
    return {"steps_per_s": steps / train_seconds, "eval_windows_per_s": eval_set.n_windows(window) / eval_seconds,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
            "n_files": n_files}


def run_parallel(spec: dict, jobs: int, threads: int, device: str, steps: int = BENCH_STEPS,
                 n_files: int = 30) -> dict:
    start_at = time.time() + 8.0 + 0.5 * jobs
    command = [sys.executable, "-m", "constellation_denoiser.resources", "--bench-worker", json.dumps(spec),
               str(threads), device, str(steps), str(start_at), str(n_files)]
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"), OMP_NUM_THREADS=str(threads))
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                 for _ in range(jobs)]
    results = []
    for process in processes:
        out, err = process.communicate(timeout=600)
        if process.returncode != 0:
            raise RuntimeError(f"benchmark worker failed: {err.strip()[-500:]}")
        results.append(json.loads(out.strip().splitlines()[-1]))
    per_process = min(r["steps_per_s"] for r in results)
    return {"jobs": jobs, "threads": threads, "device": device,
            "steps_per_s_per_process": per_process, "steps_per_s_total": per_process * jobs,
            "eval_windows_per_s_per_process": min(r["eval_windows_per_s"] for r in results),
            "peak_rss_gb_per_process": max(r["peak_rss_gb"] for r in results)}


def candidate_combos(budget: int) -> list[tuple[int, int]]:
    powers = [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= budget]
    combos = {(1, 1), (1, min(4, budget)), (budget, 1)}
    for jobs in powers + [budget]:
        for threads in powers:
            if jobs * threads <= budget:
                combos.add((jobs, threads))
    ordered = sorted(combos, key=lambda c: (c[0] * c[1], c[1]))
    # keep the benchmark short: corners plus a few in between
    if len(ordered) > 9:
        keep = {ordered[0], ordered[-1]}
        best_fill = [c for c in ordered if c[0] * c[1] == max(x[0] * x[1] for x in ordered)]
        keep.update(best_fill)
        step = max(1, len(ordered) // 6)
        keep.update(ordered[::step])
        ordered = sorted(keep, key=lambda c: (c[0] * c[1], c[1]))
    return ordered


# --------------------------------------------------------------------------- recommendation

def env_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "", "auto") else default


def probe(verbose: bool = True) -> dict:
    import torch

    from . import model_zoo

    found = detect()
    share = env_float("SHARE", 0.5)
    available = found["cpu"]["available"]
    free = found["load"]["free_cores_estimate"]
    budget = int(max(1, (free if free >= 1 else 1) * share))
    max_cores = env_float("MAX_CORES", None)
    if max_cores:
        budget = max(1, min(budget, int(max_cores)))
    usable_mem = found["memory"]["usable_gb"] or 1.0
    mem_limit = 0.5 * usable_mem
    if env_float("MAX_MEM_GB", None):
        mem_limit = min(mem_limit, env_float("MAX_MEM_GB", None))
    max_jobs = env_float("MAX_JOBS", None)

    table = []
    device = "cuda" if found["gpu"]["available"] else "cpu"
    if verbose:
        print(f"Бенчмарк: бюджет {budget} ядер (SHARE={share}), устройство {device} ...", flush=True)
    reference = REFERENCE_SPECS["complex"]
    if device == "cuda":
        combos = [(j, 1) for j in (1, 2, 4) if j <= max(1, budget)]
    else:
        combos = candidate_combos(budget)
    deadline = time.time() + float(os.environ.get("PROBE_SECONDS", 150))
    for jobs, threads in combos:
        if time.time() > deadline:
            break
        try:
            row = run_parallel(reference, jobs, threads, device)
        except Exception as exc:  # a failed combination is reported, not fatal
            row = {"jobs": jobs, "threads": threads, "device": device, "error": str(exc)[:300]}
        table.append(row)
        if verbose and "error" not in row:
            print(f"  JOBS={jobs:<3} THREADS={threads:<3} -> {row['steps_per_s_total']:7.1f} шаг/с суммарно, "
                  f"{row['steps_per_s_per_process']:6.1f} на процесс, RSS {row['peak_rss_gb_per_process']:.2f} ГБ",
                  flush=True)

    valid = [r for r in table if "error" not in r
             and r["jobs"] * (r["threads"] if device == "cpu" else 1) <= max(1, budget)
             and r["jobs"] * r["peak_rss_gb_per_process"] <= mem_limit
             and (max_jobs is None or r["jobs"] <= max_jobs)]
    best = max(valid, key=lambda r: r["steps_per_s_total"]) if valid else None
    jobs = best["jobs"] if best else 1
    threads = best["threads"] if best else 1
    single = next((r for r in table if "error" not in r and r["jobs"] == 1 and r["threads"] == threads), None)
    contention = (best["steps_per_s_per_process"] / single["steps_per_s_per_process"]) if (best and single) else 1.0

    models = {}
    for family, spec in REFERENCE_SPECS.items():
        if time.time() > deadline + 60:
            break
        try:
            row = run_parallel(spec, 1, threads, device, steps=60)
            models[family] = {"spec": spec, "macs": model_zoo.cost(spec).macs,
                              "steps_per_s": row["steps_per_s_per_process"] * contention,
                              "eval_windows_per_s": row["eval_windows_per_s_per_process"] * contention,
                              "single_steps_per_s": row["steps_per_s_per_process"],
                              "peak_rss_gb": row["peak_rss_gb_per_process"]}
        except Exception as exc:
            models[family] = {"spec": spec, "error": str(exc)[:300]}

    rec = {
        "jobs": jobs, "threads": threads, "device": device, "share": share, "core_budget": budget,
        "memory_limit_gb": mem_limit, "contention_factor": contention,
        "steps_per_s_total": best["steps_per_s_total"] if best else None,
        "foreign_load_threshold": round(max(0.0, available - jobs * (threads if device == "cpu" else 1)), 2),
        "polite_max_wait_s": 7200,
        "ok": bool(best) and free >= 1 and usable_mem >= 2.0,
    }
    payload = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hostname": os.uname().nodename,
               "torch_version": torch.__version__, "found": found, "benchmark": table,
               "recommendation": rec, "models": models}
    return payload


def human_summary(payload: dict) -> str:
    found, rec = payload["found"], payload["recommendation"]
    cpu, load, mem = found["cpu"], found["load"], found["memory"]
    lines = [
        f"Найдено {cpu['available']} доступных ядер (affinity {cpu['affinity']}, квота cgroup "
        f"{cpu['cgroup_quota'] or 'нет'}, физических {cpu['physical_cores'] or 'неизвестно'}).",
        f"Нагрузка сейчас {load['load1']:.1f}/{load['load5']:.1f}/{load['load15']:.1f}; свободно примерно "
        f"{load['free_cores_estimate']:.1f} ядер. Память: доступно {mem['usable_gb']:.1f} ГБ. "
        f"Диск: свободно {found['disk']['free_gb']:.1f} ГБ.",
        "GPU: " + (", ".join(f"{d['name']} ({d['free_gb']:.1f} ГБ свободно)" for d in found["gpu"]["devices"])
                   if found["gpu"]["available"] else "нет (будет CPU)."),
        f"Рекомендую JOBS={rec['jobs']}, THREADS={rec['threads']}, DEVICE={rec['device']}: примерно "
        f"{(rec['steps_per_s_total'] or 0):.0f} шагов/с суммарно (эталон: комплексная MLP, окно 31, hidden 48, "
        f"batch 1024).",
        f"Вежливый режим: новая задача ждёт, если чужая нагрузка выше {rec['foreign_load_threshold']}.",
    ]
    for family, info in payload["models"].items():
        if "error" in info:
            lines.append(f"  {family}: замер не удался ({info['error'][:80]})")
        else:
            lines.append(f"  {family}: ~{info['steps_per_s']:.1f} шагов/с на процесс при рекомендованных настройках")
    if not rec["ok"]:
        lines.append("ВНИМАНИЕ: ресурсов мало (почти нет свободных ядер или памяти < 2 ГБ). Лучше запускать ночью "
                     "или уменьшить сетку (SEEDS=17,29), либо задать SHARE=0.25.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- used by runner.plan

def family_of(spec: dict) -> str:
    kind = spec["model"]
    if kind.startswith("complex"):
        return "complex"
    if kind == "real_tanh":
        return "real"
    if kind == "transformer":
        return "transformer"
    return "kan"


def scaled_rates(spec: dict, resources: dict) -> dict:
    from . import model_zoo

    models = resources.get("models", {})
    families = ["complex", "kan"] if spec["model"] == "kerr_kan" else [family_of(spec)]
    seconds_step, seconds_window = 0.0, 0.0
    for family in families:
        info = models.get(family) or models.get("complex")
        if not info or "error" in info:
            info = models["complex"]
        ratio = model_zoo.cost(spec).macs / max(1, info["macs"])
        seconds_step += (0.3 + 0.7 * ratio) / info["steps_per_s"]
        seconds_window += (0.1 + 0.9 * ratio) / info["eval_windows_per_s"]
    return {"steps_per_s": 1.0 / seconds_step, "eval_windows_per_s": 1.0 / seconds_window}


def memory_per_process_gb(task: dict, resources: dict) -> float:
    info = resources.get("models", {}).get(family_of(task["spec"])) or resources["models"]["complex"]
    base = info.get("peak_rss_gb", 0.6)
    files = sum(len(v) for v in task["files"].values())
    return base + files * 16_384 * 64 / 1024 ** 3


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--bench-worker":
        spec, threads, device, steps, start_at, n_files = argv[1:7]
        print(json.dumps(bench_worker(json.loads(spec), int(threads), device, int(steps), float(start_at),
                                      int(n_files))))
        return 0
    payload = probe()
    out = PROJECT_ROOT / "resources.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(human_summary(payload))
    print(f"Записано: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
