"""Queue gates for E024-E033 (scripts/overnight_e024_e033.sh).

    python -m constellation_denoiser.gate halted                 exit 1 if results/gate_status.json has halt: true
    python -m constellation_denoiser.gate jobs [--reserve N]     JOBS for the next stage (stdout)
    python -m constellation_denoiser.gate memory                 exit 1 (and halt) if MemAvailable < MIN_FREE_GB
    python -m constellation_denoiser.gate stage --exp E025 --stage s1 [--search]
                                                                 halt if > 15 % of the stage runs failed or diverged
                                                                 (--search: tuning stage, only failures count)
    python -m constellation_denoiser.gate set --reason TEXT      halt by hand (e.g. data leakage in E024)

JOBS = min(MAX_JOBS, cores - foreign busy cores - 2 - reserve, (MemAvailable - 16 GB) / 1.5 GB); foreign = mean
number of running threads of other users over 10 samples in 5 s. The queue never stops because of foreign load,
it only lowers JOBS (and the runner's polite mode waits per task, as before).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .paths import PROJECT_ROOT

GATE_PATH = PROJECT_ROOT / "results" / "gate_status.json"
MAX_JOBS = 40
MIN_FREE_GB = float(os.environ.get("GATE_MIN_FREE_GB", "16"))   # override only for local tests of the queue
MAX_BAD_SHARE = 0.15
MEM_PER_JOB_GB = 1.5


def gate_path(smoke: bool | None = None) -> Path:
    if smoke is None:
        smoke = os.environ.get("SMOKE", "0") == "1"
    return PROJECT_ROOT / "results_smoke" / "gate_status.json" if smoke else GATE_PATH


def read_gate(smoke: bool | None = None) -> dict:
    path = gate_path(smoke)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"halt": False}


def set_halt(reason: str, detail: str = "", smoke: bool | None = None) -> None:
    from .runner import atomic_write_json

    state = read_gate(smoke)
    events = state.get("events", [])
    events.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": reason, "detail": detail})
    atomic_write_json(gate_path(smoke), {"halt": True, "reason": reason, "detail": detail, "events": events})
    print(f"ОСТАНОВ ОЧЕРЕДИ: {reason} {detail}".strip(), flush=True)


def mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return float("inf")


def running_threads() -> tuple[int, int]:
    """(running threads of this user, running threads of other users): state R in /proc/*/task/*/stat."""
    uid, own, other = os.getuid(), 0, 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            mine = entry.stat().st_uid == uid
            for task in (entry / "task").iterdir():
                state = (task / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if state == "R":
                    own, other = (own + 1, other) if mine else (own, other + 1)
        except (OSError, IndexError):
            continue
    return own, other


def foreign_busy(samples: int = 10, pause_s: float = 0.5) -> float:
    """Mean number of running threads of other users (instantaneous, unlike the 1-min load average, which
    still contains our own processes of the stage that has just finished)."""
    values = []
    for index in range(samples):
        values.append(running_threads()[1])
        if index + 1 < samples:
            time.sleep(pause_s)
    return float(np.mean(values))


def jobs_now(reserve: int = 0) -> tuple[int, dict]:
    cores = os.cpu_count() or 1
    try:
        load = float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError):
        load = 0.0
    foreign = foreign_busy()
    jobs = min(MAX_JOBS, int(cores - math.ceil(foreign) - 2 - reserve))
    free = mem_available_gb()
    if free != float("inf"):
        jobs = min(jobs, int((free - MIN_FREE_GB) / MEM_PER_JOB_GB))
    jobs = max(1, jobs)
    return jobs, {"cores": cores, "load_1min": load, "foreign_running_threads": round(foreign, 1), "reserve": reserve,
                  "mem_available_gb": round(free, 1)}


def stage_health(exp: str, stage: str, search: bool, smoke: bool) -> dict:
    from .grids import Options, build_tasks

    tasks = build_tasks(exp, stage, Options(smoke=smoke))
    failed = diverged = 0
    for task in tasks:
        out = Path(task["out_json"])
        if not out.is_file():
            failed += 1
            continue
        if json.loads(out.read_text(encoding="utf-8")).get("diverged"):
            diverged += 1
    bad = failed + (0 if search else diverged)
    return {"exp": exp, "stage": stage, "total": len(tasks), "failed_or_missing": failed, "diverged": diverged,
            "search_stage": search, "bad_share": bad / len(tasks) if tasks else 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("halted")
    jobs = sub.add_parser("jobs")
    jobs.add_argument("--reserve", type=int, default=0)
    jobs.add_argument("--max", type=int, default=MAX_JOBS)
    memory = sub.add_parser("memory")
    memory.add_argument("--wait-s", type=int, default=1800)
    stage = sub.add_parser("stage")
    stage.add_argument("--exp", required=True)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--search", action="store_true")
    halt = sub.add_parser("set")
    halt.add_argument("--reason", required=True)
    halt.add_argument("--detail", default="")
    args = parser.parse_args(argv)
    smoke = os.environ.get("SMOKE", "0") == "1"

    if args.cmd == "halted":
        state = read_gate(smoke)
        if state.get("halt"):
            print(f"очередь остановлена: {state.get('reason')} {state.get('detail', '')}")
            return 1
        return 0
    if args.cmd == "jobs":
        n, info = jobs_now(args.reserve)
        n = min(n, args.max)
        print(n)
        print(f"JOBS={n}: {info}", file=sys.stderr)
        return 0
    if args.cmd == "memory":
        waited = 0
        while mem_available_gb() < MIN_FREE_GB and waited < args.wait_s:   # a short dip of foreign memory is tolerated
            time.sleep(60)
            waited += 60
        free = mem_available_gb()
        if free < MIN_FREE_GB:
            set_halt("свободной памяти меньше 16 ГБ", f"{free:.1f} ГБ после ожидания {waited} с", smoke)
            return 1
        return 0
    if args.cmd == "stage":
        health = stage_health(args.exp, args.stage, args.search, smoke)
        print(json.dumps(health, ensure_ascii=False))
        if health["bad_share"] > MAX_BAD_SHARE and not smoke:
            set_halt(f"{args.exp} {args.stage}: упали или разошлись {100 * health['bad_share']:.0f} % прогонов",
                     json.dumps(health, ensure_ascii=False), smoke)
            return 1
        return 0
    if args.cmd == "set":
        set_halt(args.reason, args.detail, smoke)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
