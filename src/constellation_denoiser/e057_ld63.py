"""E057: DBP-X Ns = 20, direct_ld at L = 63 (the missing point of E054). The run itself is e054_prune.run unchanged.

    PYTHONPATH=src python -m constellation_denoiser.e057_ld63 fit | summarize      (SMOKE=1: Ns 8, 1 seed, tiny)

Writes results/runs/Ns20_direct_ld_L63_seed<s>.json, results/summary.csv, results/verdicts.csv.
Rule: config.json rule_draft (comparison with progressive L = 63 of E054).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import e054_prune
from .runner import exp_dir, results_root

EXP, L = "E057", 63


def tasks(smoke: bool) -> list[tuple]:
    seeds = json.loads((exp_dir(EXP) / "config.json").read_text(encoding="utf-8"))["seeds"]
    return [(8, 17, "direct_ld", True, L)] if smoke else [(20, s, "direct_ld", False, L) for s in seeds]


def _job(args):
    ns, seed, mode, smoke, target = args
    out = results_root(EXP, smoke) / "runs" / f"Ns{ns}_{mode}_L{target}_seed{seed}.json"
    if out.is_file():
        return f"{out.name}: exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = e054_prune.run(ns, seed, mode, smoke, target)
    except Exception as exc:
        return f"ОШИБКА {out.name}: {exc!r}"
    out.write_text(json.dumps(r, ensure_ascii=False, default=float), encoding="utf-8")
    lv = r["levels"][-1]
    return f"{out.name}: ok, L{lv['L']} {lv['test_reduction']:.2f} ({r['time_s']:.0f} s)"


def fit(smoke: bool) -> int:
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from .gate import MAX_JOBS, adaptive_limit

    pending, running, done = tasks(smoke), set(), 0
    todo, errors = len(pending), 0
    with ProcessPoolExecutor(max_workers=1 if smoke else MAX_JOBS) as ex:
        while pending or running:
            limit = 1 if smoke else adaptive_limit(len(running))
            while pending and len(running) < limit:
                running.add(ex.submit(_job, pending.pop(0)))
            finished, running = wait(running, timeout=60, return_when=FIRST_COMPLETED)
            running = set(running)
            for fut in finished:
                done += 1
                msg = fut.result()
                errors += msg.startswith("ОШИБКА")
                print(f"[{time.strftime('%F %T')}] [{done}/{todo}] {msg} (в работе {len(running)}, предел {limit})",
                      flush=True)
    return 1 if errors else 0


def summarize(smoke: bool) -> int:
    res = results_root(EXP, smoke)
    rows = []
    for p in sorted((res / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "levels" not in r:                      # control runs of the runner share the folder
            continue
        for lv in r["levels"]:
            rows.append(dict({k: r.get(k) for k in ("n_steps", "seed", "mode", "alpha_kind", "start_val_reduction",
                                                    "ld_before", "ld_after", "time_s")}, **lv))
    if not rows:
        print("нет прогонов")
        return 0
    t = pd.DataFrame(rows)
    t.to_csv(res / "levels_table.csv", index=False)
    g = t.groupby(["n_steps", "mode", "L"])
    s = g.agg(n=("seed", "size"), test_mean=("test_reduction", "mean"), test_std=("test_reduction", "std"),
              val_mean=("val_reduction", "mean"), q2_mean=("test_q2_gain", "mean"),
              ld_before=("ld_before", "mean"), ld_after=("ld_after", "mean"), start_val=("start_val_reduction", "mean"),
              rm_project=("rm_project", "first"), rm_xiao=("rm_xiao", "first"), rm_direct_4=("rm_direct_4", "first"),
              rm_direct_3=("rm_direct_3", "first"), rm_fft_4=("rm_fft_4", "first"), rm_fft_3=("rm_fft_3", "first"),
              sum_filter_len=("sum_filter_len", "first"), cascade_support=("cascade_support", "first"),
              time_s=("time_s", "mean")).reset_index()
    s["test_se"] = s.test_std / np.sqrt(s.n)
    s["test_range"] = g.test_reduction.agg(lambda x: x.max() - x.min()).values
    s["source"] = EXP
    ref = pd.read_csv(exp_dir("E054") / "results" / "summary.csv")
    ref = ref[(ref.n_steps == 20) & ref.L.isin([31, 63, 255])].assign(source="E054")
    both = pd.concat([s, ref[[c for c in s.columns if c in ref.columns]]], ignore_index=True)
    top = ref[(ref["mode"] == "progressive") & (ref.L == 255)]
    if len(top):
        both["share_of_progressive_L255"] = both.test_mean / top.test_mean.iloc[0]
    both.to_csv(res / "summary.csv", index=False)
    verdicts = []
    for _, a in s.iterrows():
        for mode in ("progressive", "abrupt"):
            b = ref[(ref["mode"] == mode) & (ref.L == a.L) & (ref.n_steps == a.n_steps)]
            if not len(b):
                continue
            b = b.iloc[0]
            diff = a.test_mean - b.test_mean
            se = float(np.sqrt(a.test_std ** 2 / a.n + b.test_std ** 2 / b.n))
            std = float(max(a.test_std, b.test_std))
            signal = abs(diff) > 2 * se and abs(diff) > 2 * std
            verdicts.append({"compare": f"direct_ld L{a.L} (E057) - {mode} L{b.L} (E054)", "diff_pp": diff,
                             "two_se": 2 * se, "two_std": 2 * std, "n_a": a.n, "n_b": b.n,
                             "verdict": ("direct_ld лучше" if diff > 0 else "direct_ld хуже") if signal
                             else "разница не доказана"})
    pd.DataFrame(verdicts).to_csv(res / "verdicts.csv", index=False)
    print(both.to_string(index=False))
    print(pd.DataFrame(verdicts).to_string(index=False))
    return 0


def main(argv) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    action = argv[0] if argv else ""
    if action == "fit":
        return fit(smoke)
    if action == "summarize":
        return summarize(smoke)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
