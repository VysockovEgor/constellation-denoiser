"""E059: trained DBP-X with the Kerr window R = 1, 3, 7, 15 at Ns = 8 and 20 (dbpx.train_one unchanged).

    PYTHONPATH=src python -m constellation_denoiser.e059_kerr_width train | summarize      (SMOKE=1: tiny)

Writes results/dbpx_runs/*.json (+ .pt weights, not committed), results/runs_table.csv, results/summary.csv,
results/verdicts.csv, figures/E059_kerr_width.png. Rule: config.json rule_draft.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dbpx import TRAIN, env_int, train_one
from .dbpx import results_dir as e050_results
from .e054_prune import rm_both
from .runner import exp_dir, results_root

EXP = "E059"


def config() -> dict:
    return json.loads((exp_dir(EXP) / "config.json").read_text(encoding="utf-8"))


def plan(smoke: bool) -> list[dict]:
    cfg = config()
    grid = cfg["grid"]
    best = pd.read_csv(e050_results(smoke) / "start_best.csv")
    cls = best[best.alpha_kind == "classical"]
    out = results_root(EXP, smoke) / "dbpx_runs"
    tasks = []
    for ns in ([8] if smoke else grid["n_steps"]):
        rows = cls[cls.n_steps == ns]
        row = rows.loc[rows.val_mse_trim.idxmin()]             # sign, multiplier and g of the E050 start (validation)
        for R in ([1, 3] if smoke else grid["R"]):
            for seed in (cfg["seeds"][:1] if smoke else cfg["seeds"]):
                name = f"dbpx_Ns{ns}_L{grid['L']}_R{R}_seed{seed}"
                tasks.append({"name": name, "ns": ns, "L": grid["L"], "R": R, "alpha_init": grid["alpha_init"],
                              "training": grid["training"], "g": float(row.g), "mult": float(row["mult"]),
                              "sign": float(row.sign), "seed": seed,
                              "iters": TRAIN["smoke_iters"] if smoke else TRAIN["iters"],
                              "eval_every": 10 if smoke else TRAIN["eval_every"], "batch_files": TRAIN["batch_files"],
                              "lr": TRAIN["lr"], "ld_iters": 50 if smoke else TRAIN["ld_iters"],
                              "out_json": str(out / f"{name}.json")})
    return tasks


def _job(args):
    task, smoke = args
    try:
        r = train_one(task, smoke)
        return f"[ok] {task['name']}: val={r['val_reduction_trim']:.2f} test={r['test_reduction_trim']:.2f} " \
               f"time={r['time_s']:.0f}s"
    except Exception as exc:
        return f"[fail] {task['name']}: {exc!r}"


def train(smoke: bool) -> int:
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from .gate import MAX_JOBS, adaptive_limit

    tasks = [t for t in plan(smoke) if not Path(t["out_json"]).is_file()]
    print(f"E059 train: {len(tasks)} прогонов без результата", flush=True)
    pending, running, done, failed = list(tasks), set(), 0, 0
    with ProcessPoolExecutor(max_workers=1 if smoke else MAX_JOBS) as pool:
        while pending or running:
            limit = 1 if smoke else adaptive_limit(len(running))
            while pending and len(running) < limit:
                running.add(pool.submit(_job, (pending.pop(0), smoke)))
            finished, running = wait(running, timeout=60, return_when=FIRST_COMPLETED)
            running = set(running)
            for fut in finished:
                done += 1
                msg = fut.result()
                failed += msg.startswith("[fail]")
                print(f"[{time.strftime('%F %T')}] [{done}/{len(tasks)}] {msg} (в работе {len(running)}, предел {limit})",
                      flush=True)
    return 1 if failed else 0


def compare(a, b) -> dict:
    diff = a.test_mean - b.test_mean
    se = float(np.sqrt(a.test_std ** 2 / a.n + b.test_std ** 2 / b.n))
    std = float(max(a.test_std, b.test_std))
    return {"diff_pp": diff, "two_se": 2 * se, "two_std": 2 * std,
            "better": bool(diff > 2 * se and diff > 2 * std), "worse": bool(-diff > 2 * se and -diff > 2 * std)}


def summarize(smoke: bool) -> int:
    res = results_root(EXP, smoke)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((res / "dbpx_runs").glob("*.json"))]
    if not runs:
        print("нет прогонов")
        return 0
    keys = ("name", "ns", "L", "R", "seed", "g", "mult", "best_iter", "plateau_iter_0p5pp", "diverged",
            "start_val_reduction_trim", "val_reduction_trim", "test_reduction_valbest_select", "test_q2_gain_trim",
            "test_reduction_c31", "rm_project", "rm_xiao", "rm_merged", "time_s")
    t = pd.DataFrame([{k: r.get(k) for k in keys} for r in runs])
    t.to_csv(res / "runs_table.csv", index=False)
    g = t.groupby(["ns", "L", "R"])
    s = g.agg(n=("seed", "size"), test_mean=("test_reduction_valbest_select", "mean"),
              test_std=("test_reduction_valbest_select", "std"), val_mean=("val_reduction_trim", "mean"),
              q2_mean=("test_q2_gain_trim", "mean"), start_val=("start_val_reduction_trim", "mean"),
              plateau_mean=("plateau_iter_0p5pp", "mean"), diverged=("diverged", "sum"),
              rm_project=("rm_project", "first"), rm_xiao=("rm_xiao", "first"), rm_merged=("rm_merged", "first"),
              time_s=("time_s", "mean")).reset_index()
    s["test_std"] = s.test_std.fillna(0.0)
    s["test_se"] = s.test_std / np.sqrt(s.n)
    s["test_range"] = g.test_reduction_valbest_select.agg(lambda x: x.max() - x.min()).values
    rm = pd.DataFrame([rm_both(int(a), int(b), int(c)) for a, b, c in zip(s.ns, s.L, s.R)])
    s = pd.concat([s, rm[["rm_direct_4", "rm_direct_3", "rm_fft_4", "rm_fft_3"]]], axis=1)
    s.to_csv(res / "summary.csv", index=False)
    verdicts = []
    for ns, part in s.groupby("ns"):
        part = part.set_index("R")
        if 1 in part.index:
            for R in part.index:
                if R != 1:
                    verdicts.append(dict(compare(part.loc[R], part.loc[1]), ns=ns,
                                         question=f"R={R} better than R=1 (Kerr memory helps)"))
        top = part.test_mean.idxmax()
        tol = max(2 * part.loc[top].test_se, 2 * part.loc[top].test_std)
        ok = [R for R in sorted(part.index) if part.loc[top].test_mean - part.loc[R].test_mean <= tol]
        verdicts.append({"ns": ns, "question": "sufficient R (smallest within max(2 SE, 2 std) of the best)",
                         "best_R": int(top), "sufficient_R": int(ok[0]), "tolerance_pp": tol})
    e050 = pd.read_csv(exp_dir("E050") / "results" / "train_summary.csv")
    ref = e050[(e050.ns == 20) & (e050.L == 255) & (e050.alpha_init == "classical") & (e050.training == "two")]
    mine = s[(s.ns == 20) & (s.R == 7)]
    if len(ref) and len(mine):
        c = compare(mine.iloc[0], ref.iloc[0])
        verdicts.append(dict(c, ns=20, question="reproduction: Ns 20, R 7 vs E050 classical two (10 seeds)",
                             within_2se=bool(abs(c["diff_pp"]) <= c["two_se"])))
    pd.DataFrame(verdicts).to_csv(res / "verdicts.csv", index=False)
    plot(s, exp_dir(EXP) / ("results_smoke" if smoke else "figures"))
    print(s[["ns", "R", "n", "test_mean", "test_std", "test_se", "test_range", "rm_direct_4", "rm_fft_4"]].to_string(index=False))
    print(pd.DataFrame(verdicts).to_string(index=False))
    return 0


def plot(s, figs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, col, xl in ((axes[0], "rm_direct_4", "RM, прямая свёртка (4 RM)"), (axes[1], "rm_fft_4", "RM, БПФ (Xiao, 4 RM)")):
        for (ns, p), color in zip(s.groupby("ns"), ("#2a78d6", "#eb6834")):
            p = p.sort_values("R")
            ax.errorbar(p[col], p.test_mean, yerr=p.test_se, color=color, marker="o", capsize=3, label=f"Ns = {ns}")
            for _, r in p.iterrows():
                ax.annotate(f"R={int(r.R)}", (r[col], r.test_mean), xytext=(4, -12), textcoords="offset points",
                            fontsize=8)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("снижение BER на test, % (среднее ± SE)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figs / "E059_kerr_width.png", dpi=150)
    plt.close(fig)


def main(argv) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    action = argv[0] if argv else ""
    if action == "train":
        return train(smoke)
    if action == "summarize":
        return summarize(smoke)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
