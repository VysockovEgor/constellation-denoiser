"""E054: DBP-X with trainable FIR steps, physical start at 255 taps, pruning during training to 127, 63, 31, 15, 7.

    PYTHONPATH=src python -m constellation_denoiser.e054_prune all | summarize      (SMOKE=1: tiny)

Model: dbpx.DBPX (D_total exact and fixed, Ns steps [symmetric FIR(255) by circular FFT, Kerr phase R = 7], complex
output gain). Start: physical taps of the step (L = 255), alpha = g * shape / (Ns * power) with the kernel, g, sign and
multiplier of E050 start (start_best.csv, chosen on validation as in E052). Taps are multiplied by a mask: half-length
r_active; taps beyond r_active are zero (so the active FIR has 2 r_active + 1 taps; gradients there are zero).
Schedule (Adam 1e-3, 4 whole train files per iteration, MSE on all symbols, circular shift):
  warm: WARM iterations at 255;
  for every next length L: "progressive": r_active shrinks linearly from the previous to the new half-length over
  SHRINK iterations, then FINE iterations at L; "abrupt" (control): cut to L at once, then SHRINK + FINE iterations.
  The state at the end of a level continues to the next level; at each level the checkpoint with the minimum
  validation MSE during its FINE part is evaluated on test (selection only by validation).
RM per length: dbpx.rm_columns (direct convolution and FFT / overlap-save), D_total 401 taps.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dbpx import C_TOTAL, DBPX, alpha_shape, gain, load, metrics, response, rm_columns, run_model, results_dir as e050_results
from .runner import results_root

LENGTHS = (255, 127, 63, 31, 15, 7)
SCHEDULE = {"warm": 3000, "shrink": 1000, "fine": 2000, "eval_every": 100, "batch_files": 4, "lr": 1e-3}
SMOKE_SCHEDULE = {"warm": 20, "shrink": 10, "fine": 20, "eval_every": 10, "batch_files": 2, "lr": 1e-3}
ARMS = {20: (17, 29, 43, 57, 71, 83, 97, 101, 113, 127), 40: (17, 29, 43, 57, 71)}


def disperse(z: np.ndarray, c: float) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(z, axis=-1) * response(c, None, z.shape[-1]), axis=-1)


class Masked(DBPX):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.r_active = self.L // 2

    def responses(self, T: int) -> torch.Tensor:
        keep = (torch.arange(self.L // 2 + 1) <= self.r_active).double().to(self.fir_re.dtype)
        re, im = self.fir_re * keep, self.fir_im * keep
        h = torch.complex(re, im)
        r = self.L // 2
        idx = torch.cat((torch.arange(0, r + 1), torch.arange(T - r, T)))
        vals = torch.cat((h, h[:, 1:].flip(1)), dim=1)
        return torch.fft.fft(torch.zeros(self.ns, T, dtype=h.dtype).index_copy(1, idx, vals), dim=-1)


def chosen(ns: int, smoke: bool) -> pd.Series:
    start = pd.read_csv(e050_results(smoke) / "start_best.csv")
    cls = start[(start.alpha_kind == "classical") & (start.n_steps == ns)]
    sign = cls.loc[cls.val_mse_trim.idxmin(), "sign"]
    part = start[(start.n_steps == ns) & (start.sign == sign)]
    return part.loc[part.val_mse_trim.idxmin()]


def run(ns: int, seed: int, mode: str, smoke: bool) -> dict:
    sch = SMOKE_SCHEDULE if smoke else SCHEDULE
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    row = chosen(ns, smoke)
    data = load(smoke)
    c = float(row.sign) * float(row["mult"]) * C_TOTAL
    zd = {k: disperse(d.rx_c, c) for k, d in data.items()}
    power = float(np.mean(np.abs(data["validation"].rx_c) ** 2))
    model = Masked(ns, LENGTHS[0], 7, -c / ns)
    model.init_physical()
    model.set_alpha(float(row.g) * alpha_shape(row.alpha_kind, 7) / (ns * power))
    a = gain(run_model(model, zd["train"]), data["train"].tx_c)
    with torch.no_grad():
        model.gain.copy_(torch.tensor([a.real, a.imag], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=sch["lr"])
    x_all = torch.tensor(zd["train"], dtype=torch.complex64)
    y_all = torch.tensor(data["train"].tx_c, dtype=torch.complex64)
    val, test = data["validation"], data["test"]
    t0, it, history, levels = time.time(), 0, [], []

    def step():
        nonlocal it
        it += 1
        f = torch.tensor(rng.choice(x_all.shape[0], min(sch["batch_files"], x_all.shape[0]), replace=False))
        shift = int(rng.integers(0, x_all.shape[1]))
        x, y = torch.roll(x_all[f], shift, dims=-1), torch.roll(y_all[f], shift, dims=-1)
        opt.zero_grad()
        loss = (model(x) - y).abs().square().mean()
        loss.backward()
        opt.step()
        return float(loss.detach())

    def evaluate():
        m = metrics(run_model(model, zd["validation"]), val.tx_c, val.rx_c)
        history.append({"iter": it, "r_active": model.r_active, "val_reduction_trim": m["reduction_trim"],
                        "val_mse": m["mse_trim"]})
        return m

    def level(L, n_fine):
        best = None
        for j in range(n_fine):
            step()
            if (j + 1) % sch["eval_every"] == 0 or j + 1 == n_fine:
                m = evaluate()
                if best is None or m["mse_trim"] < best[0]:
                    best = (m["mse_trim"], it, {k: v.detach().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[2])
        v = metrics(run_model(model, zd["validation"]), val.tx_c, val.rx_c)
        t = metrics(run_model(model, zd["test"]), test.tx_c, test.rx_c)
        levels.append(dict(L=L, best_iter=best[1], val_reduction=v["reduction_trim"], test_reduction=t["reduction_trim"],
                           test_q2_gain=t["q2_gain_trim"], test_reduction_c31=t["reduction_c31"],
                           **rm_columns(ns, L, 401, 7)))

    m0 = evaluate()
    start_val = m0["reduction_trim"]
    level(LENGTHS[0], sch["warm"])
    for L in LENGTHS[1:]:
        r_new, r_old = L // 2, model.r_active
        if mode == "progressive":
            for j in range(sch["shrink"]):
                model.r_active = int(round(r_old + (r_new - r_old) * (j + 1) / sch["shrink"]))
                step()
            model.r_active = r_new
            level(L, sch["fine"])
        else:
            model.r_active = r_new
            level(L, sch["shrink"] + sch["fine"])
    return {"n_steps": ns, "seed": seed, "mode": mode, "alpha_kind": row.alpha_kind, "g": float(row.g),
            "mult": float(row["mult"]), "sign": float(row.sign), "start_val_reduction": start_val, "levels": levels,
            "history": history, "schedule": sch, "time_s": time.time() - t0}


def _job(args):
    ns, seed, mode, smoke = args
    out = results_root("E054", smoke) / "runs" / f"Ns{ns}_{mode}_seed{seed}.json"
    if out.is_file():
        return f"{out.name}: exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = run(ns, seed, mode, smoke)
    except Exception as exc:
        return f"ОШИБКА {out.name}: {exc!r}"
    out.write_text(json.dumps(r, ensure_ascii=False, default=float), encoding="utf-8")
    return f"{out.name}: ok, " + ", ".join(f"L{l['L']} {l['test_reduction']:.1f}" for l in r["levels"]) + \
        f" ({r['time_s']:.0f} s)"


def tasks(smoke: bool) -> list[tuple]:
    if smoke:
        return [(8, 17, "progressive", True), (8, 17, "abrupt", True)]
    out = []
    for ns, seeds in ARMS.items():
        for seed in seeds:
            out.append((ns, seed, "progressive", False))
        for seed in seeds[:5]:
            out.append((ns, seed, "abrupt", False))
    return out


def summarize(smoke: bool) -> int:
    res = results_root("E054", smoke)
    rows = []
    for p in sorted((res / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        for l in r["levels"]:
            rows.append(dict({k: r[k] for k in ("n_steps", "seed", "mode", "alpha_kind", "start_val_reduction")}, **l))
    t = pd.DataFrame(rows)
    t.to_csv(res / "levels_table.csv", index=False)
    g = t.groupby(["n_steps", "mode", "L"])
    s = g.agg(n=("seed", "size"), test_mean=("test_reduction", "mean"), test_std=("test_reduction", "std"),
              val_mean=("val_reduction", "mean"), q2_mean=("test_q2_gain", "mean"), rm_project=("rm_project", "first"),
              rm_xiao=("rm_xiao", "first"), rm_merged=("rm_merged", "first")).reset_index()
    s["test_se"] = s.test_std / np.sqrt(s.n)
    s["test_range"] = g.test_reduction.agg(lambda x: x.max() - x.min()).values
    s.to_csv(res / "summary.csv", index=False)
    rows = []
    for (ns, mode), part in s.groupby(["n_steps", "mode"]):
        top = part[part.L == LENGTHS[0]].iloc[0]
        for _, r in part.iterrows():
            rows.append({"n_steps": ns, "mode": mode, "L": r.L, "test_mean": r.test_mean, "test_std": r.test_std,
                         "kept_share_of_L255": r.test_mean / top.test_mean if top.test_mean else np.nan,
                         "drop_pp": top.test_mean - r.test_mean, "rm_project": r.rm_project, "rm_xiao": r.rm_xiao})
    for ns in s.n_steps.unique():
        for L in LENGTHS[1:]:
            pr = s[(s.n_steps == ns) & (s["mode"] == "progressive") & (s.L == L)]
            ab = s[(s.n_steps == ns) & (s["mode"] == "abrupt") & (s.L == L)]
            if len(pr) and len(ab):
                a, b = pr.iloc[0], ab.iloc[0]
                diff = a.test_mean - b.test_mean
                se = np.sqrt(a.test_std ** 2 / a.n + b.test_std ** 2 / b.n)
                std = max(a.test_std, b.test_std)
                rows.append({"n_steps": ns, "mode": "progressive - abrupt", "L": L, "diff_pp": diff, "two_se": 2 * se,
                             "two_std": 2 * std, "signal": bool(diff > 2 * se and diff > 2 * std)})
    pd.DataFrame(rows).to_csv(res / "verdicts.csv", index=False)
    plot(s, Path(str(res).replace("results_smoke", "results_smoke")) if smoke else res.parent / "figures")
    print(s.to_string(index=False))
    return 0


def plot(s, figs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for (ns, mode), p in s.groupby(["n_steps", "mode"]):
        p = p.sort_values("L")
        ls = "-" if mode == "progressive" else ":"
        lab = f"Ns={ns}, {mode}"
        axes[0].errorbar(p.L, p.test_mean, yerr=p.test_std, ls=ls, marker="o", capsize=2, label=lab)
        axes[1].errorbar(p.rm_project, p.test_mean, yerr=p.test_std, ls=ls, marker="o", capsize=2, label=lab)
        axes[2].errorbar(p.rm_xiao, p.test_mean, yerr=p.test_std, ls=ls, marker="o", capsize=2, label=lab)
    for ax, xl in zip(axes, ("длина КИХ шага L", "RM, прямая свёртка (конвенция проекта)", "RM, БПФ / overlap-save (Xiao)")):
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("снижение BER на test, %")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "E054_pruning.png", dpi=150)
    plt.close(fig)


def main(argv) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    action = argv[0] if argv else ""
    if action == "all":
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        from .gate import MAX_JOBS, adaptive_limit

        pending, running, done, todo = tasks(smoke), set(), 0, None
        todo = len(pending)
        with ProcessPoolExecutor(max_workers=1 if smoke else MAX_JOBS) as ex:
            while pending or running:
                limit = 1 if smoke else adaptive_limit(len(running))
                while pending and len(running) < limit:
                    running.add(ex.submit(_job, pending.pop(0)))
                finished, running = wait(running, timeout=60, return_when=FIRST_COMPLETED)
                running = set(running)
                for fut in finished:
                    done += 1
                    print(f"[{time.strftime('%F %T')}] [{done}/{todo}] {fut.result()} (в работе {len(running)}, "
                          f"предел {limit})", flush=True)
        return 0
    if action == "summarize":
        return summarize(smoke)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
