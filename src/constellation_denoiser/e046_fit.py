"""E046 (phase 2): fits of architecture A (Kerr with memory + off-axis triplets) and its controls, closed form + L-BFGS.

    PYTHONPATH=src python -m constellation_denoiser.e046_fit all        # every job of config.json, JOBS processes
    PYTHONPATH=src python -m constellation_denoiser.e046_fit one A1 varpro 5 15 17
    PYTHONPATH=src python -m constellation_denoiser.e046_fit summarize
(SMOKE=1: 2/1/1 files, M = 3, one replica.) See kerr_offaxis.py for the model and the training modes.

Per job (arch, mode, M, N_c, replica): bootstrap counts of the 30 train files (numpy default_rng(replica)); theta on the
first n_sub distinct bootstrap files; C on all train files with the counts; lambda by validation for every
configuration of the sweeps (magnitude K, geometric K_th); test reported. Writes results/fits/<job>.csv.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .core import BIT_TABLE, CONSTELLATION
from .kerr_offaxis import EDGE, LAMBDAS, Kerr, features, fit_theta, kerr_rm, offaxis_box, offaxis_terms
from .q2 import q2_db
from .runner import atomic_write_json, default_files, exp_dir, git_commit, results_root
from .triplets import triplet_terms

FIR = 7
_DATA: dict = {}
POINTS = torch.tensor(CONSTELLATION, dtype=torch.complex128)
BITS = torch.tensor(BIT_TABLE, dtype=torch.int64)


EXP = {"id": os.environ.get("TRIPLET_KERR_EXP", "E046")}   # E048 reuses this module with its own config
SPLIT = {"seed": None}          # set by the "splits" action: E023 file split seed (None = the E004 split)


def data(smoke: bool):
    key = (smoke, SPLIT["seed"])
    if key not in _DATA:
        from .grids import e023_split
        from .window_data import load_fileset, resolve_paths

        files = default_files(30, smoke) if SPLIT["seed"] is None else e023_split(SPLIT["seed"], smoke)
        sets = {k: load_fileset(resolve_paths(v)) for k, v in files.items()}
        _DATA[key] = {k: (torch.tensor(np.asarray(fs.rx_c)), torch.tensor(np.asarray(fs.tx_c))) for k, fs in sets.items()}
    return _DATA[key]


def bit_errors(y: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
    """y (n, c) predictions, tx (n,) -> bit errors per column (c,)."""
    true = BITS[(tx[:, None] - POINTS).abs().argmin(-1)]
    out = []
    for start in range(0, y.shape[1], 16):                  # columns in chunks (memory)
        pred = (y[:, start:start + 16, None] - POINTS).abs().argmin(-1)
        out.append((BITS[pred] != true[:, None, :]).sum(dim=(0, 2)))
    return torch.cat(out)


def linear_features(x: torch.Tensor, L: int) -> torch.Tensor:
    from .kerr_offaxis import shift

    half = L // 2
    return torch.stack([shift(x, l) for l in range(-half, half + 1)], dim=-1)


def job_terms(arch: str, M: int, domain: str = "T3") -> list[tuple[int, int]]:
    """domain "box" (E051): off-axis terms of max(|m|, |n|) <= M for A1 / A3 / LK (the cross rule domain)."""
    if domain == "box" and arch in ("A1", "A3", "LK"):
        return offaxis_box(M)
    if arch == "K":
        return []
    return triplet_terms(M, "T3") if arch in ("T", "A3full") else offaxis_terms(M)


def run_job(arch: str, mode: str, M: int, n_c: int, replica: int, config: dict, smoke: bool) -> pd.DataFrame:
    torch.set_num_threads(1)
    d = data(smoke)
    xtr, str_ = d["train"]
    n_files = xtr.shape[0]
    rng = np.random.default_rng(replica)
    draw = rng.integers(0, n_files, n_files)
    counts = np.bincount(draw, minlength=n_files)
    sub = list(dict.fromkeys(draw.tolist()))[: config["n_sub"]]
    terms = job_terms(arch, M, config.get("domain", "T3"))
    kerr = Kerr(config["fir_len"], n_c)
    theta, c_a2 = (None, None)
    lam_fit = config["theta_lambda"]
    if arch != "T":
        x_sub, s_sub = xtr[sub], str_[sub]
        if arch == "A2":
            # C of the raw residual on the subsample, then the Kerr on x + T(x) C (sequential by definition)
            theta, c_a2 = fit_theta("A2", "seq", x_sub, s_sub, M, config["fir_len"], n_c, config["lbfgs_iters"], lam_fit)
        else:
            theta, _ = fit_theta(arch if arch != "K" else "A3", mode if arch != "K" else "seq", x_sub, s_sub, M,
                                 config["fir_len"], n_c, config["lbfgs_iters"], lam_fit,
                                 terms if arch in ("A1", "A3", "LK") else None)
    sl = slice(EDGE + M, xtr.shape[1] - EDGE - M)

    def parts(x):
        """(base, A): prediction = base + A @ C (A2: A on x, Kerr applied afterwards)."""
        if arch == "T":
            A = torch.cat((features(x, terms), linear_features(x, FIR)), dim=-1)
            return torch.zeros_like(x), A
        if arch == "K":
            return kerr.apply(theta, x)[1], torch.zeros(x.shape + (0,), dtype=x.dtype)
        if arch == "A1":
            y = kerr.apply(theta, x)[1]
            return y, features(y, terms)
        if arch in ("A3", "A3full"):
            return kerr.apply(theta, x)[1], features(x, terms)
        if arch == "LK":                                      # E051: (u + T(u) C) exp(-i phi(u))
            u, rot = kerr.rotation(theta, x)
            return u * rot, features(u, terms) * rot.unsqueeze(-1)
        return x, features(x, terms)                          # A2: correction first, Kerr later

    p = len(terms) + (FIR if arch == "T" else 0)
    G = torch.zeros((p, p), dtype=torch.complex128)
    b = torch.zeros(p, dtype=torch.complex128)
    diag = torch.zeros(p, dtype=torch.float64)
    n_sym = 0
    for f in range(n_files):
        if counts[f] == 0 or p == 0:
            continue
        base, A = parts(xtr[f:f + 1])
        A, r = A[0, sl], (str_[f, sl] - base[0, sl])
        G += counts[f] * (A.conj().T @ A)
        b += counts[f] * (A.conj().T @ r)
        diag += counts[f] * A.abs().square().sum(0)
        n_sym += counts[f] * A.shape[0]

    def solve(idx, lam):
        if not idx:
            return torch.zeros(p, dtype=torch.complex128)
        ii = torch.tensor(idx)
        g = G[ii][:, ii]
        scale = torch.real(torch.diagonal(g)).mean()
        c = torch.linalg.solve(g + lam * scale * torch.eye(len(idx), dtype=g.dtype), b[ii])
        out = torch.zeros(p, dtype=torch.complex128)
        out[ii] = c
        return out

    all_idx = list(range(p))
    trip_idx = list(range(len(terms)))
    extra = list(range(len(terms), p))
    full = [solve(all_idx, lam) for lam in LAMBDAS]
    sets = [("full", len(terms), all_idx)]
    if p:
        # magnitude pruning uses the validation-free fit at the middle lambda for the ranking (train only)
        rank_c = full[2]
        score = (rank_c.abs() * torch.sqrt(diag / max(n_sym, 1)))[: len(terms)]
        order = torch.argsort(score, descending=True).tolist()
        for k in config["K"]:
            if k < len(terms):
                sets.append((f"mag_K{k}", k, sorted(order[:k]) + extra))
        for k_th in config["K_th"]:
            sel = [i for i, (m, n) in enumerate(terms) if abs(m * n) <= k_th]
            if 0 < len(sel) < len(terms):
                sets.append((f"geo_Kth{k_th}", len(sel), sel + extra))
        for k in config.get("cross_K", []):                   # E051: Xiao cross min(|m|, |n|) <= K inside the box
            sel = [i for i, (m, n) in enumerate(terms) if min(abs(m), abs(n)) <= k]
            if sel:
                sets.append((f"cross_K{k}", len(sel), sel + extra))
    sets.append(("none", 0, extra))                            # K = 0: Kerr alone (or FIR alone for T)
    coefs = torch.stack([solve(idx, lam) for _, _, idx in sets for lam in LAMBDAS], dim=1)
    result = {}
    for split in ("validation", "test"):
        x, tx = d[split]
        errors = torch.zeros(coefs.shape[1], dtype=torch.int64)
        n_bits = 0
        for f in range(x.shape[0]):
            base, A = parts(x[f:f + 1])
            keep = slice(EDGE, x.shape[1] - EDGE)
            if arch == "A2":
                corr = x[f:f + 1].unsqueeze(-1) + (A @ coefs)
                y = torch.stack([kerr.apply(theta, corr[..., j])[1][0, keep] for j in range(coefs.shape[1])], dim=1)
            elif p:
                y = base[0, keep, None] + A[0, keep] @ coefs
            else:
                y = base[0, keep, None].expand(-1, coefs.shape[1])
            errors += bit_errors(y, tx[f, keep])
            n_bits += 4 * (x.shape[1] - 2 * EDGE)
        result[split] = errors.double() / n_bits
    raw = {k: bit_errors(d[k][0][:, EDGE:-EDGE].reshape(-1, 1), d[k][1][:, EDGE:-EDGE].reshape(-1)).item()
           / (4 * d[k][0][:, EDGE:-EDGE].numel()) for k in ("validation", "test")}
    rows = []
    n_l = len(LAMBDAS)
    for si, (name, n_terms, _) in enumerate(sets):
        val = result["validation"][si * n_l:(si + 1) * n_l]
        best = int(torch.argmin(val))
        vb, tb = float(val[best]), float(result["test"][si * n_l + best])
        kerr_part = 0.0 if arch == "T" else kerr_rm(n_c, config["fir_len"])
        rm = 12 * n_terms + (4 * FIR if arch == "T" else 0) + kerr_part
        rows.append({"arch": arch, "mode": mode, "M": M, "n_c": n_c if arch != "T" else 0, "replica": replica, "set": name,
                     "n_terms": n_terms, "rm": rm, "lambda": LAMBDAS[best], "val_ber": vb, "test_ber": tb,
                     "val_reduction": 100 * (raw["validation"] - vb) / raw["validation"],
                     "test_reduction": 100 * (raw["test"] - tb) / raw["test"], "q2_gain_db": q2_db(tb) - q2_db(raw["test"])})
    if theta is not None and SPLIT["seed"] is None:
        np.savez(results_root(EXP["id"], smoke) / "fits" / f"theta_{arch}_{mode}_M{M}_nc{n_c}_r{replica}.npz",
                 theta=theta.numpy(), coef=coefs[:, 2].numpy(), terms=np.array(terms))
    return pd.DataFrame(rows)


def jobs(config: dict, smoke: bool) -> list[tuple]:
    if "arch_modes" in config:                       # E048: explicit (arch, mode) list and controls
        Ms = config["smoke_M"] if smoke else config["M"]
        ncs = config["smoke_nc"] if smoke else config["n_c"]
        out = []
        for r in (config["replicas"][:1] if smoke else config["replicas"]):
            for M in Ms:
                if not config.get("skip_T"):          # E056: T already fitted in E046 / E048
                    out.append(("T", "closed", M, 0, r))
                for nc in ncs:
                    out += [(arch, mode, M, nc, r) for arch, mode in config["arch_modes"]]
            out += [("K", "seq", Ms[0], nc, r) for nc in ncs]
        return out
    Ms = config["smoke_M"] if smoke else config["M"]
    ncs = config["smoke_nc"] if smoke else config["n_c"]
    reps = config["replicas"][:1] if smoke else config["replicas"]
    out = []
    for r in reps:
        for M in Ms:
            out.append(("T", "closed", M, 0, r))
            for nc in ncs:
                if M == Ms[0]:
                    out.append(("K", "seq", M, nc, r))
                for arch in ("A1", "A3"):
                    for mode in ("varpro", "seq"):
                        out.append((arch, mode, M, nc, r))
                out.append(("A2", "seq", M, nc, r))
                out.append(("A3full", "varpro", M, nc, r))
    return out


def job_id(j) -> str:
    return "{}_{}_M{}_nc{}_r{}".format(*j)


def _one(j, config, smoke, split=None, exp=None):
    if exp:
        EXP["id"] = exp
    SPLIT["seed"] = split
    folder = "fits" if split is None else f"fits_split{split}"
    out = results_root(EXP["id"], smoke) / folder / f"{job_id(j)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        return job_id(j), "exists"
    frame = run_job(*j, config, smoke)
    frame["split_seed"] = split if split is not None else "E004"
    frame["git_commit"] = git_commit()
    frame.to_csv(out, index=False)
    return job_id(j), "ok"


def pool_run(todo: list, config: dict, smoke: bool, label: str) -> int:
    """Run jobs in processes; with JOBS_ADAPTIVE=1 the number in flight follows the free cores."""
    from .gate import MAX_JOBS, adaptive_limit
    from .runner import env_flag

    adaptive = env_flag("JOBS_ADAPTIVE")
    fixed = int(os.environ["JOBS"]) if os.environ.get("JOBS", "auto").isdigit() else 1
    pending, running, done = list(todo), {}, 0
    with cf.ProcessPoolExecutor(max_workers=MAX_JOBS if adaptive else max(1, fixed)) as pool:
        while pending or running:
            limit = adaptive_limit(len(running)) if adaptive else max(1, fixed)
            while pending and len(running) < limit:
                j, seed = pending.pop(0)
                running[pool.submit(_one, j, config, smoke, seed, EXP["id"])] = j
            finished, _ = cf.wait(running, timeout=60, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                running.pop(fut)
                done += 1
                try:
                    print(f"[{done}/{len(todo)}] {label}: {fut.result()} (в работе {len(running)}, предел {limit})", flush=True)
                except Exception as exc:          # keep going; the summary reports missing jobs
                    print(f"[{done}/{len(todo)}] ОШИБКА: {exc!r}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    config = json.loads((exp_dir(EXP["id"]) / "config.json").read_text(encoding="utf-8"))
    out = results_root(EXP["id"], smoke)
    (out / "fits").mkdir(parents=True, exist_ok=True)
    action = argv[0] if argv else "all"
    if action == "one":
        arch, mode, M, nc, r = argv[1], argv[2], int(argv[3]), int(argv[4]), int(argv[5])
        print(_one((arch, mode, M, nc, r), config, smoke))
        return 0
    if action == "all":
        return pool_run([(j, None) for j in jobs(config, smoke)], config, smoke, "fit")
    if action == "splits":           # best configuration by validation of each arch on the three E023 splits
        summary = pd.read_csv(out / "a_summary.csv")
        best = summary[summary.set == "full"].loc[lambda f: f.groupby("arch").val_mean.idxmax()]
        reps = config["replicas"][:1] if smoke else config["replicas"]
        todo = [((r.arch, r["mode"], int(r.M), int(r.n_c), rep), seed) for _, r in best.iterrows() for rep in reps
                for seed in (config["split_seeds"][:1] if smoke else config["split_seeds"])]
        return pool_run(todo, config, smoke, "split")
    if action == "summarize":
        from .summarize_E046 import summarize

        return summarize(config, smoke)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
