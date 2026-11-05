"""E053 task 1b/1c: S chirp features instead of the off-axis coefficient table, on the data (A1 pipeline of E046).

    PYTHONPATH=src python -m constellation_denoiser.e053_data all | splits | summarize      (SMOKE=1: tiny)

Per replica r (bootstrap of the 30 train files as E046) and base (A1 varpro M8 N_c31 of E046, A1 varpro M11 N_c63 of
E048): the Kerr part (FIR7, alpha) is the saved theta of that replica (fixed); y = Kerr(x), merged off-axis
features T_mn(y) (kerr_offaxis.features, T3 domain). Every variant is a linear parametrization of the merged
coefficients c = Phi theta, so it is fitted from the Gram matrix G = F^H F and b = F^H (s - y) of the 30 train files
(bootstrap counts): theta = (Phi^H G Phi + lam scale I)^-1 Phi^H b, lam in the E046 grid, chosen by validation BER.
  full    (i)  Phi = I (A1 with the full table; reproduces E046)
  none    (ii) Kerr alone
  chirp   S chirps, Phi_s(m, n) = mult exp(i m n / eta_s) T_s(|m| + |n|), mult = 2 for m != n (merged), A_s complex
          ("complex") or A_s = i a_s ("imag"); (eta_s, W_s) by greedy grid + joint L-BFGS on 4 files (variable
          projection in the Gram space of the 4 files, as theta in E046)
  chirp_joint  S in (3, 5), imag: then (FIR, alpha, eta, W) jointly by L-BFGS on the 4 files (A eliminated), A on 30
  random  (iv) S chirps with random eta (log-uniform |eta| in [1, 50], random sign) and W in [2, 4M], no tuning
  radial  (iii) K hat functions of r = sqrt(m^2 + n^2) (knots uniform in [1, sqrt2 M]), complex weights (2K real)
  lowrank (iii) ordered C = sum_{j<=r} u_j v_j^T with C(m,n) = C(n,m) enforced by symmetrizing, by alternating ridge
Writes results/runs/<base>_r<replica>[_split<seed>].csv (one row per variant x S), then summarize.
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
from scipy.optimize import minimize

from .runner import exp_dir, results_root

BASES = {"A1_M8_nc31": ("E046", 8, 31), "A1_M11_nc63": ("E048", 11, 63)}
S_LIST = (1, 2, 3, 4, 5, 6, 8)
ETA_GRID = np.concatenate((-np.logspace(-0.3, 1.8, 30)[::-1], np.logspace(-0.3, 1.8, 30)))
W_GRID = np.logspace(0.3, 1.7, 12)
RADIAL_K = (2, 3, 5, 8, 12)
LOWRANK_R = (1, 2, 3)
N_SUB = 4


def merged_terms(M):
    from .kerr_offaxis import offaxis_terms

    return offaxis_terms(M)


def chirp_phi(terms, etas, Ws, xp=np):
    m = xp.asarray([t[0] for t in terms], dtype=xp.float64 if xp is np else torch.float64)
    n = xp.asarray([t[1] for t in terms], dtype=m.dtype)
    mult = xp.where(m == n, 1.0, 2.0) if xp is np else torch.where(m == n, 1.0, 2.0).double()
    k = xp.abs(m) + xp.abs(n)
    cols = []
    for e, w in zip(etas, Ws):
        tri = xp.clip(1 - k / w, 0, None) if xp is np else torch.clamp(1 - k / w, min=0)
        cols.append(mult * tri * xp.exp(1j * m * n / e))
    return xp.stack(cols, axis=1) if xp is np else torch.stack(cols, dim=1)


def radial_phi(terms, K, M):
    r = np.array([np.hypot(*t) for t in terms])
    mult = np.array([1.0 if a == b else 2.0 for a, b in terms])
    knots = np.linspace(1, np.sqrt(2) * M, K)
    h = knots[1] - knots[0] if K > 1 else 1.0
    return (mult[:, None] * np.clip(1 - np.abs(r[:, None] - knots[None, :]) / h, 0, None)).astype(complex)


def solve(G, b, Phi, lam, imag=False):
    A = Phi.conj().T @ G @ Phi
    rhs = Phi.conj().T @ b
    scale = np.real(np.trace(A)) / max(A.shape[0], 1)
    if imag:          # theta = i a, a real: minimize a^T Re(Phi'^H G Phi') a - 2 a^T Re(Phi'^H b), Phi' = i Phi
        Pi = 1j * Phi
        Ar, rr = np.real(Pi.conj().T @ G @ Pi), np.real(Pi.conj().T @ b)
        reg = (lam * scale + 1e-12 * (1 + abs(scale))) * np.eye(len(rr))
        return Pi @ np.linalg.lstsq(Ar + reg, rr, rcond=None)[0]
    reg = (lam * scale + 1e-12 * (1 + abs(scale))) * np.eye(A.shape[0])
    return Phi @ np.linalg.lstsq(A + reg, rhs, rcond=None)[0]


def gram_loss(G, b, Phi, imag) -> float:
    """-(explained energy) with theta eliminated (small ridge for stability)."""
    c = solve(G, b, Phi, 1e-8, imag)
    return float(np.real(c.conj() @ G @ c) - 2 * np.real(c.conj() @ b))


def fit_chirps(G4, b4, terms, S_max, imag, M) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Greedy + joint L-BFGS; returns [(etas, Ws, n_evaluations)] for S = 1..S_max."""
    etas, logw, out, evals = [], [], [], 0
    for S in range(1, S_max + 1):
        best = None
        for e in ETA_GRID:
            for w in W_GRID:
                Phi = chirp_phi(terms, etas + [e], list(np.exp(logw)) + [w])
                val = gram_loss(G4, b4, Phi, imag)
                evals += 1
                if best is None or val < best[0]:
                    best = (val, e, np.log(w))
        x0 = np.array(etas + [best[1]] + logw + [best[2]])

        def f(p):
            if np.any(np.abs(p[:S]) < 0.2):
                return 1e30
            return gram_loss(G4, b4, chirp_phi(terms, p[:S], np.exp(p[S:])), imag)

        res = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 200})
        evals += res.nfev
        p = res.x if res.fun <= best[0] else x0
        etas, logw = list(p[:S]), list(p[S:])
        out.append((np.array(etas), np.exp(np.array(logw)), evals))
    return out


def lowrank_c(G, b, terms, r, lam, iters=15):
    """Ordered symmetric C = sum_j u_j v_j^T (then symmetrized), merged c from it; alternating ridge in Gram space."""
    idx = sorted({v for t in terms for v in t})
    pos = {v: i for i, v in enumerate(idx)}
    Q = len(idx)
    mult = np.array([1.0 if a == b_ else 2.0 for a, b_ in terms])
    rng = np.random.default_rng(0)
    U = rng.standard_normal((Q, r)) * 0.01 + 0j
    V = rng.standard_normal((Q, r)) * 0.01 + 0j

    def c_of(U, V):
        Cfull = U @ V.T
        Cs = (Cfull + Cfull.T) / 2
        return np.array([Cs[pos[a], pos[b_]] for a, b_ in terms]) * mult

    for _ in range(iters):
        # linear in U for fixed V: c_t = mult_t (U[a] V[b] + U[b] V[a]) / 2
        Phi = np.zeros((len(terms), Q * r), complex)
        for t, (a, b_) in enumerate(terms):
            for j in range(r):
                Phi[t, pos[a] * r + j] += mult[t] * V[pos[b_], j] / 2
                Phi[t, pos[b_] * r + j] += mult[t] * V[pos[a], j] / 2
        A = Phi.conj().T @ G @ Phi
        scale = np.real(np.trace(A)) / A.shape[0]
        u = np.linalg.lstsq(A + (lam * scale + 1e-12) * np.eye(A.shape[0]), Phi.conj().T @ b, rcond=None)[0]
        U = u.reshape(Q, r)
        U, V = V, U                                   # alternate (symmetric roles)
    return c_of(U, V), 2 * 2 * Q * r


def run(base: str, replica: int, split: int | None, smoke: bool) -> pd.DataFrame:
    from . import e046_fit
    from .kerr_offaxis import EDGE, LAMBDAS, Kerr, features, fit_theta

    torch.set_num_threads(1)
    exp, M, nc = BASES[base]
    if smoke:
        M, nc = 3, 7
    e046_fit.SPLIT["seed"] = split
    e046_fit._DATA.clear()
    d = e046_fit.data(smoke)
    xtr, str_ = d["train"]
    n_files = xtr.shape[0]
    rng = np.random.default_rng(replica)
    draw = rng.integers(0, n_files, n_files)
    counts = np.bincount(draw, minlength=n_files)
    sub = list(dict.fromkeys(draw.tolist()))[:N_SUB]
    terms = merged_terms(M)
    kerr = Kerr(7, nc)
    t0 = time.time()
    saved = exp_dir(exp) / "results" / "fits" / f"theta_A1_varpro_M{M}_nc{nc}_r{replica}.npz"
    if split is None and saved.is_file() and not smoke:
        theta = torch.tensor(np.load(saved)["theta"])
        theta_source = str(saved.relative_to(exp_dir(exp).parent.parent))
    else:
        theta, _ = fit_theta("A1", "varpro", xtr[sub], str_[sub], M, 7, nc, 60, 1e-4)
        theta_source = "refit (A1 varpro, as E046)"
    sl = slice(EDGE + M, xtr.shape[1] - EDGE - M)

    def gram(files, th):
        p = len(terms)
        G, b = np.zeros((p, p), complex), np.zeros(p, complex)
        for f, w in files:
            y = kerr.apply(th, xtr[f:f + 1])[1]
            A = features(y, terms)[0, sl].numpy()
            r = (str_[f, sl] - y[0, sl]).numpy()
            G += w * (A.conj().T @ A)
            b += w * (A.conj().T @ r)
        return G, b

    G, b = gram([(f, counts[f]) for f in range(n_files) if counts[f]], theta)
    G4, b4 = gram([(f, 1) for f in sub], theta)
    variants = []                                    # (name, S_or_K, n_real_params, c (merged), tuning_evals)
    eye = np.eye(len(terms), dtype=complex)
    for lam in LAMBDAS:
        variants.append(("full", len(terms), 2 * len(terms), solve(G, b, eye, lam), 0, lam))
    variants.append(("none", 0, 0, np.zeros(len(terms), complex), 0, 0.0))
    s_max = 3 if smoke else max(S_LIST)
    for amp in ("complex", "imag"):
        imag = amp == "imag"
        for S, (etas, Ws, ev) in enumerate(fit_chirps(G4, b4, terms, s_max, imag, M), start=1):
            if S not in S_LIST:
                continue
            Phi = chirp_phi(terms, etas, Ws)
            for lam in LAMBDAS:
                variants.append((f"chirp_{amp}", S, (3 if imag else 4) * S, solve(G, b, Phi, lam, imag), ev, lam,
                                 json.dumps({"eta": np.round(etas, 3).tolist(), "W": np.round(Ws, 2).tolist()})))
    rr = np.random.default_rng(1000 + replica)
    for S in S_LIST[:3] if smoke else S_LIST:
        etas = np.exp(rr.uniform(0, np.log(50), S)) * rr.choice([-1, 1], S)
        Ws = rr.uniform(2, 4 * M, S)
        Phi = chirp_phi(terms, etas, Ws)
        for lam in LAMBDAS:
            variants.append(("random_imag", S, 3 * S, solve(G, b, Phi, lam, True), 0, lam,
                             json.dumps({"eta": np.round(etas, 3).tolist(), "W": np.round(Ws, 2).tolist()})))
    for K in RADIAL_K:
        Phi = radial_phi(terms, K, M)
        for lam in LAMBDAS:
            variants.append(("radial", K, 2 * K, solve(G, b, Phi, lam), 0, lam))
    for r in LOWRANK_R[:1] if smoke else LOWRANK_R:
        for lam in LAMBDAS:
            c, npar = lowrank_c(G, b, terms, r, lam)
            variants.append(("lowrank", r, npar, c, 15, lam))
    # joint (FIR, alpha, eta, W) on the 4 files for S = 3, 5 (imag), A on the 30 files
    for S in ((2,) if smoke else (3, 5)):
        start = [v for v in variants if v[0] == "chirp_imag" and v[1] == S][0]
        par = json.loads(start[6])
        th = theta.clone().requires_grad_(True)
        eta = torch.tensor(par["eta"], dtype=torch.float64, requires_grad=True)
        lw = torch.tensor(np.log(par["W"]), dtype=torch.float64, requires_grad=True)
        xs, ss = xtr[sub], str_[sub]

        def closure():
            opt.zero_grad()
            y = kerr.apply(th, xs)[1]
            F = (features(y, terms)[:, sl].reshape(-1, len(terms)) @ chirp_phi(terms, eta, torch.exp(lw), torch)) * 1j
            r = (ss - y)[:, sl].reshape(-1)
            Fr = torch.cat((F.real, F.imag))
            rr_ = torch.cat((r.real, r.imag))
            a = torch.linalg.solve(Fr.T @ Fr + 1e-8 * torch.eye(S, dtype=Fr.dtype), Fr.T @ rr_)
            loss = (rr_ - Fr @ a).square().mean()
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([th, eta, lw], lr=1.0, max_iter=8 if smoke else 60, history_size=20,
                                line_search_fn="strong_wolfe")
        opt.step(closure)
        th_j = th.detach()
        Gj, bj = gram([(f, counts[f]) for f in range(n_files) if counts[f]], th_j)
        Phi = chirp_phi(terms, eta.detach().numpy(), np.exp(lw.detach().numpy()))
        for lam in LAMBDAS:
            variants.append(("chirp_joint_imag", S, 3 * S, solve(Gj, bj, Phi, lam, True), start[4] + 60, lam,
                             json.dumps({"eta": np.round(eta.detach().numpy(), 3).tolist(),
                                         "W": np.round(np.exp(lw.detach().numpy()), 2).tolist()}), th_j))
    fit_time = time.time() - t0
    # evaluation: validation and test BER for every variant (theta of the variant)
    C = {}
    for v in variants:
        th_v = v[7] if len(v) > 7 else theta
        C.setdefault(id(th_v), (th_v, []))[1].append(v)
    errs = {}
    raw = {}
    for split_name in ("validation", "test"):
        x, tx = d[split_name]
        keep = slice(EDGE, x.shape[1] - EDGE)
        raw[split_name] = e046_fit.bit_errors(x[:, keep].reshape(-1, 1), tx[:, keep].reshape(-1)).item() / (
            4 * x[:, keep].numel())
        for key, (th_v, vs) in C.items():
            coefs = torch.tensor(np.stack([v[3] for v in vs], axis=1))
            e = torch.zeros(coefs.shape[1], dtype=torch.int64)
            nb = 0
            for f in range(x.shape[0]):
                y = kerr.apply(th_v, x[f:f + 1])[1]
                pred = y[0, keep, None] + features(y, terms)[0, keep] @ coefs
                e += e046_fit.bit_errors(pred, tx[f, keep])
                nb += 4 * (x.shape[1] - 2 * EDGE)
            for v, err in zip(vs, (e.double() / nb).tolist()):
                errs[(id(v), split_name)] = err
    from .q2 import q2_db

    rows = []
    for v in variants:
        vb, tb = errs[(id(v), "validation")], errs[(id(v), "test")]
        rows.append({"base": base, "replica": replica, "split_seed": split if split is not None else "E004",
                     "variant": v[0], "size": v[1], "n_real_params": v[2], "lambda": v[5], "tuning_evals": v[4],
                     "params": v[6] if len(v) > 6 else "", "val_ber": vb, "test_ber": tb,
                     "val_reduction": 100 * (raw["validation"] - vb) / raw["validation"],
                     "test_reduction": 100 * (raw["test"] - tb) / raw["test"],
                     "q2_gain_db": q2_db(tb) - q2_db(raw["test"]), "theta_source": theta_source, "fit_time_s": fit_time})
    frame = pd.DataFrame(rows)
    # lambda by validation per (variant, size)
    frame = frame.loc[frame.groupby(["variant", "size"]).val_ber.idxmin()].reset_index(drop=True)
    return frame


def _job(args):
    base, replica, split, smoke = args
    out = results_root("E053", smoke) / "runs" / f"{base}_r{replica}{'' if split is None else f'_split{split}'}.csv"
    if out.is_file():
        return f"{out.name}: exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = run(base, replica, split, smoke)
    frame.to_csv(out, index=False)
    return f"{out.name}: ok ({len(frame)} rows)"


def pool(todo, smoke) -> int:
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from .gate import MAX_JOBS, adaptive_limit

    pending, running, done = list(todo), set(), 0
    with ProcessPoolExecutor(max_workers=1 if smoke else MAX_JOBS) as ex:
        while pending or running:
            limit = 1 if smoke else adaptive_limit(len(running))
            while pending and len(running) < limit:
                running.add(ex.submit(_job, pending.pop(0)))
            finished, running = wait(running, timeout=60, return_when=FIRST_COMPLETED)
            running = set(running)
            for fut in finished:
                done += 1
                try:
                    msg = fut.result()
                except Exception as exc:
                    msg = f"ОШИБКА: {exc!r}"
                print(f"[{time.strftime('%F %T')}] [{done}/{len(todo)}] {msg} (в работе {len(running)}, предел {limit})",
                      flush=True)
    return 0


REPLICAS = (17, 29, 43, 57, 71)
SPLITS = (101, 202, 303)


def main(argv) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    action = argv[0] if argv else ""
    bases = list(BASES)[:1] if smoke else list(BASES)
    reps = REPLICAS[:1] if smoke else REPLICAS
    if action == "all":
        return pool([(b, r, None, smoke) for b in bases for r in reps], smoke)
    if action == "splits":
        return pool([(bases[0], r, s, smoke) for r in reps for s in (SPLITS[:1] if smoke else SPLITS)], smoke)
    if action == "summarize":
        from .summarize_E053 import main as summ

        return summ(smoke)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
