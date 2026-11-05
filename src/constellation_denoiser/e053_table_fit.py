"""E053 task 1a: fit a sum of S chirps to existing ridge tables of off-axis coefficients C (no data, no training).

    PYTHONPATH=src python -m constellation_denoiser.e053_table_fit

Model on ordered off-axis pairs (m, n) of the table's domain:
    C_fit(m, n) = sum_s A_s exp(i m n / eta_s) T_s(|m| + |n|),  T_s(k) = max(1 - k / W_s, 0).
The tables store merged features (m <= n): coefficient of x_{p+m} x_{p+n} conj(x_{p+m+n}) counted once, so the
ordered C(m, n) = coef / 2 for m != n and coef for m = n (kerr_offaxis.features; E046 / E048 / E039).
For fixed (eta, W) the amplitudes are linear: complex least squares ("complex") or A_s = i a_s with real a_s
("imag", the symmetry C(m,n) = -conj C(-m,n)). Non-linear parameters: greedy (a chirp is added to the residual from a
grid of (eta, W)), then joint L-BFGS (scipy, numerical gradient) over all (eta_s, log W_s) with the amplitudes
eliminated (variable projection). W modes: "free", "2pi" (W = 2 pi |eta|), "pi" (W = pi |eta|).
Error: 20 log10(||C - C_fit||_F / ||C||_F) over ordered pairs, also for |m|, |n| <= 3 ("small") and the rest ("large").
Tables: middle lambda (1e-4) of the saved fits (coef index 2), not the validation-best lambda; replica r17.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .runner import exp_dir

TABLES = {
    "A1_M8_nc31 (E046)": ("E046", "fits/theta_A1_varpro_M8_nc31_r17.npz"),
    "A1_M11_nc63 (E048)": ("E048", "fits/theta_A1_varpro_M11_nc63_r17.npz"),
    "A1_M15_nc63 (E048)": ("E048", "fits/theta_A1_varpro_M15_nc63_r17.npz"),
    "T3_M21 (E039, ridge T3 + FIR7, not A1)": ("E039", "coef_M21.csv"),
}
S_LIST = (1, 2, 3, 4, 5, 6, 8)
ETA_GRID = np.concatenate((-np.logspace(-0.7, 2.7, 48)[::-1], np.logspace(-0.7, 2.7, 48)))
W_GRID = np.logspace(0.3, 3.0, 24)
ETA_PHYS_FULL = 21e-27 * 2e6 * (32e9) ** 2     # |beta2| L Rs^2 = 43 (из литературы, справочно)


def load_table(exp: str, rel: str) -> tuple[np.ndarray, np.ndarray]:
    """(pairs (P, 2) ordered off-axis, C (P,) ordered)."""
    path = exp_dir(exp) / "results" / rel
    if path.suffix == ".npz":
        z = np.load(path)
        terms, coef = z["terms"], z["coef"]
    else:
        t = pd.read_csv(path)
        t = t[(t.kind == "triplet")]
        terms, coef = t[["m", "n"]].to_numpy().astype(int), (t.re + 1j * t.im).to_numpy()
    pairs, vals = [], []
    for (m, n), c in zip(terms, coef):
        if m == 0 or n == 0:
            continue
        if m == n:
            pairs.append((m, n))
            vals.append(c)
        else:
            pairs += [(m, n), (n, m)]
            vals += [c / 2, c / 2]
    return np.array(pairs), np.array(vals)


def basis(pairs: np.ndarray, eta: float, W: float) -> np.ndarray:
    m, n = pairs[:, 0], pairs[:, 1]
    return np.exp(1j * m * n / eta) * np.maximum(1 - (np.abs(m) + np.abs(n)) / W, 0.0)


def width(eta: float, w: float, mode: str) -> float:
    return {"free": w, "2pi": 2 * np.pi * abs(eta), "pi": np.pi * abs(eta)}[mode]


def amplitudes(B: np.ndarray, C: np.ndarray, amp: str) -> np.ndarray:
    if amp == "complex":
        return np.linalg.lstsq(B, C, rcond=None)[0]
    Bi = 1j * B                                           # C ~ (i B) a, a real: stack real and imaginary parts
    A = np.concatenate((Bi.real, Bi.imag))
    a = np.linalg.lstsq(A, np.concatenate((C.real, C.imag)), rcond=None)[0]
    return 1j * a


def residual(params: np.ndarray, pairs, C, amp, mode) -> tuple[float, np.ndarray, np.ndarray]:
    S = len(params) // 2
    etas, logw = params[:S], params[S:]
    if np.any(np.abs(etas) < 0.05):
        return np.inf, None, None
    B = np.stack([basis(pairs, e, width(e, np.exp(lw), mode)) for e, lw in zip(etas, logw)], axis=1)
    A = amplitudes(B, C, amp)
    r = C - B @ A
    return float(np.vdot(r, r).real), A, r


def fit(pairs, C, S_max, amp, mode) -> list[dict]:
    params = np.zeros(0)
    out = []
    for S in range(1, S_max + 1):
        best = None
        w_grid = W_GRID if mode == "free" else (1.0,)
        for e in ETA_GRID:
            for w in w_grid:
                trial = np.concatenate((params[:S - 1], [e], params[S - 1:], [np.log(w)]))
                val = residual(trial, pairs, C, amp, mode)[0]
                if best is None or val < best[0]:
                    best = (val, trial)
        res = minimize(lambda p: residual(p, pairs, C, amp, mode)[0], best[1], method="L-BFGS-B",
                       options={"maxiter": 400})
        params = res.x if res.fun <= best[0] else best[1]
        val, A, r = residual(params, pairs, C, amp, mode)
        out.append({"S": S, "params": params.copy(), "A": A, "r": r, "sse": val})
    return out


def region_err(pairs, C, r, mask) -> float:
    return float(20 * np.log10(np.linalg.norm(r[mask]) / np.linalg.norm(C[mask]))) if mask.any() else np.nan


def main() -> int:
    out_dir = exp_dir("E053") / "results"
    fig_dir = exp_dir("E053") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, keep = [], {}
    for name, (exp, rel) in TABLES.items():
        pairs, C = load_table(exp, rel)
        small = (np.abs(pairs) <= 3).all(axis=1)
        M = int(np.abs(pairs).max())
        sym = {"rel_dev_C(-m,n)=-conj C(m,n)": None}
        lookup = {tuple(p): c for p, c in zip(pairs, C)}
        dev = [abs(lookup[(-m, n)] + np.conj(c)) / (abs(c) + abs(lookup[(-m, n)]) + 1e-30)
               for (m, n), c in zip(pairs, C) if (-m, n) in lookup]
        for amp in ("complex", "imag"):
            for mode in ("free", "2pi", "pi"):
                for f in fit(pairs, C, max(S_LIST), amp, mode):
                    if f["S"] not in S_LIST:
                        continue
                    S = f["S"]
                    etas, logw = f["params"][:S], f["params"][S:]
                    Ws = [width(e, np.exp(lw), mode) for e, lw in zip(etas, logw)]
                    rows.append({"table": name, "M": M, "n_pairs": len(C), "amp": amp, "W_mode": mode, "S": S,
                                 "n_real_params": (3 if amp == "complex" else 2) * S + (S if mode == "free" else 0) - (0 if mode == "free" else 0),
                                 "err_db": float(10 * np.log10(f["sse"] / np.vdot(C, C).real)),
                                 "err_small_db": region_err(pairs, C, f["r"], small),
                                 "err_large_db": region_err(pairs, C, f["r"], ~small),
                                 "eta": json.dumps([round(float(e), 3) for e in etas]),
                                 "W": json.dumps([round(float(w), 2) for w in Ws]),
                                 "W_over_2pi_eta": json.dumps([round(float(w / (2 * np.pi * abs(e))), 3) for e, w in zip(etas, Ws)]),
                                 "A": json.dumps([[round(float(a.real), 6), round(float(a.imag), 6)] for a in f["A"]]),
                                 "A_phase_deg": json.dumps([round(float(np.degrees(np.angle(a))), 1) for a in f["A"]]),
                                 "sym_dev_median": float(np.median(dev))})
                    if amp == "complex" and mode == "free" and S in (1, 3, 8):
                        keep[(name, S)] = (pairs, C, C - f["r"])
                print(f"{name} {amp} {mode}: " + ", ".join(f"S{r['S']} {r['err_db']:.1f}" for r in rows
                      if r["table"] == name and r["amp"] == amp and r["W_mode"] == mode), flush=True)
    table = pd.DataFrame(rows)
    table["n_real_params"] = [(2 if a == "complex" else 1) * s + (2 * s if m == "free" else s)
                              for a, m, s in zip(table.amp, table.W_mode, table.S)]
    table.to_csv(out_dir / "table_fit.csv", index=False)
    heatmaps(keep, fig_dir)
    return 0


def heatmaps(keep, fig_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted({k[0] for k in keep})
    for name in names:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        pairs, C, _ = keep[(name, 1)]
        M = int(np.abs(pairs).max())
        grids = [("таблица", C)] + [(f"S = {S}", keep[(name, S)][2]) for S in (1, 3, 8)]
        for col, (title, vals) in enumerate(grids):
            G = np.full((2 * M + 1, 2 * M + 1), np.nan, complex)
            for (m, n), v in zip(pairs, vals):
                G[m + M, n + M] = v
            im = axes[0, col].imshow(np.abs(G), origin="lower", extent=(-M - .5, M + .5, -M - .5, M + .5),
                                     vmin=0, vmax=np.nanmax(np.abs(C)))
            axes[1, col].imshow(np.angle(G), origin="lower", extent=(-M - .5, M + .5, -M - .5, M + .5),
                                cmap="twilight", vmin=-np.pi, vmax=np.pi)
            axes[0, col].set_title(f"|C|, {title}")
            axes[1, col].set_title(f"фаза C, {title}")
            for ax in axes[:, col]:
                ax.set_xlabel("n")
                ax.set_ylabel("m")
        fig.colorbar(im, ax=axes[0, -1], fraction=0.046)
        fig.suptitle(f"{name}: сумма чирпов (A комплексные, W свободные)")
        fig.tight_layout()
        tag = name.split(" ")[0]
        fig.savefig(fig_dir / f"E053_heatmap_{tag}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
