"""E050: classical DBP with the dispersion of the whole line added back first (DBP-X, the form of Xiao et al. 2025).

The received symbols are already dispersion-compensated (CDC). DBP-X first applies D_total = exp(+i c nu^2)
(the inverse of the CDC filter), then Ns steps of [linear step exp(-i c/Ns nu^2), nonlinear phase
y = u exp(-i sum_j alpha_j |u_{k+j}|^2)]. With alpha = 0 and exact filters the chain is the identity.

Link parameters (из литературы, не проверено на генераторе): Rs = 32 GBd, beta2 = -21 ps^2/km, 2000 km,
c_total = beta2 L Rs^2 / 2 = -21.5 rad (e049_physical); the multiplier of c_total and the sign are chosen on
validation. nu in rad/symbol (numpy FFT convention). Memory of a filter with parameter c: 4 pi |c| symbols
(group-delay spread over nu in [-pi, pi]); for the whole line 270 symbols (= 2 pi |beta2| B Fs L of Xiao).

Filters act on a whole file (16 384 symbols) as circular convolutions: "exact" = the sampled response
exp(i c nu^2) on the file grid; "FIR k" = the inverse DTFT truncated to k taps (what a streaming FIR or an
overlap-save block with overlap k - 1 implements). The files are close to periodic (E050 README), so circular
processing of a file is used for training and evaluation; the metric is computed on [EDGE, T - EDGE) and, as a
secondary number, on the project centres (window 31).

    python -m constellation_denoiser.dbpx unittest | start | train | summarize
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .e049_physical import BETA2_S2_PER_KM, LENGTH_KM, RS

C_TOTAL = BETA2_S2_PER_KM * LENGTH_KM * RS ** 2 / 2          # -21.504 rad
EDGE = 512
R_KERR = 7                                                     # memory of the Kerr phase as in E028
SEEDS = (17, 29, 43, 57, 71, 83, 97, 101, 113, 127)


# --------------------------------------------------------------------------- complexity

def memory(c: float) -> float:
    return 4 * math.pi * abs(c)


def odd_len(k: float) -> int:
    n = max(1, int(math.ceil(k)))
    return n if n % 2 else n + 1


def c_conv(k: int) -> float:
    """Xiao 2025: overlap-save cost per output symbol, 4 min_T T (log2 T + 1) / (T - k + 1), T a power of two."""
    if k <= 1:
        return 0.0
    return min(4 * 2 ** p * (p + 1) / (2 ** p - k + 1) for p in range(1, 24) if 2 ** p - k + 1 > 0)


def rm_columns(ns: int, step_len: int, total_len: int, n_alpha: int) -> dict:
    """RM per symbol of DBP-X in three columns.

    project: direct symmetric FIR (2(L + 1), model_zoo.macs_phys) for D_total (length total_len) and for every
             step, nonlinear phase |u|^2 2 + conv (R + 1)/2 + rotation 4 (R = 1: 7, the Kerr rule).
    xiao:    C_conv(total_len) + Ns (C_conv(L) + 2) for R = 1 (their DBP step), Ns (C_conv(L) + C_conv(R)/4 + 6)
             for R > 1 (their iSPM + iXPM term, no iCIXPM: one polarization).
    merged:  xiao with D_total and the first linear step merged into one filter of length total_len + L - 1."""
    from .model_zoo import macs_phys

    project = macs_phys(0, total_len, True) + ns * macs_phys(n_alpha, step_len, True)
    nl = 2 if n_alpha <= 1 else c_conv(n_alpha) / 4 + 6
    xiao = c_conv(total_len) + ns * (c_conv(step_len) + nl)
    merged = c_conv(total_len + step_len - 1) + (ns - 1) * c_conv(step_len) + ns * nl
    return {"total_len": total_len, "step_len": step_len, "rm_project": float(project), "rm_xiao": round(xiao, 1),
            "rm_merged": round(merged, 1)}


# --------------------------------------------------------------------------- data and metrics

def load(smoke: bool, parts=("train", "validation", "test")) -> dict:
    from .runner import default_files
    from .window_data import load_fileset, resolve_paths

    files = default_files(30, smoke)
    return {k: load_fileset(resolve_paths(files[k])) for k in parts}


def nu_grid(T: int) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(T)


def ber(z: np.ndarray, tx: np.ndarray) -> float:
    from .experiment import ber_from_iq

    z, tx = z.reshape(-1), tx.reshape(-1)
    return float(ber_from_iq(np.column_stack((z.real, z.imag)), tx))


def regions(z: np.ndarray) -> dict:
    T = z.shape[-1]
    return {"trim": z[..., EDGE:T - EDGE], "c31": z[..., 15:T - 15]}


def gain(z: np.ndarray, tx: np.ndarray) -> complex:
    """Unbiased complex gain a = <tx, tx> / <tx, z> (E[conj(tx) a z] = E|tx|^2); the MMSE gain <z, tx>/<z, z>
    shrinks the constellation and moves the 16-QAM decisions (at alpha = 0 it gave -6.5 % instead of 0)."""
    z, tx = regions(z)["trim"], regions(tx)["trim"]
    return complex(np.vdot(tx, tx) / np.vdot(tx, z))


def metrics(z: np.ndarray, tx: np.ndarray, rx: np.ndarray, a: complex = 1.0) -> dict:
    """BER reduction against raw on the same region; z is multiplied by the complex gain a."""
    from .q2 import q2_db

    out, zr, tr, rr = {}, regions(a * z), regions(tx), regions(rx)
    for key in ("trim", "c31"):
        b, raw = ber(zr[key], tr[key]), ber(rr[key], tr[key])
        out[f"ber_{key}"] = b
        out[f"raw_{key}"] = raw
        out[f"reduction_{key}"] = 100 * (raw - b) / raw
        out[f"q2_gain_{key}"] = q2_db(b) - q2_db(raw) if b > 0 else float("nan")
    out["mse_trim"] = float(np.mean(np.abs(zr["trim"] - tr["trim"]) ** 2))
    return out


# --------------------------------------------------------------------------- DBP-X (numpy)

def alpha_shape(kind: str, R: int = R_KERR) -> np.ndarray:
    """Memory kernel normalized to sum 1. classical: delta; ridge: symmetrized -Im C_{0,j} of E039 coef_M15
    (merged weight w(0, j) of x_k |x_{k+j}|^2, alpha_j = -Im w, as in E049)."""
    out = np.zeros(R)
    if kind == "classical":
        out[R // 2] = 1.0
        return out
    if kind == "ridge":
        from .runner import exp_dir

        coef = pd.read_csv(exp_dir("E039") / "results" / "coef_M15.csv")
        axial = coef[(coef.kind == "triplet") & ((coef.m == 0) | (coef.n == 0))]
        alpha = {}
        for _, r in axial.iterrows():
            j = int(r.n) if int(r.m) == 0 else int(r.m)
            alpha[j] = -float(r.im)
        r = R // 2
        full = np.array([(alpha.get(j, 0.0) + alpha.get(-j, 0.0)) / 2 if j else alpha.get(0, 0.0)
                         for j in range(-r, r + 1)])
        return full / full.sum()
    raise ValueError(kind)


def taps(c: float, k: int) -> np.ndarray:
    """Inverse DTFT of exp(i c nu^2) at n = -k//2 .. k//2 (from a fine grid, no time aliasing)."""
    n_fft = max(8192, 8 * k)
    h = np.fft.fftshift(np.fft.ifft(np.exp(1j * c * nu_grid(n_fft) ** 2)))
    centre = n_fft // 2
    return h[centre - k // 2: centre + k // 2 + 1]


def response(c: float, k: int | None, T: int) -> np.ndarray:
    """Response on the T-point file grid: exact (k None) or the k-tap FIR."""
    if k is None:
        return np.exp(1j * c * nu_grid(T) ** 2)
    padded = np.zeros(T, dtype=complex)
    h = taps(c, k)
    idx = np.arange(-(k // 2), k // 2 + 1) % T
    padded[idx] = h
    return np.fft.fft(padded)


def phase_conv(power: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    r = len(alpha) // 2
    return sum(alpha[i] * np.roll(power, -(i - r), axis=-1) for i in range(len(alpha)))


def dbpx(rx: np.ndarray, ns: int, g: float, mult: float = 1.0, sign: float = 1.0, shape: np.ndarray | None = None,
         power: float = 1.0, step_len: int | None = None, total_len: int | None = None) -> np.ndarray:
    """g: total nonlinear phase scale; per step alpha = g * shape / (Ns * power)."""
    c, T = sign * mult * C_TOTAL, rx.shape[-1]
    z = np.fft.ifft(np.fft.fft(rx, axis=-1) * response(c, total_len, T), axis=-1)
    h = response(-c / ns, step_len, T)
    a = g * (alpha_shape("classical") if shape is None else shape) / (ns * power)
    for _ in range(ns):
        z = np.fft.ifft(np.fft.fft(z, axis=-1) * h, axis=-1)
        if g:
            z = z * np.exp(-1j * phase_conv(np.abs(z) ** 2, a))
    return z


# --------------------------------------------------------------------------- torch model for training

class DBPX(torch.nn.Module):
    """Ns steps [symmetric FIR(L) as a circular convolution over the file, Kerr phase with symmetric alpha of R
    taps], then a complex output gain. Input: the file after the fixed D_total (exact), shape (B, T)."""

    def __init__(self, ns: int, L: int, R: int, c_step: float):
        super().__init__()
        self.ns, self.L, self.R, self.c_step = ns, L, R, c_step
        self.fir_re = torch.nn.Parameter(torch.zeros(ns, L // 2 + 1))
        self.fir_im = torch.nn.Parameter(torch.zeros(ns, L // 2 + 1))
        self.alpha = torch.nn.Parameter(torch.zeros(ns, R // 2 + 1)) if R else None
        self.gain = torch.nn.Parameter(torch.tensor([1.0, 0.0]))

    def init_physical(self) -> float:
        h = taps(self.c_step, self.L)[self.L // 2:]
        with torch.no_grad():
            self.fir_re.copy_(torch.tensor(h.real, dtype=torch.float32).expand_as(self.fir_re))
            self.fir_im.copy_(torch.tensor(h.imag, dtype=torch.float32).expand_as(self.fir_im))
        full = taps(self.c_step, 8 * self.L + 1)
        inside = taps(self.c_step, self.L)
        return float(np.sum(np.abs(inside) ** 2) / np.sum(np.abs(full) ** 2))

    def init_delta(self):
        with torch.no_grad():
            self.fir_re.zero_()
            self.fir_im.zero_()
            self.fir_re[:, 0] = 1.0

    def set_alpha(self, full: np.ndarray):
        with torch.no_grad():
            self.alpha.copy_(torch.tensor(full[self.R // 2:], dtype=torch.float32).expand_as(self.alpha))

    def responses(self, T: int) -> torch.Tensor:
        """(ns, T): FFT of the symmetric taps placed circularly."""
        h = torch.complex(self.fir_re, self.fir_im)                     # h_0 .. h_r
        r = self.L // 2
        padded = torch.zeros(self.ns, T, dtype=h.dtype)
        idx = torch.cat((torch.arange(0, r + 1), torch.arange(T - r, T)))
        vals = torch.cat((h, h[:, 1:].flip(1)), dim=1)                  # h_0..h_r, h_-r..h_-1 (= h_r..h_1)
        padded = padded.index_copy(1, idx, vals)
        return torch.fft.fft(padded, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        H = self.responses(z.shape[-1])
        if self.R:
            r = self.R // 2
            alpha = torch.cat((self.alpha[:, 1:].flip(1), self.alpha), dim=1)
        for s in range(self.ns):
            z = torch.fft.ifft(torch.fft.fft(z, dim=-1) * H[s], dim=-1)
            if self.R:
                p = z.real.square() + z.imag.square()
                phi = sum(alpha[s, i] * torch.roll(p, -(i - r), dims=-1) for i in range(self.R))
                z = z * torch.exp(-1j * phi)
        return z * torch.complex(self.gain[0], self.gain[1])

    def ld_loss(self, n_grid: int = 1024) -> torch.Tensor:
        """Xiao's L_D: sum_i |prod_{j<=i} H_j - exp(i c_step i nu^2)|^2 (mean over the grid)."""
        H = self.responses(n_grid)
        nu = torch.tensor(nu_grid(n_grid), dtype=torch.float32)
        cum, loss = torch.ones(n_grid, dtype=H.dtype), 0.0
        for i in range(self.ns):
            cum = cum * H[i]
            loss = loss + (cum - torch.exp(1j * self.c_step * (i + 1) * nu ** 2).to(H.dtype)).abs().square().mean()
        return loss


def pretrain_ld(model: DBPX, iters: int, lr: float = 1e-3) -> list[float]:
    """Adam on L_D (Xiao: lr 1e-3); the state with the lowest L_D is kept (L_D is a filter-design objective without
    data, so this is not a selection on validation; it prevents the Adam transient / divergence at Ns = 80 from
    replacing the physical taps by worse ones)."""
    opt = torch.optim.Adam([model.fir_re, model.fir_im], lr=lr)
    hist = []
    best = (float(model.ld_loss().detach()), model.fir_re.detach().clone(), model.fir_im.detach().clone())
    for it in range(iters):
        opt.zero_grad()
        loss = model.ld_loss()
        loss.backward()
        value = float(loss.detach())
        if value < best[0]:
            best = (value, model.fir_re.detach().clone(), model.fir_im.detach().clone())
        opt.step()
        if it % 100 == 0 or it == iters - 1:
            hist.append(value)
    with torch.no_grad():
        model.fir_re.copy_(best[1])
        model.fir_im.copy_(best[2])
    return hist


def run_model(model: DBPX, zd: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(zd, dtype=torch.complex64)).numpy().astype(np.complex128)


# --------------------------------------------------------------------------- stages

def results_dir(smoke: bool) -> Path:
    from .runner import exp_dir

    d = exp_dir("E050") / ("results_smoke" if smoke else "results")
    d.mkdir(parents=True, exist_ok=True)
    return d


UNIT_TRUNC = ((10, 31), (20, 31), (40, 31), (20, 15), (40, 15), (20, 127), (40, 127), (20, 255), (40, 255),
              (80, 255))
UNIT_TOL_PP = 2.0          # |BER reduction at zero nonlinearity| allowed for a trainable truncated config


def unittest(smoke: bool) -> int:
    """(1) exact filters, alpha = 0: identity (max error < 1e-9, BER equal to raw); hard requirement.
    (2) truncated L-tap steps (torch model, physical taps, then L_D pretraining), alpha = 0: BER reduction at the
    start within +-UNIT_TOL_PP of zero -> the config may be trained; otherwise not (truncation, not a sign error,
    when (1) holds)."""
    data = load(smoke, ("validation",))["validation"]
    rx, tx = data.rx_c, data.tx_c
    out = {"exact": [], "truncated": [], "tolerance_pp": UNIT_TOL_PP}
    ok = True
    for ns in (1, 8, 20, 40, 80):
        for sign in (1.0, -1.0):
            z = dbpx(rx, ns, 0.0, 1.0, sign)
            err = float(np.abs(z - rx).max())
            same = ber(regions(z)["trim"], regions(tx)["trim"]) == ber(regions(rx)["trim"], regions(tx)["trim"])
            out["exact"].append({"n_steps": ns, "sign": sign, "max_abs_err": err, "ber_equal_raw": bool(same)})
            ok &= err < 1e-9 and same
    zd = np.fft.ifft(np.fft.fft(rx, axis=-1) * response(C_TOTAL, None, rx.shape[-1]), axis=-1)
    for ns, L in UNIT_TRUNC:
        model = DBPX(ns, L, 0, -C_TOTAL / ns)
        share = model.init_physical()
        z0 = run_model(model, zd)
        before = float(model.ld_loss().detach())
        pretrain_ld(model, 200 if smoke else 3000)
        z1 = run_model(model, zd)
        r0, r1 = metrics(z0, tx, rx, gain(z0, tx)), metrics(z1, tx, rx, gain(z1, tx))
        out["truncated"].append({"n_steps": ns, "fir_len": L, "step_memory": memory(C_TOTAL / ns),
                                 "energy_in_window": share, "ld_before": before,
                                 "ld_after": float(model.ld_loss().detach()),
                                 "reduction_trim_physical_taps": r0["reduction_trim"],
                                 "reduction_trim_after_ld": r1["reduction_trim"],
                                 "trainable": bool(abs(r1["reduction_trim"]) <= UNIT_TOL_PP)})
        print(out["truncated"][-1], flush=True)
    out["exact_identity_ok"] = bool(ok)
    (results_dir(smoke) / "unit_test.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("exact identity:", ok)
    return 0 if ok else 1


START_NS = (1, 2, 4, 8, 10, 16, 20, 40, 80)
START_MULT = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)
START_G = tuple(float(x) for x in np.round(np.linspace(-1.2, 1.2, 25), 3))
STEP_LENS = (31, 63, 127, 255, 511, 1023)
TOTAL_LENS = (271, 301, 351, 401, 511, 1023)
LEN_TOL_PP = 0.5           # honest length: the shortest FIR within 0.5 pp of the exact filter on validation


def _start_job(args):
    ns, sign, kind, smoke = args
    torch.set_num_threads(1)
    data = load(smoke, ("validation",))["validation"]
    rx, tx = data.rx_c, data.tx_c
    power = float(np.mean(np.abs(rx) ** 2))
    shape = alpha_shape(kind)
    rows = []

    def evaluate(mult, g, step_len=None, total_len=None):
        z = dbpx(rx, ns, g, mult, sign, shape, power, step_len, total_len)
        a = gain(z, tx)
        m = metrics(z, tx, rx, a)
        rows.append(dict(n_steps=ns, sign=sign, alpha_kind=kind, mult=mult, g=g, step_len=step_len or 0,
                         total_len=total_len or 0, gain_re=a.real, gain_im=a.imag,
                         **{f"val_{k}": v for k, v in m.items()}))
        return m["mse_trim"]

    mults = (1.0,) if smoke else START_MULT
    gs = (-0.6, 0.0, 0.3, 0.6) if smoke else START_G
    for mult in mults:
        coarse = {g: evaluate(mult, g) for g in gs}
        best = min(coarse, key=coarse.get)
        step = gs[1] - gs[0]
        for g in np.round(best + step * np.linspace(-0.8, 0.8, 9), 4):
            if float(g) not in coarse:
                evaluate(mult, float(g))
    exact = [r for r in rows if r["step_len"] == 0]
    chosen = min(exact, key=lambda r: r["val_mse_trim"])
    for k in ((63, 255) if smoke else STEP_LENS):
        evaluate(chosen["mult"], chosen["g"], step_len=k)
    for k in ((401,) if smoke else TOTAL_LENS):
        evaluate(chosen["mult"], chosen["g"], total_len=k)
    return rows


def honest_len(part: pd.DataFrame, column: str, exact_val: float) -> int:
    lens = part[part[column] > 0].sort_values(column)
    ok = lens[lens.val_reduction_trim >= exact_val - LEN_TOL_PP]
    return int(ok[column].iloc[0]) if len(ok) else -1


def start(smoke: bool, jobs: int) -> int:
    """BER at the start without training: grid over (Ns, sign, c multiplier, g) for the classical kernel and the
    ridge-shaped kernel (R = 7) with exact filters; choice per (Ns, sign, kernel) by the minimum validation MSE;
    then the honest FIR lengths of a step and of D_total at the choice (validation); test at the choice, exact and
    with the honest lengths."""
    from multiprocessing import get_context

    tasks = [(ns, s, k, smoke) for ns in ((1, 8, 20) if smoke else START_NS) for s in (1.0, -1.0)
             for k in ("classical", "ridge")]
    with get_context("spawn").Pool(max(1, min(jobs, len(tasks)))) as pool:
        rows = [r for part in pool.imap_unordered(_start_job, tasks) for r in part]
    grid = pd.DataFrame(rows)
    out = results_dir(smoke)
    grid.to_csv(out / "start_grid.csv", index=False)
    data = load(smoke, ("validation", "test"))
    power = float(np.mean(np.abs(data["validation"].rx_c) ** 2))
    d = data["test"]
    chosen = []
    for (ns, sign, kind), part in grid.groupby(["n_steps", "sign", "alpha_kind"]):
        exact = part[(part.step_len == 0) & (part.total_len == 0)]
        r = exact.loc[exact.val_mse_trim.idxmin()].to_dict()
        ks, kt = honest_len(part, "step_len", r["val_reduction_trim"]), honest_len(part, "total_len",
                                                                                    r["val_reduction_trim"])
        shape, a = alpha_shape(kind), complex(r["gain_re"], r["gain_im"])
        m = metrics(dbpx(d.rx_c, int(ns), r["g"], r["mult"], sign, shape, power), d.tx_c, d.rx_c, a)
        row = dict(r, **{f"test_{k}": v for k, v in m.items()}, honest_step_len=ks, honest_total_len=kt)
        if ks > 0 and kt > 0:
            mh = metrics(dbpx(d.rx_c, int(ns), r["g"], r["mult"], sign, shape, power, ks, kt), d.tx_c, d.rx_c, a)
            row.update({f"test_honest_{k}": v for k, v in mh.items()})
            row.update(rm_columns(int(ns), ks, kt, 1 if kind == "classical" else R_KERR))
        chosen.append(row)
    best = pd.DataFrame(chosen).sort_values(["alpha_kind", "sign", "n_steps"])
    best.to_csv(out / "start_best.csv", index=False)
    cols = [c for c in ["alpha_kind", "sign", "n_steps", "mult", "g", "val_reduction_trim", "test_reduction_trim",
                        "honest_step_len", "honest_total_len", "test_honest_reduction_trim", "rm_project", "rm_xiao"]
            if c in best]
    print(best[cols].to_string(index=False))
    return 0


# --------------------------------------------------------------------------- training

TRAIN = {"ns": (20, 40), "L": (127, 255), "alpha_init": ("zero", "ridge", "classical"), "training": ("one", "two"),
         "iters": 5000, "smoke_iters": 30, "eval_every": 100, "batch_files": 4, "lr": 1e-3, "ld_iters": 3000,
         "delta_arm": {"ns": 20, "L": 255, "alpha_init": "zero"}}


def plan_train(smoke: bool) -> list[dict]:
    out = results_dir(smoke)
    unit = json.loads((out / "unit_test.json").read_text(encoding="utf-8"))
    if not unit["exact_identity_ok"]:
        raise SystemExit("E050 train: точное тождество не выполнено (unit_test.json), обучение не запускается")
    trainable = {(u["n_steps"], u["fir_len"]) for u in unit["truncated"] if u["trainable"]}
    best = pd.read_csv(out / "start_best.csv")
    cls_all = best[best.alpha_kind == "classical"]
    sign_of = {ns: float(p.loc[p.val_mse_trim.idxmin(), "sign"]) for ns, p in cls_all.groupby("n_steps")}
    best = best[best.sign == best.n_steps.map(sign_of)]          # sign chosen per Ns on validation (classical)
    seeds = SEEDS[:1] if smoke else SEEDS
    tasks = []

    def add(ns, L, kind, training):
        rows = best[(best.n_steps == ns)]
        if not len(rows):
            return
        cls = rows[rows.alpha_kind == "classical"].iloc[0]
        g = 0.0 if kind == "zero" else float(rows[rows.alpha_kind == kind].iloc[0]["g"])
        for seed in seeds:
            name = f"dbpx_Ns{ns}_L{L}_{kind}_{training}_seed{seed}"
            tasks.append({"name": name, "ns": ns, "L": L, "R": R_KERR, "alpha_init": kind, "training": training,
                          "g": g, "mult": float(cls["mult"]), "sign": float(cls["sign"]), "seed": seed,
                          "iters": TRAIN["smoke_iters"] if smoke else TRAIN["iters"],
                          "eval_every": 10 if smoke else TRAIN["eval_every"], "batch_files": TRAIN["batch_files"],
                          "lr": TRAIN["lr"], "ld_iters": 50 if smoke else TRAIN["ld_iters"],
                          "out_json": str(out / "dbpx_runs" / f"{name}.json")})

    for ns in TRAIN["ns"]:
        for L in TRAIN["L"]:
            if (ns, L) not in trainable:
                print(f"пропуск Ns={ns} L={L}: юнит-тест усечения не пройден")
                continue
            for kind in TRAIN["alpha_init"]:
                for training in TRAIN["training"]:
                    add(ns, L, kind, training)
    da = TRAIN["delta_arm"]
    add(da["ns"], da["L"], da["alpha_init"], "delta")
    return tasks


def train_one(task: dict, smoke: bool) -> dict:
    torch.set_num_threads(env_int("THREADS", 1))
    torch.manual_seed(task["seed"])
    rng = np.random.default_rng(task["seed"])
    data = load(smoke)
    c = task["sign"] * task["mult"] * C_TOTAL
    zd = {k: np.fft.ifft(np.fft.fft(d.rx_c, axis=-1) * response(c, None, d.length), axis=-1) for k, d in data.items()}
    power = float(np.mean(np.abs(data["validation"].rx_c) ** 2))
    model = DBPX(task["ns"], task["L"], task["R"], -c / task["ns"])
    t0, ld_hist, share = time.time(), [], None
    if task["training"] == "delta":
        model.init_delta()
    else:
        share = model.init_physical()
        if task["training"] == "two":
            ld_hist = pretrain_ld(model, task["ld_iters"], 1e-3)
    if task["alpha_init"] != "zero":
        model.set_alpha(task["g"] * alpha_shape(task["alpha_init"], task["R"]) / (task["ns"] * power))
    a = gain(run_model(model, zd["train"]), data["train"].tx_c)        # output gain by least squares on train
    with torch.no_grad():
        model.gain.copy_(torch.tensor([a.real, a.imag], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=task["lr"] / (10 if task["training"] == "two" else 1))
    x_all = torch.tensor(zd["train"], dtype=torch.complex64)
    y_all = torch.tensor(data["train"].tx_c, dtype=torch.complex64)
    start_val = metrics(run_model(model, zd["validation"]), data["validation"].tx_c, data["validation"].rx_c)
    history = [{"iter": 0, "val_reduction_trim": start_val["reduction_trim"], "val_mse": start_val["mse_trim"]}]
    best = (start_val["mse_trim"], 0, {k: v.detach().clone() for k, v in model.state_dict().items()})
    diverged = False
    for it in range(1, task["iters"] + 1):
        f = torch.tensor(rng.choice(x_all.shape[0], min(task["batch_files"], x_all.shape[0]), replace=False))
        shift = int(rng.integers(0, x_all.shape[1]))          # circular files: random rotation, loss on all symbols
        x, y = torch.roll(x_all[f], shift, dims=-1), torch.roll(y_all[f], shift, dims=-1)
        opt.zero_grad()
        loss = (model(x) - y).abs().square().mean()
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        opt.step()
        if it % task["eval_every"] == 0 or it == task["iters"]:
            m = metrics(run_model(model, zd["validation"]), data["validation"].tx_c, data["validation"].rx_c)
            history.append({"iter": it, "val_reduction_trim": m["reduction_trim"], "val_mse": m["mse_trim"],
                            "train_loss": float(loss.detach())})
            if m["mse_trim"] < best[0]:
                best = (m["mse_trim"], it, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    val = metrics(run_model(model, zd["validation"]), data["validation"].tx_c, data["validation"].rx_c)
    test = metrics(run_model(model, zd["test"]), data["test"].tx_c, data["test"].rx_c)
    top = max(h["val_reduction_trim"] for h in history)
    plateau = next(h["iter"] for h in history if h["val_reduction_trim"] >= top - 0.5)
    total_len = odd_len(memory(c))
    result = dict(task, best_iter=best[1], plateau_iter_0p5pp=plateau, diverged=diverged, time_s=time.time() - t0,
                  energy_in_window=share, ld_history=ld_hist, history=history,
                  start_val_reduction_trim=start_val["reduction_trim"],
                  **{f"val_{k}": v for k, v in val.items()}, **{f"test_{k}": v for k, v in test.items()},
                  test_reduction_valbest_select=test["reduction_trim"],
                  **rm_columns(task["ns"], task["L"], 401, task["R"]))
    result["d_total_note"] = f"D_total exact in training; RM with the honest length 401 (memory {total_len})"
    path = Path(task["out_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, default=float), encoding="utf-8")
    torch.save(best[2], path.with_suffix(".pt"))
    return result


def _train_job(args):
    task, smoke = args
    try:
        r = train_one(task, smoke)
        return f"[ok] {task['name']}: val={r['val_reduction_trim']:.2f} test={r['test_reduction_trim']:.2f} " \
               f"plateau={r['plateau_iter_0p5pp']} time={r['time_s']:.0f}s"
    except Exception as exc:                       # one failed run does not stop the stage
        return f"[fail] {task['name']}: {exc!r}"


def train(smoke: bool) -> int:
    """All planned runs that have no result yet, in an adaptive pool (free cores, gate.adaptive_limit)."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from .gate import MAX_JOBS, adaptive_limit

    tasks = [t for t in plan_train(smoke) if not Path(t["out_json"]).is_file()]
    print(f"E050 train: {len(tasks)} прогонов без результата", flush=True)
    pending, running, done = list(tasks), set(), 0
    with ProcessPoolExecutor(max_workers=1 if smoke else MAX_JOBS) as pool:
        while pending or running:
            limit = 1 if smoke else adaptive_limit(len(running))
            while pending and len(running) < limit:
                running.add(pool.submit(_train_job, (pending.pop(0), smoke)))
            finished, running = wait(running, timeout=60, return_when=FIRST_COMPLETED)
            running = set(running)
            for fut in finished:
                done += 1
                print(f"[{time.strftime('%F %T')}] [{done}/{len(tasks)}] {fut.result()} (в работе {len(running)}, "
                      f"предел {limit})", flush=True)
    return 0


PROBE = [  # parameters of the probes in README (classical kernel, sign +1), chosen on 2 validation files
    (1, 0.02, 1.0), (2, 0.06, 1.0), (8, 0.15, 1.0), (8, 0.15, 0.9), (20, 0.25, 1.0), (20, 0.30, 1.1), (20, 0.30, 0.9),
    (40, 0.45, 1.0), (40, 0.45, 1.1), (40, 0.40, 0.9)]


def probe_test(smoke: bool) -> int:
    """The probe settings on the test files without any new selection: the complex gain is fitted on the same 2
    validation files as in the probe; exact filters and the honest lengths (255 step, 401 D_total)."""
    from .runner import default_files
    from .window_data import load_fileset, resolve_paths, split_files

    probe_files = split_files()["validation"][:2]
    pr = load_fileset(resolve_paths(probe_files))
    test = load(smoke, ("test",))["test"]
    power = float(np.mean(np.abs(pr.rx_c) ** 2))       # the probe normalized by its own files
    rows = []
    for ns, g, mult in PROBE[:2] if smoke else PROBE:
        for step_len, total_len in ((None, None), (255, 401)):
            zp = dbpx(pr.rx_c, ns, g, mult, 1.0, None, power, step_len, total_len)
            a = gain(zp, pr.tx_c)
            mp = metrics(zp, pr.tx_c, pr.rx_c, a)
            zt = dbpx(test.rx_c, ns, g, mult, 1.0, None, power, step_len, total_len)
            mt = metrics(zt, test.tx_c, test.rx_c, a)
            rows.append({"n_steps": ns, "g": g, "mult": mult, "filters": "exact" if step_len is None else "FIR 255 / 401",
                         "probe_files": ";".join(probe_files), "probe_reduction_trim": mp["reduction_trim"],
                         "test_reduction_trim": mt["reduction_trim"], "test_reduction_c31": mt["reduction_c31"],
                         "test_q2_gain_trim": mt["q2_gain_trim"], "test_ber_trim": mt["ber_trim"],
                         "test_raw_trim": mt["raw_trim"], "n_test_files": test.n_files})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(results_dir(smoke) / "probe_test.csv", index=False)
    return 0


def rm_detail(ns: int, step_len: int, total_len: int, R: int = R_KERR) -> dict:
    """Multiplications and additions per output symbol, 4 or 3 real multiplications per complex product.

    Complex product by a fixed coefficient: 4M + 2A, or 3M + 3A (precomputed c + d, d - c). Symmetric FIR of length L
    per output: (L - 1)/2 complex pre-additions (L - 1 real), (L + 1)/2 complex products, L - 1 real additions of
    the accumulation. Overlap-save with a radix-2 FFT of size T (T chosen to minimize the multiplications, as Xiao's
    C_conv): per block 2 FFTs of (T/2) log2 T butterflies (one twiddle product + 2 complex additions = 4 real) and T
    pointwise products, divided by T - k + 1 outputs. Kerr step (direct in both columns): |u|^2 2M + 1A, symmetric
    phase convolution (R + 1)/2 M + R - 1 A, rotation one complex product (exp from a table: 0). D_total one filter,
    output gain one complex product."""
    out = {"n_steps": ns, "step_len": step_len, "total_len": total_len, "R": R}
    for cm, ca in ((4, 2), (3, 3)):
        def direct(L):
            return cm * (L + 1) / 2, (L - 1) + ca * (L + 1) / 2 + (L - 1)

        def fft(k):
            best = None
            for p in range(1, 24):
                T = 2 ** p
                if T - k + 1 <= 0:
                    continue
                m = cm * (T * p + T) / (T - k + 1)
                a = (T * p * (4 + ca) + T * ca) / (T - k + 1)
                if best is None or m < best[0]:
                    best = (m, a, T)
            return best

        kerr_m, kerr_a = 2 + (R + 1) / 2 + cm, 1 + (R - 1) + ca
        dm_s, da_s = direct(step_len)
        dm_t, da_t = direct(total_len)
        fm_s, fa_s, T_s = fft(step_len)
        fm_t, fa_t, T_t = fft(total_len)
        out[f"rm_direct_{cm}"] = round(dm_t + ns * (dm_s + kerr_m) + cm, 1)
        out[f"adds_direct_{cm}"] = round(da_t + ns * (da_s + kerr_a) + ca, 1)
        out[f"rm_fft_{cm}"] = round(fm_t + ns * (fm_s + kerr_m) + cm, 1)
        out[f"adds_fft_{cm}"] = round(fa_t + ns * (fa_s + kerr_a) + ca, 1)
        out["fft_size_step"], out["fft_size_total"] = T_s, T_t
    return out


def file_stats(smoke: bool) -> int:
    """DBP-X with the ridge kernel at Ns = 8, 12, 20 on the test files, spread over files.

    Configuration chosen on validation only: Ns 8 and 20 from start_best.csv (ridge rows, sign of the classical rows),
    Ns 12 by the same grid as start (_start_job, both signs, minimum validation MSE, honest lengths on validation).
    Per test file: BER and raw BER on [EDGE, T - EDGE), reduction; mean and SE over the files; pooled reduction and
    its bootstrap over files (2000 resamples, seed 0); effective SNR 10 log10(E|tx|^2 / MSE) of the output and of the
    raw signal. Exact filters and honest FIR lengths."""
    out_dir = results_dir(smoke)
    start = pd.read_csv(out_dir / "start_best.csv")
    data = load(smoke, ("validation", "test"))
    val, test = data["validation"], data["test"]
    power = float(np.mean(np.abs(val.rx_c) ** 2))
    shape = alpha_shape("ridge")
    rows, files = [], []
    for ns in ((8,) if smoke else (8, 12, 20)):
        if ns in set(start.n_steps):
            cls = start[(start.alpha_kind == "classical") & (start.n_steps == ns)]
            sign = cls.loc[cls.val_mse_trim.idxmin(), "sign"]
            r = start[(start.alpha_kind == "ridge") & (start.n_steps == ns) & (start.sign == sign)].iloc[0]
            cfg = {"sign": float(sign), "mult": float(r["mult"]), "g": float(r.g), "gain": complex(r.gain_re, r.gain_im),
                   "step_len": int(r.honest_step_len), "total_len": int(r.honest_total_len), "source": "start_best.csv"}
        else:
            grid = pd.DataFrame([x for sg in (1.0, -1.0) for x in _start_job((ns, sg, "ridge", smoke))])
            exact = grid[(grid.step_len == 0) & (grid.total_len == 0)]
            b = exact.loc[exact.val_mse_trim.idxmin()]
            part = grid[grid.sign == b.sign]
            cfg = {"sign": float(b.sign), "mult": float(b["mult"]), "g": float(b.g), "gain": complex(b.gain_re, b.gain_im),
                   "step_len": honest_len(part, "step_len", b.val_reduction_trim),
                   "total_len": honest_len(part, "total_len", b.val_reduction_trim), "source": "grid on validation here"}
        for filters in ("exact", "honest"):
            if filters == "honest" and (cfg["step_len"] < 1 or cfg["total_len"] < 1):
                continue                                         # no honest length found (smoke grid)
            sl, tl = (None, None) if filters == "exact" else (cfg["step_len"], cfg["total_len"])
            z = cfg["gain"] * dbpx(test.rx_c, ns, cfg["g"], cfg["mult"], cfg["sign"], shape, power, sl, tl)
            zr, tr, rr = regions(z)["trim"], regions(test.tx_c)["trim"], regions(test.rx_c)["trim"]
            per = []
            for f in range(test.n_files):
                b_f, raw_f = ber(zr[f], tr[f]), ber(rr[f], tr[f])
                per.append((b_f, raw_f, 100 * (raw_f - b_f) / raw_f))
                files.append({"n_steps": ns, "filters": filters, "file": test.names[f], "ber": b_f, "raw": raw_f,
                              "reduction": per[-1][2]})
            per = np.array(per)
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(200 if smoke else 2000):
                i = rng.integers(0, len(per), len(per))
                boot.append(100 * (per[i, 1].sum() - per[i, 0].sum()) / per[i, 1].sum())
            sig = float(np.mean(np.abs(tr) ** 2))
            rows.append({"n_steps": ns, "kernel": "ridge", "filters": filters, **{k: v for k, v in cfg.items() if k != "gain"},
                         "n_files": test.n_files, "reduction_mean_over_files": per[:, 2].mean(),
                         "reduction_se_over_files": per[:, 2].std(ddof=1) / np.sqrt(len(per)),
                         "reduction_pooled": 100 * (per[:, 1].sum() - per[:, 0].sum()) / per[:, 1].sum(),
                         "pooled_boot_se": float(np.std(boot, ddof=1)), "pooled_boot_lo": float(np.percentile(boot, 2.5)),
                         "pooled_boot_hi": float(np.percentile(boot, 97.5)), "ber_pooled": per[:, 0].mean(),
                         "raw_pooled": per[:, 1].mean(),
                         "snr_eff_out_db": 10 * np.log10(sig / np.mean(np.abs(zr - tr) ** 2)),
                         "snr_eff_raw_db": 10 * np.log10(sig / np.mean(np.abs(rr - tr) ** 2)),
                         **rm_detail(ns, cfg["step_len"], cfg["total_len"])})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "file_stats.csv", index=False)
    pd.DataFrame(files).to_csv(out_dir / "file_stats_per_file.csv", index=False)
    return 0


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    return int(value) if value.isdigit() else default


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    action = argv[0] if argv else ""
    force = os.environ.get("FORCE", "0") == "1"
    if action in ("unittest", "start") and not force:      # stages run once: the queue does not overwrite results
        done = results_dir(smoke) / ("unit_test.json" if action == "unittest" else "start_best.csv")
        if done.is_file():
            print(f"{done} уже есть: стадия {action} пропущена (FORCE=1 для пересчёта)")
            return 0
    if action == "unittest":
        return unittest(smoke)
    if action == "start":
        from .gate import jobs_now

        return start(smoke, env_int("JOBS", 1 if smoke else jobs_now(total=True)[0]))
    if action == "file_stats":
        return file_stats(smoke)
    if action == "probe_test":
        return probe_test(smoke)
    if action == "plan":
        tasks = plan_train(smoke)
        print(f"{len(tasks)} прогонов")
        return 0
    if action == "train":
        return train(smoke)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
