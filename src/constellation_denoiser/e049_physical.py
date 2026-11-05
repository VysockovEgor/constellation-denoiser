"""E049: physical initialization of the DBP FIR steps and its calibration against trained DBP (E018/E028).

Link parameters (из литературы, не проверено на генераторе): Rs = 32 GBd, beta2 = -21 ps^2/km, 2000 km
(Bogdanov, Sidelnikov 2021). One DBP step of length dz = 2000 km / Ns has the frequency response
H(nu) = exp(j c nu^2), nu in rad/symbol in [-pi, pi], c = s * beta2 * dz * Rs^2 / 2 (s = sign convention);
c_total = beta2 * 2000 km * Rs^2 / 2 = -21.5 rad. The taps are the inverse DTFT (h[n] = h[-n]), truncated to L.

    PYTHONPATH=src python -m constellation_denoiser.e049_physical calib     # on the server (checkpoints there)

calib: for every trained phys_dbp checkpoint of E018/E028 (``*_best.pt``) and every step, c of the learned
symmetric taps by a grid search of min_a ||h - a h_phys(c)||^2 / ||h||^2 (a complex), fit quality r2, and the
energy of the physical response inside the L-tap window. Writes results/calib.csv, results/calib_summary.csv,
results/calib_summary.json (sign used by the physical initialization).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

BETA2_S2_PER_KM = -21e-24
RS = 32e9
LENGTH_KM = 2000.0
N_FFT = 8192


def c_step(n_steps: int, sign: float = 1.0) -> float:
    return sign * BETA2_S2_PER_KM * (LENGTH_KM / n_steps) * RS ** 2 / 2


def physical_taps(c: float, L: int, n_fft: int = N_FFT) -> tuple[np.ndarray, float]:
    """(h[-L//2 .. L//2], share of the response energy inside the window)."""
    nu = 2 * np.pi * np.fft.fftfreq(n_fft)                  # rad / symbol in [-pi, pi)
    h = np.fft.ifft(np.exp(1j * c * nu ** 2))                # h[n] at n = 0, 1, ..., -1 (wrapped)
    h = np.fft.fftshift(h)
    centre = n_fft // 2
    half = L // 2
    window = h[centre - half: centre + half + 1]
    return window, float(np.sum(np.abs(window) ** 2) / np.sum(np.abs(h) ** 2))


_BANK: dict = {}


def bank(L: int, grid: np.ndarray) -> np.ndarray:
    """Physical taps for every c of the grid, shape (len(grid), L); cached per (L, grid)."""
    key = (L, len(grid), float(grid[0]), float(grid[-1]))
    if key not in _BANK:
        _BANK[key] = np.stack([physical_taps(c, L, 2048)[0] for c in grid])
    return _BANK[key]


def fit_c(h: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    """Best c on the grid and r2 = 1 - residual share (complex scale a fitted for every c), vectorized."""
    P = bank(len(h), grid)
    a = (P.conj() @ h) / np.sum(np.abs(P) ** 2, axis=1)
    resid = np.sum(np.abs(h[None, :] - a[:, None] * P) ** 2, axis=1) / np.vdot(h, h).real
    i = int(np.argmin(resid))
    return float(grid[i]), float(1 - resid[i])


def full_taps(state: dict, k: int, sym: bool) -> np.ndarray | None:
    re, im = state.get(f"steps.{k}.fir_re"), state.get(f"steps.{k}.fir_im")
    if re is None:
        return None
    re, im = re.numpy(), im.numpy()
    if sym:
        re, im = np.r_[re[1:][::-1], re], np.r_[im[1:][::-1], im]
    return re + 1j * im


def calib(out) -> pd.DataFrame:
    from .paths import CHECKPOINTS_DIR

    grid = np.round(np.arange(-12, 12.0001, 0.02), 4)
    rows = []
    for exp in ("E018", "E025", "E028"):
        for path in sorted((CHECKPOINTS_DIR / exp).glob("*_best.pt")):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            spec = ckpt.get("spec", {})
            if spec.get("model") not in ("phys_dbp",):
                continue
            ns, L, sym = int(spec["n_steps"]), int(spec["fir_len"]), bool(spec["sym"])
            if L <= 1:
                continue
            for k in range(ns):
                h = full_taps(ckpt["model_state"], k, sym)
                if h is None:
                    continue
                c, r2 = fit_c(h, grid)
                pred = c_step(ns)
                _, energy = physical_taps(pred, L)
                rows.append({"experiment": exp, "run_id": ckpt.get("run_id", path.stem), "n_steps": ns, "fir_len": L,
                             "n_alpha": int(spec["n_alpha"]), "step": k, "c_fit": c, "r2": r2,
                             "c_pred_abs": abs(pred), "energy_in_window_pred": energy})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "calib.csv", index=False)
    if frame.empty:
        return frame
    good = frame[frame.r2 >= 0.5]
    summary = frame.groupby(["experiment", "n_steps", "fir_len"]).agg(
        runs=("run_id", "nunique"), steps=("step", "count"), c_fit_median=("c_fit", "median"),
        c_fit_sum_per_run=("c_fit", lambda s: float(s.sum()) / max(1, frame.loc[s.index, "run_id"].nunique())),
        r2_median=("r2", "median"), c_pred_abs=("c_pred_abs", "first"),
        energy_in_window_pred=("energy_in_window_pred", "first")).reset_index()
    summary["c_fit_times_ns"] = summary.c_fit_median * summary.n_steps
    summary["c_pred_total_abs"] = abs(c_step(1))
    summary.to_csv(out / "calib_summary.csv", index=False)
    sign = float(np.sign(good.c_fit.median())) if len(good) else 0.0
    from .runner import atomic_write_json

    atomic_write_json(out / "calib_summary.json", {
        "sign_of_learned_c": sign, "steps_with_r2_ge_0.5": int(len(good)), "steps_total": int(len(frame)),
        "c_total_literature_rad": c_step(1), "note": "Rs 32 GBd, beta2 -21 ps^2/km: из литературы, не проверено на генераторе"})
    return summary


def physical_dbp(spec: dict) -> torch.nn.Module:
    """PhysDBP with physical FIR taps (c per step from spec) and optionally alpha_centre on every step."""
    from .phys_models import PhysDBP

    window, L, R, sym, ns = int(spec["window"]), int(spec["fir_len"]), int(spec["n_alpha"]), bool(spec["sym"]), int(spec["n_steps"])
    model = PhysDBP(window, L, R, sym, ns)
    c = c_step(ns, float(spec["c_sign"]))
    taps, _ = physical_taps(c, L)
    half = taps[L // 2:] if sym else taps
    with torch.no_grad():
        for step in model.steps:
            if spec.get("fir_init", "physical") == "physical":
                step.fir_re.copy_(torch.tensor(half.real, dtype=torch.float32))
                step.fir_im.copy_(torch.tensor(half.imag, dtype=torch.float32))
            if spec.get("alpha_centre") is not None:
                step.alpha.zero_()
                step.alpha[0 if sym else R // 2] = float(spec["alpha_centre"]) / ns
    return model


def full_alpha(state: dict, k: int, sym: bool) -> np.ndarray | None:
    a = state.get(f"steps.{k}.alpha")
    if a is None:
        return None
    a = a.numpy()
    return np.r_[a[1:][::-1], a] if sym else a


def alpha_width(out) -> pd.DataFrame:
    """Check 1 (гипотеза: ближе к приёмнику шире): RMS width of |alpha_j| per step of trained DBP, and the Spearman
    correlation of the width with the step index within each run (step 0 acts first on the received signal)."""
    from scipy.stats import spearmanr

    from .paths import CHECKPOINTS_DIR

    rows, corr = [], []
    for exp in ("E018", "E025", "E028"):
        for path in sorted((CHECKPOINTS_DIR / exp).glob("*_best.pt")):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            spec = ckpt.get("spec", {})
            if spec.get("model") != "phys_dbp" or int(spec.get("n_alpha", 0)) < 3 or int(spec["n_steps"]) < 2:
                continue
            ns, R, sym = int(spec["n_steps"]), int(spec["n_alpha"]), bool(spec["sym"])
            j = np.arange(R) - R // 2
            widths = []
            for k in range(ns):
                a = full_alpha(ckpt["model_state"], k, sym)
                w = np.abs(a)
                width = float(np.sqrt(np.sum(j ** 2 * w) / np.sum(w))) if w.sum() > 0 else float("nan")
                widths.append(width)
                rows.append({"experiment": exp, "run_id": ckpt.get("run_id", path.stem), "n_steps": ns, "fir_len": int(spec["fir_len"]),
                             "n_alpha": R, "step": k, "step_frac": k / (ns - 1), "rms_width": width,
                             "centre_share": float(w[R // 2] / w.sum()) if w.sum() > 0 else float("nan"), "sum_alpha": float(a.sum())})
            ok = [x for x in widths if x == x]
            if len(ok) >= 3 and np.ptp(ok) > 0:
                corr.append({"experiment": exp, "n_steps": ns, "fir_len": int(spec["fir_len"]), "n_alpha": R,
                             "run_id": ckpt.get("run_id", path.stem), "spearman_width_vs_step": float(spearmanr(range(len(widths)), widths).statistic)})
    frame, corr = pd.DataFrame(rows), pd.DataFrame(corr)
    frame.to_csv(out / "alpha_width.csv", index=False)
    corr.to_csv(out / "alpha_width_corr.csv", index=False)
    if len(corr):
        summary = corr.groupby(["experiment", "n_steps", "fir_len", "n_alpha"]).agg(
            runs=("run_id", "count"), rho_median=("spearman_width_vs_step", "median"),
            share_negative=("spearman_width_vs_step", lambda s: float((s < 0).mean()))).reset_index()
        summary.to_csv(out / "alpha_width_summary.csv", index=False)
        return summary
    return corr


def start_ber(out, smoke: bool) -> pd.DataFrame:
    """Check 2: BER without training for alpha_j(s) = scale * alpha_total_j / Ns, FIR = delta.

    alpha_total_j from the axial ridge coefficients: the merged weight w(0, j) multiplies x_k |x_{k+j}|^2, and
    x exp(-i sum_j alpha_j |x_{k+j}|^2) ~ x - i sum_j alpha_j |x_{k+j}|^2 x, so alpha_j = -Im w(0, j) (Re w ~ 0, E044);
    symmetrized over +-j. The scale is chosen by the minimum validation MSE on a grid; test reported."""
    from .experiment import ber_from_iq
    from .phys_models import PhysDBP
    from .runner import default_files, exp_dir
    from .window_data import load_fileset, predict_iq, resolve_paths

    coef = pd.read_csv(exp_dir("E039") / "results" / "coef_M15.csv")
    axial = coef[(coef.kind == "triplet") & ((coef.m == 0) | (coef.n == 0))]
    alpha_of = {}
    for _, r in axial.iterrows():
        j = int(r.n) if int(r.m) == 0 else int(r.m)
        alpha_of[j] = -float(r.im)
    files = default_files(30, smoke)
    data = {k: load_fileset(resolve_paths(files[k])) for k in ("validation", "test")}
    raw = {k: d.raw_ber(31) for k, d in data.items()}
    scales = [-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0]   # sign of the factor is not fixed
    rows = []
    for ns, R in ((8, 7), (16, 15), (32, 15), (8, 31)):
        r = R // 2
        half = np.array([(alpha_of.get(j, 0.0) + alpha_of.get(-j, 0.0)) / (1 if j == 0 else 2) for j in range(r + 1)])
        best = None
        for scale in scales:
            model = PhysDBP(31, 1, R, True, ns)
            with torch.no_grad():
                for step in model.steps:
                    step.alpha.copy_(torch.tensor(scale * half / ns, dtype=torch.float32))
            res = {}
            for k, d in data.items():
                pred = predict_iq(model, d, 31)
                rx_c, tx_c = d.centers(31)
                ber = ber_from_iq(pred, tx_c)
                mse = float(np.mean(np.abs(pred[:, 0] + 1j * pred[:, 1] - tx_c) ** 2))
                res[k] = (ber, mse)
            row = {"n_steps": ns, "n_alpha": R, "scale": scale, "val_mse": res["validation"][1],
                   "val_reduction": 100 * (raw["validation"] - res["validation"][0]) / raw["validation"],
                   "test_reduction": 100 * (raw["test"] - res["test"][0]) / raw["test"], "alpha_total_sum": float(scale * (half[0] + 2 * half[1:].sum()))}
            rows.append(row)
        part = [x for x in rows if x["n_steps"] == ns and x["n_alpha"] == R]
        chosen = min(part, key=lambda x: x["val_mse"])
        for x in part:
            x["chosen_by_val_mse"] = x is chosen
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "start_ber.csv", index=False)
    return frame


def main(argv: list[str]) -> int:
    from .runner import exp_dir, results_root

    smoke = os.environ.get("SMOKE", "0") == "1"
    out = results_root("E049", smoke)
    out.mkdir(parents=True, exist_ok=True)
    action = (argv or ["calib"])[0]
    if action == "alpha_width":
        s = alpha_width(out)
        print(s.round(3).to_string(index=False) if len(s) else "нет данных")
        return 0
    if action == "start_ber":
        f = start_ber(out, smoke)
        print(f.round(3).to_string(index=False))
        return 0
    if action == "calib":
        s = calib(out)
        print(s.round(3).to_string(index=False) if len(s) else "нет чекпоинтов phys_dbp")
        print(json.dumps(json.loads((out / "calib_summary.json").read_text()) if (out / "calib_summary.json").is_file() else {},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
