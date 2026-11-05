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


def fit_c(h: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    """Best c on the grid and r2 = 1 - residual share (complex scale a fitted for every c)."""
    L = len(h)
    norm = np.vdot(h, h).real
    best = (np.nan, -np.inf)
    for c in grid:
        p, _ = physical_taps(c, L, 2048)
        a = np.vdot(p, h) / np.vdot(p, p)
        r2 = 1 - np.vdot(h - a * p, h - a * p).real / norm
        if r2 > best[1]:
            best = (float(c), float(r2))
    return best


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


def main(argv: list[str]) -> int:
    from .runner import exp_dir, results_root

    smoke = os.environ.get("SMOKE", "0") == "1"
    out = results_root("E049", smoke)
    out.mkdir(parents=True, exist_ok=True)
    if (argv or ["calib"])[0] == "calib":
        s = calib(out)
        print(s.round(3).to_string(index=False) if len(s) else "нет чекпоинтов phys_dbp")
        print(json.dumps(json.loads((out / "calib_summary.json").read_text()) if (out / "calib_summary.json").is_file() else {},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
