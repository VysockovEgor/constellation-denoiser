"""E015: does the phase error of a received symbol depend on the power of its neighbours?

    PYTHONPATH=src python -m constellation_denoiser.physics_diagnostics   (SMOKE=1: 2/1/1 files, 20 bootstraps)

No neural network is trained. On the received symbols after phase recovery
(the same "raw" signal as in the BER metric) and the transmitted symbols:

1. dphi_k = angle(z_k * conj(a_k)) in (-pi, pi];
2. binned dphi (20 quantile bins, mean and std) against own power |z_k|^2 and
   neighbour power P_nb(R) = sum_{0<|j|<=R} |z_{k+j}|^2, R in {1, 3, 7, 15};
   10 x 10 map of mean dphi over (own power, P_nb(7));
3. Spearman rho(dphi, |z|^2) and rho(dphi, P_nb(R)), overall and per ring |a|;
4. raw residual MSE |z - a|^2 against P_nb(R);
5. direct fit of z_k * exp(-i * sum_j alpha_j |z_{k+j}|^2) to a_k for R in
   {1, 3, 7, 15, 31} coefficients (j = -(R-1)/2 .. (R-1)/2) on train with
   scipy.optimize (L-BFGS-B, gradient from torch, float64, alpha = 0 start);
   BER reduction and Q2 gain on validation and test.

Confidence intervals: bootstrap over files (200 resamples; no seeds exist here).
Spearman is computed from ranks over the whole split; the bootstrap reuses
those ranks through per-file sufficient statistics (ranks are not recomputed
inside each resample - a documented approximation).
Validation decides; test is reported only.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from . import plot_style
from .core import BIT_TABLE, CONSTELLATION, get_indexes
from .q2 import q2_db
from .runner import atomic_write_json, default_files, exp_dir, log, results_root
from .window_data import load_fileset, resolve_paths

EDGE = 15                       # window 31 as in E010-E014: the same symbols as the BER metric
NB_RADII = (1, 3, 7, 15)
FIT_R = (1, 3, 7, 15, 31)
RINGS = {"inner": 0.447, "middle": 1.0, "outer": 1.342}
RAW_TEST_BER_E004 = 0.009551859551859552   # experiments/E004_training_file_curve/config.json


# --------------------------------------------------------------------------- data

def split_arrays(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    data = load_fileset(resolve_paths(names))
    return data.rx_c, data.tx_c          # (F, L) complex128


def centre(a: np.ndarray) -> np.ndarray:
    return a[:, EDGE: a.shape[1] - EDGE]


def neighbour_power(power: np.ndarray, radius: int) -> np.ndarray:
    """sum_{0<|j|<=radius} power[k+j] for the centre symbols of every file."""
    kernel = np.ones(2 * radius + 1)
    kernel[radius] = 0.0
    full = np.stack([np.convolve(row, kernel, mode="same") for row in power])
    return centre(full)


def ring_of(a: np.ndarray) -> np.ndarray:
    amplitude = np.abs(a)
    names = np.array(list(RINGS))
    levels = np.array(list(RINGS.values()))
    return names[np.argmin(np.abs(amplitude[..., None] - levels), axis=-1)]


def bit_errors(decided: np.ndarray, sent: np.ndarray) -> np.ndarray:
    """Per-file bit-error counts (F,) for complex arrays (F, n)."""
    d = BIT_TABLE[get_indexes(decided.reshape(-1))]
    s = BIT_TABLE[get_indexes(sent.reshape(-1))]
    return (d != s).sum(axis=1).reshape(decided.shape).sum(axis=1)


# --------------------------------------------------------------------------- bootstrap helpers

class FileBootstrap:
    def __init__(self, n_files: int, reps: int, seed: int = 20261023):
        rng = np.random.default_rng(seed)
        self.weights = np.stack([np.bincount(rng.integers(0, n_files, n_files), minlength=n_files)
                                 for _ in range(reps)])          # (reps, F)

    def ci(self, per_file: dict[str, np.ndarray], statistic) -> tuple[float, float]:
        values = [statistic({k: (w[:, None] * v if v.ndim > 1 else w * v).sum(axis=0) for k, v in per_file.items()})
                  for w in self.weights]
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return float("nan"), float("nan")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def pearson_from_sums(s: dict) -> float:
    n = s["n"]
    cov = s["xy"] - s["x"] * s["y"] / n
    vx, vy = s["xx"] - s["x"] ** 2 / n, s["yy"] - s["y"] ** 2 / n
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def spearman_with_ci(x: np.ndarray, y: np.ndarray, boot: FileBootstrap, mask: np.ndarray | None = None) -> dict:
    """x, y: (F, n). Spearman over all selected points; CI by file bootstrap of rank sums."""
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    rx = np.full(x.shape, np.nan)
    ry = np.full(y.shape, np.nan)
    rx[mask] = rankdata(x[mask])
    ry[mask] = rankdata(y[mask])
    m = mask.astype(float)
    rx0, ry0 = np.nan_to_num(rx), np.nan_to_num(ry)
    per_file = {"n": m.sum(1), "x": rx0.sum(1), "y": ry0.sum(1), "xx": (rx0 ** 2).sum(1),
                "yy": (ry0 ** 2).sum(1), "xy": (rx0 * ry0).sum(1)}
    rho = pearson_from_sums({k: v.sum() for k, v in per_file.items()})
    low, high = boot.ci(per_file, pearson_from_sums)
    return {"value": rho, "ci_low": low, "ci_high": high}


def binned(x: np.ndarray, y: np.ndarray, bins: int, boot: FileBootstrap) -> pd.DataFrame:
    """Quantile bins of x: mean/std of y per bin, CI of the mean by file bootstrap."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = which == b
        n_f, s_f, ss_f = sel.sum(1).astype(float), np.where(sel, y, 0).sum(1), np.where(sel, y ** 2, 0).sum(1)
        n = n_f.sum()
        mean = s_f.sum() / n
        std = np.sqrt(max(ss_f.sum() / n - mean ** 2, 0.0))
        low, high = boot.ci({"n": n_f, "s": s_f}, lambda t: t["s"] / t["n"] if t["n"] else np.nan)
        rows.append({"bin": b, "x_low": edges[b], "x_high": edges[b + 1], "x_mean": float(x[sel].mean()),
                     "n": int(n), "mean": float(mean), "std": float(std), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- alpha fit

def windows_power(z: np.ndarray, n_alpha: int) -> np.ndarray:
    """(F, n, R) powers |z_{k+j}|^2, j = -(R-1)/2 .. (R-1)/2, for the centre symbols."""
    power = np.abs(z) ** 2
    r = n_alpha // 2
    n = z.shape[1] - 2 * EDGE
    return np.stack([power[:, EDGE + j: EDGE + j + n] for j in range(-r, r + 1)], axis=-1)


def apply_alpha(z: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    phi = windows_power(z, len(alpha)) @ alpha
    return centre(z) * np.exp(-1j * phi)


def fit_alpha(z: np.ndarray, a: np.ndarray, n_alpha: int, maxiter: int) -> tuple[np.ndarray, dict]:
    import torch

    p = torch.from_numpy(windows_power(z, n_alpha).reshape(-1, n_alpha))
    zc = torch.from_numpy(centre(z).reshape(-1))
    ac = torch.from_numpy(centre(a).reshape(-1))

    def objective(alpha_np):
        alpha = torch.tensor(alpha_np, dtype=torch.float64, requires_grad=True)
        y = zc * torch.exp(-1j * (p @ alpha))
        loss = (y - ac).abs().square().mean()
        loss.backward()
        return float(loss.detach()), alpha.grad.numpy().copy()

    started = time.perf_counter()
    result = minimize(objective, np.zeros(n_alpha), jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "gtol": 1e-12, "ftol": 1e-15})
    return result.x, {"loss": float(result.fun), "iterations": int(result.nit), "converged": bool(result.success),
                      "message": str(result.message), "seconds": time.perf_counter() - started}


def ber_with_ci(decided: np.ndarray, sent: np.ndarray, raw: np.ndarray, boot: FileBootstrap) -> dict:
    model_err, raw_err = bit_errors(decided, sent), bit_errors(raw, sent)
    bits = np.full(sent.shape[0], 4.0 * sent.shape[1])
    ber, raw_ber = model_err.sum() / bits.sum(), raw_err.sum() / bits.sum()

    def red(t):
        return 100 * (t["r"] - t["m"]) / t["r"] if t["r"] else np.nan

    def gain(t):
        return q2_db(t["m"] / t["b"]) - q2_db(t["r"] / t["b"])

    per_file = {"m": model_err.astype(float), "r": raw_err.astype(float), "b": bits}
    low, high = boot.ci(per_file, red)
    g_low, g_high = boot.ci(per_file, gain)
    return {"ber": ber, "raw_ber": raw_ber, "reduction": 100 * (raw_ber - ber) / raw_ber, "reduction_ci_low": low,
            "reduction_ci_high": high, "q2_db_raw": q2_db(raw_ber), "q2_db_model": q2_db(ber),
            "q2_gain_db": q2_db(ber) - q2_db(raw_ber), "q2_gain_ci_low": g_low, "q2_gain_ci_high": g_high}


# --------------------------------------------------------------------------- main

def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E015")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    results = results_root("E015", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reps = 20 if smoke else config["bootstrap_reps"]
    maxiter = 60 if smoke else config["fit_maxiter"]
    files = default_files(config["n_train_files"], smoke)
    log(f"E015: train {len(files['train'])}, validation {len(files['validation'])}, test {len(files['test'])} файлов; "
        f"bootstrap {reps}")

    arrays = {name: split_arrays(files[name]) for name in ("train", "validation", "test")}
    rows, tables = [], []

    # control: the raw signal is exactly the one of E004-E014
    z_test, a_test = arrays["test"]
    raw_err = bit_errors(centre(z_test), centre(a_test)).sum() / (4.0 * centre(a_test).size)
    control_ok = smoke or abs(raw_err - RAW_TEST_BER_E004) < 1e-12
    rows.append({"metric": "control_raw_test_ber", "split": "test", "value": raw_err,
                 "target": RAW_TEST_BER_E004, "ok": control_ok})
    log(f"контроль: raw test BER {raw_err:.12f} (E004: {RAW_TEST_BER_E004:.12f}) -> {'OK' if control_ok else 'ОШИБКА'}")
    if not control_ok:
        pd.DataFrame(rows).to_csv(results / "diagnostics_summary.csv", index=False)
        log("ОШИБКА: сырой сигнал не совпадает с E004-E014; остальные расчёты E015 не выполняются")
        return 3

    for split in ("validation", "test"):
        z, a = arrays[split]
        boot = FileBootstrap(z.shape[0], reps)
        zc, ac = centre(z), centre(a)
        dphi = np.angle(zc * np.conj(ac))
        own = np.abs(zc) ** 2
        rings = ring_of(ac)
        residual = np.abs(zc - ac) ** 2
        predictors = {"own_power": own} | {f"P_nb_R{r}": neighbour_power(np.abs(z) ** 2, r) for r in NB_RADII}
        for ring in ["all"] + list(RINGS):
            mask = None if ring == "all" else rings == ring
            for name, values in predictors.items():
                stat = spearman_with_ci(dphi, values, boot, mask)
                rows.append({"metric": "spearman_dphi", "split": split, "predictor": name, "ring": ring, **stat})
        for name, values in predictors.items():
            for target, label in ((dphi, "dphi"), (residual, "raw_residual_mse")):
                table = binned(values, target, 20, boot)
                table.insert(0, "split", split)
                table.insert(1, "predictor", name)
                table.insert(2, "target", label)
                tables.append(table)
        if split == "validation":
            heat = heatmap(own, predictors["P_nb_R7"], dphi)
            heat.to_csv(results / "dphi_heatmap_R7.csv", index=False)
    binned_frame = pd.concat(tables, ignore_index=True)
    binned_frame.to_csv(results / "binned_dphi_and_residual.csv", index=False)

    fits = []
    z_tr, a_tr = arrays["train"]
    for n_alpha in FIT_R:
        alpha, info = fit_alpha(z_tr, a_tr, n_alpha, maxiter)
        r = n_alpha // 2
        pd.DataFrame({"j": np.arange(-r, r + 1), "alpha": alpha}).to_csv(results / f"alpha_fit_R{n_alpha}.csv",
                                                                         index=False)
        log(f"fit R={n_alpha}: loss {info['loss']:.6f}, {info['iterations']} итераций, {info['seconds']:.1f} с")
        for split in ("validation", "test"):
            z, a = arrays[split]
            boot = FileBootstrap(z.shape[0], reps)
            stats = ber_with_ci(apply_alpha(z, alpha), centre(a), centre(z), boot)
            fits.append({"n_alpha": n_alpha, "split": split, **stats, **{f"fit_{k}": v for k, v in info.items()}})
    fit_frame = pd.DataFrame(fits)
    fit_frame.to_csv(results / "alpha_fit_summary.csv", index=False)
    for _, item in fit_frame.iterrows():
        rows.append({"metric": "fit_ber_reduction", "split": item.split, "n_alpha": item.n_alpha,
                     "value": item.reduction, "ci_low": item.reduction_ci_low, "ci_high": item.reduction_ci_high,
                     "q2_gain_db": item.q2_gain_db})

    verdict = decide(pd.DataFrame(rows), fit_frame, config)
    summary = pd.DataFrame(rows)
    summary.to_csv(results / "diagnostics_summary.csv", index=False)
    atomic_write_json(results / "verdict.json", verdict)
    log(f"вердикт по черновику правила: {verdict['verdict']}")
    plots(binned_frame, results, figures, fit_frame)
    return 0


def heatmap(own: np.ndarray, nb: np.ndarray, dphi: np.ndarray, bins: int = 10) -> pd.DataFrame:
    e1 = np.quantile(own, np.linspace(0, 1, bins + 1))
    e2 = np.quantile(nb, np.linspace(0, 1, bins + 1))
    i = np.clip(np.searchsorted(e1, own, side="right") - 1, 0, bins - 1).ravel()
    j = np.clip(np.searchsorted(e2, nb, side="right") - 1, 0, bins - 1).ravel()
    frame = pd.DataFrame({"own_bin": i, "nb_bin": j, "dphi": dphi.ravel()})
    out = frame.groupby(["own_bin", "nb_bin"]).dphi.agg(["mean", "std", "count"]).reset_index()
    out["own_mid"] = [(e1[b] + e1[b + 1]) / 2 for b in out.own_bin]
    out["nb_mid"] = [(e2[b] + e2[b + 1]) / 2 for b in out.nb_bin]
    return out


def decide(rows: pd.DataFrame, fits: pd.DataFrame, config: dict) -> dict:
    """Draft rule of config.json -> verdict.json (validation decides, test must agree)."""
    rule = config["rule_draft"]
    sp = rows[(rows.metric == "spearman_dphi") & (rows.ring == "all")]

    def rho(split, predictor):
        return float(sp[(sp.split == split) & (sp.predictor == predictor)].value.iloc[0])

    nb_names = [f"P_nb_R{r}" for r in NB_RADII]
    a_val = {n: abs(rho("validation", n)) >= abs(rho("validation", "own_power")) for n in nb_names}
    cond_a = any(a_val.values())

    def red(split, n_alpha):
        return float(fits[(fits.split == split) & (fits.n_alpha == n_alpha)].reduction.iloc[0])

    gap_val = red("validation", 7) - red("validation", 1)
    gap_test = red("test", 7) - red("test", 1)
    cond_b = gap_val >= rule["fit_gap_pp"]
    low, high = rule["test_ratio_range"]

    def agrees(v, t):
        return bool(np.sign(v) == np.sign(t) and v != 0 and low <= abs(t / v) <= high)

    best_nb = max(nb_names, key=lambda n: abs(rho("validation", n)))
    test_ok = agrees(rho("validation", best_nb), rho("test", best_nb)) and agrees(gap_val, gap_test)
    confirmed = cond_a and cond_b and test_ok
    return {"rule_status": rule["status"], "condition_a_by_R": a_val, "condition_a": cond_a,
            "rho_own_validation": rho("validation", "own_power"),
            "rho_nb_validation": {n: rho("validation", n) for n in nb_names},
            "rho_nb_test": {n: rho("test", n) for n in nb_names},
            "fit_gap_R7_minus_R1_validation_pp": gap_val, "fit_gap_R7_minus_R1_test_pp": gap_test,
            "condition_b": cond_b, "test_agrees": test_ok,
            "verdict": "память есть" if confirmed else "не доказано"}


def plots(binned_frame: pd.DataFrame, results, figures, fits: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply()
    val = binned_frame[(binned_frame.split == "validation") & (binned_frame.target == "dphi")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for index, name in enumerate(["own_power"] + [f"P_nb_R{r}" for r in NB_RADII]):
        part = val[val.predictor == name]
        ax = axes[0] if name == "own_power" else axes[1]
        ax.plot(part.x_mean, part["mean"], marker="o", color=plot_style.color(index), label=name)
        ax.fill_between(part.x_mean, part.ci_low, part.ci_high, color=plot_style.color(index), alpha=0.2, linewidth=0)
    axes[0].set_xlabel("own power |z_k|^2 (bin mean)")
    axes[1].set_xlabel("neighbour power P_nb(R) (bin mean)")
    axes[0].set_ylabel("mean phase error dphi, rad (band: 95% file bootstrap)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Phase error vs own and neighbour power (validation)")
    fig.tight_layout()
    fig.savefig(figures / "E015_dphi_vs_power.png", dpi=160)
    plt.close(fig)

    heat = pd.read_csv(results / "dphi_heatmap_R7.csv").pivot(index="nb_bin", columns="own_bin", values="mean")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    limit = np.nanmax(np.abs(heat.values))
    image = ax.imshow(heat.values, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    fig.colorbar(image, ax=ax, label="mean dphi, rad")
    ax.set_xlabel("own power decile")
    ax.set_ylabel("neighbour power P_nb(7) decile")
    ax.set_title("Mean phase error (validation)")
    fig.tight_layout()
    fig.savefig(figures / "E015_dphi_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for index, n_alpha in enumerate(FIT_R):
        frame = pd.read_csv(results / f"alpha_fit_R{n_alpha}.csv")
        ax.plot(frame.j, frame.alpha, marker="o", markersize=4, color=plot_style.color(index), label=f"R={n_alpha}")
    ax.axhline(0, color=plot_style.MUTED, linewidth=0.8)
    ax.set_xlabel("neighbour offset j")
    ax.set_ylabel("fitted alpha_j")
    ax.set_title("Fitted nonlinear-phase coefficients (train)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E015_alpha_profile.png", dpi=160)
    plt.close(fig)

    def whiskers(value, low, high):
        return [np.clip(np.nan_to_num(value - low), 0, None), np.clip(np.nan_to_num(high - value), 0, None)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for index, split in enumerate(("validation", "test")):
        part = fits[fits.split == split]
        axes[0].errorbar(part.n_alpha, part.reduction,
                         yerr=whiskers(part.reduction, part.reduction_ci_low, part.reduction_ci_high),
                         marker="o", capsize=2, color=plot_style.color(index), label=split)
        axes[1].errorbar(part.n_alpha, part.q2_gain_db,
                         yerr=whiskers(part.q2_gain_db, part.q2_gain_ci_low, part.q2_gain_ci_high),
                         marker="o", capsize=2, color=plot_style.color(index), label=split)
    for ax, label in zip(axes, ("BER reduction vs raw, %", "Q2 gain, dB")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of alpha coefficients R")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle("Direct fit of the nonlinear phase formula (no network)")
    fig.tight_layout()
    fig.savefig(figures / "E015_fit_vs_R.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
