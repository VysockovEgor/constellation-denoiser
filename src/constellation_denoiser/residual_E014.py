"""Optional E014 analysis: where does the best Kerr MLP (k0_kerr) leave structure?

    PYTHONPATH=src python -m constellation_denoiser.residual_E014   (SMOKE=1 reads results_smoke/)

Takes the k0_kerr run with the best validation reduction, loads its
validation-best weights and computes, on the VALIDATION files (test is not
used), the error e = prediction - transmitted symbol by constellation point
and by received-symbol amplitude. Writes results/residual_by_point.csv,
results/residual_by_amplitude.csv and figures/E014_residual.png.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from . import model_zoo, plot_style
from .core import BIT_TABLE, CONSTELLATION, get_indexes
from .paths import CHECKPOINTS_DIR
from .runner import exp_dir, results_root
from .window_data import load_fileset, predict_iq, resolve_paths, synthetic_fileset


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E014")
    results = results_root("E014", smoke)
    figures = results / "figures" if smoke else directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(p.read_text()) for p in (results / "runs").glob("*.json")]
    runs = [r for r in runs if r.get("variant") == "k0_kerr" and r.get("stage") == "main" and not r.get("diverged")]
    if not runs:
        print("нет прогонов k0_kerr")
        return 1
    best = max(runs, key=lambda r: r["best_val_reduction"])
    task_ckpt = (results / "checkpoints") if smoke else CHECKPOINTS_DIR / "E014"
    payload = torch.load(task_ckpt / f"{best['run_id']}_best.pt", map_location="cpu", weights_only=False)
    model = model_zoo.build(payload["spec"])
    model.load_state_dict(payload["model_state"])
    window = int(payload["spec"]["window"])

    if best.get("synthetic"):
        data = synthetic_fileset(best["n_val_files"], 2)
    else:
        from .runner import default_files

        files = default_files(best["n_train_nominal"], smoke)["validation"]
        data = load_fileset(resolve_paths(files))
    pred_iq = predict_iq(model, data, window)
    rx_c, tx_c = data.centers(window)
    pred = pred_iq[:, 0] + 1j * pred_iq[:, 1]
    tx_index = get_indexes(tx_c)
    error = pred - CONSTELLATION[tx_index]
    raw_error = rx_c - CONSTELLATION[tx_index]
    bit_errors = (BIT_TABLE[get_indexes(pred)] != BIT_TABLE[tx_index]).sum(axis=1)
    raw_bit_errors = (BIT_TABLE[get_indexes(rx_c)] != BIT_TABLE[tx_index]).sum(axis=1)

    by_point = pd.DataFrame({"point": tx_index, "err2": np.abs(error) ** 2, "raw_err2": np.abs(raw_error) ** 2,
                             "err_re": error.real, "err_im": error.imag, "bits": bit_errors, "raw_bits": raw_bit_errors})
    table_point = by_point.groupby("point").agg(
        n=("err2", "size"), mse=("err2", "mean"), raw_mse=("raw_err2", "mean"), bias_re=("err_re", "mean"),
        bias_im=("err_im", "mean"), ber=("bits", lambda b: b.sum() / (4 * len(b))),
        raw_ber=("raw_bits", lambda b: b.sum() / (4 * len(b)))).reset_index()
    table_point["point_re"] = CONSTELLATION[table_point.point].real
    table_point["point_im"] = CONSTELLATION[table_point.point].imag
    table_point["amplitude"] = np.abs(CONSTELLATION[table_point.point])
    table_point.insert(0, "run_id", best["run_id"])
    table_point.to_csv(results / "residual_by_point.csv", index=False)

    amplitude = np.abs(rx_c)
    bins = np.quantile(amplitude, np.linspace(0, 1, 11))
    which = np.clip(np.digitize(amplitude, bins[1:-1]), 0, 9)
    by_amp = pd.DataFrame({"bin": which, "amp": amplitude, "err2": np.abs(error) ** 2, "raw_err2": np.abs(raw_error) ** 2,
                           "bits": bit_errors, "raw_bits": raw_bit_errors})
    table_amp = by_amp.groupby("bin").agg(
        amp_mean=("amp", "mean"), amp_min=("amp", "min"), amp_max=("amp", "max"), n=("err2", "size"),
        mse=("err2", "mean"), raw_mse=("raw_err2", "mean"), ber=("bits", lambda b: b.sum() / (4 * len(b))),
        raw_ber=("raw_bits", lambda b: b.sum() / (4 * len(b)))).reset_index()
    table_amp.insert(0, "run_id", best["run_id"])
    table_amp.to_csv(results / "residual_by_amplitude.csv", index=False)
    plot(table_point, table_amp, figures, best["run_id"])
    print(f"остаток {best['run_id']}: {results / 'residual_by_point.csv'}, {results / 'residual_by_amplitude.csv'}")
    return 0


def plot(points: pd.DataFrame, amps: pd.DataFrame, figures, run_id: str) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    scatter = left.scatter(points.point_re, points.point_im, c=points.mse, s=260, cmap="Blues",
                           edgecolors=plot_style.MUTED, linewidths=0.6)
    for _, row in points.iterrows():
        left.annotate(f"{row.mse:.3f}", (row.point_re, row.point_im), ha="center", va="center", fontsize=6,
                      color=plot_style.TEXT if row.mse < points.mse.median() else "white")
    fig.colorbar(scatter, ax=left, label="MSE after model")
    left.set_title("residual MSE by transmitted point (validation)")
    left.set_xlabel("I")
    left.set_ylabel("Q")
    left.set_aspect("equal")
    right.plot(amps.amp_mean, amps.raw_mse, marker="o", color=plot_style.color(1), label="raw (after phase recovery)")
    right.plot(amps.amp_mean, amps.mse, marker="o", color=plot_style.color(0), label="best k0_kerr")
    right.set_xlabel("received amplitude |rx| (decile means)")
    right.set_ylabel("MSE")
    right.set_title("residual MSE vs received amplitude")
    right.legend(fontsize=8)
    fig.suptitle(run_id, fontsize=8, color=plot_style.MUTED)
    fig.tight_layout()
    fig.savefig(figures / "E014_residual.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
