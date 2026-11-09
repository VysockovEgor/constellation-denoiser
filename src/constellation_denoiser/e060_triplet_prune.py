"""E060: pruning the ready closed-form triplets T (T3 + FIR7, M = 15, E035 warm_M15.npz) without refitting.

    PYTHONPATH=src python -m constellation_denoiser.e060_triplet_prune      (SMOKE=1: 2 files, closed form here)

The K terms with the largest score are kept with their coefficients unchanged, the others are set to zero; the
FIR7 taps are always kept. Scores: "abs" = |C_mn|; "abs_rms" = |C_mn| * rms(T_mn) on the 30 train files (the
ranking of triplets_ext.importance). K = 368, 360, ..., 8, 0. BER on the window-31 centres of validation and test.
Rule (config.json rule_draft): K_val = smallest K with val reduction >= full - 0.5 p.p.; test reported at K_val.
Writes results/prune_curve.csv, results/verdict.json, figures/E060_triplet_prune.png.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .runner import default_files, exp_dir, results_root
from .triplets import EDGE, ber_columns, features, triplet_terms

EXP = "E060"
M, FIR = 15, 7


def padded(rx: np.ndarray, pad: int) -> np.ndarray:
    return np.concatenate([np.zeros(pad, complex), rx, np.zeros(pad, complex)])


def main() -> int:
    from .triplets_ext import closed_form_coefficients
    from .window_data import load_fileset, resolve_paths

    smoke = os.environ.get("SMOKE", "0") == "1"
    if os.environ.get("PLOT_ONLY", "0") == "1":                # redraw from the saved results, no data needed
        res = results_root(EXP, smoke)
        frame = pd.read_csv(res / "prune_curve.csv")
        plot(frame, json.loads((res / "verdict.json").read_text(encoding="utf-8")), int(frame.K.max()), smoke)
        return 0
    cfg = json.loads((exp_dir(EXP) / "config.json").read_text(encoding="utf-8"))
    files = default_files(30, smoke)
    data = {k: load_fileset(resolve_paths(v)) for k, v in files.items()}
    terms = triplet_terms(M, "T3")
    n = len(terms)
    pad = max(max(max(abs(m), abs(nn), abs(m + nn)) for m, nn in terms), FIR // 2) + 1
    if smoke:
        coef, lam = closed_form_coefficients(data["train"], data["validation"], M, FIR, cfg["lambda_grid"], 4096)
        source = f"closed form on smoke files, lambda {lam:g}"
    else:
        coef = np.load(exp_dir("E035") / "results" / "warm_M15.npz")["coef"]
        source = "experiments/E035_triplet_hybrid/results/warm_M15.npz (lambda 1e-4, warm_info.json)"
    assert len(coef) == n + FIR

    # rms of every triplet feature on train (centres of window 31)
    power, count = np.zeros(n), 0
    tr = data["train"]
    for f in range(tr.n_files):
        c = np.arange(EDGE, len(tr.rx_c[f]) - EDGE)
        a = features(padded(tr.rx_c[f], pad), c, terms, 0, pad)
        power += (np.abs(a) ** 2).sum(0)
        count += len(c)
    rms = np.sqrt(power / count)
    scores = {"abs": np.abs(coef[:n]), "abs_rms": np.abs(coef[:n]) * rms}

    ks = list(range(n, 0, -8))
    if ks[-1] != 8:
        ks.append(8)
    ks.append(0)
    cols, labels = [], []
    for name, s in scores.items():
        order = np.argsort(s)[::-1]
        for k in ks:
            c = coef.copy()
            c[order[k:]] = 0
            cols.append(c)
            labels.append((name, k))
    C = np.stack(cols, axis=1)

    out_rows = {}
    for split in ("validation", "test"):
        d = data[split]
        preds, txs = [], []
        for f in range(d.n_files):
            c = np.arange(EDGE, len(d.rx_c[f]) - EDGE)
            a = features(padded(d.rx_c[f], pad), c, terms, FIR, pad)
            preds.append(d.rx_c[f][c][:, None] + a @ C)
            txs.append(d.tx_c[f][c])
        pred, tx = np.concatenate(preds), np.concatenate(txs)
        rx = np.concatenate([d.rx_c[f][EDGE:-EDGE] for f in range(d.n_files)])
        raw = float(ber_columns(rx[:, None], tx)[0])
        ber = ber_columns(pred, tx)
        out_rows[split] = (raw, ber)

    rows = []
    for j, (name, k) in enumerate(labels):
        row = {"score": name, "K": k, "rm_repo": 12 * k + 4 * FIR}
        for split, (raw, ber) in out_rows.items():
            key = "val" if split == "validation" else "test"
            row[f"{key}_ber"] = float(ber[j])
            row[f"{key}_reduction"] = 100 * (raw - ber[j]) / raw
        rows.append(row)
    frame = pd.DataFrame(rows)
    res = results_root(EXP, smoke)
    res.mkdir(parents=True, exist_ok=True)
    frame.to_csv(res / "prune_curve.csv", index=False)

    verdict = {"rule": cfg["rule_draft"], "coefficients": source, "n_terms_full": n, "tolerance_pp": 0.5}
    for name in scores:
        g = frame[frame.score == name].sort_values("K")
        full = g[g.K == n].iloc[0]
        ok_val = g[g.val_reduction >= full.val_reduction - 0.5]
        kv = ok_val.iloc[0]
        ok_test = g[g.test_reduction >= full.test_reduction - 0.5]
        verdict[name] = {
            "full_val": full.val_reduction, "full_test": full.test_reduction,
            "K_val": int(kv.K), "removed_K_val": int(n - kv.K), "rm_K_val": int(kv.rm_repo),
            "test_at_K_val": kv.test_reduction, "test_drop_at_K_val": full.test_reduction - kv.test_reduction,
            "K_test_report_only": int(ok_test.iloc[0].K),
        }
    (res / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    plot(frame, verdict, n, smoke)
    print(json.dumps(verdict, ensure_ascii=False, indent=1, default=float))
    return 0


def plot(frame: pd.DataFrame, verdict: dict, n: int, smoke: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    names = {"abs": "по |C|", "abs_rms": "по |C|·rms(T)"}
    for ax, split, title in ((axes[0], "val", "валидация"), (axes[1], "test", "test")):
        for name, color in (("abs", "#2a78d6"), ("abs_rms", "#eb6834")):
            g = frame[frame.score == name].sort_values("K")
            ax.plot(g.K, g[f"{split}_reduction"], "-o", ms=3, color=color, label=f"отбор {names[name]}")
            full = g[g.K == n].iloc[0][f"{split}_reduction"]
            kv = verdict[name]["K_val"]
            ax.axvline(kv, color=color, ls=":", lw=1.2)
        ax.axhline(full - 0.5, color="0.4", ls="--", lw=1, label="полный набор − 0.5 п.п.")
        ax.set_xlabel("оставлено триплетов K (из 368)")
        ax.set_ylim(30, full + 2)
        low = frame[(frame.score == "abs") & (frame[f"{split}_reduction"] < 30)].K.max()
        ax.text(0.02, 0.96, title, transform=ax.transAxes, va="top")
        ax.text(0.45, 0.04, f"K ≤ {low}: ниже 30 %\n(K = 0: {frame[frame.K == 0][f'{split}_reduction'].iloc[0]:.0f} %)",
                transform=ax.transAxes, va="bottom", fontsize=11, color="0.3")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("снижение BER, %")
    axes[0].legend(loc="center right", fontsize=11)
    fig.tight_layout()
    out = (exp_dir(EXP) / ("results_smoke" if smoke else "figures"))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "E060_triplet_prune.png", dpi=150)


if __name__ == "__main__":
    sys.exit(main())
