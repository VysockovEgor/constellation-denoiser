"""E052: DBP-X at small Ns followed by external off-axis triplets (cross rule, closed form).

    PYTHONPATH=src python -m constellation_denoiser.e052_dbpx_triplets all | summarize     (SMOKE=1: tiny)

DBP-X (dbpx.dbpx, no training) with the configuration chosen on validation in E050 start (start_best.csv: sign per
Ns from the classical rows, kernel classical or ridge by the lower validation MSE, its g, multiplier and complex
gain), exact circular filters; RM of DBP-X with the honest FIR lengths of E050 (two conventions).
Then y = z + F(z) c with off-axis merged triplets of z (kerr_offaxis.features) in the box max(|m|,|n|) <= M and the
cross rule min(|m|,|n|) <= K (Xiao 2025), K in (1, 2, 3), M in (8, 15); c by ridge on the 30 train files (bootstrap
counts per replica, 5 replicas as E046), lambda by validation BER (6 values), test reported. Metric: BER reduction on
the trimmed region [512, T - 512) against raw on the same region (as E050).
Question: does Ns = 8 + triplets reach the level of DBP-X Ns = 40 (no triplets) at the same or lower RM?
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from .dbpx import EDGE, alpha_shape, dbpx, metrics, results_dir as e050_results
from .kerr_offaxis import LAMBDAS, features, offaxis_box
from .runner import exp_dir, results_root

NS = (4, 8, 10, 20)
REF_NS = (40, 80)
M_LIST = (8, 15)
K_LIST = (1, 2, 3)
REPLICAS = (17, 29, 43, 57, 71)


def chosen(start: pd.DataFrame, ns: int) -> pd.Series:
    cls = start[(start.alpha_kind == "classical") & (start.n_steps == ns)]
    sign = cls.loc[cls.val_mse_trim.idxmin(), "sign"]
    part = start[(start.n_steps == ns) & (start.sign == sign)]
    return part.loc[part.val_mse_trim.idxmin()]


def dbpx_out(d, row, power) -> np.ndarray:
    z = dbpx(d.rx_c, int(row.n_steps), float(row.g), float(row["mult"]), float(row.sign),
             alpha_shape(row.alpha_kind), power)
    return z * complex(row.gain_re, row.gain_im)


def run_ns(ns: int, smoke: bool) -> pd.DataFrame:
    from .dbpx import load

    torch.set_num_threads(1)
    start = pd.read_csv(e050_results(smoke) / "start_best.csv")
    row = chosen(start, ns)
    data = load(smoke)
    power = float(np.mean(np.abs(data["validation"].rx_c) ** 2))
    z = {k: dbpx_out(d, row, power) for k, d in data.items()}
    T = data["train"].length
    sl = slice(EDGE, T - EDGE)
    rows = []
    base_val = metrics(z["validation"], data["validation"].tx_c, data["validation"].rx_c)
    base_test = metrics(z["test"], data["test"].tx_c, data["test"].rx_c)
    rm0 = {"rm_project": row.get("rm_project"), "rm_xiao": row.get("rm_xiao")}
    rows.append({"n_steps": ns, "alpha_kind": row.alpha_kind, "M": 0, "K": 0, "n_terms": 0, "replica": -1,
                 "lambda": 0.0, "val_reduction": base_val["reduction_trim"], "test_reduction": base_test["reduction_trim"],
                 "test_q2_gain": base_test["q2_gain_trim"], **rm0, "rm_triplets": 0})
    Ms = (3,) if smoke else M_LIST
    big = [(m, n) for (m, n) in offaxis_box(max(Ms)) if min(abs(m), abs(n)) <= max(K_LIST)]   # union of the sets
    sets = {}
    for M in Ms:
        for K in K_LIST:
            sets[(M, K)] = [i for i, (m, n) in enumerate(big) if max(abs(m), abs(n)) <= M and min(abs(m), abs(n)) <= K]

    def feats(x):
        return features(torch.tensor(x), big)[0, sl].numpy()        # (T', p)

    n_files = data["train"].n_files
    Gf, bf = [], []
    for f in range(n_files):
        A = feats(z["train"][f:f + 1])
        Gf.append(A.conj().T @ A)
        bf.append(A.conj().T @ (data["train"].tx_c[f] - z["train"][f])[sl])
    cols, meta = [], []
    for rep in (REPLICAS[:1] if smoke else REPLICAS):
        counts = np.bincount(np.random.default_rng(rep).integers(0, n_files, n_files), minlength=n_files)
        G = sum(c * g for c, g in zip(counts, Gf))
        b = sum(c * v for c, v in zip(counts, bf))
        for (M, K), idx in sets.items():
            ii = np.array(idx)
            g, bb = G[np.ix_(ii, ii)], b[ii]
            scale = np.real(np.trace(g)) / len(ii)
            for lam in LAMBDAS:
                c = np.zeros(len(big), complex)
                c[ii] = np.linalg.solve(g + lam * scale * np.eye(len(ii)), bb)
                cols.append(c)
                meta.append((rep, M, K, len(ii), lam))
    C = np.stack(cols, axis=1)
    from .e046_fit import bit_errors

    errs, raw = {}, {}
    for k in ("validation", "test"):
        d = data[k]
        e = torch.zeros(C.shape[1], dtype=torch.int64)
        e_raw, nb = 0, 0
        for f in range(d.n_files):
            pred = z[k][f][sl, None] + feats(z[k][f:f + 1]) @ C
            tx = torch.tensor(d.tx_c[f][sl])
            e += bit_errors(torch.tensor(pred), tx)
            e_raw += int(bit_errors(torch.tensor(d.rx_c[f][sl, None]), tx)[0])
            nb += 4 * pred.shape[0]
        errs[k], raw[k] = (e.double() / nb).numpy(), e_raw / nb
    from .q2 import q2_db

    frame = pd.DataFrame(meta, columns=["replica", "M", "K", "n_terms", "lambda"])
    frame["val_ber"], frame["test_ber"] = errs["validation"], errs["test"]
    frame = frame.loc[frame.groupby(["replica", "M", "K"]).val_ber.idxmin()]
    for _, r in frame.iterrows():
        rows.append({"n_steps": ns, "alpha_kind": row.alpha_kind, "M": int(r.M), "K": int(r.K), "n_terms": int(r.n_terms),
                     "replica": int(r.replica), "lambda": r["lambda"],
                     "val_reduction": 100 * (raw["validation"] - r.val_ber) / raw["validation"],
                     "test_reduction": 100 * (raw["test"] - r.test_ber) / raw["test"],
                     "test_q2_gain": q2_db(r.test_ber) - q2_db(raw["test"]), **rm0, "rm_triplets": 12 * int(r.n_terms)})
    return pd.DataFrame(rows)


def run_ref(ns: int, smoke: bool) -> pd.DataFrame:
    start = pd.read_csv(e050_results(smoke) / "start_best.csv")
    row = chosen(start, ns)
    return pd.DataFrame([{"n_steps": ns, "alpha_kind": row.alpha_kind, "M": 0, "K": 0, "n_terms": 0, "replica": -1,
                          "lambda": 0.0, "val_reduction": row.val_reduction_trim,
                          "test_reduction": row.get("test_honest_reduction_trim", row.test_reduction_trim),
                          "test_q2_gain": row.get("test_honest_q2_gain_trim", row.test_q2_gain_trim),
                          "rm_project": row.get("rm_project"), "rm_xiao": row.get("rm_xiao"), "rm_triplets": 0}])


def _job(args):
    kind, ns, smoke = args
    out = results_root("E052", smoke) / "runs" / f"{kind}_Ns{ns}.csv"
    if out.is_file():
        return f"{out.name}: exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frame = run_ns(ns, smoke) if kind == "trip" else run_ref(ns, smoke)
    frame["time_s"] = time.time() - t0
    frame.to_csv(out, index=False)
    return f"{out.name}: ok"


def summarize(smoke: bool) -> int:
    res = results_root("E052", smoke)
    t = pd.concat([pd.read_csv(f) for f in sorted((res / "runs").glob("*.csv"))], ignore_index=True)
    t["rm_project_total"] = t.rm_project + t.rm_triplets
    t["rm_fft_total"] = t.rm_xiao + t.rm_triplets
    g = t.groupby(["n_steps", "alpha_kind", "M", "K"])
    s = g.agg(n=("test_reduction", "size"), n_terms=("n_terms", "first"), test_mean=("test_reduction", "mean"),
              test_std=("test_reduction", "std"), val_mean=("val_reduction", "mean"), q2_mean=("test_q2_gain", "mean"),
              rm_project_total=("rm_project_total", "first"), rm_fft_total=("rm_fft_total", "first")).reset_index()
    s["test_se"] = s.test_std / np.sqrt(s.n)
    s.to_csv(res / "summary.csv", index=False)
    rows = []
    ref = s[(s.n_steps == 40) & (s.M == 0)]
    if len(ref):
        ref = ref.iloc[0]
        for _, r in s[(s.n_steps == 8) & (s.M > 0)].iterrows():
            for conv in ("rm_project_total", "rm_fft_total"):
                std = 0.0 if np.isnan(r.test_std) else r.test_std
                se = 0.0 if np.isnan(r.test_se) else r.test_se
                gap = ref.test_mean - r.test_mean
                rows.append({"question": "Ns 8 + triplets reaches DBP-X Ns 40 (no triplets)", "convention": conv,
                             "arm": f"Ns8 M{r.M} K{r.K}", "arm_test": r.test_mean, "arm_std": std,
                             "arm_rm": r[conv], "ref_test": ref.test_mean, "ref_rm": ref[conv],
                             "rm_not_higher": bool(r[conv] <= ref[conv]), "gap_pp": gap,
                             "reached": bool(gap <= 0), "within_2std_and_2se": bool(gap < 2 * std and gap < 2 * se)
                             if std else False})
    pd.DataFrame(rows).to_csv(res / "verdicts.csv", index=False)
    plot(s, exp_dir("E052") / ("results_smoke" if smoke else "figures"))
    print(s.to_string(index=False))
    return 0


def plot(s, figs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, col, title in ((axes[0], "rm_project_total", "RM, прямая свёртка (конвенция проекта)"),
                           (axes[1], "rm_fft_total", "RM, БПФ / overlap-save (Xiao)")):
        base = s[s.M == 0].sort_values("n_steps")
        ax.plot(base[col], base.test_mean, "k-o", label="DBP-X без триплетов")
        for x, y, ns in zip(base[col], base.test_mean, base.n_steps):
            ax.annotate(f"Ns={ns}", (x, y), fontsize=7)
        for ns, mk in zip(NS, "sD^v"):
            p = s[(s.n_steps == ns) & (s.M > 0)].sort_values(col)
            if len(p):
                ax.errorbar(p[col], p.test_mean, yerr=p.test_std, fmt=mk, capsize=2, label=f"Ns={ns} + триплеты (крест)")
        ax.set_xscale("log")
        ax.set_xlabel(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("снижение BER на test, %")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "E052_dbpx_triplets_vs_rm.png", dpi=150)
    plt.close(fig)


def main(argv) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    action = argv[0] if argv else ""
    if action == "all":
        from concurrent.futures import ProcessPoolExecutor

        todo = [("trip", ns, smoke) for ns in ((8,) if smoke else NS)] + \
               [("ref", ns, smoke) for ns in (() if smoke else REF_NS)]
        jobs = 1 if smoke else len(todo)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for msg in ex.map(_job, todo):
                print(f"[{time.strftime('%F %T')}] {msg}", flush=True)
        return 0
    if action == "summarize":
        return summarize(smoke)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
