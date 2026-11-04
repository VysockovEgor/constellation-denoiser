"""E039: diagnostics of the triplet models without training (existing results and closed-form fits).

    PYTHONPATH=src python -m constellation_denoiser.e039_diagnostics [defs|rm|structure|raw|all]
    (SMOKE=1: 2/1/1 files, M = 5 and 7 only; writes results_smoke/)

defs       table of the E038 and K2048 definitions (config, README, summaries) -> results/definitions.csv
rm         three RM conventions for every triplet model of E031-E038 (rm_group_convention) -> results/rm_conventions.csv
structure  coefficients C_mn (M = 15, 21 from E035 warm starts; M = 31, 51 fitted here in closed form on all 30 train
           files, lambda by validation): singular values, parametric fits g(u, v) with u = mn, v = (m - n)^2
           (polynomials of degree 2-4, 5 x 5 Gaussian grid), explained variance and the BER of the substituted
           coefficients (no retraining) against M0 = FIR7 only; |C| maps in dB -> results/structure_*.csv, figures
raw        raw BER of the splits and the conversion table reduction -> Delta Q2 -> results/raw_ber.json, q2_table.csv
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT
from .q2 import q2_db
from .runner import atomic_write_json, default_files, exp_dir, results_root
from .triplets import EDGE, FileGram, ber_columns, predict, solve, triplet_terms

FIR = 7
LAMBDAS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
BLOCK = 4096


# --------------------------------------------------------------------------- 1. definitions

def definitions(out) -> pd.DataFrame:
    rows = []
    e038 = exp_dir("E038") / "results" / "dbp_triplets_summary.csv"
    cfg = json.loads((exp_dir("E038") / "config.json").read_text(encoding="utf-8"))
    if e038.is_file():
        for _, r in pd.read_csv(e038).iterrows():
            rows.append({"name": f"E038 dbp_Ns{int(r.n_steps)}_K{int(r.k)}",
                         "inside": f"PhysDBP Ns={int(r.n_steps)} (L31, R7, sym) + {int(r.k)} strongest T3 terms of the "
                                   f"closed-form M={cfg['select_M']} fit, C = 0 at start, trained jointly",
                         "source": "experiments/E038_dbp_triplets/config.json, triplets_ext.DBPTriplets",
                         "rm": r.rm, "test_reduction": r.test_mean, "test_min": r.test_min, "test_max": r.test_max,
                         "seeds": int(r.n_seeds)})
    else:
        rows.append({"name": "E038", "inside": "не найдено", "source": "", "rm": np.nan, "test_reduction": np.nan,
                     "seeds": 0})
    e036 = exp_dir("E036") / "results" / "triplet_ext_summary.csv"
    if e036.is_file():
        s = pd.read_csv(e036)
        hit = s[s.variant == "T3FIR7_M51_K2048"]
        if len(hit):
            r = hit.iloc[0]
            rows.append({"name": "K2048 (E036 T3FIR7_M51_K2048)",
                         "inside": "closed-form T3 + FIR7: the 2048 strongest of 3 979 merged triplet terms of the M = 51 fit "
                                   "(|C| * rms feature on train), refitted with lambda by validation",
                         "source": "experiments/E036_triplets_ceiling/config.json, triplets_ext.prune_replica",
                         "rm": r.rm, "test_reduction": r.test_mean, "test_min": r.test_min, "test_max": r.test_max,
                         "seeds": int(r.n_replicas)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "definitions.csv", index=False)
    return frame


# --------------------------------------------------------------------------- 2. RM conventions

def rm_table(out) -> pd.DataFrame:
    from .rm_group_convention import group_dualpol_complex, rm_arxiv_2409_20023, rm_buffered, rm_buffered_bound

    rows = []

    def add(source, name, M, terms=None, n_terms=None, fir=FIR, extra_rm=0, test=np.nan):
        if terms is not None:
            n_terms = len(terms)
            rb, exact = rm_buffered(terms, fir), True
            dual = group_dualpol_complex(terms)
        else:
            rb, exact, dual = rm_buffered_bound(n_terms, M, fir), False, np.nan
        rows.append({"source": source, "model": name, "M": M, "n_terms": n_terms, "fir_len": fir,
                     "rm_repo": 12 * n_terms + 4 * fir + extra_rm, "rm_buffered_my_estimate": rb + extra_rm,
                     "buffered_exact": exact, "rm_arxiv_2409_20023": rm_arxiv_2409_20023(M, fir) + extra_rm,
                     "group_dualpol_complex_products": dual, "test_reduction": test})

    e031 = exp_dir("E031") / "results" / "triplet_summary.csv"
    if e031.is_file():
        for _, r in pd.read_csv(e031).iterrows():
            arm = "T3" if r.arm.startswith("T3") else r.arm
            fir = int(r.arm.replace("T3+FIR", "")) if "FIR" in r.arm else 0
            add("E031", f"{r.arm}_M{int(r.M)}", int(r.M), terms=triplet_terms(int(r.M), arm if arm != "T3" else "T3"),
                fir=fir, test=r.test_mean)
    for exp in ("E034", "E036"):
        path = exp_dir(exp) / "results" / "triplet_ext_summary.csv"
        if path.is_file():
            for _, r in pd.read_csv(path).iterrows():
                M = int(r.variant.split("_M")[1].split("_")[0])
                if r.variant.endswith("_full"):
                    add(exp, r.variant, M, terms=triplet_terms(M, "T3"), test=r.test_mean)
                else:
                    add(exp, r.variant, M, n_terms=int(r.K), test=r.test_mean)
    e038 = exp_dir("E038") / "results" / "terms.npz"
    s38 = exp_dir("E038") / "results" / "dbp_triplets_summary.csv"
    if e038.is_file() and s38.is_file():
        from .model_zoo import macs_phys

        terms = [tuple(int(v) for v in t) for t in np.load(e038)["terms"]]
        for _, r in pd.read_csv(s38).iterrows():
            k = int(r.k)
            if k:
                dbp = macs_phys(7, 31, True) * int(r.n_steps)
                add("E038", f"dbp_Ns{int(r.n_steps)}_K{k}", 15, terms=terms[:k], fir=0, extra_rm=dbp, test=r.test_mean)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "rm_conventions.csv", index=False)
    return frame


# --------------------------------------------------------------------------- 3. structure of C

def load_split(smoke: bool):
    from .window_data import load_fileset, resolve_paths

    files = default_files(30, smoke)
    return {k: load_fileset(resolve_paths(v)) for k, v in files.items()}


def fit_closed_form(train, val, M: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(coef of T3 + FIR7, coef of FIR7 only, lambda) on all train files; one normal matrix in memory."""
    terms = triplet_terms(M, "T3")
    gram = rhs = None
    for f in range(train.n_files):
        g = FileGram(train.rx_c[f], train.tx_c[f], terms, FIR, False, BLOCK)
        gram = g.gram if gram is None else gram + g.gram
        rhs = g.rhs if rhs is None else rhs + g.rhs
        del g
    val_tx = np.concatenate([val.tx_c[f][EDGE:-EDGE] for f in range(val.n_files)])

    def best(idx, sel_terms):
        coefs = np.stack([solve(gram[np.ix_(idx, idx)], rhs[idx], lam) for lam in LAMBDAS], axis=1)
        pv = np.concatenate([predict(val.rx_c[f], coefs, sel_terms, FIR, False, BLOCK) for f in range(val.n_files)])
        b = int(np.argmin(ber_columns(pv, val_tx)))
        return coefs[:, b], LAMBDAS[b]

    n = len(terms)
    full, lam = best(list(range(n + FIR)), terms)
    fir_only, _ = best(list(range(n, n + FIR)), [])
    return full, fir_only, lam


def coefficients(M: int, data, out) -> tuple[np.ndarray, np.ndarray, str]:
    terms = triplet_terms(M, "T3")
    warm = exp_dir("E035") / "results" / f"warm_M{M}.npz"
    if warm.is_file() and os.environ.get("SMOKE", "0") != "1":
        full = np.load(warm)["coef"]
        _, fir_only, _ = fit_closed_form(data["train"], data["validation"], 3)   # FIR-only part does not depend on M
        source = f"E035 warm start ({warm.relative_to(PROJECT_ROOT)})"
    else:
        full, fir_only, lam = fit_closed_form(data["train"], data["validation"], M)
        source = f"closed form here, lambda {lam:g}"
    pd.DataFrame({"m": [t[0] for t in terms] + [None] * FIR, "n": [t[1] for t in terms] + [None] * FIR,
                  "kind": ["triplet"] * len(terms) + ["fir"] * FIR, "re": full.real, "im": full.imag}) \
        .to_csv(out / f"coef_M{M}.csv", index=False)
    return full, fir_only, source


def design(terms, M: int, form: str) -> np.ndarray:
    u = np.array([m * n for m, n in terms], float) / (M * M)
    v = np.array([(m - n) ** 2 for m, n in terms], float) / (4 * M * M)
    if form.startswith("poly"):
        d = int(form[4:])
        cols = [u ** a * v ** b for a in range(d + 1) for b in range(d + 1 - a)]
        return np.stack(cols, axis=1).astype(complex)
    if form == "rbf5x5":
        cu, cv = np.linspace(u.min(), u.max(), 5), np.linspace(v.min(), v.max(), 5)
        su, sv = max(np.ptp(u) / 4, 1e-6), max(np.ptp(v) / 4, 1e-6)
        cols = [np.exp(-((u - a) / su) ** 2 - ((v - b) / sv) ** 2) for a in cu for b in cv]
        return np.stack(cols, axis=1).astype(complex)
    raise ValueError(form)


FORMS = ("poly2", "poly3", "poly4", "rbf5x5")


def structure(smoke: bool, out, figures) -> pd.DataFrame:
    data = load_split(smoke)
    Ms = [5, 7] if smoke else [15, 21, 31, 51]
    raw = {k: data[k].raw_ber(31) for k in ("validation", "test")}
    tx = {k: np.concatenate([data[k].tx_c[f][EDGE:-EDGE] for f in range(data[k].n_files)]) for k in ("validation", "test")}
    rows, svd_rows = [], []
    for M in Ms:
        terms = triplet_terms(M, "T3")
        n = len(terms)
        full, fir_only, source = coefficients(M, data, out)
        c = full[:n]
        mat = np.zeros((2 * M + 1, 2 * M + 1), complex)
        for (m, k), value in zip(terms, c):
            mat[m + M, k + M] = mat[k + M, m + M] = value
        sv = np.linalg.svd(mat, compute_uv=False)
        energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
        svd_rows.append({"M": M, **{f"energy_rank{r}": float(energy[r - 1]) for r in range(1, 6)}})
        plot_map(mat, M, figures)
        variants = {"free": full, "M0_fir_only": np.concatenate([np.zeros(n, complex), fir_only])}
        info = {"free": (2 * n, 1.0), "M0_fir_only": (0, np.nan)}
        for form in FORMS:
            phi = design(terms, M, form)
            theta, *_ = np.linalg.lstsq(phi, c, rcond=None)
            fit = phi @ theta
            ev = 1 - np.sum(np.abs(c - fit) ** 2) / np.sum(np.abs(c - c.mean()) ** 2)
            variants[form] = np.concatenate([fit, full[n:]])
            info[form] = (2 * phi.shape[1], float(ev))
        coefs = np.stack(list(variants.values()), axis=1)
        red = {}
        for split, fs in (("validation", data["validation"]), ("test", data["test"])):
            pred = np.concatenate([predict(fs.rx_c[f], coefs, terms, FIR, False, BLOCK) for f in range(fs.n_files)])
            bers = ber_columns(pred, tx[split])
            red[split] = 100 * (raw[split] - bers) / raw[split]
            red[split + "_ber"] = bers
        for j, name in enumerate(variants):
            params, ev = info[name]
            rows.append({"M": M, "variant": name, "coef_source": source, "real_params": params,
                         "param_share_of_free": params / (2 * n) if params else 0.0, "explained_variance": ev,
                         "val_reduction": red["validation"][j], "test_reduction": red["test"][j],
                         "q2_gain_db_test": q2_db(red["test_ber"][j]) - q2_db(raw["test"])})
        print(f"structure M={M}: {source}", flush=True)
    frame = pd.DataFrame(rows)
    for split in ("val", "test"):
        base = frame[frame.variant == "M0_fir_only"].set_index("M")[f"{split}_reduction"]
        free = frame[frame.variant == "free"].set_index("M")[f"{split}_reduction"]
        frame[f"{split}_gain_retained"] = [(r - base[M]) / (free[M] - base[M]) if free[M] != base[M] else np.nan
                                           for M, r in zip(frame.M, frame[f"{split}_reduction"])]
    frame.to_csv(out / "structure_parametric.csv", index=False)
    pd.DataFrame(svd_rows).to_csv(out / "structure_svd.csv", index=False)
    return frame


def plot_map(mat: np.ndarray, M: int, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    amp = np.abs(mat)
    db = 20 * np.log10(np.where(amp > 0, amp / amp.max(), np.nan))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(db, origin="lower", extent=(-M - 0.5, M + 0.5, -M - 0.5, M + 0.5), cmap="viridis", vmin=-40, vmax=0)
    ax.axhline(0, color="white", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="white", linewidth=0.6, linestyle="--")
    ax.plot([0], [0], marker="*", color="red", markersize=10, linestyle="", label="SPM (m = n = 0)")
    ax.plot([], [], color="white", linestyle="--", label="XPM (m = 0 or n = 0)")
    ax.plot([], [], marker="s", color=plot_style.TEXT, linestyle="", markersize=4, label="IFWM (other m, n)")
    fig.colorbar(im, ax=ax, label="|C_mn|, dB relative to max")
    ax.set_xlabel("n")
    ax.set_ylabel("m")
    ax.set_title(f"Closed-form triplet coefficients, M = {M}")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / f"E039_coef_map_M{M}.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- 4. raw BER

def raw_ber(smoke: bool, out) -> dict:
    from .experiment import ber_from_iq
    from .grids import split_files_all
    from .window_data import load_fileset, resolve_paths

    data = load_split(smoke)
    info = {k: {"files": fs.n_files, "raw_ber_window31": fs.raw_ber(31),
                "raw_ber_all_symbols": float(np.mean([ber_from_iq(np.column_stack((fs.rx_c[f].real, fs.rx_c[f].imag)),
                                                                   fs.tx_c[f]) for f in range(fs.n_files)]))}
            for k, fs in data.items()}
    if not smoke:
        allfs = load_fileset(resolve_paths(split_files_all()))
        info["all_300_files"] = {"files": allfs.n_files, "raw_ber_window31": allfs.raw_ber(31)}
    raw = info["test"]["raw_ber_window31"]
    levels = [60, 66, 72, 78, 85, 90, 95]
    table = pd.DataFrame({"reduction_pct": levels, "ber": [raw * (1 - r / 100) for r in levels],
                          "delta_q2_db": [q2_db(raw * (1 - r / 100)) - q2_db(raw) for r in levels]})
    table.to_csv(out / "q2_table.csv", index=False)
    info["q2_raw_db_test"] = q2_db(raw)
    info["plan_value"] = "about 9.58e-3"
    atomic_write_json(out / "raw_ber.json", info)
    return info


# --------------------------------------------------------------------------- verdict and gate

def verdict(config: dict, out, smoke: bool) -> dict:
    frame = pd.read_csv(out / "structure_parametric.csv")
    rule = config["structure_rule"]
    M = rule["M"] if not smoke else int(frame.M.max())
    part = frame[(frame.M == M) & frame.variant.isin(FORMS)]
    ok = part[(part.param_share_of_free <= rule["max_param_share"])]
    best = ok.loc[ok.val_gain_retained.idxmax()] if len(ok) else None
    passed = bool(best is not None and best.val_gain_retained >= rule["min_gain_retained"])
    result = {"rule": config["rule_draft"]["structure"], "M": int(M),
              "best_form_by_validation": None if best is None else best.variant,
              "param_share": None if best is None else float(best.param_share_of_free),
              "val_gain_retained": None if best is None else float(best.val_gain_retained),
              "test_gain_retained": None if best is None else float(best.test_gain_retained),
              "passed": passed, "e042": "open" if passed else "closed"}
    atomic_write_json(out / "structure_verdict.json", result)
    pd.DataFrame([{"hypothesis": f"структура C эксплуатируема (M = {M}: ≤ 5 % параметров, ≥ 95 % выигрыша, валидация)",
                   "value": result["val_gain_retained"], "verdict": "подтверждено" if passed else "не доказано",
                   "detail": f"{result['best_form_by_validation']}, доля параметров {result['param_share']}"}]) \
        .assign(rule_status=config["rule_draft"]["status"]).to_csv(out / "e039_verdicts.csv", index=False)
    gate = (PROJECT_ROOT / ("results_smoke" if smoke else "results") / "gate_e039.json")
    atomic_write_json(gate, result)
    return result


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E039")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    out = results_root("E039", smoke)
    figures = out / "figures" if smoke else directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    parts = set(argv or ["all"])
    if parts & {"defs", "all"}:
        print(definitions(out).to_string(index=False))
    if parts & {"rm", "all"}:
        print(f"RM: {len(rm_table(out))} моделей -> {out / 'rm_conventions.csv'}")
    if parts & {"raw", "all"}:
        r = raw_ber(smoke, out)
        print(f"сырой BER test {r['test']['raw_ber_window31']:.6g}")
    if parts & {"structure", "all"}:
        structure(smoke, out, figures)
        v = verdict(config, out, smoke)
        print(f"структура C: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
