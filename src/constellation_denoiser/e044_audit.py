"""E044 (phase 0): audit of the triplet definitions and of the ridge coefficients, no training.

    PYTHONPATH=src python -m constellation_denoiser.e044_audit

1. definitions of T1, T2, T3 (triplets.triplet_terms) and of the "K strongest" selection (E034/E036/E038), checked
   in code: number of terms, share of axial terms, hypothesis "T1/T2 at M = 3 are the 2M + 1 axial triplets";
2. data: what the repository knows about the generator (nothing beyond the files) and the raw BER values;
3. symmetries C_mn = C_nm = C_{-m,-n} = -C*_{-m,n} of the closed-form ridge coefficients (E039 results/coef_M*.csv),
   per ordered coefficient C = merged weight / multiplicity (the model merges (m, n) and (n, m));
4. |C_mn| maps in dB relative to |C_00| with the boundaries |mn| = K_th, and the shares of axial terms (count and
   sum |C|) in the full sets and in the E038 top-64 set.
Writes results/definitions.csv, symmetry.csv, axis_shares.csv, data_facts.json and figures E044_coef_map_M*.png.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .runner import atomic_write_json, exp_dir, results_root
from .triplets import triplet_terms


def is_axis(m: int, n: int) -> bool:
    return m == 0 or n == 0


def definitions(out) -> pd.DataFrame:
    rows = []
    for arm in ("T1", "T2", "T3"):
        for M in (3, 5, 21):
            terms = triplet_terms(M, arm)
            rows.append({"arm": arm, "M": M, "n_terms": len(terms), "n_axis": sum(is_axis(m, n) for m, n in terms),
                         "two_M_plus_1": 2 * M + 1,
                         "definition": {"T1": "axial (m = 0 or n = 0), C purely imaginary (real form b, C = i b)",
                                        "T2": "axial, complex C", "T3": "all merged (m <= n) with |m|, |n|, |m + n| <= M"}[arm],
                         "source": "из кода: triplets.triplet_terms, triplets.FileGram (imaginary=True for T1)"})
    frame = pd.DataFrame(rows)
    frame["hypothesis_T1T2_equal_axes"] = [(r.arm in ("T1", "T2")) and r.n_terms == r.two_M_plus_1 == r.n_axis
                                           if r.arm != "T3" else None for r in frame.itertuples()]
    frame.to_csv(out / "definitions.csv", index=False)
    return frame


SELECTION = {
    "E034/E036 K strongest": "из кода (triplets_ext.prune_replica): per bootstrap replica the full T3 + FIR7 ridge fit "
                             "(lambda by validation) ranks the merged triplet terms by |C_j| * rms(feature_j) on train; the K "
                             "strongest + the 7 FIR taps are refitted from the sub-matrix of the same normal equations; the "
                             "selected sets were not saved",
    "E038 K strongest": "из кода (triplets_ext.rank_terms, grids_e024.e038_select): the same ranking on all 30 train files, "
                        "M = 15; the first 64 terms are saved in experiments/E038_dbp_triplets/results/terms.npz",
}


def load_coef(M: int) -> dict[tuple[int, int], complex]:
    path = exp_dir("E039") / "results" / f"coef_M{M}.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    frame = frame[frame.kind == "triplet"]
    return {(int(m), int(n)): complex(re, im) for m, n, re, im in zip(frame.m, frame.n, frame.re, frame.im)}


def ordered(merged: dict) -> dict:
    """Ordered C_mn from merged weights: C = w (m = n), w / 2 for both orders (m < n)."""
    out = {}
    for (m, n), w in merged.items():
        c = w if m == n else w / 2
        out[(m, n)] = out[(n, m)] = c
    return out


def symmetry(out) -> pd.DataFrame:
    rows = []
    for M in (15, 21, 31, 51):
        merged = load_coef(M)
        if not merged:
            continue
        c = ordered(merged)
        scale = np.median([abs(v) for v in c.values()])
        axis = [(k, v) for k, v in c.items() if is_axis(*k)]
        re_rel = np.array([abs(v.real) / abs(v) for _, v in axis if abs(v) > 0])
        neg = np.array([abs(c[(-m, -n)] - v) / abs(v) for (m, n), v in c.items() if (-m, -n) in c and abs(v) > scale * 1e-3])
        flip = np.array([abs(c[(-m, n)] + np.conj(v)) / abs(v) for (m, n), v in c.items()
                         if (-m, n) in c and m != 0 and abs(v) > scale * 1e-3])
        rows.append({"M": M, "axis_terms": len(axis), "axis_re_over_abs_median": float(np.median(re_rel)),
                     "axis_re_over_abs_max": float(re_rel.max()),
                     "C_minus_m_minus_n_rel_dev_median": float(np.median(neg)), "C_minus_m_minus_n_rel_dev_max": float(neg.max()),
                     "C_minus_m_n_plus_conj_rel_dev_median": float(np.median(flip)),
                     "C_minus_m_n_plus_conj_rel_dev_max": float(flip.max()),
                     "C_mn_eq_C_nm": "by construction (merged terms), not testable",
                     "note": "ridge fitted on the received x, not on clean z: deviations are a result"})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "symmetry.csv", index=False)
    return frame


def axis_shares(out) -> pd.DataFrame:
    rows = []
    for M in (15, 21, 31, 51):
        merged = load_coef(M)
        if not merged:
            continue
        total = sum(abs(v) for v in merged.values())
        ax = [k for k in merged if is_axis(*k)]
        rows.append({"set": f"full T3 M={M}", "n_terms": len(merged), "axis_share_count": len(ax) / len(merged),
                     "axis_share_abs_C": sum(abs(merged[k]) for k in ax) / total})
    terms_path = exp_dir("E038") / "results" / "terms.npz"
    merged15 = load_coef(15)
    if terms_path.is_file() and merged15:
        top = [tuple(int(v) for v in t) for t in np.load(terms_path)["terms"]]
        for k in (16, 32, 64):
            sel = top[:k]
            total = sum(abs(merged15[t]) for t in sel)
            ax = [t for t in sel if is_axis(*t)]
            rows.append({"set": f"E038 top-{k} (M=15)", "n_terms": k, "axis_share_count": len(ax) / k,
                         "axis_share_abs_C": sum(abs(merged15[t]) for t in ax) / total})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "axis_shares.csv", index=False)
    return frame


def plot_maps(figures, k_th_fracs=(0.25, 1.0)) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    for M in (21, 31):
        merged = load_coef(M)
        if not merged:
            continue
        c = ordered(merged)
        grid = np.full((2 * M + 1, 2 * M + 1), np.nan)
        for (m, n), v in c.items():
            grid[m + M, n + M] = abs(v)
        ref = abs(c[(0, 0)])
        db = 20 * np.log10(grid / ref)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(db, origin="lower", extent=(-M - 0.5, M + 0.5, -M - 0.5, M + 0.5), cmap="viridis", vmin=-60, vmax=0)
        nn = np.linspace(0.5, M, 200)
        for frac, style in zip(k_th_fracs, ("--", ":")):
            k = frac * M * M
            for sx in (1, -1):
                for sy in (1, -1):
                    mm = k / nn
                    keep = mm <= M
                    ax.plot(sx * nn[keep], sy * mm[keep], style, color="white", linewidth=0.8)
            ax.plot([], [], style, color="white", label=f"|mn| = {frac:g} M^2")
        ax.set_xlabel("n")
        ax.set_ylabel("m")
        ax.set_title(f"|C_mn| relative to |C_00|, dB (ridge, M = {M})")
        fig.colorbar(im, ax=ax, label="dB")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(figures / f"E044_coef_map_M{M}.png", dpi=160)
        plt.close(fig)


def data_facts(out) -> dict:
    e039 = exp_dir("E039") / "results" / "raw_ber.json"
    raw = json.loads(e039.read_text()) if e039.is_file() else {}
    facts = {
        "generator_in_repository": "нет: в репозитории только CSV (tx_I, tx_Q, rx_I, rx_Q) без метаданных (из логов E024)",
        "link_from_literature": "Богданов, Сидельников, Квантовая электроника 51 (2021): 20 x 100 км SSMF, NF 4.5 дБ, "
                                "alpha 0.2 дБ/км, beta2 -21 пс^2/км, gamma 1.3 1/(Вт км), 16-QAM 32 Гбод, RRC 0.1, "
                                "идеальная КХД; что это те же данные — не проверено",
        "launch_power": "не известна (вопрос научному руководителю)",
        "noise": "только NF усилителей по статье; уровень шума в файлах не известен",
        "raw_ber_test_window31": raw.get("test", {}).get("raw_ber_window31"),
        "raw_ber_all_300_files": raw.get("all_300_files", {}).get("raw_ber_window31"),
        "raw_ber_test_all_symbols": raw.get("test", {}).get("raw_ber_all_symbols"),
        "value_9.58e-3": "источник не найден в репозитории; ближайшее — test по всем символам 9.570e-3",
    }
    atomic_write_json(out / "data_facts.json", facts)
    return facts


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E044")
    out = results_root("E044", smoke)
    figures = out / "figures" if smoke else directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    print(definitions(out).to_string(index=False))
    pd.DataFrame([{"selection": k, "definition": v} for k, v in SELECTION.items()]).to_csv(out / "selection.csv", index=False)
    print(symmetry(out).round(3).to_string(index=False))
    print(axis_shares(out).round(3).to_string(index=False))
    print(json.dumps(data_facts(out), ensure_ascii=False, indent=1))
    plot_maps(figures)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
