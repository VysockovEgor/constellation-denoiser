"""E061: triplets T3 + FIR7 (M = 15) made cheaper - tied coefficients (mirror, g(m n)) and pruning with refit.

    PYTHONPATH=src python -m constellation_denoiser.e061_triplet_opt            (SMOKE=1: 2 files)
    PLOT_ONLY=1 ... e061_triplet_opt                                            (redraw from results.csv)

One complex Gram G = A^H A, b = A^H r over the 30 train files (A: 368 triplet + 7 FIR features of the window-31
centres, r = s - x). A variant is a kept term set S and a tying B: c = B theta, theta real (Re, Im per group),
theta = solve(Re(B^H G B) + lambda * mean diag * I, Re(B^H b)). lambda and K by validation; test once at the end.
Writes results/results.csv, results/symmetry.json, results/verdicts.json, figures/E061_triplet_optimization.png.
Rule and conventions: config.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .core import BIT_TABLE, CONSTELLATION
from .runner import default_files, exp_dir, results_root
from .triplets import EDGE, features, triplet_terms

EXP = "E061"
M, FIR = 15, 7
P_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 225)]


# --------------------------------------------------------------------------- data, features, BER

def padded(rx: np.ndarray, pad: int) -> np.ndarray:
    return np.concatenate([np.zeros(pad, complex), rx, np.zeros(pad, complex)])


class Split:
    """Window-31 centres of every file of a split; features are built on demand per file."""

    def __init__(self, fs, terms, pad):
        self.fs, self.terms, self.pad = fs, terms, pad

    def __len__(self):
        return self.fs.n_files

    def file(self, f):
        rx = self.fs.rx_c[f]
        c = np.arange(EDGE, len(rx) - EDGE)
        return features(padded(rx, self.pad), c, self.terms, FIR, self.pad), rx[c], self.fs.tx_c[f][c]


def bit_errors(pred: np.ndarray, tx: np.ndarray) -> np.ndarray:
    """(N, C) complex predictions -> (C,) bit errors against (N,) transmitted symbols."""
    idx = np.argmin(np.abs(pred[..., None] - CONSTELLATION), axis=-1)
    ref = BIT_TABLE[np.argmin(np.abs(tx[:, None] - CONSTELLATION), axis=-1)]
    return (BIT_TABLE[idx] != ref[:, None, :]).sum(axis=(0, 2))


def per_file_errors(split: Split, coefs: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(errors per file and column (F, C), raw errors per file (F,), bits per file)."""
    errs, raw = [], []
    for f in range(len(split)):
        a, x, tx = split.file(f)
        errs.append(np.concatenate([bit_errors(x[:, None] + a @ coefs[:, i:i + 24], tx)
                                    for i in range(0, coefs.shape[1], 24)]))
        raw.append(bit_errors(x[:, None], tx)[0])
    return np.array(errs), np.array(raw), 4 * len(x)


# --------------------------------------------------------------------------- tying and solving

def groups_free(kept):
    return [[(j, 1)] for j in kept]


def partner(t):
    m, n = t
    return (-n, -m)


def groups_mirror(kept, terms, conj: bool):
    pos = {t: i for i, t in enumerate(terms)}
    kept_set, seen, out = set(kept), set(), []
    for j in kept:
        if j in seen:
            continue
        q = pos[partner(terms[j])]
        if q != j and q in kept_set:
            out.append([(j, 1), (q, -1 if conj else 1)])
            seen.update((j, q))
        else:
            out.append([(j, 1)])
            seen.add(j)
    return out


def p_key(t, coarse: bool):
    p = t[0] * t[1]
    if not coarse:
        return p
    a = abs(p)
    return (int(np.sign(p)), next(i for i, (lo, hi) in enumerate(P_BINS) if lo <= a <= hi))


def groups_p(kept, terms, coarse: bool):
    axial, by = [], {}
    for j in kept:
        m, n = terms[j]
        if m == 0 or n == 0:
            axial.append([(j, 1)])
        else:
            by.setdefault(p_key(terms[j], coarse), []).append((j, 1))
    return axial + list(by.values())


def basis(groups, n_terms: int) -> np.ndarray:
    """Complex (n_terms + FIR, 2 (n_groups + FIR)): c = B theta; conj flag -1 means c = Re - i Im."""
    g_all = groups + [[(n_terms + l, 1)] for l in range(FIR)]
    B = np.zeros((n_terms + FIR, 2 * len(g_all)), complex)
    for gi, members in enumerate(g_all):
        for j, s in members:
            B[j, 2 * gi] = 1
            B[j, 2 * gi + 1] = 1j * s
    return B


def solve_all(G, b, B, lambdas) -> np.ndarray:
    H = np.real(B.conj().T @ G @ B)
    h = np.real(B.conj().T @ b)
    scale = float(np.mean(np.diag(H)))
    I = np.eye(len(H))
    return np.stack([B @ np.linalg.solve(H + lam * scale * I, h) for lam in lambdas], axis=1)


# --------------------------------------------------------------------------- orders for pruning

def orders(terms, c0):
    off = [j for j, (m, n) in enumerate(terms) if m != 0 and n != 0]
    amp = sorted(off, key=lambda j: -abs(c0[j]))
    mn = sorted(off, key=lambda j: (abs(terms[j][0] * terms[j][1]), max(map(abs, terms[j])),
                                    abs(terms[j][0]) + abs(terms[j][1]), terms[j]))
    mx = sorted(off, key=lambda j: (max(map(abs, terms[j])), abs(terms[j][0] * terms[j][1]), terms[j]))
    return {"T4": amp, "T5mn": mn, "T5max": mx}


def symmetry_stats(terms, c0) -> dict:
    pos = {t: i for i, t in enumerate(terms)}
    eq, cj, mod = [], [], []
    for j, t in enumerate(terms):
        q = pos[partner(t)]
        if q <= j:
            continue                  # each pair once; self-partners (q == j) skipped
        r = c0[j] / c0[q]
        eq.append(abs(np.angle(r)))
        cj.append(abs(np.angle(c0[j] / np.conj(c0[q]))))
        mod.append(abs(abs(r) - 1))
    q = lambda v, p: float(np.percentile(v, p))
    return {"n_pairs": len(eq), "abs_ratio_minus_1_median": q(mod, 50), "abs_ratio_minus_1_p90": q(mod, 90),
            "phase_equal_median_rad": q(eq, 50), "phase_equal_p90_rad": q(eq, 90),
            "phase_conj_median_rad": q(cj, 50), "phase_conj_p90_rad": q(cj, 90),
            "tie": "equality" if np.median(eq) < np.median(cj) else "conjugation"}


def p_spread(terms, c0) -> dict:
    """Spread of the free T0 coefficients inside the groups of equal p = m n (off-axis)."""
    by = {}
    for j, (m, n) in enumerate(terms):
        if m and n:
            by.setdefault(m * n, []).append(c0[j])
    num = den = 0.0
    rel = []
    for v in by.values():
        if len(v) < 2:
            continue
        v = np.array(v)
        d = np.abs(v - v.mean()) ** 2
        num += d.sum()
        den += (np.abs(v) ** 2).sum()
        rel.append(np.sqrt(d.mean()) / np.abs(v).mean())
    # the same statistic for a random grouping with the same group sizes (baseline)
    rng = np.random.default_rng(0)
    vals = np.array([c0[j] for j, (m, n) in enumerate(terms) if m and n])
    sizes = [len(v) for v in by.values()]
    base = []
    for _ in range(200):
        perm, k, nb = rng.permutation(vals), 0, 0.0
        for s in sizes:
            g = perm[k:k + s]
            k += s
            if s > 1:
                nb += (np.abs(g - g.mean()) ** 2).sum()
        base.append(np.sqrt(nb / den))
    return {"n_groups": len(by), "n_groups_multi": len(rel), "pooled_rel_rms": float(np.sqrt(num / den)),
            "median_group_rel_rms": float(np.median(rel)), "random_grouping_pooled_rel_rms": float(np.median(base))}


# --------------------------------------------------------------------------- main

def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    res = results_root(EXP, smoke)
    if os.environ.get("PLOT_ONLY", "0") == "1":
        plot(pd.read_csv(res / "results.csv"), smoke)
        return 0
    from .window_data import load_fileset, resolve_paths

    cfg = json.loads((exp_dir(EXP) / "config.json").read_text(encoding="utf-8"))
    lambdas, ks = cfg["lambda_grid"], cfg["K_offaxis"]
    files = default_files(30, smoke)
    data = {k: load_fileset(resolve_paths(v)) for k, v in files.items()}
    terms = triplet_terms(M, "T3")
    n = len(terms)
    pad = max(max(max(abs(m), abs(q), abs(m + q)) for m, q in terms), FIR // 2) + 1
    train, val, test = (Split(data[k], terms, pad) for k in ("train", "validation", "test"))

    G = np.zeros((n + FIR, n + FIR), complex)
    b = np.zeros(n + FIR, complex)
    for f in range(len(train)):
        a, x, s = train.file(f)
        G += a.conj().T @ a
        b += a.conj().T @ (s - x)
    print("Gram done", flush=True)

    axial = [j for j, (m, q) in enumerate(terms) if m == 0 or q == 0]
    full = list(range(n))
    configs = []          # (variant, K_off, kept, groups)

    def add(variant, k, kept, groups):
        configs.append({"variant": variant, "K": k, "kept": kept, "groups": groups})

    # T0 first: its validation-chosen coefficients define the |C| order and the symmetry statistics
    add("T0", n - len(axial), full, groups_free(full))
    val_cache = {}

    def evaluate_val(cfgs):
        cols, owners = [], []
        for ci, c in enumerate(cfgs):
            coefs = solve_all(G, b, basis(c["groups"], n), lambdas)
            c["coefs"] = coefs
            cols.append(coefs)
            owners += [ci] * len(lambdas)
        C = np.concatenate(cols, axis=1)
        errs, raw, bits = per_file_errors(val, C)
        val_cache["raw"] = raw.sum()
        red = 100 * (raw.sum() - errs.sum(0)) / raw.sum()
        for ci, c in enumerate(cfgs):
            r = red[[i for i, o in enumerate(owners) if o == ci]]
            best = int(np.argmax(r))
            c.update(lam=lambdas[best], val=float(r[best]), coef=c["coefs"][:, best])

    evaluate_val(configs)
    t0 = configs[0]
    control = {"T0_val": t0["val"], "E060_full_val": 67.88375295737843, "tol": 0.05,
               "passed": smoke or abs(t0["val"] - 67.88375295737843) <= 0.05}
    print("control", control, flush=True)
    res.mkdir(parents=True, exist_ok=True)
    (res / "control.json").write_text(json.dumps(control, indent=1), encoding="utf-8")
    if not control["passed"]:
        print("T0 does not reproduce E060: stop")
        return 3
    c0 = t0["coef"][:n]
    sym = symmetry_stats(terms, c0)
    spread = p_spread(terms, c0)
    (res / "symmetry.json").write_text(json.dumps({"mirror": sym, "p_groups": spread}, indent=1), encoding="utf-8")
    print(json.dumps({"mirror": sym, "p_groups": spread}, indent=1), flush=True)

    conj = sym["tie"] == "conjugation"
    stage = []
    stage.append({"variant": "T1", "K": n - len(axial), "kept": full, "groups": groups_mirror(full, terms, conj)})
    stage.append({"variant": "T2", "K": n - len(axial), "kept": full, "groups": groups_p(full, terms, False)})
    stage.append({"variant": "T2c", "K": n - len(axial), "kept": full, "groups": groups_p(full, terms, True)})
    ords = orders(terms, c0)
    for name, order in ords.items():
        for k in ks:
            kept = sorted(axial + order[:k])
            stage.append({"variant": name, "K": k, "kept": kept, "groups": groups_free(kept)})
    evaluate_val(stage)
    configs += stage

    def rm_b(c):
        return 8 * len(c["kept"]) + 4 * len(c["groups"]) + 4 * FIR

    thr = t0["val"] - 0.5
    ties = [c for c in configs if c["variant"] in ("T1", "T2")]
    ok = [c for c in ties if c["val"] >= thr]
    tie = min(ok, key=rm_b)["variant"] if ok else max(ties, key=lambda c: c["val"])["variant"]
    pr = {}
    for name in ords:
        rows = [c for c in configs if c["variant"] == name]
        good = [rm_b(c) for c in rows if c["val"] >= thr]
        pr[name] = (min(good) if good else np.inf, -np.mean([c["val"] for c in rows]))
    prune = min(pr, key=lambda k: pr[k])
    t6 = []
    for k in ks:
        kept = sorted(axial + ords[prune][:k])
        groups = groups_mirror(kept, terms, conj) if tie == "T1" else groups_p(kept, terms, False)
        t6.append({"variant": "T6", "K": k, "kept": kept, "groups": groups})
    evaluate_val(t6)
    configs += t6
    choice = {"T6_tying": tie, "T6_pruning": prune, "pruning_scores": {k: [float(v[0]), float(-v[1])] for k, v in pr.items()}}
    print("choice", choice, flush=True)

    # test: once, at the validation-chosen lambda of every (variant, K)
    C = np.stack([c["coef"] for c in configs], axis=1)
    errs, raw, bits = per_file_errors(test, C)
    raw_pool = raw.sum()
    red_pool = 100 * (raw_pool - errs.sum(0)) / raw_pool
    red_file = 100 * (raw[:, None] - errs) / raw[:, None]
    ber_t0 = errs[:, 0] / bits
    from .q2 import q2_db

    rows = []
    for i, c in enumerate(configs):
        diff = (errs[:, 0] - errs[:, i]) / bits                       # paired per-file BER difference vs T0
        n_off = sum(1 for j in c["kept"] if terms[j][0] and terms[j][1])
        rows.append({"variant": c["variant"], "K_offaxis": c["K"], "n_terms": len(c["kept"]), "n_axial": len(c["kept"]) - n_off,
                     "lambda": c["lam"], "n_coef_triplet": len(c["groups"]), "n_coef_fir": FIR,
                     "rm_naive_a": 12 * len(c["kept"]) + 4 * FIR, "rm_shared_b": rm_b(c),
                     "val_reduction": c["val"], "test_reduction": float(red_pool[i]),
                     "test_file_mean": float(red_file[:, i].mean()), "test_file_std": float(red_file[:, i].std(ddof=1)),
                     "test_file_min": float(red_file[:, i].min()), "test_file_max": float(red_file[:, i].max()),
                     "test_ber": float(errs[:, i].sum() / (bits * len(raw))), "test_q2_db": q2_db(errs[:, i].sum() / (bits * len(raw))),
                     "diff_vs_T0_pp": float(red_pool[i] - red_pool[0]),
                     "diff_vs_T0_se_pp": float(100 * diff.std(ddof=1) / np.sqrt(len(diff)) / (raw_pool / (bits * len(raw))))})
    frame = pd.DataFrame(rows)
    frame.to_csv(res / "results.csv", index=False)
    verdicts(frame, cfg, choice, sym, res)
    plot(frame, smoke)
    return 0


def verdicts(frame: pd.DataFrame, cfg: dict, choice: dict, sym: dict, res) -> None:
    t0 = frame[frame.variant == "T0"].iloc[0]
    thr = t0.val_reduction - 0.5
    out = {"rule": cfg["rule_draft"], "T1_tie": sym["tie"], **choice, "T0": t0.to_dict(), "variants": {}}
    for v, g in frame.groupby("variant", sort=False):
        best = g.loc[g.val_reduction.idxmax()]
        ok = g[g.val_reduction >= thr]
        cheap = ok.loc[ok.rm_shared_b.idxmin()] if len(ok) else None
        out["variants"][v] = {"best_by_val": best.to_dict(), "cheapest_without_loss": None if cheap is None else cheap.to_dict()}
    t2 = frame[frame.variant == "T2"].iloc[0]
    out["g_mn_helps"] = bool(t2.val_reduction >= thr and t2.rm_shared_b < t0.rm_shared_b)
    out["g_mn_val_loss_pp"] = float(t0.val_reduction - t2.val_reduction)
    cheap = frame[(frame.rm_shared_b <= 1400)]
    out["comparable_to_dbpx_at_1400"] = bool((cheap.test_reduction >= 99.07 - 5).any())
    out["best_test_at_rm_b_le_1400"] = cheap.loc[cheap.val_reduction.idxmax()].to_dict() if len(cheap) else None
    (res / "verdicts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


# --------------------------------------------------------------------------- figure

def plot(frame: pd.DataFrame, smoke: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from .q2 import q2_db

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    RAW = 0.009551859551859552
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
    fig, (a, q) = plt.subplots(1, 2, figsize=(15, 6.4), facecolor=SURF, sharex=True)
    to_q2 = lambda red: [q2_db(RAW * (1 - r / 100)) for r in red]

    def both(x, red, **kw):
        a.plot(x, red, **kw)
        kw.pop("label", None)
        q.plot(x, to_q2(red), **kw)

    # E056 arms: upper envelope by validation (as E056_ablation_vs_rm.png), grey-ish
    old = pd.read_csv(exp_dir("E056") / "results" / "ablation_all_configs.csv")
    arms = {1: ("E056: 1 все триплеты", "#9a9a96", "o"), 2: ("E056: 2 Керр без памяти + внеосевые", "#c9b37e", "v"),
            3: ("E056: 3 Керр с памятью, без внеосевых", "#9fbf9a", "^"),
            4: ("E056: 4 КИХ7 + Керр с памятью + внеосевые (A1)", "#d08c8c", "D"),
            5: ("E056: 5 Керр без памяти, без внеосевых", "#b8a6cc", "P")}
    for arm, (lab, col, mk) in arms.items():
        part, env, best = old[old.arm == arm].sort_values("rm"), [], -1e9
        for _, r in part.iterrows():
            if r.val_mean > best:
                best = r.val_mean
                env.append(r)
        env = pd.DataFrame(env)
        both(env.rm, env.test_mean, color=col, lw=1.4, marker=mk, ms=4, label=lab, zorder=2)

    new = {"T0": ("T0 свободные 368", "#0b0b0b", "*", 14), "T1": ("T1 зеркальная связь", "#2a78d6", "s", 9),
           "T2": ("T2 общий g(m·n)", "#eb6834", "D", 9), "T2c": ("T2c g(|m·n|), корзины", "#eb6834", "d", 9)}
    curves = {"T4": ("T4 отброс по |C| + пересчёт", "#1f9e89", "o"), "T5mn": ("T5 отброс по |m·n| + пересчёт", "#7a4fc4", "o"),
              "T5max": ("T5 отброс по max(|m|,|n|) + пересчёт", "#c44f9a", "o"), "T6": ("T6 связь + отброс", "#173f73", "o")}
    for v, (lab, col, mk) in curves.items():
        g = frame[frame.variant == v].sort_values("rm_shared_b")
        if len(g):
            both(g.rm_shared_b, g.test_reduction, color=col, lw=2.2 if v == "T6" else 1.8, marker=mk, ms=5,
                 ls="-" if v in ("T4", "T6") else "--", label=lab, zorder=4)
    for v, (lab, col, mk, ms) in new.items():
        g = frame[frame.variant == v]
        if len(g):
            both(g.rm_shared_b, g.test_reduction, color=col, lw=0, marker=mk, ms=ms, mec=SURF, mew=1.2, label=lab, zorder=6)
    dbp = pd.DataFrame({"rm": [1235.5, 1421.2], "red": [99.07, 99.12]})
    both(dbp.rm, dbp.red, color="#0b0b0b", lw=0, marker="X", ms=11, label="DBP-X обучен, Ns 20 (L 63 / 255)", zorder=7)
    unt = pd.DataFrame({"rm": [618.6, 894.4, 1449.5], "red": [91.72, 94.91, 96.70]})
    both(unt.rm, unt.red, color="#52514e", lw=1.2, ls=":", marker="x", ms=9, mew=2,
         label="DBP-X без обучения, Ns 8 / 12 / 20", zorder=7)

    for ax in (a, q):
        ax.set_facecolor(SURF)
        ax.set_xscale("log")
        ax.grid(color=GRID, lw=0.8)
        ax.axvline(1400, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("RM на символ (триплеты — прямой счёт, общие коэффициенты;\nDBP-X — БПФ-подсчёт)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a.set_ylim(-10, 102)
    a.set_ylabel("снижение BER на тесте, %")
    q.set_ylabel("Q², дБ")
    q.axhline(q2_db(RAW), color=INK2, lw=0.8, ls="--")
    q.text(32, q2_db(RAW) + 0.1, "без компенсации", color=INK2, fontsize=10)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    out = exp_dir(EXP) / ("results_smoke" if smoke else "figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "E061_triplet_optimization.png", dpi=200, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
