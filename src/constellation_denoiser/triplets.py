"""E031: first-order perturbation (PB-NLC) triplets fitted in closed form.

    y_k = x_k + sum_{(m,n)} C_{m,n} x_{k+m} x*_{k+m+n} x_{k+n},   |m|, |n|, |m+n| <= M   (+ optional FIR on L symbols)

x are the received symbols (after CDC and phase recovery), the target is the transmitted symbol. Because
x_{k+m} x_{k+n} is symmetric in (m, n), the terms (m, n) and (n, m) are identical and are merged (m <= n).
Arms: T1 XPM type (m = 0 or n = 0) with purely imaginary C, T2 XPM type with complex C, T3 all triplets,
T3+FIR. Fit: ridge least squares on the residual s - x; the normal equations are accumulated per file in
blocks, so a bootstrap replica (train files resampled with replacement) is a weighted sum of per-file
matrices. Ridge lambda is relative to the mean diagonal and chosen by validation BER; test is reported.
Symbols outside a file are zero; BER is computed on the window-31 centres, as for the networks.
"""

from __future__ import annotations

import numpy as np

EDGE = 15          # window 31 centres: the same symbols as runner / FileSet.raw_ber(31)


def triplet_terms(M: int, arm: str) -> list[tuple[int, int]]:
    terms = []
    for m in range(-M, M + 1):
        for n in range(m, M + 1):
            if abs(m + n) > M:
                continue
            if arm in ("T1", "T2") and not (m == 0 or n == 0):
                continue
            terms.append((m, n))
    return terms


def ordered_count(terms: list[tuple[int, int]]) -> int:
    return sum(1 if m == n else 2 for m, n in terms)


def rm_counts(M: int, arm: str, fir_len: int = 0) -> dict:
    terms = triplet_terms(M, arm)
    fir = 4 * fir_len
    return {"n_terms": len(terms), "n_ordered": ordered_count(terms), "rm_naive": 12 * ordered_count(terms) + fir,
            "rm_sym": 12 * len(terms) + fir}


def features(x: np.ndarray, centres: np.ndarray, terms: list[tuple[int, int]], fir_len: int, pad: int) -> np.ndarray:
    """(len(centres), n_terms + fir_len) complex features of the padded symbol array ``x``."""
    k = centres + pad
    cols = [x[k + m] * np.conj(x[k + m + n]) * x[k + n] for m, n in terms]
    half = fir_len // 2
    cols += [x[k + l] for l in range(-half, half + 1)] if fir_len else []
    return np.stack(cols, axis=1)


class FileGram:
    """Per-file normal equations A^H A, A^H r (or their real form for purely imaginary C)."""

    def __init__(self, rx: np.ndarray, tx: np.ndarray, terms, fir_len: int, imaginary: bool, block: int):
        pad = max(max((max(abs(m), abs(n), abs(m + n)) for m, n in terms), default=0), fir_len // 2) + 1
        x = np.concatenate([np.zeros(pad, complex), rx, np.zeros(pad, complex)])
        centres = np.arange(EDGE, len(rx) - EDGE)
        p = len(terms) + fir_len
        dtype = float if imaginary else complex
        self.gram = np.zeros((p, p), dtype)
        self.rhs = np.zeros(p, dtype)
        for start in range(0, len(centres), block):
            c = centres[start:start + block]
            a = features(x, c, terms, fir_len, pad)
            r = tx[c] - rx[c]
            if imaginary:      # r = i A b with real b: [Re r; Im r] = [-Im A; Re A] b
                g = np.concatenate([-a.imag, a.real])
                t = np.concatenate([r.real, r.imag])
                self.gram += g.T @ g
                self.rhs += g.T @ t
            else:
                self.gram += a.conj().T @ a
                self.rhs += a.conj().T @ r


def solve(gram: np.ndarray, rhs: np.ndarray, lam: float) -> np.ndarray:
    scale = float(np.real(np.trace(gram))) / len(gram) if len(gram) else 1.0
    return np.linalg.solve(gram + lam * scale * np.eye(len(gram)), rhs)


def predict(rx: np.ndarray, coef: np.ndarray, terms, fir_len: int, imaginary: bool, block: int) -> np.ndarray:
    """Equalized window-31 centres of one file for one or several coefficient vectors (columns)."""
    pad = max(max((max(abs(m), abs(n), abs(m + n)) for m, n in terms), default=0), fir_len // 2) + 1
    x = np.concatenate([np.zeros(pad, complex), rx, np.zeros(pad, complex)])
    centres = np.arange(EDGE, len(rx) - EDGE)
    coef = coef.reshape(len(coef), -1)
    out = []
    for start in range(0, len(centres), block):
        c = centres[start:start + block]
        a = features(x, c, terms, fir_len, pad)
        correction = a @ (1j * coef if imaginary else coef)
        out.append(rx[c][:, None] + correction)
    return np.concatenate(out)


def ber_columns(pred: np.ndarray, tx_centres: np.ndarray) -> np.ndarray:
    from .experiment import ber_from_iq

    return np.array([ber_from_iq(np.column_stack((pred[:, j].real, pred[:, j].imag)), tx_centres)
                     for j in range(pred.shape[1])])


def fit_arm(train, val, test, M: int, arm: str, fir_len: int, lambdas: list[float], replicas: list[int],
            block: int) -> list[dict]:
    """One arm and M: per-file Grams once, then per bootstrap replica the validation-best lambda and test."""
    terms = triplet_terms(M, arm)
    imaginary = arm == "T1"
    grams = [FileGram(train.rx_c[f], train.tx_c[f], terms, fir_len, imaginary, block) for f in range(train.n_files)]
    rows = []
    n_files = train.n_files
    for replica in replicas:
        rng = np.random.default_rng(replica)
        counts = np.bincount(rng.integers(0, n_files, n_files), minlength=n_files)
        gram = sum(c * g.gram for c, g in zip(counts, grams) if c)
        rhs = sum(c * g.rhs for c, g in zip(counts, grams) if c)
        coefs = np.stack([solve(gram, rhs, lam) for lam in lambdas], axis=1)
        val_pred = np.concatenate([predict(val.rx_c[f], coefs, terms, fir_len, imaginary, block) for f in range(val.n_files)])
        val_tx = np.concatenate([val.tx_c[f][EDGE:-EDGE] for f in range(val.n_files)])
        val_ber = ber_columns(val_pred, val_tx)
        best = int(np.argmin(val_ber))
        test_pred = np.concatenate([predict(test.rx_c[f], coefs[:, best], terms, fir_len, imaginary, block)
                                    for f in range(test.n_files)])
        test_tx = np.concatenate([test.tx_c[f][EDGE:-EDGE] for f in range(test.n_files)])
        test_ber = float(ber_columns(test_pred, test_tx)[0])
        rows.append({"arm": arm, "M": M, "fir_len": fir_len, "replica": replica, "lambda": lambdas[best],
                     "val_ber": float(val_ber[best]), "test_ber": test_ber, "val_ber_by_lambda": val_ber.tolist(),
                     **rm_counts(M, arm, fir_len)})
    return rows


def synthetic_check(seed: int = 0, n: int = 20000, M: int = 3) -> float:
    """Max |C_fit - C_true| when s = x + sum C T(x) exactly (no noise): least squares must recover C."""
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    terms = triplet_terms(M, "T3")
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    true = (rng.normal(size=len(terms)) + 1j * rng.normal(size=len(terms))) * 0.01
    pad = M + 1
    xp = np.concatenate([np.zeros(pad, complex), x, np.zeros(pad, complex)])
    s = x + features(xp, np.arange(n), terms, 0, pad) @ true
    data = SimpleNamespace(rx_c=x[None], tx_c=s[None], n_files=1)
    gram = FileGram(data.rx_c[0], data.tx_c[0], terms, 0, False, 4096)
    return float(np.abs(solve(gram.gram, gram.rhs, 0.0) - true).max())
