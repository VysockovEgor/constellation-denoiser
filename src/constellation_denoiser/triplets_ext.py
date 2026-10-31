"""E034/E035: perturbation triplets beyond E031 - larger memory M, pruning to K terms, and a torch module.

Closed form (E034): the same per-file normal equations as triplets.py. Pruning: the full T3 + FIR fit of a
bootstrap replica (lambda chosen by validation) ranks the triplet terms by |C_j| * rms(feature_j) on train;
the K strongest terms (the FIR taps are always kept) are refitted from the sub-matrix of the same normal
equations, lambda again by validation, test reported. RM (sym) = 12 K + 4 L.

Torch (E035): ``TripletNet`` computes the same features inside a window W >= 2M + 1 and adds an optional
one-hidden Kerr MLP on the residual (output layer at zero); ``warm`` loads closed-form coefficients.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .triplets import EDGE, FileGram, ber_columns, predict, solve, triplet_terms


def importance(gram: np.ndarray, coef: np.ndarray, n: int) -> np.ndarray:
    """|C_j| times the rms of feature j (diag of A^H A / n)."""
    return np.abs(coef) * np.sqrt(np.real(np.diag(gram)) / max(n, 1))


def prune_replica(train, val, test, M: int, fir_len: int, ks: list[int], lambdas: list[float], replica: int,
                  grams: list[FileGram], terms, block: int) -> list[dict]:
    rng = np.random.default_rng(replica)
    n_files = train.n_files
    counts = np.bincount(rng.integers(0, n_files, n_files), minlength=n_files)
    gram = sum(c * g.gram for c, g in zip(counts, grams) if c)
    rhs = sum(c * g.rhs for c, g in zip(counts, grams) if c)
    n_symbols = int(counts.sum()) * (train.length - 2 * EDGE)
    val_tx = np.concatenate([val.tx_c[f][EDGE:-EDGE] for f in range(val.n_files)])
    test_tx = np.concatenate([test.tx_c[f][EDGE:-EDGE] for f in range(test.n_files)])

    def evaluate(sel_terms, idx):
        g, r = gram[np.ix_(idx, idx)], rhs[idx]
        coefs = np.stack([solve(g, r, lam) for lam in lambdas], axis=1)
        pv = np.concatenate([predict(val.rx_c[f], coefs, sel_terms, fir_len, False, block) for f in range(val.n_files)])
        vb = ber_columns(pv, val_tx)
        best = int(np.argmin(vb))
        pt = np.concatenate([predict(test.rx_c[f], coefs[:, best], sel_terms, fir_len, False, block)
                             for f in range(test.n_files)])
        return coefs[:, best], lambdas[best], float(vb[best]), float(ber_columns(pt, test_tx)[0])

    n_terms = len(terms)
    fir_idx = list(range(n_terms, n_terms + fir_len))
    full_coef, lam, vb, tb = evaluate(terms, list(range(n_terms)) + fir_idx)
    rows = [{"variant": f"T3FIR{fir_len}_M{M}_full", "K": n_terms, "replica": replica, "lambda": lam, "val_ber": vb,
             "test_ber": tb, "rm_sym": 12 * n_terms + 4 * fir_len}]
    score = importance(gram, full_coef, n_symbols)[:n_terms]
    order = np.argsort(score)[::-1]
    for k in ks:
        if k >= n_terms:
            continue
        keep = sorted(order[:k].tolist())
        _, lam, vb, tb = evaluate([terms[i] for i in keep], keep + fir_idx)
        rows.append({"variant": f"T3FIR{fir_len}_M{M}_K{k}", "K": k, "replica": replica, "lambda": lam, "val_ber": vb,
                     "test_ber": tb, "rm_sym": 12 * k + 4 * fir_len})
    return rows


def closed_form_coefficients(train, val, M: int, fir_len: int, lambdas: list[float], block: int) -> tuple[np.ndarray, float]:
    """T3 + FIR coefficients on all train files, lambda by validation BER (warm start of E035)."""
    terms = triplet_terms(M, "T3")
    grams = [FileGram(train.rx_c[f], train.tx_c[f], terms, fir_len, False, block) for f in range(train.n_files)]
    gram, rhs = sum(g.gram for g in grams), sum(g.rhs for g in grams)
    coefs = np.stack([solve(gram, rhs, lam) for lam in lambdas], axis=1)
    val_tx = np.concatenate([val.tx_c[f][EDGE:-EDGE] for f in range(val.n_files)])
    pv = np.concatenate([predict(val.rx_c[f], coefs, terms, fir_len, False, block) for f in range(val.n_files)])
    best = int(np.argmin(ber_columns(pv, val_tx)))
    return coefs[:, best], lambdas[best]


class TripletNet(nn.Module):
    """y = x_c + sum_j C_j T_j(x) + sum_l h_l x_{c+l} (+ MLP_res(x)); T_j = x_{c+m} x*_{c+m+n} x_{c+n}."""

    def __init__(self, window: int, M: int, fir_len: int, mlp_hidden: int = 0, warm: str | None = None):
        super().__init__()
        if window < 2 * M + 1 or window < fir_len:
            raise ValueError("window must cover 2M + 1 and the FIR")
        self.window, self.centre, self.fir_len = window, window // 2, fir_len
        terms = triplet_terms(M, "T3")
        c = self.centre
        self.register_buffer("i_m", torch.tensor([c + m for m, _ in terms]))
        self.register_buffer("i_mn", torch.tensor([c + m + n for m, n in terms]))
        self.register_buffer("i_n", torch.tensor([c + n for _, n in terms]))
        half = fir_len // 2
        self.register_buffer("i_fir", torch.arange(c - half, c + half + 1))
        p = len(terms) + fir_len
        self.coef_re = nn.Parameter(torch.zeros(p))
        self.coef_im = nn.Parameter(torch.zeros(p))
        if warm:
            values = np.load(warm)["coef"]
            if len(values) != p:
                raise ValueError(f"warm start has {len(values)} coefficients, the model {p}")
            with torch.no_grad():
                self.coef_re.copy_(torch.tensor(values.real, dtype=torch.float32))
                self.coef_im.copy_(torch.tensor(values.imag, dtype=torch.float32))
        self.mlp = None
        if mlp_hidden:
            from .complex_mlp_ablation import AblationComplexMLP

            self.mlp = AblationComplexMLP(window=window, hidden=mlp_hidden, variant="d")
            for name in ("weight_real", "weight_imag", "bias_real", "bias_imag"):
                nn.init.zeros_(getattr(self.mlp.fc3, name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], self.window)
        feats = z[:, self.i_m] * torch.conj(z[:, self.i_mn]) * z[:, self.i_n]
        feats = torch.cat((feats, z[:, self.i_fir]), dim=1)
        y = z[:, self.centre] + feats @ torch.complex(self.coef_re, self.coef_im)
        if self.mlp is not None:
            y = y + self.mlp(x)
        return y


def triplet_net_macs(window: int, M: int, fir_len: int, mlp_hidden: int) -> int:
    return 12 * len(triplet_terms(M, "T3")) + 4 * fir_len + mlp_hidden * (4 * window + 11)
