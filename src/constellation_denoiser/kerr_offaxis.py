"""E046 (phase 2, architecture A): Kerr with memory + off-axis triplets (IFWM-type residual).

Kerr stage (EnDBP / PhysKerr type): u = FIR_L(x) (complex taps, unit impulse at start), y = u exp(-i phi),
phi_k = sum_{j=-r..r} alpha_j |u_{k+j}|^2, alpha symmetric, N_c = 2r + 1 coefficients.
Off-axis triplets: T_mn(z)_k = z_{k+m} z_{k+n} z*_{k+m+n} for merged (m <= n) terms of the T3 domain
(|m|, |n|, |m + n| <= M) with m != 0 and n != 0; in first order the Kerr phase equals the axial triplets.
Architectures: A1 y = u_K + T(u_K) C (Kerr first), A2 y = Kerr(x + T(x) C) (triplets first), A3 y = Kerr(x) + T(x) C
(parallel); controls: T (all T3 terms + FIR7, no Kerr), K (Kerr only = K = 0 point), A3full (Kerr + T3 with axes).
Training: "varpro" (A1, A3): theta = (FIR, alpha) by L-BFGS on a file subsample with C eliminated by ridge
(variable projection); "seq": Kerr fitted alone (L-BFGS), then ridge for C on the residual (A2: ridge on the raw
residual first, then the Kerr on x + T(x) C). After theta is fixed, C is refitted in closed form on all train files
(bootstrap counts per replica), lambda by validation; then sweeps: K strongest off-axis terms by |C| rms(feature)
(magnitude pruning) and the geometric cut |mn| <= K_th. RM = 4 L + 2 + (N_c + 1)/2 + 4 + 12 K (repository convention).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .triplets import triplet_terms

EDGE = 15
LAMBDAS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]


def offaxis_terms(M: int) -> list[tuple[int, int]]:
    return [(m, n) for m, n in triplet_terms(M, "T3") if m != 0 and n != 0]


def offaxis_box(M: int) -> list[tuple[int, int]]:
    """E051: off-axis merged terms (m <= n, m, n != 0) of the box max(|m|, |n|) <= M (the domain of the cross rule
    S = {min(|m|, |n|) <= K, max(|m|, |n|) <= L} of Xiao 2025); T_mn = T_nm, so m <= n."""
    return [(m, n) for m in range(-M, M + 1) for n in range(m, M + 1) if m != 0 and n != 0]


def kerr_rm(n_c: int, L: int) -> float:
    return 4 * L + 2 + (n_c + 1) / 2 + 4


def shift(z: torch.Tensor, d: int) -> torch.Tensor:
    """z[..., k + d] with zeros outside."""
    if d == 0:
        return z
    out = torch.zeros_like(z)
    if d > 0:
        out[..., :-d] = z[..., d:]
    else:
        out[..., -d:] = z[..., :d]
    return out


def features(z: torch.Tensor, terms) -> torch.Tensor:
    """(F, L) -> (F, L, p) triplet features."""
    return torch.stack([shift(z, m) * shift(z, n) * torch.conj(shift(z, m + n)) for m, n in terms], dim=-1)


class Kerr:
    """Parameters theta = [fir_re (L), fir_im (L), alpha half (r + 1)] as one flat float64 tensor."""

    def __init__(self, L: int, n_c: int):
        self.L, self.r = L, (n_c - 1) // 2

    def init(self) -> torch.Tensor:
        t = torch.zeros(2 * self.L + self.r + 1, dtype=torch.float64)
        t[self.L // 2] = 1.0
        return t

    def apply(self, theta: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(u, y): FIR output and Kerr output for (F, L) complex x."""
        L = self.L
        taps = torch.complex(theta[:L], theta[L:2 * L]).flip(0)
        re, im = x.real.unsqueeze(1), x.imag.unsqueeze(1)
        kr, ki = taps.real.view(1, 1, -1), taps.imag.view(1, 1, -1)
        pad = L // 2
        u = torch.complex(F.conv1d(re, kr, padding=pad) - F.conv1d(im, ki, padding=pad),
                          F.conv1d(re, ki, padding=pad) + F.conv1d(im, kr, padding=pad)).squeeze(1)
        half = theta[2 * L:]
        alpha = torch.cat((half[1:].flip(0), half)).view(1, 1, -1)
        power = (u.real.square() + u.imag.square()).unsqueeze(1)
        phi = F.conv1d(power, alpha, padding=self.r).squeeze(1)
        return u, u * torch.exp(-1j * phi)

    def rotation(self, theta: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(u, exp(-i phi)): E051 A-M form v = (u + T(u) C) exp(-i phi(u)) (Liang, Kumar 2014)."""
        u, y = self.apply(theta, x)
        half = theta[2 * self.L:]
        alpha = torch.cat((half[1:].flip(0), half)).view(1, 1, -1)
        power = (u.real.square() + u.imag.square()).unsqueeze(1)
        phi = F.conv1d(power, alpha, padding=self.r).squeeze(1)
        return u, torch.exp(-1j * phi)


def ridge(A: torch.Tensor, r: torch.Tensor, lam: float) -> torch.Tensor:
    G = A.conj().T @ A
    scale = torch.real(torch.diagonal(G)).mean()
    return torch.linalg.solve(G + lam * scale * torch.eye(G.shape[0], dtype=G.dtype), A.conj().T @ r)


def centre_mask(n: int, length: int) -> slice:
    return slice(EDGE, length - EDGE)


def fit_theta(arch: str, mode: str, x: torch.Tensor, s: torch.Tensor, M: int, L: int, n_c: int, iters: int = 60,
              lam: float = 1e-4, terms=None) -> tuple[torch.Tensor, np.ndarray | None]:
    """theta on the subsample (F files); for A2/seq also the closed-form C on the subsample.
    arch LK (E051): v = (u + T(u) C) exp(-i phi(u)); varpro: C by ridge on the rotated features."""
    kerr = Kerr(L, n_c)
    theta = kerr.init().requires_grad_(True)
    sl = slice(EDGE + M, x.shape[1] - EDGE - M)
    if terms is None:
        terms = offaxis_terms(M) if arch != "A3full" else triplet_terms(M, "T3")
    c_fixed = None
    if arch == "A2":                                   # ridge on the raw residual first (seq only)
        A = features(x, terms)[:, sl].reshape(-1, len(terms))
        c_fixed = ridge(A, (s - x)[:, sl].reshape(-1), lam).detach()
        x_in = x + features(x, terms) @ c_fixed
    else:
        x_in = x
    A_fixed = features(x, terms)[:, sl].reshape(-1, len(terms)) if arch in ("A3", "A3full") else None

    def loss_fn():
        if arch == "LK" and mode == "varpro":
            u, rot = kerr.rotation(theta, x_in)
            A = (features(u, terms) * rot.unsqueeze(-1))[:, sl].reshape(-1, len(terms))
            r = (s - u * rot)[:, sl].reshape(-1)
            return (r - A @ ridge(A, r, lam)).abs().square().mean()
        u, y = kerr.apply(theta, x_in)
        if arch in ("A1", "A3", "A3full") and mode == "varpro":
            if arch == "A1":
                A = features(y, terms)[:, sl].reshape(-1, len(terms))
            else:
                A = A_fixed
            r = (s - y)[:, sl].reshape(-1)
            c = ridge(A, r, lam)
            return (r - A @ c).abs().square().mean()
        return (s - y)[:, sl].abs().square().mean()

    optimizer = torch.optim.LBFGS([theta], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    optimizer.step(closure)
    return theta.detach(), (None if c_fixed is None else c_fixed.numpy())


def stage_input(arch: str, kerr: Kerr, theta: torch.Tensor, x: torch.Tensor, c_a2: np.ndarray | None, terms):
    """(base, z): output = base + T(z) C for A1 / A3 / A3full / T; for A2 the output is Kerr(x + T(x) C)."""
    if arch == "T":
        return x, x
    u, y = kerr.apply(theta, x)
    if arch == "A1":
        return y, y
    if arch in ("A3", "A3full"):
        return y, x
    raise ValueError(arch)


def fir_only(theta_L: int) -> None:
    return None
