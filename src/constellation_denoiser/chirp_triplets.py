"""E053: off-axis triplet sum as a sum of S "chirps" (H1) and its fast evaluation by recurrences (H2).

Notation (code convention of E044/E046, kerr_offaxis.features): T_mn(x)_p = x_{p+m} x_{p+n} conj(x_{p+m+n}),
off-axis m, n != 0. One chirp with parameters (eta, W):
    Delta_p(eta, W) = sum_{(m, n) in D} exp(i m n / eta) T(|m| + |n|) T_mn(x)_p,   T(k) = max(1 - k / W, 0),
D = off-axis pairs of the domain: "T3" (|m|, |n|, |m + n| <= M, as A1) or "box" (|m|, |n| <= M). Ordered pairs
(m, n) and (n, m) are both in the sum (merged feature (m < n) has weight 2).

Unfolding (algebra: (p+m)^2 + (p+n)^2 - (p+m+n)^2 = p^2 - 2 m n): with xt_a = x_a exp(-i (a - o)^2 / (2 eta)) for any
origin o,  Delta_p = exp(i (p - o)^2 / (2 eta)) sum_D T(|m| + |n|) xt_{p+m} xt_{p+n} conj(xt_{p+m+n}).

Fast sum of the unfolded weights (H2), per symbol p:
  quadrant A (m, n >= 1): weight depends on k = m + n only; S_p[k] = sum_{j=0..k} xt_{p+j} xt_{p+k-j},
      S_p[k] = 2 xt_p xt_{p+k} + S_{p+1}[k-2] (S_p[0] = xt_p^2, S_p[1] = 2 xt_p xt_{p+1});
      sum over m, n >= 1, m + n = k equals S_{p+1}[k-2];  Q^A_p = sum_{k=2..K} T(k) conj(xt_{p+k}) S_{p+1}[k-2],
      K = M for T3 (|m + n| <= M), 2M for the box (then the box also needs m, n <= M: S has to be truncated, see
      quadrant_a_box).
  quadrant A' (m, n <= -1): the mirror (indices p - j).
  quadrant B (m > 0 > n, both orders): with e = m + n and d = m - n = |m| + |n| (d > |e|, d = e mod 2),
      pair products P_c[d] = T(d) xt_{(c+d)/2} xt_{(c-d)/2} with c = 2p + e (shared by all p with the same c),
      G = sum_{d = |e|+2, step 2}^{hi(e)} P_c[d] (prefix sums over d: additions only), hi = 2M - |e|,
      Q^B_p = 2 sum_e conj(xt_{p+e}) G,   |e| <= M - 1 (T3: also |e| <= M, inactive) or <= 2M - 2 (box).
"""

from __future__ import annotations

import numpy as np


def tri(k, W: float):
    return np.maximum(1.0 - np.asarray(k, dtype=float) / W, 0.0)


def domain(M: int, kind: str = "T3") -> list[tuple[int, int]]:
    """Ordered off-axis pairs (m, n)."""
    out = []
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            if m == 0 or n == 0:
                continue
            if kind == "T3" and abs(m + n) > M:
                continue
            out.append((m, n))
    return out


def pad(x: np.ndarray, M: int) -> np.ndarray:
    return np.concatenate((np.zeros(2 * M, x.dtype), x, np.zeros(2 * M, x.dtype)))


def direct(x: np.ndarray, M: int, eta: float, W: float, kind: str = "T3") -> np.ndarray:
    """Delta_p for all p by the definition (zeros outside the sequence)."""
    z, o, P = pad(x, M), 2 * M, len(x)
    p = np.arange(P) + o
    out = np.zeros(P, complex)
    for m, n in domain(M, kind):
        out += np.exp(1j * m * n / eta) * tri(abs(m) + abs(n), W) * z[p + m] * z[p + n] * np.conj(z[p + m + n])
    return out


def unfold(x: np.ndarray, eta: float, origin: float = 0.0) -> np.ndarray:
    a = np.arange(len(x)) - origin
    return x * np.exp(-1j * a ** 2 / (2 * eta))


def via_unfold(x: np.ndarray, M: int, eta: float, W: float, kind: str = "T3", origin: float | None = None) -> np.ndarray:
    """Same sum through the unfolded samples, pair weights T only (no exp(i m n / eta))."""
    o = len(x) / 2 if origin is None else origin
    xt = pad(unfold(x, eta, o), M)
    off, P = 2 * M, len(x)
    p = np.arange(P) + off
    acc = np.zeros(P, complex)
    for m, n in domain(M, kind):
        acc += tri(abs(m) + abs(n), W) * xt[p + m] * xt[p + n] * np.conj(xt[p + m + n])
    return np.exp(1j * (np.arange(P) - o) ** 2 / (2 * eta)) * acc


# --------------------------------------------------------------------------- fast (recurrence) evaluation

def quadrant_a(xt: np.ndarray, M: int, W: float, kind: str, sign: int = 1) -> np.ndarray:
    """Q^A_p (sign +1: m, n >= 1) or Q^A'_p (sign -1: m, n <= -1) on the padded unfolded sequence; returns values at
    the padded positions (valid in the interior). Recurrence S_p[k] = 2 xt_p xt_{p+sign k} + S_{p+sign}[k-2]."""
    N = len(xt)
    K = M if kind == "T3" else 2 * M
    y = xt if sign == 1 else xt[::-1]
    S = np.zeros((N + 2, K + 1), complex)              # S[p, k], p over positions (reversed for sign -1)
    for p in range(N - 1, -1, -1):
        for k in range(K + 1):
            if p + k >= N:
                continue
            if k == 0:
                S[p, 0] = y[p] * y[p]
            elif k == 1:
                S[p, 1] = 2 * y[p] * y[p + 1]
            else:
                S[p, k] = 2 * y[p] * y[p + k] + S[p + 1, k - 2]
    Q = np.zeros(N, complex)
    w = tri(np.arange(K + 1), W)
    for p in range(N):
        for k in range(2, K + 1):
            if p + k < N and w[k]:
                Q[p] += w[k] * np.conj(y[p + k]) * S[p + 1, k - 2]
    return Q if sign == 1 else Q[::-1]


def quadrant_a_box(xt: np.ndarray, M: int, W: float, sign: int = 1) -> np.ndarray:
    """Box domain: m, n in [1, M] with k = m + n up to 2M; the inner sum over m in [k - M, M] is a truncated S, i.e.
    S_{p+1}[k-2] minus 2 sum_{j < k-M} ...: computed here directly from the pair products (reference for the RM)."""
    N = len(xt)
    y = xt if sign == 1 else xt[::-1]
    Q = np.zeros(N, complex)
    for p in range(N):
        for k in range(2, 2 * M + 1):
            if p + k >= N or tri(k, W) == 0:
                continue
            s = sum(y[p + m] * y[p + k - m] for m in range(max(1, k - M), min(M, k - 1) + 1))
            Q[p] += tri(k, W) * np.conj(y[p + k]) * s
    return Q if sign == 1 else Q[::-1]


def quadrant_b(xt: np.ndarray, M: int, W: float, kind: str) -> np.ndarray:
    """Q^B_p for m > 0 > n and m < 0 < n through the pair table P_c[d] and prefix sums over d."""
    N = len(xt)
    D = 2 * M
    emax = M - 1 if kind == "T3" else 2 * M - 2
    Q = np.zeros(N, complex)
    w = tri(np.arange(D + 1), W)
    # pair table: c in [0, 2N - 2], d in [0, D] with (c + d) even: P[c, d] = w[d] xt[(c+d)/2] xt[(c-d)/2]
    P = np.zeros((2 * N - 1, D + 1), complex)
    for c in range(2 * N - 1):
        for d in range(c % 2, D + 1, 2):
            a, b = (c + d) // 2, (c - d) // 2
            if 0 <= b and a < N:
                P[c, d] = w[d] * xt[a] * xt[b]
    PS = np.cumsum(P, axis=1)                               # prefix sums over d (other-parity entries are zero)
    for p in range(N):
        for e in range(-emax, emax + 1):
            lo, hi = abs(e) + 2, 2 * M - abs(e)
            if lo > hi or not 0 <= p + e < N:
                continue
            c = 2 * p + e
            G = PS[c, hi] - PS[c, lo - 2]                   # entries of the other parity are zero
            Q[p] += 2 * np.conj(xt[p + e]) * G
    return Q


def fast(x: np.ndarray, M: int, eta: float, W: float, kind: str = "T3", origin: float | None = None) -> np.ndarray:
    o = len(x) / 2 if origin is None else origin
    xt = pad(unfold(x, eta, o), M)
    if kind == "T3":
        qa, qa2 = quadrant_a(xt, M, W, kind, 1), quadrant_a(xt, M, W, kind, -1)
    else:
        qa, qa2 = quadrant_a_box(xt, M, W, 1), quadrant_a_box(xt, M, W, -1)
    q = qa + qa2 + quadrant_b(xt, M, W, kind)
    P = len(x)
    return np.exp(1j * (np.arange(P) - o) ** 2 / (2 * eta)) * q[2 * M: 2 * M + P]


# --------------------------------------------------------------------------- complexity (RM per symbol, one chirp)

def rm_direct(M: int, kind: str = "T3", Kth: float | None = None) -> dict:
    """Table of C: 12 RM per merged off-axis term (repository convention), additions = 2 complex adds per term."""
    pairs = domain(M, kind)
    merged = {(min(m, n), max(m, n)) for m, n in pairs if Kth is None or abs(m * n) <= Kth}
    return {"terms": len(merged), "rm": 12 * len(merged), "adds_real": 2 * 2 * len(merged)}


def rm_fast_chirp(M: int, W: float, kind: str = "T3", exp_computed: bool = False) -> dict:
    """One chirp, T3 domain (quadrants A, A' by the recurrence, B by pair tables and prefix sums). Counted per new
    symbol in a streaming implementation:
      unfold xt = x exp(-i a^2 / 2 eta): 1 complex product (4 RM; the table exp costs 0 RM, as in the repository);
      quadrant A: S_p[k] for k = 0..K needs xt_p xt_{p+k}: K + 1 complex products; the weighted sum Q^A: for every
                  k = 2..K one complex product conj(xt) * S (4) and one real weight T(k) (2): 6 (K - 1);  A' the same;
      quadrant B: a new sample x_N gives the new pair products x_N x_b, d = N - b in [2, 2M]: 2M - 1 complex products
                  + T(d) (2 RM each); prefix sums are additions; Q^B: 2M - 1 complex products (|e| <= M - 1) and the
                  factor 2 (free, an addition);
      refold exp(+i p^2 / 2 eta) (4) and the amplitude A_s (4, or 2 if A_s = i a_s).
    Terms where T = 0 (k >= W, d >= W) are skipped. Additions are reported separately (real additions)."""
    if kind != "T3":
        raise ValueError("box: quadrant A needs truncated S (quadrant_a_box), not counted")
    K = min(M, int(np.ceil(W)) - 1 if W < M + 1 else M)
    Kb = min(2 * M, int(np.ceil(W)) - 1 if W < 2 * M + 1 else 2 * M)
    unfold_rm = 4
    qa = 4 * (K + 1) + 6 * max(K - 1, 0)
    pairs_new = max(Kb - 1, 0)                      # new x_N x_b, d = N - b in [2, Kb]
    qb = pairs_new * (4 + 2) + 4 * (2 * M - 1)
    total = unfold_rm + 2 * qa + qb + 4 + 4
    adds = 2 * (2 * (K + 1) + 2 * max(K - 1, 0)) + 2 * pairs_new + 2 * 2 * (2 * M - 1) + 2 * 2 * 3
    out = {"K_A": K, "rm": total, "adds_real": adds}
    if exp_computed:   # exp(-i a^2 / 2 eta) by a running phasor r_{a+1} = r_a q_a, q_{a+1} = q_a q0: 2 complex products;
        out["rm_exp_computed"] = total + 2 * 4   # the refold uses conj(r_p): free
    return out


def rm_oc_reuse(M: int, kind: str = "T3") -> dict:
    """Optics Continuum 4(12) 2896, sec. 3.3.3 adapted to one polarization (моя адаптация): the double products
    x_{k+n} conj(x_{k+m+n}) are reused from the previous sample, a new sample adds one product per lag m != 0 (2M);
    per merged term C * x_{k+m} * (double product): 2 complex products (8 RM). Grouped variant (моё добавление):
    sum_n C_mn y_m[k+n] first (4 RM per term), then one product by x_{k+m} per distinct m (4 RM)."""
    d = rm_direct(M, kind)
    lags = 2 * M
    first = {min(m, n) for m, n in domain(M, kind)}
    return {"rm_oc": 8 * d["terms"] + 4 * lags, "rm_grouped": 4 * d["terms"] + 4 * lags + 4 * len(first),
            "adds_real": d["adds_real"]}


def rm_xiao_offaxis(K: int, M: int, nf: str = "2M+1") -> float:
    """Xiao 2025 (из литературы, по пересказу задания): iFWM part of one SNSE step, K (2 C_conv(N_f) + 12) RM per
    symbol for the cross rule min(|m|,|n|) <= K, max(|m|,|n|) <= L, arbitrary coefficients along the cross lines.
    N_f is not defined in the task text: "2M+1" (full filter over -L..L) or "M+1" (one side); both reported."""
    from .dbpx import c_conv

    n_f = 2 * M + 1 if nf == "2M+1" else M + 1
    return K * (2 * c_conv(n_f) + 12)


def check(seed: int = 0, P: int = 120) -> list[dict]:
    """Identity and fast evaluation on random complex data, several (M, eta, W, domain)."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(P) + 1j * rng.standard_normal(P)) / np.sqrt(2)
    rows = []
    for M, eta, W, kind in ((3, 4.0, 7.0, "T3"), (6, -9.5, 30.0, "T3"), (8, 43.0, 2 * np.pi * 43.0, "T3"),
                            (8, 2.3, 5.0, "T3"), (5, 7.0, 12.0, "box"), (6, -3.0, 40.0, "box")):
        ref = direct(x, M, eta, W, kind)
        u = via_unfold(x, M, eta, W, kind)
        f = fast(x, M, eta, W, kind)
        nrm = np.linalg.norm(ref)
        rows.append({"M": M, "eta": eta, "W": W, "domain": kind, "rel_err_unfold": np.linalg.norm(u - ref) / nrm,
                     "rel_err_fast": np.linalg.norm(f - ref) / nrm})
    return rows


if __name__ == "__main__":
    for r in check():
        print(r)
