"""Complexity of the triplet (perturbation) models under three conventions, single polarization (E039).

The repository number ``macs_per_symbol`` stays the main one; the functions here are for comparison only.

(a) repository (triplets.rm_counts, "sym"): every merged term T_mn = x_{k+m} x_{k+n} x*_{k+m+n} (m <= n) is
    computed directly: two complex products for the triplet and one for the coefficient = 3 complex = 12 RM,
    plus 4 RM per FIR tap.

(b) buffered evaluation, single polarization - MY ESTIMATE, not the formula of the group. Derivation:
    T_mn(k) = P(k + m, n) * x_{k+n} with P(j, n) = x_j * conj(x_{j+n}).
    P(j, n) depends on (j, n) only, so it is computed once when symbol j arrives and kept in a cyclic buffer:
    |N| complex products per symbol, N = set of n used by the terms.
    S_n(k) = sum_{m: (m,n) used} C_mn P(k + m, n): one complex product per term.
    y_k - x_k = sum_{n in N} x_{k+n} S_n(k): |N| complex products.
    Total per symbol: n_terms + 2 |N| complex products, i.e. 4 (n_terms + 2 |N|) RM, plus 4 RM per FIR tap.
    For pruned models whose term list was not saved, |N| <= min(n_terms, 2M + 1) is used (an upper bound).

(c) arXiv 2409.20023 (as quoted in the plan): 9 M^2 / 2 + 19 M / 2 + 4 complex products, times 4 RM; the memory
    parameter M there is taken equal to ours, which is an assumption (their definition was not checked).

(d) for reference only, the per-coefficient counts of the group (Optics Continuum 4(12), 2896, 2025) as quoted in
    the plan: 25 complex products for a general coefficient, 13 for axial (m = 0 or n = 0) and diagonal (m = n)
    ones, 4 for C_00; dual polarization, before the cyclic buffer of their eqs. (10)-(13), which were not checked.
"""

from __future__ import annotations

from collections.abc import Iterable


def rm_repo(terms: list[tuple[int, int]], fir_len: int = 0) -> int:
    return 12 * len(terms) + 4 * fir_len


def distinct_n(terms: Iterable[tuple[int, int]]) -> int:
    return len({n for _, n in terms})


def rm_buffered(terms: list[tuple[int, int]], fir_len: int = 0, n_distinct: int | None = None) -> int:
    nd = distinct_n(terms) if n_distinct is None else n_distinct
    return 4 * (len(terms) + 2 * nd) + 4 * fir_len


def rm_buffered_bound(n_terms: int, M: int, fir_len: int = 0) -> int:
    """Upper bound of (b) when only the number of terms is known (pruned models)."""
    return 4 * (n_terms + 2 * min(n_terms, 2 * M + 1)) + 4 * fir_len


def rm_arxiv_2409_20023(M: int, fir_len: int = 0) -> float:
    return 4 * (9 * M * M / 2 + 19 * M / 2 + 4) + 4 * fir_len


def group_dualpol_complex(terms: list[tuple[int, int]]) -> int:
    total = 0
    for m, n in terms:
        if m == 0 and n == 0:
            total += 4
        elif m == 0 or n == 0 or m == n:
            total += 13
        else:
            total += 25
    return total


class Counter:
    """Counts complex multiplications done through ``mul``."""

    def __init__(self):
        self.count = 0

    def mul(self, a, b):
        self.count += 1
        return a * b


def direct_eval(x, terms, coef, k: int, counter: Counter):
    """Repository convention executed literally for one output symbol k."""
    total = 0j
    for (m, n), c in zip(terms, coef):
        t = counter.mul(counter.mul(x[k + m], x[k + n]), x[k + m + n].conjugate())
        total += counter.mul(c, t)
    return x[k] + total


def buffered_eval(x, terms, coef, ks: range, counter: Counter):
    """Convention (b) executed literally for consecutive outputs ``ks``; returns outputs and products per symbol
    in the steady state (the buffer is filled before the first output, not counted)."""
    ns = sorted({n for _, n in terms})
    by_n: dict[int, list[tuple[int, complex]]] = {}
    for (m, n), c in zip(terms, coef):
        by_n.setdefault(n, []).append((m, c))
    reach = max(max(abs(m), abs(n), abs(m + n)) for m, n in terms)
    buffer: dict[tuple[int, int], complex] = {}
    warm = Counter()
    for j in range(ks.start - reach, ks.start + reach):            # fill the buffer (not counted)
        for n in ns:
            buffer[(j, n)] = warm.mul(x[j], x[j + n].conjugate())
    out = []
    for k in ks:
        j = k + reach                                               # the symbol that arrives now
        for n in ns:
            buffer[(j, n)] = counter.mul(x[j], x[j + n].conjugate())
        y = x[k]
        for n in ns:
            s = 0j
            for m, c in by_n[n]:
                s += counter.mul(c, buffer[(k + m, n)])
            y += counter.mul(x[k + n], s)
        out.append(y)
    return out
