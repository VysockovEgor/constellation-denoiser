"""E040/E041: cascades of triplet and phase stages (single polarization), PB-DBP-like.

Stage s (types: "T" triplets, "K" nonlocal Kerr phase, "KT" both on the same input, "skip"):
    T:  y_k = x_k + sum_{(m,n)} C_mn x_{k+m} x_{k+n} x*_{k+m+n},   |m|, |n| <= M, |mn| <= K_th
    K:  y_k = x_k exp(i sum_{j=-r..r} c_j |x_{k+j}|^2),             R = 2r + 1 real coefficients, c_j = c_-j
    KT: y_k = x_k exp(i ...) + sum C_mn ...                         (both on the stage input)
    then x^(s) = FIR_s(y) with L_s complex taps (L_s = 1: no filter); skip: the stage is the identity.
Symmetries of C (as in PB-DBP): C_mn = C_nm = C_{-m,-n} = -C*_{-m,n}. They are imposed by a base value
b(a, b) on 0 <= a <= b: C_mn = b if mn >= 0 else -conj(b); b is purely imaginary when a = 0. The merged term
(m <= n) carries the weight C_mn (m = n) or 2 C_mn (m < n), since T_mn = T_nm.
Initialization: C = 0, c = 0, FIR = unit impulse, i.e. the identity. ``shared``: one set of C (and c) for all
stages (FIR stays per stage). The network window W covers the receptive field; every stage is evaluated only on
the positions the next stages need (the last one on the centre symbol).

Real multiplies per symbol (repository convention): T 12 per merged term, K macs_phys(R, 1, sym) = (R + 1)/2 + 6,
FIR 4 L (0 for L = 1); E039 conventions in ``rm_variants``.
"""

from __future__ import annotations

import torch
from torch import nn

CHUNK = 48


def stage_terms(M: int, k_th: int) -> list[tuple[int, int]]:
    return [(m, n) for m in range(-M, M + 1) for n in range(m, M + 1) if abs(m * n) <= k_th]


def reach_of(stage: dict) -> int:
    reach = 0
    if stage["type"] in ("T", "KT"):
        reach = max(max(abs(m), abs(n), abs(m + n)) for m, n in stage_terms(stage["M"], stage["K_th"]))
    if stage["type"] in ("K", "KT"):
        reach = max(reach, (stage["R"] - 1) // 2)
    return reach


def half_fir(stage: dict) -> int:
    return stage.get("L", 1) // 2 if stage["type"] != "skip" else 0


def window_for(stages: list[dict]) -> int:
    return 2 * sum(reach_of(s) + half_fir(s) for s in stages if s["type"] != "skip") + 1


def stage_rm(stage: dict) -> int:
    if stage["type"] == "skip":
        return 0
    rm = 0
    if stage["type"] in ("T", "KT"):
        rm += 12 * len(stage_terms(stage["M"], stage["K_th"]))
    if stage["type"] in ("K", "KT"):
        rm += (stage["R"] + 1) // 2 + 6
    L = stage.get("L", 1)
    return rm + (4 * L if L > 1 else 0)


def cascade_rm(stages: list[dict]) -> int:
    return sum(stage_rm(s) for s in stages)


def rm_variants(stages: list[dict]) -> dict:
    from .rm_group_convention import rm_arxiv_2409_20023, rm_buffered

    buffered = arxiv = 0.0
    for s in stages:
        if s["type"] == "skip":
            continue
        fir = 4 * s.get("L", 1) if s.get("L", 1) > 1 else 0
        kerr = (s["R"] + 1) // 2 + 6 if s["type"] in ("K", "KT") else 0
        if s["type"] in ("T", "KT"):
            terms = stage_terms(s["M"], s["K_th"])
            buffered += rm_buffered(terms) + kerr + fir
            arxiv += rm_arxiv_2409_20023(s["M"]) + kerr + fir
        else:
            buffered += kerr + fir
            arxiv += kerr + fir
    return {"rm_repo": cascade_rm(stages), "rm_buffered_my_estimate": buffered, "rm_arxiv_2409_20023": arxiv}


class TripletCoefficients(nn.Module):
    """C_mn of one (M, K_th) domain with the PB-DBP symmetries; returns merged weights for stage_terms."""

    def __init__(self, M: int, k_th: int):
        super().__init__()
        terms = stage_terms(M, k_th)
        bases = sorted({tuple(sorted((abs(m), abs(n)))) for m, n in terms})
        index = {b: i for i, b in enumerate(bases)}
        self.register_buffer("base_index", torch.tensor([index[tuple(sorted((abs(m), abs(n))))] for m, n in terms]))
        self.register_buffer("flip", torch.tensor([m * n < 0 for m, n in terms]))
        self.register_buffer("weight", torch.tensor([1.0 if m == n else 2.0 for m, n in terms]))
        self.register_buffer("real_allowed", torch.tensor([0.0 if a == 0 else 1.0 for a, _ in bases]))
        self.base_re = nn.Parameter(torch.zeros(len(bases)))
        self.base_im = nn.Parameter(torch.zeros(len(bases)))
        self.n_free_real = int(2 * len(bases) - (self.real_allowed == 0).sum())

    def forward(self) -> torch.Tensor:
        re = (self.base_re * self.real_allowed)[self.base_index]
        im = self.base_im[self.base_index]
        re = torch.where(self.flip, -re, re)          # -conj(b) = -re + i im
        return torch.complex(re, im) * self.weight


class FreeCoefficients(nn.Module):
    """E047 variant: merged weights as free complex numbers (no PB-DBP symmetries)."""

    def __init__(self, M: int, k_th: int):
        super().__init__()
        n = len(stage_terms(M, k_th))
        self.re = nn.Parameter(torch.zeros(n))
        self.im = nn.Parameter(torch.zeros(n))

    def forward(self) -> torch.Tensor:
        return torch.complex(self.re, self.im)


def load_init(coef: nn.Module, weights) -> None:
    """Set the coefficients from merged weights (ridge solution); symmetric form: orbit average of the base values."""
    w = torch.as_tensor(weights, dtype=torch.complex64)
    with torch.no_grad():
        if isinstance(coef, FreeCoefficients):
            coef.re.copy_(w.real)
            coef.im.copy_(w.imag)
            return
        c = w / coef.weight
        c = torch.where(coef.flip, -torch.conj(c), c)                 # back to the base value b
        n = len(coef.base_re)
        total = torch.zeros(n, dtype=torch.complex64).index_add_(0, coef.base_index, c)
        count = torch.zeros(n).index_add_(0, coef.base_index, torch.ones(len(c)))
        b = total / count
        coef.base_re.copy_(b.real * coef.real_allowed)
        coef.base_im.copy_(b.imag)


class Stage(nn.Module):
    def __init__(self, stage: dict, coef: TripletCoefficients | None, kerr: nn.Parameter | None):
        super().__init__()
        self.kind = stage["type"]
        self.reach, self.half = reach_of(stage), half_fir(stage)
        self.coef = coef
        self.kerr = kerr
        if self.kind in ("T", "KT"):
            terms = stage_terms(stage["M"], stage["K_th"])
            self.register_buffer("tm", torch.tensor([m for m, _ in terms]))
            self.register_buffer("tn", torch.tensor([n for _, n in terms]))
        self.r = (stage.get("R", 1) - 1) // 2
        L = stage.get("L", 1)
        self.fir = None
        if L > 1:
            re = torch.zeros(L)
            re[L // 2] = 1.0
            self.fir_re = nn.Parameter(re)
            self.fir_im = nn.Parameter(torch.zeros(L))
            self.fir = L

    def forward(self, z: torch.Tensor, h_out: int) -> torch.Tensor:
        h_in = (z.shape[1] - 1) // 2
        h_mid = h_out + self.half
        base = h_in - h_mid
        pos = torch.arange(2 * h_mid + 1, device=z.device) + base
        mid = z[:, base: base + 2 * h_mid + 1]
        y = mid
        if self.kind in ("K", "KT"):
            power = z.real.square() + z.imag.square()
            c = torch.cat((self.kerr.flip(0)[:-1], self.kerr))                          # c_{-r..r}, symmetric
            offsets = torch.arange(-self.r, self.r + 1, device=z.device)
            phase = (power[:, pos[:, None] + offsets[None, :]] * c).sum(-1)
            y = mid * torch.exp(1j * phase)
        if self.kind in ("T", "KT"):
            weights = self.coef()
            total = torch.zeros_like(mid)
            for start in range(0, len(self.tm), CHUNK):
                m, n = self.tm[start:start + CHUNK], self.tn[start:start + CHUNK]
                a = z[:, pos[:, None] + m[None, :]]
                b = z[:, pos[:, None] + n[None, :]]
                c = z[:, pos[:, None] + (m + n)[None, :]]
                total = total + (a * b * torch.conj(c)) @ weights[start:start + CHUNK]
            y = y + total
        if self.fir is None:
            return y[:, self.half: self.half + 2 * h_out + 1] if self.half else y
        taps = torch.complex(self.fir_re, self.fir_im)
        idx = torch.arange(2 * h_out + 1, device=z.device)[:, None] + torch.arange(self.fir, device=z.device)[None, :]
        return (y[:, idx] * taps).sum(-1)


class Cascade(nn.Module):
    def __init__(self, window: int, stages: list[dict], shared: bool = False, free: bool = False,
                 init_first: str | None = None):
        super().__init__()
        coef_class = FreeCoefficients if free else TripletCoefficients
        active = [s for s in stages if s["type"] != "skip"]
        if window < window_for(active):
            raise ValueError(f"window {window} < receptive field {window_for(active)}")
        self.window = window
        shared_coef = shared_kerr = None
        if shared and active:
            first = active[0]
            if any((s["type"], s.get("M"), s.get("K_th"), s.get("R")) != (first["type"], first.get("M"), first.get("K_th"),
                                                                          first.get("R")) for s in active):
                raise ValueError("shared coefficients need identical stages")
            shared_coef = coef_class(first["M"], first["K_th"]) if first["type"] in ("T", "KT") else None
            shared_kerr = nn.Parameter(torch.zeros(first.get("R", 1) // 2 + 1)) if first["type"] in ("K", "KT") else None
        self.shared_coef, self.shared_kerr = shared_coef, shared_kerr
        mods = []
        for s in active:
            coef = shared_coef if shared else (coef_class(s["M"], s["K_th"]) if s["type"] in ("T", "KT") else None)
            kerr = shared_kerr if shared else (nn.Parameter(torch.zeros(s.get("R", 1) // 2 + 1))
                                               if s["type"] in ("K", "KT") else None)
            mods.append(Stage(s, coef, kerr))
        self.stages = nn.ModuleList(mods)
        if init_first:                                   # E047: ridge initialization of the first triplet stage
            import numpy as np

            load_init(mods[0].coef, np.load(init_first)["weights"])
        outs, h = [], 0
        for st in reversed(mods):                    # half-span each stage must output
            outs.append(h)
            h = h + st.reach + st.half
        self.h_out = list(reversed(outs))
        self.extra = (window - 1) // 2 - h           # unused margin of the window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], self.window)
        if self.extra:
            z = z[:, self.extra: self.window - self.extra]
        for stage, h_out in zip(self.stages, self.h_out):
            z = stage(z, h_out)
        return z[:, 0]
