"""Physics-motivated equalizers for E016-E018 (digital back-propagation style).

One step = complex FIR filter of length L followed by a nonlinear phase rotation
with memory:   u = F(z),   y_k = u_k * exp(-i * sum_{j=-r..r} alpha_j * |u_{k+j}|^2),
r = (R - 1) / 2, alpha real and learnable (R coefficients; R = 1 is the classical
Kerr / SPM term, R = 0 disables the nonlinearity). FIR taps start as a unit
impulse and alpha as zero, so every step is the identity at initialization.

The network sees one window of W received symbols (W = 31 as in E010-E014).
Filtering and the power convolution use zero padding inside the window; the
centre symbol is the output. A streaming implementation would filter every
symbol once and reuse it for all outputs, which is what ``macs_phys`` counts.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _symmetric(half: torch.Tensor) -> torch.Tensor:
    """[h_0, h_1, .., h_m] -> [h_m, .., h_1, h_0, h_1, .., h_m]."""
    return torch.cat((half[1:].flip(0), half))


class PhysStep(nn.Module):
    def __init__(self, fir_len: int, n_alpha: int, sym: bool):
        super().__init__()
        if fir_len < 1 or fir_len % 2 == 0 or (n_alpha and n_alpha % 2 == 0):
            raise ValueError("fir_len and n_alpha must be odd (n_alpha may be 0)")
        self.fir_len, self.n_alpha, self.sym = fir_len, n_alpha, sym
        if fir_len > 1:
            n_taps = (fir_len + 1) // 2 if sym else fir_len
            centre = 0 if sym else fir_len // 2
            real = torch.zeros(n_taps)
            real[centre] = 1.0
            self.fir_re = nn.Parameter(real)
            self.fir_im = nn.Parameter(torch.zeros(n_taps))
        if n_alpha:
            self.alpha = nn.Parameter(torch.zeros((n_alpha + 1) // 2 if sym else n_alpha))

    def taps(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sym:
            return _symmetric(self.fir_re), _symmetric(self.fir_im)
        return self.fir_re, self.fir_im

    def alphas(self) -> torch.Tensor:
        return _symmetric(self.alpha) if self.sym else self.alpha

    def filter(self, z: torch.Tensor) -> torch.Tensor:
        if self.fir_len == 1:
            return z
        h_re, h_im = self.taps()
        pad = self.fir_len // 2
        x_re, x_im = z.real.unsqueeze(1), z.imag.unsqueeze(1)
        k_re, k_im = h_re.view(1, 1, -1), h_im.view(1, 1, -1)
        out_re = F.conv1d(x_re, k_re, padding=pad) - F.conv1d(x_im, k_im, padding=pad)
        out_im = F.conv1d(x_re, k_im, padding=pad) + F.conv1d(x_im, k_re, padding=pad)
        return torch.complex(out_re.squeeze(1), out_im.squeeze(1))

    def phase(self, u: torch.Tensor, centre_only: bool) -> torch.Tensor:
        power = u.real.square() + u.imag.square()
        alpha = self.alphas()
        r = self.n_alpha // 2
        if centre_only:
            c = u.shape[1] // 2
            return (power[:, c - r: c + r + 1] * alpha).sum(dim=1, keepdim=True)
        return F.conv1d(power.unsqueeze(1), alpha.view(1, 1, -1), padding=r).squeeze(1)

    def forward(self, z: torch.Tensor, centre_only: bool = False) -> torch.Tensor:
        u = self.filter(z)
        if centre_only:
            c = u.shape[1] // 2
            centre = u[:, c: c + 1]
            if not self.n_alpha:
                return centre
            return centre * torch.exp(-1j * self.phase(u, True))
        if not self.n_alpha:
            return u
        return u * torch.exp(-1j * self.phase(u, False))


class PhysDBP(nn.Module):
    """``n_steps`` x [FIR(L) + nonlinear phase with R coefficients]; ``n_steps = 1`` is PhysKerr."""

    def __init__(self, window: int, fir_len: int, n_alpha: int, sym: bool, n_steps: int = 1):
        super().__init__()
        if n_alpha > window:
            raise ValueError("n_alpha cannot exceed the window")
        self.window = window
        self.steps = nn.ModuleList(PhysStep(fir_len, n_alpha, sym) for _ in range(n_steps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], self.window)
        for index, step in enumerate(self.steps):
            z = step(z, centre_only=index == len(self.steps) - 1)
        return z.squeeze(-1)


class PhysPlusMLP(nn.Module):
    """E017: y = PhysKerr(z) + MLP_res(z); MLP_res is a one-hidden-layer complex Kerr MLP
    whose output layer starts at zero, so the hybrid starts equal to PhysKerr."""

    def __init__(self, window: int, fir_len: int, n_alpha: int, sym: bool, hidden: int):
        super().__init__()
        from .complex_mlp_ablation import AblationComplexMLP

        self.phys = PhysDBP(window, fir_len, n_alpha, sym, 1)
        self.mlp = AblationComplexMLP(window=window, hidden=hidden, variant="d")
        for name in ("weight_real", "weight_imag", "bias_real", "bias_imag"):
            nn.init.zeros_(getattr(self.mlp.fc3, name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.phys(x) + self.mlp(x)


def build_phys_model(spec: dict) -> nn.Module:
    window = int(spec["window"])
    fir_len, n_alpha, sym = int(spec["fir_len"]), int(spec["n_alpha"]), bool(spec["sym"])
    if spec["model"] == "phys_kerr":
        return PhysDBP(window, fir_len, n_alpha, sym, 1)
    if spec["model"] == "phys_dbp":
        return PhysDBP(window, fir_len, n_alpha, sym, int(spec["n_steps"]))
    if spec["model"] == "phys_plus_mlp":
        return PhysPlusMLP(window, fir_len, n_alpha, sym, int(spec["mlp_hidden"]))
    raise ValueError(spec["model"])
