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

    @staticmethod
    def _band(taps: torch.Tensor, window: int) -> torch.Tensor:
        """(W, W) matrix T with (x @ T)_k = sum_t taps_t x_{k + t - len//2}, zero padding."""
        length = taps.shape[0]
        rows = torch.arange(window, device=taps.device)
        offset = rows[:, None] - rows[None, :] + length // 2
        valid = (offset >= 0) & (offset < length)
        return torch.where(valid, taps[offset.clamp(0, length - 1)], torch.zeros((), dtype=taps.dtype,
                                                                                  device=taps.device))

    def toeplitz(self, window: int) -> torch.Tensor:
        h_re, h_im = self.taps()
        return torch.complex(self._band(h_re, window), self._band(h_im, window))

    def filter(self, z: torch.Tensor, columns: slice | None = None) -> torch.Tensor:
        """FIR inside the window; ``columns`` restricts the output positions (cheaper for the last step)."""
        if self.fir_len == 1:
            return z if columns is None else z[:, columns]
        matrix = self.toeplitz(z.shape[1])
        return z @ (matrix if columns is None else matrix[:, columns])

    def filter_conv(self, z: torch.Tensor) -> torch.Tensor:
        """Reference implementation with conv1d (used by tests)."""
        if self.fir_len == 1:
            return z
        h_re, h_im = self.taps()
        pad = self.fir_len // 2
        x_re, x_im = z.real.unsqueeze(1), z.imag.unsqueeze(1)
        k_re, k_im = h_re.view(1, 1, -1), h_im.view(1, 1, -1)
        out_re = F.conv1d(x_re, k_re, padding=pad) - F.conv1d(x_im, k_im, padding=pad)
        out_im = F.conv1d(x_re, k_im, padding=pad) + F.conv1d(x_im, k_re, padding=pad)
        return torch.complex(out_re.squeeze(1), out_im.squeeze(1))

    def forward(self, z: torch.Tensor, centre_only: bool = False) -> torch.Tensor:
        alpha = self.alphas() if self.n_alpha else None
        r = self.n_alpha // 2
        if centre_only:
            c = z.shape[1] // 2
            u = self.filter(z, slice(c - r, c + r + 1))          # only the positions the phase needs
            centre = u[:, r: r + 1]
            if alpha is None:
                return centre
            power = u.real.square() + u.imag.square()
            return centre * torch.exp(-1j * (power * alpha).sum(dim=1, keepdim=True))
        u = self.filter(z)
        if alpha is None:
            return u
        power = u.real.square() + u.imag.square()
        phase = power @ self._band(alpha, power.shape[1])     # same as conv1d(power, alpha, padding=r)
        return u * torch.exp(-1j * phase)


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
        model = PhysDBP(window, fir_len, n_alpha, sym, int(spec["n_steps"]))
        if spec.get("alpha_init") is not None:   # E022: nonlinear phase set from outside (kept frozen by phases)
            values = torch.tensor([float(v) for v in spec["alpha_init"]])
            for step in model.steps:
                if step.alpha.shape != values.shape:
                    raise ValueError("alpha_init length must match the stored alpha (use sym=False for full R)")
                with torch.no_grad():
                    step.alpha.copy_(values)
        return model
    if spec["model"] == "phys_plus_mlp":
        return PhysPlusMLP(window, fir_len, n_alpha, sym, int(spec["mlp_hidden"]))
    raise ValueError(spec["model"])
