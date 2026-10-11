"""Kerr activation ablations, linear-layer merge, and complexity accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .paper_complex_mlp import ComplexDense, NonlinearPhase, PaperComplexMLP, predict_complex

PHASE_REAL_MULTS_PER_NEURON = 7
VARIANTS = ("baseline", "a", "b", "c", "d")
ABLATION_VARIANTS = ("a", "b", "c", "d")


class IdentityPhase(nn.Module):
    """No-op activation used when Kerr is disabled at a layer boundary."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z

    @property
    def c(self) -> torch.Tensor:
        return torch.tensor(float("nan"))


def _complex_weight_bias(layer: ComplexDense) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.complex(layer.weight_real, layer.weight_imag)
    bias = torch.complex(layer.bias_real, layer.bias_imag)
    return weight, bias


def merge_complex_dense(outer: ComplexDense, inner: ComplexDense) -> tuple[torch.Tensor, torch.Tensor]:
    """W = W_outer W_inner, b = W_outer b_inner + b_outer for y = W x + b."""
    w_outer, b_outer = _complex_weight_bias(outer)
    w_inner, b_inner = _complex_weight_bias(inner)
    return w_outer @ w_inner, w_outer @ b_inner + b_outer


class AblationComplexMLP(nn.Module):
    """Two-hidden-layer ablation network; variant ``d`` uses a single hidden layer."""

    def __init__(self, window: int = 31, hidden: int = 48, variant: str = "baseline"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.window = window
        self.hidden = hidden
        self.variant = variant

        if variant == "d":
            self.fc1 = ComplexDense(window, hidden)
            self.act1 = NonlinearPhase()
            self.fc2 = None
            self.act2 = IdentityPhase()
            self.fc3 = ComplexDense(hidden, 1)
        else:
            self.fc1 = ComplexDense(window, hidden)
            self.fc2 = ComplexDense(hidden, hidden)
            self.fc3 = ComplexDense(hidden, 1)
            if variant in ("baseline", "b"):
                self.act1 = NonlinearPhase()
            else:
                self.act1 = IdentityPhase()
            if variant in ("baseline", "a"):
                self.act2 = NonlinearPhase()
            else:
                self.act2 = IdentityPhase()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], self.window)
        z = self.act1(self.fc1(z))
        if self.fc2 is not None:
            z = self.act2(self.fc2(z))
        return self.fc3(z).squeeze(-1)


class MergedAblationMLP(nn.Module):
    """Inference graph after folding consecutive linear maps where activations allow it."""

    def __init__(self, source: AblationComplexMLP | PaperComplexMLP):
        super().__init__()
        self.window = source.window
        self.hidden = source.hidden
        variant = getattr(source, "variant", "baseline")

        if variant == "d":
            self.mode = "d"
            self.fc1 = source.fc1
            self.act1 = source.act1
            self.fc3 = source.fc3
            return

        if variant in ("a", "c"):
            self.mode = variant
            w_merged, b_merged = merge_complex_dense(source.fc2, source.fc1)
            self.register_buffer("merged_weight", w_merged)
            self.register_buffer("merged_bias", b_merged)
            self.act2 = source.act2 if variant == "a" else IdentityPhase()
            self.fc3 = source.fc3
            return

        if variant == "b":
            self.mode = "b"
            w_merged, b_merged = merge_complex_dense(source.fc3, source.fc2)
            self.fc1 = source.fc1
            self.act1 = source.act1
            self.register_buffer("merged_weight", w_merged)
            self.register_buffer("merged_bias", b_merged)
            return

        self.mode = "none"
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], self.window)
        if self.mode in ("a", "c"):
            z = z @ self.merged_weight.transpose(0, 1) + self.merged_bias
            z = self.act2(z)
            return self.fc3(z).squeeze(-1)
        if self.mode == "b":
            z = self.act1(self.fc1(z))
            return (z @ self.merged_weight.transpose(0, 1) + self.merged_bias).squeeze(-1)
        if self.mode == "d":
            z = self.act1(self.fc1(z))
            return self.fc3(z).squeeze(-1)
        return self.source(x)


@dataclass(frozen=True)
class Complexity:
    parameters: int
    train_dense_real_multiplies: int
    train_phase_real_multiplies: int
    train_total_real_multiplies: int
    merged_dense_real_multiplies: int
    merged_phase_real_multiplies: int
    merged_total_real_multiplies: int


def _dense_real(window: int, hidden: int, *, two_hidden: bool) -> int:
    if two_hidden:
        complex_products = window * hidden + hidden * hidden + hidden
    else:
        complex_products = window * hidden + hidden
    return 4 * complex_products


def complexity(window: int, hidden: int, variant: str) -> Complexity:
    if variant == "d":
        train_dense = _dense_real(window, hidden, two_hidden=False)
        train_phase = PHASE_REAL_MULTS_PER_NEURON * hidden
        merged_dense = train_dense
        merged_phase = train_phase
        return Complexity(
            parameters=-1,
            train_dense_real_multiplies=train_dense,
            train_phase_real_multiplies=train_phase,
            train_total_real_multiplies=train_dense + train_phase,
            merged_dense_real_multiplies=merged_dense,
            merged_phase_real_multiplies=merged_phase,
            merged_total_real_multiplies=merged_dense + merged_phase,
        )

    train_dense = _dense_real(window, hidden, two_hidden=True)
    if variant == "baseline":
        train_phase = PHASE_REAL_MULTS_PER_NEURON * 2 * hidden
    elif variant in ("a", "b"):
        train_phase = PHASE_REAL_MULTS_PER_NEURON * hidden
    else:
        train_phase = 0

    if variant in ("a", "c"):
        merged_dense = _dense_real(window, hidden, two_hidden=False)
        merged_phase = PHASE_REAL_MULTS_PER_NEURON * hidden if variant == "a" else 0
    elif variant == "b":
        merged_dense = 4 * (window * hidden + hidden)
        merged_phase = PHASE_REAL_MULTS_PER_NEURON * hidden
    else:
        merged_dense = train_dense
        merged_phase = train_phase

    return Complexity(
        parameters=-1,
        train_dense_real_multiplies=train_dense,
        train_phase_real_multiplies=train_phase,
        train_total_real_multiplies=train_dense + train_phase,
        merged_dense_real_multiplies=merged_dense,
        merged_phase_real_multiplies=merged_phase,
        merged_total_real_multiplies=merged_dense + merged_phase,
    )


def build_model(window: int, hidden: int, variant: str) -> nn.Module:
    if variant == "baseline":
        return PaperComplexMLP(window=window, hidden=hidden)
    return AblationComplexMLP(window=window, hidden=hidden, variant=variant)


@torch.inference_mode()
def activation_diagnostics(model: nn.Module, x_probe: torch.Tensor, window: int) -> dict[str, float]:
    z = torch.complex(x_probe[..., 0], x_probe[..., 1]).reshape(x_probe.shape[0], window)
    z_pre_act1 = model.fc1(z)
    if getattr(model, "fc2", None) is None:
        z_pre_act2 = torch.empty(0)
    else:
        z_pre_act2 = model.fc2(model.act1(z_pre_act1))
    return {
        "c1": float(model.act1.c.detach().item()),
        "c2": float(model.act2.c.detach().item()),
        "mean_abs_z_act1_sq": float(z_pre_act1.abs().square().mean().item()),
        "mean_abs_z_act2_sq": float(z_pre_act2.abs().square().mean().item()) if z_pre_act2.numel() else float("nan"),
    }


@torch.inference_mode()
def verify_merge(model: nn.Module, x: np.ndarray, *, atol: float = 1e-5) -> tuple[float, int]:
    merged = MergedAblationMLP(model)
    merged.eval()
    model.eval()
    x_t = torch.from_numpy(x[:1000])
    original = model(x_t)
    approx = merged(x_t)
    diff = (original - approx).abs().max().item()
    return diff, min(1000, len(x))
