"""Training losses on the same complex symbol estimate y (E030); the model and its multiplies are unchanged.

Logits over the 16 constellation points: l_k = -beta * |y - s_k|^2 with a learnable beta > 0
(beta = exp(log_beta), initial beta = 1 / sigma^2 of the raw received symbols of the train files).
The decision stays the nearest point, which equals argmax_k l_k for any beta > 0.

- ``mse``      |y - s|^2 (what runner.run_task uses when no loss is given);
- ``ce_dist``  cross-entropy over the 16 classes;
- ``bit_bce``  binary cross-entropy of the 4 Gray bits (core.BIT_TABLE), log P(bit = 1) is the
               log-sum-exp of the logits of the classes with that bit minus that of all classes;
- ``mse+ce``   mse + 0.1 * ce_dist.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .core import BIT_TABLE, CONSTELLATION

LOSSES = ("mse", "ce_dist", "bit_bce", "mse+ce")
CE_WEIGHT = 0.1


def _as_complex(prediction: torch.Tensor) -> torch.Tensor:
    return prediction if torch.is_complex(prediction) else torch.complex(prediction[..., 0], prediction[..., 1])


class SymbolLoss(nn.Module):
    def __init__(self, name: str, beta_init: float):
        super().__init__()
        if name not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        self.name = name
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))
        points = torch.tensor(CONSTELLATION, dtype=torch.complex64)
        self.register_buffer("points", points)
        self.register_buffer("bits", torch.tensor(BIT_TABLE, dtype=torch.bool))       # (16, 4)

    def logits(self, y: torch.Tensor) -> torch.Tensor:
        distance = (y[:, None] - self.points[None, :]).abs().square()
        return -self.log_beta.exp() * distance

    def classes(self, target: torch.Tensor) -> torch.Tensor:
        return (target[:, None] - self.points[None, :]).abs().argmin(dim=1)

    def forward(self, prediction: torch.Tensor, target_iq: torch.Tensor) -> torch.Tensor:
        y = _as_complex(prediction)
        target = torch.complex(target_iq[:, 0], target_iq[:, 1]).to(y.dtype)
        mse = (y - target).abs().square().mean()
        if self.name == "mse":
            return mse
        logits = self.logits(y)
        labels = self.classes(target)
        if self.name in ("ce_dist", "mse+ce"):
            ce = nn.functional.cross_entropy(logits, labels)
            return ce if self.name == "ce_dist" else mse + CE_WEIGHT * ce
        log_all = torch.logsumexp(logits, dim=1, keepdim=True)
        masked = logits[:, :, None].masked_fill(~self.bits[None], float("-inf"))     # (B, 16, 4)
        log_p1 = torch.logsumexp(masked, dim=1) - log_all                            # log P(bit = 1)
        log_p0 = torch.log1p(-log_p1.exp().clamp(max=1 - 1e-7))
        truth = self.bits[labels].float()
        return -(truth * log_p1 + (1 - truth) * log_p0).mean()


class WithLoss(nn.Module):
    """The equalizer plus its loss module (so beta is trained by the same optimizer)."""

    def __init__(self, model: nn.Module, loss: SymbolLoss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_loss(self, prediction: torch.Tensor, target_iq: torch.Tensor) -> torch.Tensor:
        return self.loss(prediction, target_iq)


def beta_from_train(train) -> float:
    """1 / mean |rx - tx|^2 over the train files (noise variance of the raw estimate)."""
    sigma2 = float(np.mean(np.abs(np.asarray(train.rx_c) - np.asarray(train.tx_c)) ** 2))
    return 1.0 / max(sigma2, 1e-6)


def attach_loss(model: nn.Module, name: str, train) -> WithLoss:
    return WithLoss(model, SymbolLoss(name, beta_from_train(train)))
