"""Baseline models for E022 (added for comparison at equal complexity; existing models untouched).

WindowCNN — a simplified real-valued 1D CNN in the spirit of CNN equalizers from the literature:
input channels [Re, Im] of the received window, three convolution layers with tanh between them,
output channels [Re, Im] at the centre position. It is an approximation for a complexity-matched
comparison, not a reproduction of any specific published network.

Multiplication convention (real multiplies per predicted symbol, streaming): each layer produces one
new output position per symbol, so the cost is sum over layers of C_in * C_out * k; tanh, biases and
additions are not counted (as tanh/GELU elsewhere in the project).
"""

from __future__ import annotations

import torch
from torch import nn


class WindowCNN(nn.Module):
    def __init__(self, window: int, channels: int, kernel: int):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        if 3 * (kernel - 1) + 1 > window:
            raise ValueError("receptive field of three layers exceeds the window")
        pad = kernel // 2
        self.window = window
        self.net = nn.Sequential(
            nn.Conv1d(2, channels, kernel, padding=pad), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel, padding=pad), nn.Tanh(),
            nn.Conv1d(channels, 2, kernel, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.transpose(1, 2))           # (B, 2, W)
        return out[:, :, self.window // 2]          # (B, 2): [Re, Im] of the centre symbol


def cnn_macs(channels: int, kernel: int) -> int:
    """2*C*k (first layer) + C*C*k (second) + C*2*k (third) real multiplies per symbol."""
    return 2 * channels * kernel + channels * channels * kernel + channels * 2 * kernel


def width_for_macs(kernel: int, target: int) -> int:
    return min(range(1, 400), key=lambda c: abs(cnn_macs(c, kernel) - target))
