"""Learning-rate schedules and optimizer parameter groups (E010+).

Defaults reproduce the historical setting of E005-E009: Adam, constant lr 1e-3,
no warmup, one learning rate for every parameter (``c_mult=1``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

SCHEDULES = ("const", "cosine_to_zero", "cosine_to_floor")


@dataclass(frozen=True)
class LRConfig:
    peak_lr: float = 1e-3
    schedule: str = "const"
    warmup: int = 0
    c_mult: float = 1.0
    floor_frac: float = 0.1          # cosine_to_floor ends at floor_frac * peak
    optimizer: str = "adam"          # adam | adamw
    weight_decay: float = 0.0
    horizon: int | None = None       # decay horizon; None = total steps of the run

    def __post_init__(self):
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError("optimizer must be 'adam' or 'adamw'")

    @classmethod
    def from_dict(cls, payload: dict | None) -> "LRConfig":
        payload = dict(payload or {})
        payload.pop("name", None)
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)

    def tag(self) -> str:
        parts = [f"{self.schedule}", f"lr{self.peak_lr:g}", f"wu{self.warmup}", f"cm{self.c_mult:g}"]
        if self.optimizer != "adam":
            parts.append(self.optimizer)
        if self.horizon is not None:
            parts.append(f"hz{self.horizon}")
        return "_".join(parts)


def lr_factor(step: int, total: int, config: LRConfig) -> float:
    """Multiplier of the peak lr used for optimizer update number ``step + 1``.

    ``step`` counts completed updates (0 for the first update), as in
    ``torch.optim.lr_scheduler`` after ``step`` calls. Warmup is linear from
    ``1/warmup`` to 1; the decay runs over ``horizon - warmup`` updates after it.
    """
    horizon = config.horizon or total
    if config.warmup > 0 and step < config.warmup:
        return (step + 1) / config.warmup
    if config.schedule == "const":
        return 1.0
    span = max(1, horizon - config.warmup)
    progress = min(1.0, (step - config.warmup) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = 0.0 if config.schedule == "cosine_to_zero" else config.floor_frac
    return floor + (1.0 - floor) * cosine


def is_kerr_coefficient(name: str, module_types: dict[str, type]) -> bool:
    owner = name.rsplit(".", 1)[0] if "." in name else ""
    from .paper_complex_mlp import NonlinearPhase

    return name.endswith(".c") and issubclass(module_types.get(owner, object), NonlinearPhase)


def build_optimizer(model: nn.Module, config: LRConfig) -> tuple[torch.optim.Optimizer, list[float]]:
    """Optimizer with two groups (weights, Kerr ``c``); returns it with per-group base lr."""
    module_types = {name: type(module) for name, module in model.named_modules()}
    kerr, other = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (kerr if is_kerr_coefficient(name, module_types) else other).append(parameter)
    groups, base = [], []
    if other:
        groups.append({"params": other, "lr": config.peak_lr, "name": "weights"})
        base.append(config.peak_lr)
    if kerr:
        groups.append({"params": kerr, "lr": config.peak_lr * config.c_mult, "name": "kerr_c"})
        base.append(config.peak_lr * config.c_mult)
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(groups, lr=config.peak_lr)
    else:
        optimizer = torch.optim.AdamW(groups, lr=config.peak_lr, weight_decay=config.weight_decay)
    return optimizer, base


def apply_lr(optimizer: torch.optim.Optimizer, base: list[float], step: int, total: int, config: LRConfig) -> float:
    factor = lr_factor(step, total, config)
    for group, lr in zip(optimizer.param_groups, base):
        group["lr"] = lr * factor
    return optimizer.param_groups[0]["lr"]
