"""Model registry and real-multiplication accounting for E010+.

Reuses the historical classes unchanged:
- ``complex_both``        PaperComplexMLP (Kerr after both hidden layers, E007/E008 baseline);
- ``complex_a``           AblationComplexMLP variant ``a`` (Kerr only after layer 2, E009);
- ``complex_one_hidden``  AblationComplexMLP variant ``d`` (one hidden layer with Kerr);
- ``real_tanh``           WindowMLP tanh, Xavier normal, zero bias (E002/E005 real baseline);
- ``transformer``         TransformerEqualizer from ``models.py`` (E001), any odd window.

Multiplication convention (real multiplies per predicted symbol): complex
product = 4, Kerr activation = 7 per neuron, biases/additions/tanh/GELU/softmax/
LayerNorm are not counted.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .complex_mlp_ablation import complexity
from .experiment import set_seed


@dataclass(frozen=True)
class Cost:
    macs: int            # as trained
    macs_merged: int     # after folding linear layers that have no activation between them


def complex_two_hidden_macs(window: int, hidden: int) -> int:
    """Two hidden layers, Kerr on both: 4h(W + h + 1) + 14h."""
    return 4 * hidden * (window + hidden + 1) + 14 * hidden


def complex_one_hidden_macs(window: int, hidden: int) -> int:
    """One hidden layer with Kerr (or variant a after merging): h(4W + 11)."""
    return hidden * (4 * window + 11)


def width_matching_macs(window: int, target_macs: int) -> int:
    """Integer one-hidden-layer width whose cost is closest to ``target_macs``."""
    return max(1, round(target_macs / (4 * window + 11)))


def real_mlp_macs(window: int, hidden: int) -> int:
    return 2 * window * hidden + hidden * hidden + 2 * hidden


def transformer_macs(window: int, width: int, layers: int, ffn_mult: int = 2) -> int:
    """Full cost of one window (no sliding reuse for bidirectional attention).

    Per token: input projection 2d; per layer QKV 3d^2, output projection d^2,
    FFN 2*d*(ffn_mult*d). Per window and layer: QK^T W^2 d and AV W^2 d.
    Head (E001): Linear(d, d) + Linear(d, 2) on the centre token = d^2 + 2d.
    """
    d, w = width, window
    per_layer = w * (3 * d * d + d * d + 2 * d * ffn_mult * d) + 2 * w * w * d
    return w * 2 * d + layers * per_layer + d * d + 2 * d


def build(spec: dict, seed: int | None = None) -> nn.Module:
    """Construct a model from a spec dict; ``set_seed`` first so init matches E005-E009."""
    if seed is not None:
        set_seed(seed)
    kind, window = spec["model"], int(spec["window"])
    if kind == "complex_both":
        from .paper_complex_mlp import PaperComplexMLP

        return PaperComplexMLP(window=window, hidden=int(spec["hidden"]))
    if kind in ("complex_a", "complex_one_hidden"):
        from .complex_mlp_ablation import AblationComplexMLP

        variant = "a" if kind == "complex_a" else "d"
        return AblationComplexMLP(window=window, hidden=int(spec["hidden"]), variant=variant)
    if kind == "real_tanh":
        from .models import WindowMLP

        model = WindowMLP(window, hidden=int(spec["hidden"]), activation="tanh")
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        return model
    if kind == "transformer":
        from .models import TransformerEqualizer

        if int(spec.get("ffn_mult", 2)) != 2:
            raise ValueError("TransformerEqualizer (E001) has dim_feedforward = 2*width; ffn_mult must be 2")
        return TransformerEqualizer(window, width=int(spec["d_model"]), heads=int(spec.get("heads", 4)),
                                    layers=int(spec["layers"]))
    if kind in ("kan", "kerr_kan"):
        from .kan import build_kan_model

        return build_kan_model(spec)
    raise ValueError(f"unknown model kind {kind!r}")


def cost(spec: dict) -> Cost:
    kind, window = spec["model"], int(spec["window"])
    if kind == "complex_both":
        c = complexity(window, int(spec["hidden"]), "baseline")
        return Cost(c.train_total_real_multiplies, c.merged_total_real_multiplies)
    if kind == "complex_a":
        c = complexity(window, int(spec["hidden"]), "a")
        return Cost(c.train_total_real_multiplies, c.merged_total_real_multiplies)
    if kind == "complex_one_hidden":
        c = complexity(window, int(spec["hidden"]), "d")
        return Cost(c.train_total_real_multiplies, c.merged_total_real_multiplies)
    if kind == "real_tanh":
        m = real_mlp_macs(window, int(spec["hidden"]))
        return Cost(m, m)
    if kind == "transformer":
        m = transformer_macs(window, int(spec["d_model"]), int(spec["layers"]), int(spec.get("ffn_mult", 2)))
        return Cost(m, m)
    if kind in ("kan", "kerr_kan"):
        from .kan import kan_model_cost

        return kan_model_cost(spec)
    raise ValueError(f"unknown model kind {kind!r}")


def n_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def outputs_complex(spec: dict) -> bool:
    return spec["model"].startswith("complex") or spec["model"] == "kerr_kan"
