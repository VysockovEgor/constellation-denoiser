"""Minimal B-spline KAN layer (efficient-KAN style) and Kerr + KAN hybrids for E014.

KANLinear(n -> m): y_j = sum_i [ w_base[j,i] * silu(x_i) + sum_g c[j,i,g] * B_g(x_i) ],
B_g are order-k B-splines on a uniform grid of G intervals over ``grid_range``
extended by k knots on each side (G + k basis functions per input edge).
No external packages (pykan, efficient-kan) are used.

Operation convention (real multiplications per predicted symbol), see
``kan_layer_cost``: SiLU = 1 per input (sigmoid not counted, like tanh/GELU
elsewhere); Cox-de Boor recursion = 4 per basis function and level with
precomputed reciprocal knot spans; base linear m*n; spline linear m*n*(G+k)
("dense", as implemented) or m*n*(k+1) ("sparse": only k+1 bases are nonzero
for any input). Both numbers are written to the run JSON.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class KANLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple[float, float] = (-2.0, 2.0), zero_init: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        step = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step + grid_range[0]
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        if zero_init:
            nn.init.zeros_(self.base_weight)
            nn.init.zeros_(self.spline_weight)
        else:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            nn.init.normal_(self.spline_weight, std=0.1 / math.sqrt(in_features * (grid_size + spline_order)))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in) -> (B, in, G + k) basis values (Cox-de Boor)."""
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(order + 1)]) / (grid[:, order:-1] - grid[:, : -(order + 1)])
            right = (grid[:, order + 1:] - x) / (grid[:, order + 1:] - grid[:, 1:-order])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(F.silu(x), self.base_weight)
        spline = F.linear(self.b_splines(x).reshape(x.shape[0], -1),
                          self.spline_weight.reshape(self.out_features, -1))
        return base + spline


class KANNet(nn.Module):
    def __init__(self, widths: list[int], grid_size: int, spline_order: int = 3, zero_last: bool = False):
        super().__init__()
        self.widths = widths
        self.layers = nn.ModuleList(
            KANLinear(a, b, grid_size, spline_order, zero_init=zero_last and i == len(widths) - 2)
            for i, (a, b) in enumerate(zip(widths[:-1], widths[1:]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def local_features(x: torch.Tensor, half: int) -> torch.Tensor:
    """Centre symbol and ``half`` neighbours on each side, flattened [Re, Im]: (B, W, 2) -> (B, 2(2h+1))."""
    centre = x.shape[1] // 2
    return x[:, centre - half: centre + half + 1, :].reshape(x.shape[0], -1)


class KANOnly(nn.Module):
    """k1: KAN [2W -> hidden -> 2] on the real features of the whole window."""

    def __init__(self, window: int, hidden: int, grid_size: int):
        super().__init__()
        self.kan = KANNet([2 * window, hidden, 2], grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x.reshape(x.shape[0], -1))


class KerrKAN(nn.Module):
    """Kerr MLP (one hidden layer) plus a KAN correction; complex output.

    mode ``parallel``: out = kerr(x) + KAN(local)       (k2, k3)
    mode ``cascade``:  out = kerr(x) + KAN([kerr(x), local])   (k4)
    The last KAN layer starts at zero, so the hybrid initially equals the Kerr MLP.
    """

    def __init__(self, window: int, hidden: int, kan_hidden: int, grid_size: int, local_half: int, mode: str):
        super().__init__()
        from .complex_mlp_ablation import AblationComplexMLP

        self.kerr = AblationComplexMLP(window=window, hidden=hidden, variant="d")
        self.mode, self.local_half = mode, local_half
        n_local = 2 * (2 * local_half + 1)
        n_in = n_local + (2 if mode == "cascade" else 0)
        self.kan = KANNet([n_in, kan_hidden, 2], grid_size, zero_last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.kerr(x)
        features = local_features(x, self.local_half)
        if self.mode == "cascade":
            features = torch.cat((torch.stack((base.real, base.imag), dim=-1), features), dim=-1)
        correction = self.kan(features)
        return base + torch.complex(correction[:, 0], correction[:, 1])


def build_kan_model(spec: dict) -> nn.Module:
    window, grid = int(spec["window"]), int(spec["grid"])
    if spec["model"] == "kan":
        return KANOnly(window, int(spec.get("kan_hidden", 16)), grid)
    return KerrKAN(window, int(spec.get("hidden", 48)), int(spec.get("kan_hidden", 8)), grid,
                   int(spec.get("local_half", 3)), spec.get("mode", "parallel"))


def kan_layer_cost(n: int, m: int, grid: int, order: int = 3) -> dict:
    basis_dense = n * sum(4 * (grid + 2 * order - level) for level in range(1, order + 1))
    basis_sparse = n * sum(4 * (level + 1) for level in range(1, order + 1))
    silu = n
    return {
        "params": m * n * (1 + grid + order),
        "dense": silu + basis_dense + m * n + m * n * (grid + order),
        "sparse": silu + basis_sparse + m * n + m * n * (order + 1),
    }


def kan_net_cost(widths: list[int], grid: int, order: int = 3) -> dict:
    parts = [kan_layer_cost(a, b, grid, order) for a, b in zip(widths[:-1], widths[1:])]
    return {key: sum(p[key] for p in parts) for key in ("params", "dense", "sparse")}


def kan_part_widths(spec: dict) -> list[int]:
    window = int(spec["window"])
    if spec["model"] == "kan":
        return [2 * window, int(spec.get("kan_hidden", 16)), 2]
    n_local = 2 * (2 * int(spec.get("local_half", 3)) + 1)
    n_in = n_local + (2 if spec.get("mode", "parallel") == "cascade" else 0)
    return [n_in, int(spec.get("kan_hidden", 8)), 2]


def kan_cost_details(spec: dict) -> dict:
    from .model_zoo import complex_one_hidden_macs

    kan = kan_net_cost(kan_part_widths(spec), int(spec["grid"]))
    kerr = 0 if spec["model"] == "kan" else complex_one_hidden_macs(int(spec["window"]), int(spec.get("hidden", 48)))
    return {"kan_widths": kan_part_widths(spec), "kan_params": kan["params"], "kan_macs_dense": kan["dense"],
            "kan_macs_sparse": kan["sparse"], "kerr_macs": kerr, "macs_dense_total": kerr + kan["dense"],
            "macs_sparse_total": kerr + kan["sparse"]}


def kan_model_cost(spec: dict):
    from .model_zoo import Cost

    details = kan_cost_details(spec)
    # macs = dense (as implemented); macs_merged = sparse convention (k+1 nonzero bases)
    return Cost(details["macs_dense_total"], details["macs_sparse_total"])
