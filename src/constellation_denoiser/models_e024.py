"""Model kinds added in E024-E033 (the historical classes in model_zoo stay unchanged).

- ``act_mlp``     one-hidden-layer complex MLP (layout of AblationComplexMLP variant ``d``) with a chosen
                  complex activation (E027): spec ``activation`` in ACTIVATIONS, optional ``c_fixed``;
- ``real_mlp``    real MLP on [Re, Im] of the window with 1 or 2 hidden layers (E025 F10/F11), spec
                  ``layers``, ``hidden``, ``activation`` (tanh | relu | gelu | silu), ``init_scale``;
- ``dbp_shared``  PhysDBP whose Ns steps share one PhysStep (same multiplies, fewer parameters; E028);
- ``dbp_mlp``     PhysDBP(Ns) + one-hidden-layer Kerr MLP on the residual, output layer at zero (E028;
                  Ns = 1 is the E017 hybrid ``phys_plus_mlp``);
- ``dbp_noise``   PhysDBP whose taps and alpha start at identity + N(0, ``init_noise``^2) (E028 variant c).

Real multiplies per symbol (same convention as model_zoo: complex product 4, Kerr 7 per neuron,
real activations, additions and biases 0; activation costs per neuron are in ACT_RM).
"""

from __future__ import annotations

import torch
from torch import nn

from .paper_complex_mlp import ComplexDense, NonlinearPhase

KINDS = ("act_mlp", "real_mlp", "dbp_shared", "dbp_mlp", "dbp_noise", "triplet_net")

# real multiplies per hidden neuron of each activation (|z|^2 = 2, real x complex = 2, complex x complex = 4,
# a division counts as 1, exp/cos/sin/sqrt/sigmoid/tanh/ReLU from lookup or comparison = 0)
ACT_RM = {
    "kerr_c_train": 7,    # |z|^2 2, c*|z|^2 1, rotation 4
    "kerr_c_fixed": 7,
    "kerr_taylor1": 8,    # |z|^2 2, |z|^2 * z 2, a * (.) 4 (complex a)
    "modrelu": 5,         # |z|^2 2, relu(|z| + b) / |z| 1, scale * z 2
    "crelu": 0,
    "split_tanh": 0,
    "cardioid": 6,        # |z|^2 2, Re z / |z| 1, 0.5 (1 + cos) 1, scale * z 2
    "sigmoid_gate": 5,    # |z|^2 2, a * |z|^2 1, sigma * z 2
    "identity": 0,
}
ACTIVATIONS = tuple(ACT_RM)


class FixedKerr(nn.Module):
    """exp(i c |z|^2) z with c a buffer: never trained (set_trainable only touches parameters)."""

    def __init__(self, c: float):
        super().__init__()
        self.register_buffer("c", torch.tensor(float(c)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(1j * self.c * z.abs().square()) * z


class TaylorKerr(NonlinearPhase):
    """First term of the Kerr expansion with a complex coefficient a = c[0] + i c[1]: z + a |z|^2 z.

    Subclass of NonlinearPhase with the parameter named ``c``, so the optimizer gives it the
    ``c_mult`` learning rate like the Kerr coefficient (lr_schedules.is_kerr_coefficient)."""

    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + torch.complex(self.c[0], self.c[1]) * z.abs().square() * z


class ModReLU(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = z.abs().clamp_min(1e-8)
        return torch.relu(magnitude + self.bias) / magnitude * z


class CReLU(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))


class SplitTanh(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.tanh(z.real), torch.tanh(z.imag))


class Cardioid(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cos = z.real / z.abs().clamp_min(1e-8)
        return 0.5 * (1.0 + cos) * z


class SigmoidGate(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(hidden))
        self.b = nn.Parameter(torch.full((hidden,), 2.0))   # gate ~0.88 at start: close to identity

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * z.abs().square() + self.b) * z


class Identity(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


def make_activation(name: str, hidden: int, c_fixed: float | None = None) -> nn.Module:
    if name == "kerr_c_train":
        return NonlinearPhase()
    if name == "kerr_c_fixed":
        if c_fixed is None:
            raise ValueError("kerr_c_fixed needs spec['c_fixed']")
        return FixedKerr(c_fixed)
    if name == "kerr_taylor1":
        return TaylorKerr()
    if name == "modrelu":
        return ModReLU(hidden)
    if name == "crelu":
        return CReLU()
    if name == "split_tanh":
        return SplitTanh()
    if name == "cardioid":
        return Cardioid()
    if name == "sigmoid_gate":
        return SigmoidGate(hidden)
    if name == "identity":
        return Identity()
    raise ValueError(f"unknown activation {name!r}; expected one of {ACTIVATIONS}")


class ActivationMLP(nn.Module):
    """fc1 (W -> h) -> activation -> fc3 (h -> 1); with ``kerr_c_train`` equal to AblationComplexMLP('d')."""

    def __init__(self, window: int, hidden: int, activation: str, c_fixed: float | None = None):
        super().__init__()
        self.window = window
        self.fc1 = ComplexDense(window, hidden)
        self.act1 = make_activation(activation, hidden, c_fixed)
        self.fc3 = ComplexDense(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], self.window)
        return self.fc3(self.act1(self.fc1(z))).squeeze(-1)


REAL_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class RealMLP(nn.Module):
    """Real MLP: 2W -> h (-> h) -> 2; Xavier normal (times ``init_scale``), zero bias as real_tanh."""

    def __init__(self, window: int, hidden: int, layers: int, activation: str, init_scale: float = 1.0):
        super().__init__()
        if layers not in (1, 2):
            raise ValueError("layers must be 1 or 2")
        act = REAL_ACTIVATIONS[activation]
        blocks: list[nn.Module] = [nn.Flatten(), nn.Linear(2 * window, hidden), act()]
        if layers == 2:
            blocks += [nn.Linear(hidden, hidden), act()]
        blocks.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*blocks)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                with torch.no_grad():
                    layer.weight.mul_(float(init_scale))
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedDBP(nn.Module):
    """Ns applications of one PhysStep (DBP with parameters shared between steps)."""

    def __init__(self, window: int, fir_len: int, n_alpha: int, sym: bool, n_steps: int):
        super().__init__()
        from .phys_models import PhysStep

        self.window, self.n_steps = window, n_steps
        self.step = PhysStep(fir_len, n_alpha, sym)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], self.window)
        for index in range(self.n_steps):
            z = self.step(z, centre_only=index == self.n_steps - 1)
        return z.squeeze(-1)


class DBPPlusMLP(nn.Module):
    """y = PhysDBP_Ns(z) + MLP_res(z); MLP_res output layer starts at zero (starts equal to the DBP)."""

    def __init__(self, window: int, fir_len: int, n_alpha: int, sym: bool, n_steps: int, hidden: int):
        super().__init__()
        from .complex_mlp_ablation import AblationComplexMLP
        from .phys_models import PhysDBP

        self.phys = PhysDBP(window, fir_len, n_alpha, sym, n_steps)
        self.mlp = AblationComplexMLP(window=window, hidden=hidden, variant="d")
        for name in ("weight_real", "weight_imag", "bias_real", "bias_imag"):
            nn.init.zeros_(getattr(self.mlp.fc3, name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.phys(x) + self.mlp(x)


def build(spec: dict) -> nn.Module:
    kind, window = spec["model"], int(spec["window"])
    if kind == "act_mlp":
        return ActivationMLP(window, int(spec["hidden"]), spec["activation"], spec.get("c_fixed"))
    if kind == "real_mlp":
        return RealMLP(window, int(spec["hidden"]), int(spec["layers"]), spec.get("activation", "tanh"),
                       float(spec.get("init_scale", 1.0)))
    if kind == "dbp_shared":
        return SharedDBP(window, int(spec["fir_len"]), int(spec["n_alpha"]), bool(spec["sym"]), int(spec["n_steps"]))
    if kind == "dbp_mlp":
        return DBPPlusMLP(window, int(spec["fir_len"]), int(spec["n_alpha"]), bool(spec["sym"]), int(spec["n_steps"]),
                          int(spec["mlp_hidden"]))
    if kind == "triplet_net":            # E035: perturbation triplets (+ Kerr MLP), triplets_ext.TripletNet
        from .paths import PROJECT_ROOT
        from .triplets_ext import TripletNet

        warm = spec.get("warm")
        return TripletNet(window, int(spec["M"]), int(spec["fir_len"]), int(spec.get("mlp_hidden", 0)),
                          str(PROJECT_ROOT / warm) if warm else None)
    if kind == "dbp_noise":
        from .phys_models import PhysDBP

        model = PhysDBP(window, int(spec["fir_len"]), int(spec["n_alpha"]), bool(spec["sym"]), int(spec["n_steps"]))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * float(spec["init_noise"]))
        return model
    raise ValueError(kind)


def act_mlp_macs(window: int, hidden: int, activation: str) -> int:
    """4Wh (fc1) + 4h (fc3) + ACT_RM * h; kerr_c_train gives h(4W + 11) as complex_one_hidden."""
    return 4 * window * hidden + 4 * hidden + ACT_RM[activation] * hidden


def real_mlp_macs(window: int, hidden: int, layers: int) -> int:
    return 2 * window * hidden + (hidden * hidden if layers == 2 else 0) + 2 * hidden


def macs(spec: dict) -> tuple[int, int]:
    """(as trained, after merging) real multiplies per symbol."""
    from .model_zoo import complex_one_hidden_macs, macs_phys

    kind, window = spec["model"], int(spec["window"])
    if kind == "act_mlp":
        m = act_mlp_macs(window, int(spec["hidden"]), spec["activation"])
        merged = 4 * window if spec["activation"] == "identity" else m     # a linear network folds to one FIR
        return m, merged
    if kind == "real_mlp":
        m = real_mlp_macs(window, int(spec["hidden"]), int(spec["layers"]))
        return m, m
    if kind == "triplet_net":
        from .triplets_ext import triplet_net_macs

        m = triplet_net_macs(window, int(spec["M"]), int(spec["fir_len"]), int(spec.get("mlp_hidden", 0)))
        return m, m
    if kind in ("dbp_shared", "dbp_mlp", "dbp_noise"):
        total = macs_phys(int(spec["n_alpha"]), int(spec["fir_len"]), bool(spec["sym"])) * int(spec["n_steps"])
        if kind == "dbp_mlp":
            total += complex_one_hidden_macs(window, int(spec["mlp_hidden"]))
        return total, total
    raise ValueError(kind)
