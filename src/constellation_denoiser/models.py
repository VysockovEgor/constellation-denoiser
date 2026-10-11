"""Small neural equalizers that share a sliding-window I/Q input/output."""

import torch
from torch import nn


class WindowMLP(nn.Module):
    def __init__(self, window: int, hidden: int = 64, activation: str = "relu"):
        super().__init__()
        activation_layer = {"relu": nn.ReLU, "tanh": nn.Tanh}.get(activation.lower())
        if activation_layer is None:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * 2, hidden),
            activation_layer(),
            nn.Linear(hidden, hidden),
            activation_layer(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, window: int, channels: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, 2))
        self.center = window // 2

    def forward(self, x):
        return self.head(self.features(x.transpose(1, 2))[:, :, self.center])


class LSTMEqualizer(nn.Module):
    def __init__(self, window: int, hidden: int = 32):
        super().__init__()
        self.sequence = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.center = window // 2

    def forward(self, x):
        states, _ = self.sequence(x)
        return self.head(states[:, self.center, :])


class TransformerEqualizer(nn.Module):
    def __init__(self, window: int, width: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(2, width)
        self.position = nn.Parameter(torch.zeros(1, window, width))
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        self.center = window // 2

    def forward(self, x):
        z = self.input_projection(x) + self.position
        return self.head(self.encoder(z)[:, self.center, :])


def build_model(name: str, window: int, hidden: int = 64) -> nn.Module:
    constructors = {
        "MLP": lambda w: WindowMLP(w, hidden=hidden, activation="relu"),
        "MLP-Tanh": lambda w: WindowMLP(w, hidden=hidden, activation="tanh"),
        "1D-CNN": CNN1D,
        "LSTM": LSTMEqualizer,
        "Transformer": TransformerEqualizer,
    }
    try:
        return constructors[name](window)
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}. Choose from {tuple(constructors)}") from exc
