"""Pilot implementation of the complex MLP proposed in Bogdanov & Sidelnikov."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .experiment import ber_from_iq, evaluate_raw, load_split, predict, set_seed
from .paths import E002_CHECKPOINTS, E002_RESULTS


class ComplexDense(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Complex Glorot normal: each Cartesian component has this std.
        std = (1.0 / (in_features + out_features)) ** 0.5
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features).normal_(0, std))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features).normal_(0, std))
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.complex(self.weight_real, self.weight_imag)
        b = torch.complex(self.bias_real, self.bias_imag)
        return x @ w.transpose(0, 1) + b


class NonlinearPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(1j * self.c * z.abs().square()) * z


class PaperComplexMLP(nn.Module):
    def __init__(self, window: int = 31, hidden: int = 32):
        super().__init__()
        self.fc1 = ComplexDense(window, hidden)
        self.act1 = NonlinearPhase()
        self.fc2 = ComplexDense(hidden, hidden)
        self.act2 = NonlinearPhase()
        self.fc3 = ComplexDense(hidden, 1)
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], self.window)
        z = self.act1(self.fc1(z))
        z = self.act2(self.fc2(z))
        return self.fc3(z).squeeze(-1)


def train(seed: int, train_data, val_data, *, window: int, hidden: int,
          epochs: int, patience: int, batch_size: int, learning_rate: float,
          threads: int):
    set_seed(seed)
    torch.set_num_threads(threads)
    model = PaperComplexMLP(window, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_train, y_train = train_data[:2]
    x_val, y_val = val_data[:2]
    y_train_complex = torch.complex(torch.from_numpy(y_train[:, 0]), torch.from_numpy(y_train[:, 1]))
    y_val_complex = torch.complex(torch.from_numpy(y_val[:, 0]), torch.from_numpy(y_val[:, 1]))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), y_train_complex), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    best_ber, best_mse, best_monitor, best_state = float("inf"), float("inf"), float("inf"), None
    bad_epochs, best_epoch = 0, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = (pred - yb).abs().square().mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_pred = model(torch.from_numpy(x_val))
            pred_iq = torch.stack((val_pred.real, val_pred.imag), dim=1).numpy()
            val_mse = float((val_pred - y_val_complex).abs().square().mean().item())
        val_ber = ber_from_iq(pred_iq, val_data[3])
        if (val_ber, val_mse) < (best_ber, best_mse):
            best_ber, best_mse = val_ber, val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if val_mse < best_monitor - 1e-12:
            best_monitor, bad_epochs = val_mse, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, epoch, best_ber, best_mse, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--window", type=int, default=31)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    train_data, val_data, test_data = load_split(a.window)
    print(f"Complex MLP: context={a.window}, {a.hidden} complex neurons/layer, 2 hidden layers, learned phase activation; Adam")
    print(f"Files: train={train_data[5]}, validation={val_data[5]}, test={test_data[5]}")
    print(f"Raw BER: validation={evaluate_raw(val_data):.6f}; test={evaluate_raw(test_data):.6f}")
    rows, models = [], []
    for seed in a.seeds:
        print(f"Training seed={seed} ...", flush=True)
        model, best_epoch, epochs_run, val_ber, val_mse, seconds = train(
            seed, train_data, val_data, window=a.window, hidden=a.hidden,
            epochs=a.epochs, patience=a.patience, batch_size=a.batch_size,
            learning_rate=a.learning_rate, threads=a.threads,
        )
        test_pred = predict_complex(model, test_data[0])
        test_ber = ber_from_iq(test_pred, test_data[3])
        rows.append({
            "seed": seed, "window": a.window, "complex_hidden_per_layer": a.hidden,
            "parameters": sum(x.numel() for x in model.parameters()),
            "activation": "exp(i*c*|z|^2)*z", "initializer": "complex Glorot normal",
            "optimizer": "Adam", "best_epoch": best_epoch, "epochs_run": epochs_run,
            "validation_ber": val_ber, "validation_mse": val_mse,
            "test_ber": test_ber, "test_raw_ber": evaluate_raw(test_data),
            "test_improvement_pct_vs_raw": 100 * (evaluate_raw(test_data) - test_ber) / evaluate_raw(test_data),
            "test_ber_PR_9": ber_from_iq(test_pred[test_data[4] == "Symbols_1m_1ch_PR_9"], test_data[3][test_data[4] == "Symbols_1m_1ch_PR_9"]),
            "test_ber_PR_10": ber_from_iq(test_pred[test_data[4] == "Symbols_1m_1ch_PR_10"], test_data[3][test_data[4] == "Symbols_1m_1ch_PR_10"]),
            "train_seconds": seconds,
        })
        models.append((model, val_ber, val_mse, seed))
        print(f"  best epoch={best_epoch}/{epochs_run}; val BER={val_ber:.6f}; test BER={test_ber:.6f}; time={seconds:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    out = E002_RESULTS
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "paper_complex_mlp.csv", index=False)
    selected, val_ber, val_mse, seed = min(models, key=lambda x: (x[1], x[2]))
    torch.save({
        "model": "paper-complex-MLP", "window": a.window, "hidden_per_layer": a.hidden,
        "activation": "exp(i*c*|z|^2)*z", "optimizer": "Adam", "seed": seed,
        "state_dict": selected.state_dict(), "validation_ber": val_ber,
        "train_files": train_data[5], "validation_files": val_data[5], "test_files": test_data[5],
    }, E002_CHECKPOINTS / "paper_complex_mlp.pt")
    print("\nResults by seed:")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nMean test BER={frame.test_ber.mean():.6f}; mean improvement vs raw={frame.test_improvement_pct_vs_raw.mean():.2f}%; selected seed={seed} by validation BER.")


def predict_complex(model, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            p = model(torch.from_numpy(x[start:start + batch_size]))
            out.append(torch.stack((p.real, p.imag), dim=1).numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    main()
