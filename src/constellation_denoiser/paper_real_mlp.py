"""Paper-like reproduction of the real-valued tanh MLP baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .experiment import ber_from_iq, load_split, predict, set_seed, evaluate_raw
from .paths import E002_CHECKPOINTS, E002_RESULTS
from .models import WindowMLP


def train(seed: int, train_data, validation_data, *, window: int, hidden: int,
          epochs: int, patience: int, batch_size: int, learning_rate: float,
          threads: int):
    set_seed(seed)
    torch.set_num_threads(threads)
    device = torch.device("cpu")
    model = WindowMLP(window, hidden=hidden, activation="tanh")
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    x_train, y_train = train_data[:2]
    x_val, y_val = validation_data[:2]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_mse, best_ber, best_monitor_mse, best_state = float("inf"), float("inf"), float("inf"), None
    bad_epochs, best_epoch = 0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_pred = predict(model, x_val, device)
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        val_ber = ber_from_iq(val_pred, validation_data[3])
        if (val_ber, val_mse) < (best_ber, best_mse):
            best_ber, best_mse = val_ber, val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if val_mse < best_monitor_mse - 1e-12:
            best_monitor_mse = val_mse
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, epoch, best_ber, best_mse, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    train_data, val_data, test_data = load_split(args.window)
    results, models = [], []
    print(f"Paper-like real MLP: window={args.window}, hidden={args.hidden}x2, tanh, Xavier normal, Adam")
    print(f"Files: train={train_data[5]}, validation={val_data[5]}, test={test_data[5]}")
    print(f"Raw BER: validation={evaluate_raw(val_data):.6f}; test={evaluate_raw(test_data):.6f}")
    for seed in args.seeds:
        print(f"Training seed={seed} ...", flush=True)
        model, best_epoch, epochs_run, val_ber, val_mse, seconds = train(
            seed, train_data, val_data, window=args.window, hidden=args.hidden,
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            learning_rate=args.learning_rate, threads=args.threads,
        )
        test_pred = predict(model, test_data[0], torch.device("cpu"))
        test_ber = ber_from_iq(test_pred, test_data[3])
        test_ber_pr9 = ber_from_iq(test_pred[test_data[4] == "Symbols_1m_1ch_PR_9"], test_data[3][test_data[4] == "Symbols_1m_1ch_PR_9"])
        test_ber_pr10 = ber_from_iq(test_pred[test_data[4] == "Symbols_1m_1ch_PR_10"], test_data[3][test_data[4] == "Symbols_1m_1ch_PR_10"])
        results.append({
            "seed": seed, "window": args.window, "hidden_per_layer": args.hidden,
            "activation": "tanh", "initializer": "Xavier normal", "optimizer": "Adam",
            "best_epoch": best_epoch, "epochs_run": epochs_run,
            "validation_ber": val_ber, "validation_mse": val_mse,
            "test_ber": test_ber, "test_raw_ber": evaluate_raw(test_data),
            "test_improvement_pct_vs_raw": 100 * (evaluate_raw(test_data) - test_ber) / evaluate_raw(test_data),
            "test_ber_PR_9": test_ber_pr9, "test_ber_PR_10": test_ber_pr10,
            "train_seconds": seconds,
        })
        models.append((model, val_ber, val_mse, seed))
        print(f"  best epoch={best_epoch}/{epochs_run}; val BER={val_ber:.6f}; test BER={test_ber:.6f}; time={seconds:.1f}s", flush=True)

    output = E002_RESULTS
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "paper_real_mlp.csv", index=False)
    selected, selected_ber, selected_mse, selected_seed = min(models, key=lambda x: (x[1], x[2]))
    torch.save({
        "model": "paper-like-real-MLP", "window": args.window,
        "hidden_per_layer": args.hidden, "activation": "tanh",
        "initializer": "Xavier normal", "optimizer": "Adam", "seed": selected_seed,
        "state_dict": selected.state_dict(), "validation_ber": selected_ber,
        "train_files": train_data[5], "validation_files": val_data[5], "test_files": test_data[5],
    }, E002_CHECKPOINTS / "paper_real_mlp.pt")
    print("\nResults by seed:")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nMean test BER={frame.test_ber.mean():.6f}; selected seed={selected_seed} by validation BER.")


if __name__ == "__main__":
    main()
