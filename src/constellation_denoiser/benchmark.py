"""Evaluate validation-selected checkpoints and measure CPU inference cost."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .experiment import MODEL_NAMES, ber_from_iq, evaluate, load_split
from .models import build_model
from .paths import E001_CHECKPOINTS, E001_RESULTS


def benchmark():
    window = 7
    torch.set_num_threads(1)
    device = torch.device("cpu")
    _, _, test_data = load_split(window)
    x_test = test_data[0]
    raw_per_run = {}
    for name in test_data[5]:
        mask = test_data[4] == name
        raw_iq = np.column_stack((test_data[2][mask].real, test_data[2][mask].imag))
        raw_per_run[name] = ber_from_iq(raw_iq, test_data[3][mask])
    run_rows = []
    latency_rows = []

    for name in MODEL_NAMES:
        checkpoint_path = E001_CHECKPOINTS / f"{name.lower().replace('-', '')}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(name, window).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        metrics = evaluate(model, test_data, device)
        for run, ber in metrics["per_run"].items():
            run_rows.append(
                {"model": name, "run": run, "raw_ber": raw_per_run[run], "test_ber": ber}
            )

        sample = torch.from_numpy(x_test[:1]).to(device)
        batch = torch.from_numpy(x_test[: min(1024, len(x_test))]).to(device)
        with torch.inference_mode():
            for _ in range(20):
                model(sample)
            samples = []
            for _ in range(200):
                started = time.perf_counter()
                model(sample)
                samples.append((time.perf_counter() - started) * 1000)
            for _ in range(3):
                model(batch)
            batch_times = []
            for _ in range(10):
                started = time.perf_counter()
                model(batch)
                batch_times.append(time.perf_counter() - started)

        throughput = len(batch) / float(np.median(batch_times))
        latency_rows.append(
            {
                "model": name,
                "checkpoint_seed": checkpoint["seed"],
                "validation_ber": checkpoint["validation_ber"],
                "test_ber_checkpoint": metrics["ber"],
                "test_raw_ber": metrics["raw_ber"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "single_window_latency_ms_median": float(np.median(samples)),
                "batch_size": len(batch),
                "batch_throughput_symbols_per_sec": throughput,
            }
        )

    results_dir = E001_RESULTS
    pd.DataFrame(run_rows).to_csv(results_dir / "selected_checkpoint_per_run.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(results_dir / "inference_benchmark.csv", index=False)
    print(pd.DataFrame(latency_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    benchmark()
