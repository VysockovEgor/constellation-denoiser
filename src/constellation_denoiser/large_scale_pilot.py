"""Compare paper-like real and complex MLPs on deduplicated PR files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .core import RAW_DATA_DIR, load_channel_data
from .experiment import PROJECT_ROOT, ber_from_iq, evaluate, make_windows, train_one
from .paper_complex_mlp import predict_complex, train as train_complex


def get_unique_files(data_dir: Path, audit_csv: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv)
    required = {"relative_path", "sha256", "pr_id", "error"}
    if not required.issubset(audit.columns):
        raise ValueError(f"Audit file needs columns {sorted(required)}; rerun audit_data first")
    # Use the newly provided top-level archive as canonical; the original PR1–10
    # under data/raw are byte-for-byte copies of the archive's PR1–10.
    root_files = audit[
        audit.relative_path.astype(str).map(lambda p: "/" not in p and "\\" not in p)
        & audit.sha256.notna()
        & (audit.error.fillna("") == "")
    ].copy()
    root_files = root_files.drop_duplicates("sha256", keep="first")
    root_files["path"] = root_files.relative_path.map(lambda p: data_dir / p)
    root_files = root_files[root_files.path.map(Path.is_file)]
    return root_files.sort_values("pr_id").reset_index(drop=True)


def collect(paths: list[Path], window: int):
    xs, ys, rx_centers, tx_centers, names = [], [], [], [], []
    edge = window // 2
    for path in paths:
        tx, rx = load_channel_data(path)
        x, y = make_windows(rx, tx, window)
        xs.append(x)
        ys.append(y)
        rx_centers.append(rx[edge:len(rx) - edge])
        tx_centers.append(tx[edge:len(tx) - edge])
        names.extend([path.stem] * len(y))
    return (
        np.concatenate(xs), np.concatenate(ys), np.concatenate(rx_centers),
        np.concatenate(tx_centers), np.asarray(names), [p.stem for p in paths],
    )


def summarize_per_file(model, data, kind: str):
    x, _, rx_center, tx_center, names, files = data
    if kind == "real":
        from .experiment import predict
        pred = predict(model, x, torch.device("cpu"))
    else:
        pred = predict_complex(model, x)
    raw_iq = np.column_stack((rx_center.real, rx_center.imag))
    rows = []
    for name in files:
        mask = names == name
        rows.append({
            "file": name,
            "raw_ber": ber_from_iq(raw_iq[mask], tx_center[mask]),
            "model_ber": ber_from_iq(pred[mask], tx_center[mask]),
            "symbols": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--audit-csv", type=Path, default=PROJECT_ROOT / "reports/results/data_audit_all.csv")
    parser.add_argument("--sample-files", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--split-seed", type=int, default=20261003)
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--complex-hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    audit_path = args.audit_csv.expanduser().resolve()
    candidates = get_unique_files(data_dir, audit_path)
    if len(candidates) < 30:
        raise ValueError(f"Only {len(candidates)} unique root-level files found")
    sample_n = min(args.sample_files, len(candidates))
    rng = np.random.default_rng(args.split_seed)
    sampled = candidates.iloc[rng.permutation(len(candidates))[:sample_n]].reset_index(drop=True)
    perm = rng.permutation(sample_n)
    n_train = int(sample_n * 0.70)
    n_val = int(sample_n * 0.15)
    train_rows = sampled.iloc[perm[:n_train]]
    val_rows = sampled.iloc[perm[n_train:n_train + n_val]]
    test_rows = sampled.iloc[perm[n_train + n_val:]]
    file_splits = {
        "train": [Path(p) for p in train_rows.path],
        "validation": [Path(p) for p in val_rows.path],
        "test": [Path(p) for p in test_rows.path],
    }
    print(f"Unique files available={len(candidates)}; sampled={sample_n}; split sizes={len(file_splits['train'])}/{len(file_splits['validation'])}/{len(file_splits['test'])}", flush=True)
    print("Split by SHA-256 content, not filename; no exact duplicate can cross split boundaries.", flush=True)

    print("Loading 31-symbol windows...", flush=True)
    train_data = collect(file_splits["train"], args.window)
    val_data = collect(file_splits["validation"], args.window)
    test_data = collect(file_splits["test"], args.window)
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    base = SimpleNamespace(
        window=args.window, hidden=args.hidden, learning_rate=args.learning_rate,
        weight_decay=0.0, batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, optimizer="adam", initializer="xavier_normal",
    )

    output_dir = PROJECT_ROOT / "reports" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_file_rows, run_rows = [], []
    for kind in ("real", "complex"):
        for seed in args.seeds:
            print(f"Training {kind} MLP seed={seed} ...", flush=True)
            if kind == "real":
                result = train_one("MLP-Tanh", seed, train_data, val_data, None, base, device)
                model = result["model"]
                best_epoch, epochs_run = result["best_epoch"], result["epochs_run"]
                val_ber, val_mse, seconds = result["val"]["ber"], result["val"]["mse"], result["seconds"]
                torch.save({
                    "model": "paper-real-tanh-MLP", "seed": seed, "window": args.window,
                    "hidden_per_layer": args.hidden, "initializer": "Xavier normal", "optimizer": "Adam",
                    "state_dict": model.state_dict(),
                }, PROJECT_ROOT / "models" / f"large_real_tanh_seed_{seed}.pt")
            else:
                model, best_epoch, epochs_run, val_ber, val_mse, seconds = train_complex(
                    seed, train_data, val_data, window=args.window,
                    hidden=args.complex_hidden, epochs=args.epochs,
                    patience=args.patience, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, threads=args.threads,
                )
                torch.save({
                    "model": "paper-complex-MLP", "seed": seed, "window": args.window,
                    "hidden_per_layer": args.complex_hidden, "activation": "exp(i*c*|z|^2)*z",
                    "optimizer": "Adam", "state_dict": model.state_dict(),
                }, PROJECT_ROOT / "models" / f"large_complex_seed_{seed}.pt")
            per_file = summarize_per_file(model, test_data, kind)
            per_file["model"] = kind
            per_file["seed"] = seed
            per_file_rows.extend(per_file.to_dict("records"))
            mean_raw = float(per_file.raw_ber.mean())
            mean_model = float(per_file.model_ber.mean())
            run_rows.append({
                "model": kind, "seed": seed, "window": args.window,
                "hidden_per_layer": args.hidden if kind == "real" else args.complex_hidden,
                "best_epoch": best_epoch, "epochs_run": epochs_run,
                "validation_ber": val_ber, "validation_mse": val_mse,
                "test_raw_ber_mean_over_files": mean_raw,
                "test_model_ber_mean_over_files": mean_model,
                "improvement_pct_vs_raw": 100 * (mean_raw - mean_model) / mean_raw,
                "train_seconds": seconds,
            })
            print(f"  best epoch={best_epoch}/{epochs_run}; val BER={val_ber:.6f}; mean test BER={mean_model:.6f}; raw={mean_raw:.6f}; reduction={100*(mean_raw-mean_model)/mean_raw:.2f}%; time={seconds:.1f}s", flush=True)

    pd.DataFrame(run_rows).to_csv(output_dir / "large_scale_pilot_summary.csv", index=False)
    pd.DataFrame(per_file_rows).to_csv(output_dir / "large_scale_pilot_per_file.csv", index=False)
    config = {
        "dataset_root": str(data_dir), "audit_csv": str(audit_path),
        "available_unique_files": len(candidates), "sample_files": sample_n,
        "split_seed": args.split_seed, "train_files": [p.name for p in file_splits["train"]],
        "validation_files": [p.name for p in file_splits["validation"]],
        "test_files": [p.name for p in file_splits["test"]],
        "seeds": args.seeds, "window": args.window,
        "real_model": {"hidden": args.hidden, "activation": "tanh", "initializer": "Xavier normal", "optimizer": "Adam"},
        "complex_model": {"hidden": args.complex_hidden, "activation": "exp(i*c*|z|^2)*z", "initializer": "complex Glorot normal", "optimizer": "Adam"},
        "epochs": args.epochs, "patience": args.patience,
    }
    (output_dir / "large_scale_pilot_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(pd.DataFrame(run_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote pilot summary and per-file results under {output_dir}")


if __name__ == "__main__":
    main()
