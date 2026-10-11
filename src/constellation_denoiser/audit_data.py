"""Audit any folder of PR CSVs before model training."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .core import RAW_DATA_DIR, load_channel_data, symbols_to_bits
from .paths import MANIFEST_DIR


PR_PATTERN = re.compile(r"PR[ _-]*(\d+)", re.IGNORECASE)


def audit_file(path: Path, root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    file_hash = digest.hexdigest()

    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 4:
        raise ValueError(f"expected 4 columns, found {frame.shape[1]}")
    values = frame.to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    result = {
        "filename": path.name,
        "relative_path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": frame.shape[1],
        "all_finite": finite,
        "tx_levels_I": None,
        "tx_levels_Q": None,
        "tx_rms": None,
        "rx_rms": None,
        "raw_ber": None,
        "sha256": file_hash,
        "error": "",
    }
    match = PR_PATTERN.search(path.stem)
    result["pr_id"] = int(match.group(1)) if match else None
    if not finite:
        result["error"] = "non-finite numeric values"
        return result

    tx, rx = load_channel_data(path)
    result["tx_levels_I"] = len(np.unique(np.round(tx.real, 6)))
    result["tx_levels_Q"] = len(np.unique(np.round(tx.imag, 6)))
    result["tx_rms"] = float(np.sqrt(np.mean(np.abs(tx) ** 2)))
    result["rx_rms"] = float(np.sqrt(np.mean(np.abs(rx) ** 2)))
    result["raw_ber"] = float(np.mean(symbols_to_bits(tx) != symbols_to_bits(rx)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_dir.expanduser().resolve()
    files = sorted(p for p in root.rglob("*.csv") if PR_PATTERN.search(p.stem))
    if not files:
        raise SystemExit(f"No CSV files with a PR number found under {root}")

    rows = []
    for i, path in enumerate(files, start=1):
        try:
            rows.append(audit_file(path, root))
        except Exception as exc:
            rows.append({
                "filename": path.name, "relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "rows": None, "columns": None, "all_finite": False,
                "tx_levels_I": None, "tx_levels_Q": None, "tx_rms": None,
                "rx_rms": None, "raw_ber": None, "sha256": "",
                "pr_id": int(PR_PATTERN.search(path.stem).group(1)),
                "error": f"{type(exc).__name__}: {exc}",
            })
        if i % 100 == 0 or i == len(files):
            print(f"Audited {i}/{len(files)} CSV files", flush=True)

    frame = pd.DataFrame(rows).sort_values(["pr_id", "filename"], na_position="last")
    by_id = defaultdict(list)
    by_hash = defaultdict(list)
    for row in rows:
        if row.get("pr_id") is not None:
            by_id[row["pr_id"]].append(row["relative_path"])
        if row.get("sha256"):
            by_hash[row["sha256"]].append(row["relative_path"])
    duplicate_ids = {k: v for k, v in by_id.items() if len(v) > 1}
    duplicate_content = {k: v for k, v in by_hash.items() if len(v) > 1}

    output = args.output or (MANIFEST_DIR / "data_audit.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"\nFiles: {len(files)}; unique PR numbers: {len(by_id)}; unique contents: {len(by_hash)}; rows with errors: {(frame.error != '').sum()}")
    print(f"Duplicate PR-number groups: {len(duplicate_ids)}; duplicate content groups: {len(duplicate_content)}; redundant copies: {len(files) - len(by_hash)}")
    if duplicate_ids:
        print(f"Duplicate PR-number examples: {dict(list(duplicate_ids.items())[:5])}")
    if duplicate_content:
        print(f"Duplicate content examples: {list(duplicate_content.values())[:5]}")
    if frame.raw_ber.notna().any():
        print(f"Raw BER range: {frame.raw_ber.min():.6f}–{frame.raw_ber.max():.6f}; mean={frame.raw_ber.mean():.6f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
