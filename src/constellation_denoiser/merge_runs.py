"""Merge ``<exp>/results/runs/*.json`` into one CSV (one row per run).

    PYTHONPATH=src python -m constellation_denoiser.merge_runs --exp E010 --out lr_sweep.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

SCALAR_SKIP = {"spec", "phases", "lr_config", "steps_to_target"}


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((results_dir / "runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        row = {k: v for k, v in record.items() if k not in SCALAR_SKIP and not isinstance(v, (dict, list))}
        for key, value in (record.get("lr_config") or {}).items():
            row[f"lr_{key}"] = value
        for key, value in (record.get("spec") or {}).items():
            row.setdefault(f"spec_{key}", value)
        for key, value in (record.get("steps_to_target") or {}).items():
            row[key] = value
        row["curve_csv"] = str(Path("curves") / f"{record['run_id']}.csv")
        rows.append(row)
    return pd.DataFrame(rows)


def results_dir_for(exp: str, smoke: bool) -> Path:
    from .runner import results_root

    return results_root(exp, smoke)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--out", required=True, help="file name inside the results folder")
    args = parser.parse_args()
    smoke = os.environ.get("SMOKE", "0") == "1"
    results = results_dir_for(args.exp, smoke)
    frame = load_runs(results)
    out = results / args.out
    frame.to_csv(out, index=False)
    print(f"{len(frame)} прогонов -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
