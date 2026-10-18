"""Check that every CSV used by E010-E014 is present and matches data/manifest SHA-256.

    PYTHONPATH=src python scripts/check_data.py            # check presence and size
    PYTHONPATH=src python scripts/check_data.py --sha256   # also verify SHA-256 (slower)
    PYTHONPATH=src python scripts/check_data.py --write-list experiments/E010_E014_required_files.txt

The split is experiments/E004_training_file_curve/config.json: 210 train-pool
files (N=30 uses the first 30), 45 validation and 45 test files, all in
data/raw/archive/.
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from constellation_denoiser.window_data import split_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--write-list", type=Path)
    args = parser.parse_args()
    split = split_files()
    names = list(dict.fromkeys(split["train_pool"] + split["validation"] + split["test"]))
    audit = pd.read_csv(ROOT / "data" / "manifest" / "data_audit_all.csv")
    audit = audit[~audit.relative_path.str.contains("/")].drop_duplicates("filename").set_index("filename")
    if args.write_list:
        lines = [f"data/raw/archive/{name}" for name in names]
        args.write_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
        size = sum(int(audit.loc[name, "bytes"]) for name in names)
        print(f"{len(lines)} файлов, {size / 1e6:.0f} МБ -> {args.write_list}")
        return 0
    missing, bad = [], []
    for name in names:
        path = ROOT / "data" / "raw" / "archive" / name
        if not path.is_file():
            missing.append(name)
            continue
        if path.stat().st_size != int(audit.loc[name, "bytes"]):
            bad.append(name)
        elif args.sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != audit.loc[name, "sha256"]:
            bad.append(name)
    total = sum(int(audit.loc[name, "bytes"]) for name in names)
    print(f"Нужно файлов: {len(names)} (train-пул 210, validation 45, test 45), ~{total / 1e6:.0f} МБ.")
    print(f"Отсутствуют: {len(missing)}; не совпадают размер{'/SHA-256' if args.sha256 else ''}: {len(bad)}.")
    for name in (missing + bad)[:10]:
        print(f"  {name}")
    if missing or bad:
        print("Скопируйте недостающее по списку experiments/E010_E014_required_files.txt (см. RUNBOOK, раздел B).")
        return 1
    print("Данные на месте.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
