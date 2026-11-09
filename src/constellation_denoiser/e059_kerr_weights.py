"""E059: trained Kerr weights alpha of every step, read from the saved best-validation weights (no training).

    PYTHONPATH=src python -m constellation_denoiser.e059_kerr_weights      (SMOKE=1: smoke results)

Reads results/dbpx_runs/*.pt, writes results/kerr_weights.csv: one row per (Ns, R, seed, step, j), j = -R//2..R//2,
alpha_j of the full symmetric window (stored half mirrored as in dbpx.DBPX.forward), plus the sum over j.
"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd
import torch

from .runner import results_root

EXP = "E059"
NAME = re.compile(r"dbpx_Ns(\d+)_L(\d+)_R(\d+)_seed(\d+)$")


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    res = results_root(EXP, smoke)
    rows = []
    for pt in sorted((res / "dbpx_runs").glob("*.pt")):
        m = NAME.match(pt.stem)
        if not m:
            continue
        ns, L, R, seed = map(int, m.groups())
        half = torch.load(pt, map_location="cpu")["alpha"].double().numpy()   # (ns, R//2 + 1), index 0 = centre
        r = R // 2
        for s in range(ns):
            full = list(half[s, 1:][::-1]) + list(half[s])
            total = sum(full)
            for i, a in enumerate(full):
                rows.append(dict(ns=ns, L=L, R=R, seed=seed, step=s + 1, j=i - r, alpha=a, alpha_sum=total))
    if not rows:
        print("no .pt files in", res / "dbpx_runs")
        return 1
    out = res / "kerr_weights.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
