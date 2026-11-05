"""E055 summary: small real tanh MLP, test BER reduction and absolute BER per (window, hidden), 55 % rule.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E055
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from .runner import exp_dir, results_root


def main() -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    res = results_root("E055", smoke)
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((res / "runs").glob("*.json"))]
    rows = [r for r in rows if r.get("stage") == "main"]
    if not rows:
        print("нет прогонов")
        return 0
    t = pd.DataFrame([{"variant": r["variant"], "seed": r["seed"], "window": r["spec"]["window"], "hidden": r["spec"]["hidden"],
                       "test": r["test_reduction_valbest_select"], "val": r.get("best_val_reduction"),
                       "raw": r.get("raw_ber"), "rm": r.get("macs_per_symbol"), "diverged": r.get("diverged")} for r in rows])
    t["params"] = t.hidden * (2 * t.window + 3) + 2
    t["test_ber"] = t.raw * (1 - t.test / 100)
    g = t.groupby(["window", "hidden"])
    s = g.agg(n=("seed", "size"), params=("params", "first"), rm=("rm", "first"), test_mean=("test", "mean"),
              test_std=("test", "std"), test_min=("test", "min"), test_max=("test", "max"), val_mean=("val", "mean"),
              ber_mean=("test_ber", "mean"), raw=("raw", "first"), diverged=("diverged", "sum")).reset_index()
    s["test_se"] = s.test_std / np.sqrt(s.n)
    s["reached_55"] = s.test_mean >= 55
    s["close_55"] = (~s.reached_55) & (55 - s.test_mean < 2 * s.test_std)
    s.to_csv(res / "summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
