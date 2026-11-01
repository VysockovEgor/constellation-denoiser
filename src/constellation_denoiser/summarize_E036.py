"""E036: where does the growth of the triplets stop? T3 + FIR7 at M = 41, 51 and pruning of the M = 51 pool.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E036 [fit|summarize|all]   (SMOKE=1: small M)

Same procedure as E034 (summarize_E034.fit). Writes results/triplet_ext_fits.csv, runs_table.csv,
triplet_ext_summary.csv, triplet_ceiling_verdicts.csv and figures/E036_triplets_ceiling.png.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

from .runner import atomic_write_json, exp_dir, results_root
from .summarize_E034 import fit, plot, summarize as summarize_e034


def verdicts(config: dict, out) -> None:
    s = pd.read_csv(out / "triplet_ext_summary.csv").set_index("variant")
    fir, base = config["fir_len"], config["prune_M"]
    top = max(config["M"] + [base])
    rows = []
    prev = exp_dir("E034") / "results" / "triplet_ext_summary.csv"
    if prev.is_file():
        e034 = pd.read_csv(prev).set_index("variant")
        key = f"T3FIR{fir}_M31_full"
        if key in e034.index and f"T3FIR{fir}_M{top}_full" in s.index:
            gain = float(s.loc[f"T3FIR{fir}_M{top}_full", "test_mean"] - e034.loc[key, "test_mean"])
            rows.append({"hypothesis": f"рост с M продолжается (M={top} − M=31 ≥ 1 п.п.)", "value": gain,
                         "verdict": "подтверждено" if gain >= 1 else "не доказано", "detail": "E034 M=31 против E036"})
    k = config["rule_k"]
    full, pruned = f"T3FIR{fir}_M{base}_full", f"T3FIR{fir}_M{base}_K{k}"
    if full in s.index and pruned in s.index:
        share = float(s.loc[pruned, "test_mean"] / s.loc[full, "test_mean"])
        rows.append({"hypothesis": f"K={k} членов сохраняют ≥ 95 % снижения полной модели M={base}", "value": 100 * share,
                     "verdict": "подтверждено" if share >= 0.95 else "не доказано", "detail": ""})
    v = pd.DataFrame(rows, columns=["hypothesis", "value", "verdict", "detail"])
    v["rule_status"] = config["rule_draft"]["status"]
    v.to_csv(out / "triplet_ceiling_verdicts.csv", index=False)


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    directory = exp_dir("E036")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    out = results_root("E036", smoke)
    figures = out / "figures" if smoke else directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    action = argv[0] if argv else "all"
    if action in ("fit", "all"):
        fit(config, smoke, out)
    if action in ("summarize", "all"):
        summarize_e034(config, out, figures)          # runs_table, summary, (E034-specific verdict file, unused)
        (out / "triplet_ext_verdicts.csv").unlink(missing_ok=True)
        verdicts(config, out)
        summary = pd.read_csv(out / "triplet_ext_summary.csv")
        (figures / "E034_triplets_pruned_vs_rm.png").unlink(missing_ok=True)
        plot(summary, figures)
        (figures / "E034_triplets_pruned_vs_rm.png").rename(figures / "E036_triplets_ceiling.png")
        rt = pd.read_csv(out / "runs_table.csv")
        rt["experiment"] = "E036"
        rt.to_csv(out / "runs_table.csv", index=False)
    atomic_write_json(out / "status.json", {"done": action})
    print(f"E036: {out / 'triplet_ext_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
