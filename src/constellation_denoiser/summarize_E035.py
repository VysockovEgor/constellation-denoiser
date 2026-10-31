"""E035 summary: triplets trained by gradient (from scratch / from the closed form) and triplets + Kerr MLP.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E035   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/triplet_net_summary.csv, results/triplet_net_verdicts.csv and
figures/E035_triplet_net.png.
"""

from __future__ import annotations

import pandas as pd

from .runner import exp_dir
from .summary_e024 import aggregate, bar_compare, better, setup, verdict, write_verdicts


def closed_form_test(M: int, fir: int) -> float | None:
    for exp, path, arm_col, name in (("E031", "results/triplet_summary.csv", "arm", f"T3+FIR{fir}"),
                                     ("E034", "results/triplet_ext_summary.csv", "variant", f"T3FIR{fir}_M{M}_full")):
        f = exp_dir(exp) / path
        if f.is_file():
            s = pd.read_csv(f)
            hit = s[(s[arm_col] == name) & (s.M == M)] if exp == "E031" else s[s[arm_col] == name]
            if len(hit):
                return float(hit.test_mean.iloc[0])
    return None


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E035")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage.isin(["s1", "s2"])]
    table = aggregate(runs, ["e035_config", "lr_id", "stage", "budget", "spec_M", "spec_mlp_hidden"])
    table.to_csv(results / "triplet_net_summary.csv", index=False)
    s2 = table[table.stage == "s2"].set_index("e035_config")
    rows = []
    for name, item in config["configs"].items():
        if name not in s2.index or not item.get("mlp_hidden"):
            continue
        base = f"tg_M{item['M']}_warm"
        if base in s2.index:
            diff, ok = better(s2.loc[name], s2.loc[base], 1.0)
            rows.append({"hypothesis": f"{name}: сеть на остаток помогает (≥ 1 п.п. к {base}, интервалы не пересекаются)",
                         "value": diff, "verdict": verdict(ok), "detail": f"RM {int(s2.loc[name].rm)} против {int(s2.loc[base].rm)}"})
    for M in sorted({c["M"] for c in config["configs"].values()}):
        ref = closed_form_test(M, config["fir_len"])
        for name in (f"tg_M{M}_warm", f"tg_M{M}_scratch"):
            if name in s2.index and ref is not None:
                diff = float(s2.loc[name].test_mean - ref)
                rule = abs(diff) < 1.0 if name.endswith("warm") else diff >= -1.0
                rows.append({"hypothesis": f"{name}: градиент не хуже закрытой формы ({'|Δ| < 1' if name.endswith('warm') else 'Δ ≥ −1'} п.п.)",
                             "value": diff, "verdict": verdict(rule), "detail": f"закрытая форма M={M}: {ref:.2f} %"})
    write_verdicts(rows, config, results / "triplet_net_verdicts.csv")
    if len(s2):
        frame = s2.reset_index().assign(group="S2, 200k")
        bar_compare(frame, "e035_config", ["S2, 200k"], figures / "E035_triplet_net.png",
                    "Triplets by gradient and triplets + Kerr MLP (5 seeds, 200k; lr chosen in S1 by validation)",
                    "test BER reduction, % (min-max)")
    print(f"E035: {results / 'triplet_net_summary.csv'}; вердикты: {results / 'triplet_net_verdicts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
