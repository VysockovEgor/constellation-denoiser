"""E027 summary: Kerr activation against other complex activations at equal width.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E027   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/activation_summary.csv, results/activation_verdicts.csv and
figures E027_activations_by_width.png, E027_activations_200k.png.
"""

from __future__ import annotations

import pandas as pd

from .summary_e024 import aggregate, bar_compare, better, setup, verdict, write_verdicts

KERR = "kerr_c_train"


def best_by_val(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.groupby(["activation", "spec_hidden"]).val_mean.idxmax()]


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E027")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage != "control"]
    table = aggregate(runs, ["activation", "spec_hidden", "lr_id", "e027_stage", "budget"])
    table.to_csv(results / "activation_summary.csv", index=False)
    rule = config["rule_draft"]
    rows = []
    for stage_name, label in (("s2", "S2, 100k"), ("s3", "S3, 200k")):
        part = table[table.e027_stage == stage_name]
        if part.empty:
            continue
        best = best_by_val(part)
        for hidden, group in best.groupby("spec_hidden"):
            kerr = group[group.activation == KERR]
            if kerr.empty:
                continue
            kerr = kerr.iloc[0]
            for _, other in group[group.activation != KERR].iterrows():
                diff, ok = better(kerr, other, 1.0)
                rows.append({"hypothesis": f"{label}, h={int(hidden)}: Kerr лучше {other.activation}", "value": diff,
                             "verdict": verdict(ok), "detail": f"RM {int(kerr.rm_trained)} против {int(other.rm_trained)}"})
    s3 = best_by_val(table[table.e027_stage == "s3"]) if (table.e027_stage == "s3").any() else pd.DataFrame()
    if len(s3):
        get = {a: r for a, r in zip(s3.activation, s3.itertuples())}
        if KERR in get and "kerr_c_fixed" in get:
            diff = get[KERR].test_mean - get["kerr_c_fixed"].test_mean
            rows.append({"hypothesis": "обучаемость c нужна (kerr_c_train − kerr_c_fixed ≥ 0.5 п.п., S3)", "value": diff,
                         "verdict": verdict(diff >= 0.5), "detail": rule["trainable_c"]})
        if "identity" in get:
            nonlin = s3[s3.activation != "identity"]
            top = nonlin.loc[nonlin.val_mean.idxmax()]
            diff = top.test_mean - get["identity"].test_mean
            rows.append({"hypothesis": f"нелинейность нужна (лучшая по валидации {top.activation} − identity ≥ 20 п.п.)",
                         "value": diff, "verdict": verdict(diff >= 20), "detail": rule["nonlinearity"]})
    write_verdicts(rows, config, results / "activation_verdicts.csv")
    s2 = table[table.e027_stage == "s2"]
    if len(s2):
        best = best_by_val(s2).assign(group=lambda f: "h=" + f.spec_hidden.astype(int).astype(str))
        bar_compare(best, "activation", list(dict.fromkeys(best.group)), figures / "E027_activations_by_width.png",
                    "Activations at equal width (S2, 100k, 5 seeds; best of 2 recipes by validation)",
                    "test BER reduction, % (min-max)")
    if len(s3):
        bar_compare(s3.assign(group="h=48"), "activation", ["h=48"], figures / "E027_activations_200k.png",
                    "Activations, h = 48, 200k steps, 5 seeds (S3)", "test BER reduction, % (min-max)")
    print(f"активации: {results / 'activation_summary.csv'}; вердикты: {results / 'activation_verdicts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
