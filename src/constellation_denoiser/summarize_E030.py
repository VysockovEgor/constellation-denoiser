"""E030 summary: losses on the same output against MSE.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E030   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/loss_summary.csv, results/loss_verdicts.csv, figures/E030_losses.png.
"""

from __future__ import annotations

from .summary_e024 import aggregate, bar_compare, better, setup, verdict, write_verdicts


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E030")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage.isin(["s1", "s2"])]
    table = aggregate(runs, ["model_key", "loss_name", "lr_scale", "stage", "budget"])
    table.to_csv(results / "loss_summary.csv", index=False)
    s2 = table[table.stage == "s2"]
    rows = []
    for loss in config["losses"]:
        if loss == "mse":
            continue
        wins, details = 0, []
        for model in config["models"]:
            a = s2[(s2.model_key == model) & (s2.loss_name == loss)]
            b = s2[(s2.model_key == model) & (s2.loss_name == "mse")]
            if a.empty or b.empty:
                continue
            diff, ok = better(a.iloc[0], b.iloc[0], 0.7)
            wins += ok
            details.append(f"{model}: {diff:+.2f}")
        rows.append({"hypothesis": f"потеря {loss} лучше MSE (≥ 0.7 п.п., интервалы не пересекаются, ≥ 2 из 4 моделей)",
                     "value": wins, "verdict": verdict(wins >= 2), "detail": "; ".join(details)})
    write_verdicts(rows, config, results / "loss_verdicts.csv")
    if len(s2):
        bar_compare(s2.rename(columns={"loss_name": "group"}), "model_key", list(config["losses"]),
                    figures / "E030_losses.png", "Losses on the same output (S2, 200k, 5 seeds; lr scale chosen in S1 by validation)",
                    "test BER reduction, % (min-max)")
    print(f"E030: {results / 'loss_summary.csv'}; вердикты: {results / 'loss_verdicts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
