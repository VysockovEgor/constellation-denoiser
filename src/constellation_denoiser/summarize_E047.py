"""E047 summary: cascade training variants against the reference v1 (one factor at a time).

    PYTHONPATH=src python -m constellation_denoiser.summarize_E047   (SMOKE=1 reads results_smoke/)
"""

from __future__ import annotations

from .summary_e024 import aggregate, bar_compare, setup, verdict, write_verdicts


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E047")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage == "main"]
    table = aggregate(runs, ["variant"])
    std = runs.groupby("variant").test_reduction_valbest_select.std()
    table["test_std"] = table.variant.map(std)
    table.to_csv(results / "variants_summary.csv", index=False)
    base = table[table.variant == "v1_independent_joint"]
    rows = []
    if len(base):
        b = base.iloc[0]
        for _, r in table[table.variant != b.variant].iterrows():
            diff = float(r.test_mean - b.test_mean)
            sd = max(r.test_std, b.test_std)
            rows.append({"hypothesis": f"{r.variant} отличается от v1 (> 2 std)", "value": diff,
                         "verdict": verdict(abs(diff) > 2 * sd), "detail": f"2 std = {2 * sd:.2f}; {r.test_mean:.2f} против {b.test_mean:.2f}"})
    write_verdicts(rows, config, results / "variants_verdicts.csv")
    bar_compare(table.assign(group="200k, 5 seeds"), "variant", ["200k, 5 seeds"], figures / "E047_cascade_variants.png",
                "Cascade (2 stages, M = 4, K_th = 16, FIR7; 1 136 RM): training variants", "test BER reduction, % (min-max)")
    print(f"E047: {results / 'variants_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
