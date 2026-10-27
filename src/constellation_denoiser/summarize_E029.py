"""E029 summary: hybrid PhysKerr + one-hidden Kerr MLP against the pure MLP at each width.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E029   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/width_summary.csv, results/width_verdicts.csv, figures/E029_hybrid_vs_width.png.
"""

from __future__ import annotations

from .summary_e024 import aggregate, better, setup, verdict, write_verdicts


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E029")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage == "main"]
    table = aggregate(runs, ["variant", "arm", "width", "spec_n_alpha", "spec_fir_len"])
    table.to_csv(results / "width_summary.csv", index=False)
    fe = config["front_end"]
    main_hyb = table[(table.arm == "hybrid") & (table.spec_n_alpha == fe["n_alpha"]) & (table.spec_fir_len == fe["fir_len"])]
    pure = table[table.arm == "pure"]
    rows, diffs = [], {}
    for _, p in pure.iterrows():
        h = main_hyb[main_hyb.width == p.width]
        if h.empty:
            continue
        diff, ok = better(h.iloc[0], p, 1.0)
        diffs[int(p.width)] = diff
        rows.append({"hypothesis": f"h={int(p.width)}: физический front-end окупается (гибрид − чистая ≥ 1 п.п.)",
                     "value": diff, "verdict": verdict(ok), "detail": f"RM {int(h.iloc[0].rm)} против {int(p.rm)}"})
    if 16 in diffs and 119 in diffs:
        gap = diffs[16] - diffs[119]
        rows.append({"hypothesis": "выигрыш убывает с h (разность при h=16 больше, чем при h=119, на ≥ 1 п.п.)",
                     "value": gap, "verdict": verdict(gap >= 1.0), "detail": ""})
    write_verdicts(rows, config, results / "width_verdicts.csv")
    plot(pure, main_hyb, table, figures)
    print(f"E029: {results / 'width_summary.csv'}; вердикты: {results / 'width_verdicts.csv'}")
    return 0


def plot(pure, hyb, table, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for index, (frame, label, marker) in enumerate(((pure, "pure one-hidden Kerr MLP", "o"),
                                                   (hyb, "PhysKerr(R31, L7) + MLP", "s"))):
        frame = frame.sort_values("width")
        ax.errorbar(frame.width, frame.test_mean, yerr=[frame.test_mean - frame.test_min, frame.test_max - frame.test_mean],
                    marker=marker, color=plot_style.color(index), capsize=2, label=label)
    variants = table[(table.arm == "hybrid") & (table.width == 48)]
    for index, (_, r) in enumerate(variants.iterrows()):
        ax.plot([48], [r.test_mean], "^v<>"[index % 4], color=plot_style.color(2 + index), linestyle="",
                label=f"front-end R{int(r.spec_n_alpha)} L{int(r.spec_fir_len)} (h=48)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden width h")
    ax.set_ylabel("test BER reduction, % (5 seeds, min-max)")
    ax.set_title("Physics front-end vs plain MLP across widths (200k)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "E029_hybrid_vs_width.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
