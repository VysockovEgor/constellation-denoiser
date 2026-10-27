"""E026 summary: window x width map of the one-hidden Kerr MLP, depth (both) and saturation.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E026   (SMOKE=1 reads results_smoke/)

Writes results/runs_table.csv, results/map_summary.csv, results/instability.csv, results/map_verdicts.csv and
figures E026_heatmap_val.png, E026_pareto_by_window.png, E026_instability.png.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .pareto_v2 import LEVELS, level_of
from .summary_e024 import aggregate, better, setup, verdict, write_verdicts


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E026")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage.isin(["grid", "confirm"])]
    table = aggregate(runs, ["variant", "stage", "budget", "spec_model", "spec_window", "spec_hidden"])
    table["kind"] = np.where(table.spec_model == "complex_both", "both", "one_hidden")
    table["level"] = [level_of(m) for m in table.rm]
    table.to_csv(results / "map_summary.csv", index=False)
    grid = table[table.stage == "grid"]
    inst = runs[runs.stage == "grid"].groupby("spec_window").agg(runs=("seed", "count"), diverged=("diverged", "sum")).reset_index()
    inst["diverged_share"] = inst.diverged / inst.runs
    inst.to_csv(results / "instability.csv", index=False)

    rows = []
    oh = grid[grid.kind == "one_hidden"]
    for level in LEVELS:
        part = oh[oh.level == level]
        base, wide = part[part.spec_window == 31], part[part.spec_window > 31]
        if len(base) and len(wide):
            b, w = base.loc[base.val_mean.idxmax()], wide.loc[wide.val_mean.idxmax()]
            diff, ok = better(w, b, 0.5)
            rows.append({"hypothesis": f"{level} RM: окно шире 31 окупается ({w.variant} против {b.variant})",
                         "value": diff, "verdict": verdict(ok), "detail": "лучшие по валидации в окне ±25 %, тест"})
        both = grid[(grid.kind == "both") & (grid.level == level)]
        if len(both) and len(part):
            d, o = both.loc[both.val_mean.idxmax()], part.loc[part.val_mean.idxmax()]
            diff, ok = better(d, o, 0.5)
            rows.append({"hypothesis": f"{level} RM: глубина (both) окупается ({d.variant} против {o.variant})",
                         "value": diff, "verdict": verdict(ok), "detail": "тест"})
    env = envelope_curve(grid)
    sat = saturation(env)
    rows.append({"hypothesis": "насыщение: наименьший RM, после которого удвоение RM даёт < 0.5 п.п.", "value": sat,
                 "verdict": "подтверждено" if sat == sat else "не доказано",
                 "detail": "огибающая средних валидаций (100k, 3 seeds)"})
    write_verdicts(rows, config, results / "map_verdicts.csv")
    plots(grid, table, inst, figures)
    print(f"карта: {results / 'map_summary.csv'}; вердикты: {results / 'map_verdicts.csv'}")
    return 0


def envelope_curve(frame: pd.DataFrame) -> pd.DataFrame:
    order = frame.sort_values("rm")
    best, rows = -math.inf, []
    for _, r in order.iterrows():
        best = max(best, r.val_mean)
        rows.append((r.rm, best))
    return pd.DataFrame(rows, columns=["rm", "val"])


def saturation(env: pd.DataFrame) -> float:
    for rm, val in env.values:
        later = env[env.rm <= 2 * rm]
        if later.val.max() - val < 0.5 and (env.rm > 2 * rm).any():
            return float(rm)
    return float("nan")


def plots(grid: pd.DataFrame, table: pd.DataFrame, inst: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    oh = grid[grid.kind == "one_hidden"]
    if len(oh):
        pivot = oh.pivot_table(index="spec_hidden", columns="spec_window", values="val_mean")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        values = pivot.to_numpy(float)
        top = np.nanmax(values)
        vmin = max(np.nanmin(values), top - 40)            # outliers (diverged cells) do not flatten the scale
        im = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=top)
        ax.set_xticks(range(len(pivot.columns)), [int(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [int(i) for i in pivot.index])
        for (i, j), v in np.ndenumerate(values):
            if v == v:
                bright = (min(max(v, vmin), top) - vmin) / max(top - vmin, 1e-9) > 0.6
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="black" if bright else "white")
        w = np.array(pivot.columns, float)
        h = np.array(pivot.index, float)
        rm = h[:, None] * (4 * w[None, :] + 11)
        cs = ax.contour(np.log10(rm), levels=np.log10([300, 1000, 4000, 16000, 40000]), colors=plot_style.TEXT,
                        linewidths=0.8, linestyles="dashed")
        ax.clabel(cs, fmt=lambda v: f"{10 ** v:.0f} RM", fontsize=6)
        fig.colorbar(im, ax=ax, label="validation BER reduction, % (100k, mean of 3 seeds)")
        ax.set_xlabel("window w")
        ax.set_ylabel("hidden width h")
        ax.set_title("One-hidden Kerr MLP: validation map with RM isolines h(4w + 11)")
        fig.tight_layout()
        fig.savefig(figures / "E026_heatmap_val.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (w, part) in enumerate(grid.groupby("spec_window")):
        for kind, marker in (("one_hidden", "o"), ("both", "s")):
            p = part[part.kind == kind].sort_values("rm")
            if len(p):
                ax.plot(p.rm, p.val_mean, marker + "-", color=plot_style.color(index), markersize=4, linewidth=0.8,
                        label=f"w={int(w)} {kind}")
    ax.set_xscale("log")
    ax.set_xlabel("RM per symbol (log)")
    ax.set_ylabel("validation BER reduction, %")
    ax.set_title("Pareto by window (100k, 3 seeds); squares = two hidden layers (both)")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "E026_pareto_by_window.png", dpi=160)
    plt.close(fig)
    if len(inst):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar([str(int(w)) for w in inst.spec_window], inst.diverged_share, color=plot_style.color(0))
        ax.set_xlabel("window w")
        ax.set_ylabel("share of diverged runs")
        ax.set_title("Instability by window (grid stage)")
        fig.tight_layout()
        fig.savefig(figures / "E026_instability.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
