"""Slide figure "BER reduction against the cost in multiplications": windowed DBP -> DBP-X without training -> trained DBP-X.
No computation: numbers from the result files only. Run from the repository root:
    .venv/bin/python experiments/E050_dbpx_physical/make_slide_path.py
Writes figures/E050_slide_path.png (2560 x 1100 px) and results/slide_path_points.csv (every drawn point with its source).
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path("experiments")
E050 = ROOT / "E050_dbpx_physical"
PX = 72 / 100                                   # dpi 100: points per pixel
rows = []


def keep(series, fft, direct, red, spread, source, env_fft=None, env_direct=None, variant=None):
    n = len(list(red))
    env_fft = [None] * n if env_fft is None else env_fft
    env_direct = [None] * n if env_direct is None else env_direct
    variant = [None] * n if variant is None else variant
    for f, d, y, s, ef, ed, v in zip(fft, direct, red, spread, env_fft, env_direct, variant):
        rows.append({"series": series, "config": v, "rm_fft": float(f), "rm_direct": float(d), "test_reduction_pct": float(y),
                     "test_std_pct": None if s is None else float(s), "in_envelope_fft": ef, "in_envelope_direct": ed,
                     "source": source})


def envelope(frame: pd.DataFrame, col: str) -> pd.Series:
    """Upper envelope over cost: a configuration is on it if its validation reduction beats every cheaper one
    (selection by validation, as E056; test values are drawn)."""
    best, on = -1e9, pd.Series(False, index=frame.index)
    for i, r in frame.sort_values([col, "val_mean"], ascending=[True, False]).iterrows():
        if r.val_mean > best:
            best, on[i] = r.val_mean, True
    return on


# (a) windowed DBP: E050 windowed_dbp_summary.csv, E050 reference (10 seeds) where it exists, else E028 grid a (3 seeds)
w = pd.read_csv(E050 / "results/windowed_dbp_summary.csv")
w = w.assign(_p=(w.source != "E050 ref").astype(int)).sort_values("_p").drop_duplicates(["n_steps", "L", "R"])
w["env_fft"], w["env_direct"] = envelope(w, "rm_xiao"), envelope(w, "rm_project")
keep("Старый DBP в окне", w.rm_xiao, w.rm_project, w.test_mean, w.test_std,
     "E050/results/windowed_dbp_summary.csv (E050 ref where present, else E028 a); rm_xiao, rm_project, test_mean ± test_std; "
     "envelope by validation (val_mean)", w.env_fft, w.env_direct, w.variant)
# long training of the same DBP: E028 a_summary.csv, stage a_long, 400 000 steps; FFT cost of the same configuration
a28 = pd.read_csv(ROOT / "E028_dbp_depth_layerwise/results/a_summary.csv")
lg = a28[(a28.stage == "a_long") & (a28.variant == "dbp_Ns32_L31_R15")].iloc[0]
lg_fft = float(pd.read_csv(E050 / "results/windowed_dbp_summary.csv").query("variant == 'dbp_Ns32_L31_R15'").rm_xiao.iloc[0])
keep("Старый DBP в окне, долгое обучение", [lg_fft], [lg.rm], [lg.test_mean], [None],
     "E028/results/a_summary.csv (a_long, 400000 steps, 5 seeds; rm); rm_xiao of the same config from E050 windowed_dbp_summary.csv")
# (b) DBP-X without training: E050 file_stats.csv, ridge kernel, honest filters, Ns 8 / 12 / 20
fs = pd.read_csv(E050 / "results/file_stats.csv")
fs = fs[(fs.kernel == "ridge") & (fs.filters == "honest") & fs.n_steps.isin([8, 12, 20])].sort_values("n_steps")
keep("DBP-X без обучения", fs.rm_fft_4, fs.rm_direct_4, fs.reduction_mean_over_files, [None] * len(fs),
     "E050/results/file_stats.csv (ridge, honest); rm_fft_4, rm_direct_4, reduction_mean_over_files")
# (c) trained DBP-X: E054 summary.csv, Ns 20: L 63 / 255 progressive, L 31 direct_ld (joint least-squares start)
s54 = pd.read_csv(ROOT / "E054_dbpx_pruning/results/summary.csv")
tr = pd.concat([s54[(s54.n_steps == 20) & (s54.L == L) & (s54["mode"] == m)]
                for L, m in ((31, "direct_ld"), (63, "progressive"), (255, "progressive"))])
keep("DBP-X обученный", tr.rm_fft_4, tr.rm_direct_4, tr.test_mean, tr.test_std,
     "E054/results/summary.csv (Ns 20; L 31 direct_ld, L 63 / 255 progressive); rm_fft_4, rm_direct_4, test_mean ± test_std")

INK, INK2, GRID = "#111111", "#4a4a4a", "#e2e2e2"
GREY, BLUE = "#7f7f7f", "#0072b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 28 * PX, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(25.6, 11.0), dpi=100, facecolor="white", sharey=True)
MS, LW = 16 * PX + 3, 4 * PX + 1
t63 = tr[tr.L == 63].iloc[0]


def note(ax, text, xy, xytext, color, ha="left"):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=28 * PX, color=color, fontweight="bold", ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=2),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))


for ax, col_w, col_fs, col_tr, xl in ((axes[0], "rm_xiao", "rm_fft_4", "rm_fft_4", "RM на символ, цена по БПФ"),
                                       (axes[1], "rm_project", "rm_direct_4", "rm_direct_4", "RM на символ, прямой счёт")):
    e = w[w["env_fft" if col_w == "rm_xiao" else "env_direct"]].sort_values(col_w)
    ax.errorbar(e[col_w], e.test_mean, yerr=e.test_std, fmt="-o", color=GREY, lw=LW, ms=MS, elinewidth=3, capsize=6,
                capthick=3, mec="white", mew=1.5, zorder=3)
    top = e.loc[e.test_mean.idxmax()]
    note(ax, f"лучший старый DBP: {top.test_mean:.1f} %", (top[col_w], top.test_mean),
         (400 if col_w == "rm_xiao" else top[col_w] * 0.45, 30), GREY)
    x_long = lg_fft if col_w == "rm_xiao" else lg.rm
    ax.plot([x_long], [lg.test_mean], "s", color=GREY, ms=MS * 1.1, mec=INK, mew=2.5, zorder=4)
    ax.plot(fs[col_fs], fs.reduction_mean_over_files, color=BLUE, lw=LW, marker="^", ms=MS * 1.15, mec="white",
            mew=1.5, zorder=5)
    ax.plot(tr[col_tr], tr.test_mean, "D", color=INK, ms=MS * 1.05, zorder=6, clip_on=False)
    ax.set_xscale("log")
    ax.set_xlabel(xl, fontsize=28 * PX, labelpad=10)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.grid(color=GRID, lw=1.2)
    ax.tick_params(labelsize=28 * PX, width=1.5, length=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    note(ax, f"после долгого обучения: {lg.test_mean:.1f} %", (x_long, lg.test_mean),
         (x_long * (0.09 if col_w == "rm_xiao" else 0.12), 70), INK2)

a, b = axes
a.set_xlim(150, 3000)
a.xaxis.set_major_locator(FixedLocator([200, 300, 500, 1000, 2000, 3000]))
b.set_xlim(250, 30000)
b.xaxis.set_major_locator(FixedLocator([300, 1000, 3000, 10000, 30000]))
a.set_ylim(-10, 105)
a.set_ylabel("снижение BER на тесте, %", fontsize=28 * PX)
a.axvline(1400, color=INK2, lw=3 * PX + 0.3, ls=(0, (4, 4)), zorder=0)
a.text(1400 * 1.04, -6, "1.4k RM", fontsize=28 * PX, color=INK2)
note(a, "до 97 %", (float(fs.rm_fft_4.iloc[-1]), float(fs.reduction_mean_over_files.iloc[-1])), (560, 80), BLUE)
note(b, "до 97 %", (float(fs.rm_direct_4.iloc[-1]), float(fs.reduction_mean_over_files.iloc[-1])), (2600, 80), BLUE)
note(a, f"{t63.test_mean:.2f} % при {t63.rm_fft_4:,.0f} RM (БПФ)".replace(",", " "), (t63.rm_fft_4, t63.test_mean),
     (165, 104), INK)
note(b, f"{t63.test_mean:.2f} % при {t63.rm_direct_4:,.0f} RM".replace(",", " "), (t63.rm_direct_4, t63.test_mean),
     (290, 104), INK)

handles = [
    Line2D([], [], color=GREY, lw=LW, marker="o", ms=MS, label="Старый DBP в окне"),
    Line2D([], [], color=BLUE, lw=LW, marker="^", ms=MS * 1.15, label="DBP-X без обучения (физические веса, ридж-ядро Керра)"),
    Line2D([], [], color=INK, lw=0, marker="D", ms=MS * 1.05, label="DBP-X обученный"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=28 * PX, bbox_to_anchor=(0.5, 0.075),
           columnspacing=2.5, handlelength=2.5)
fig.text(0.5, 0.02, "Разные старты Керра (ноль, классическое ядро, ридж-ядро) и обучение в 1 или 2 этапа дают 99.1–99.3 %, "
         "на графике не показаны.", ha="center", fontsize=28 * PX, color=INK2)
fig.subplots_adjust(left=0.06, right=0.985, top=0.96, bottom=0.27, wspace=0.08)
fig.savefig(E050 / "figures/E050_slide_path.png", dpi=100, facecolor="white")
pd.DataFrame(rows).to_csv(E050 / "results/slide_path_points.csv", index=False)
print("ok", len(rows))
