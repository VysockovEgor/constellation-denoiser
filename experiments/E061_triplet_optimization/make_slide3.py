"""E061: slide figure (triplet variants vs DBP-X at equal cost). No computation: numbers from the result files only.
Run from the repository root:
    .venv/bin/python experiments/E061_triplet_optimization/make_slide3.py
Writes figures/E061_slide3.png (2560 x 1100 px) and results/slide3_points.csv (every drawn point with its source).
Q^2 = 20 log10(sqrt(10) erfcinv(8 BER / 3)) with BER = raw * (1 - reduction / 100), raw test BER of the project
(window-31 centres, 0.009551859551859552) for every series.
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, "src")
from constellation_denoiser.q2 import q2_db  # noqa: E402

ROOT = Path("experiments")
E061 = ROOT / "E061_triplet_optimization"
RAW = 0.009551859551859552
PX = 72 / 100                       # dpi 100: points per pixel

rows = []                           # drawn points: series, rm, reduction, source


def keep(series, rm, red, source):
    for x, y in zip(rm, red):
        rows.append({"series": series, "rm": float(x), "test_reduction_pct": float(y),
                     "q2_db": q2_db(RAW * (1 - y / 100)), "source": source})
    return list(rm), list(red)


# ---- E056 arms: upper envelope by validation over RM (as E056_ablation_vs_rm.png)
old = pd.read_csv(ROOT / "E056_kerr_offaxis_ablation/results/ablation_all_configs.csv")


def envelope(arm):
    part, env, best = old[old.arm == arm].sort_values("rm"), [], -1e9
    for _, r in part.iterrows():
        if r.val_mean > best:
            best = r.val_mean
            env.append(r)
    return pd.DataFrame(env)


SRC56 = "E056/results/ablation_all_configs.csv, arm {}, upper envelope by validation"
arm = {a: keep(name, envelope(a).rm, envelope(a).test_mean, SRC56.format(a)) for a, name in
       ((1, "Все триплеты"), (2, "Керр без памяти + внеосевые триплеты"), (3, "Керр с памятью, без внеосевых"),
        (4, "КИХ7 + Керр с памятью + внеосевые (A1)"))}

# ---- E061 variants
r61 = pd.read_csv(E061 / "results/results.csv")
SRC61 = "E061/results/results.csv, variant {}, rm_shared_b, test_reduction"
t6 = r61[r61.variant == "T6"].sort_values("rm_shared_b")
t6xy = keep("Связка по m·n + отбор членов (T6)", t6.rm_shared_b, t6.test_reduction, SRC61.format("T6"))
t0 = r61[r61.variant == "T0"]
t0xy = keep("Свободные коэффициенты, 368 (T0)", t0.rm_shared_b, t0.test_reduction, SRC61.format("T0"))
t2 = r61[r61.variant == "T2"]
t2xy = keep("Общий коэффициент на группу m·n, 159 (T2)", t2.rm_shared_b, t2.test_reduction, SRC61.format("T2"))
others = {}
for v in ("T4", "T5mn", "T5max"):
    g = r61[r61.variant == v].sort_values("rm_shared_b")
    others[v] = keep(f"Другие способы отбора ({v})", g.rm_shared_b, g.test_reduction, SRC61.format(v))

# ---- DBP-X trained (E054 summary, Ns 20): L 63 / 255 progressive, L 31 direct_ld (joint least-squares start)
s54 = pd.read_csv(ROOT / "E054_dbpx_pruning/results/summary.csv")
pick = [(63, "progressive"), (255, "progressive"), (31, "direct_ld")]
d = pd.concat([s54[(s54.n_steps == 20) & (s54.L == L) & (s54["mode"] == m)] for L, m in pick])
SRC54 = "E054/results/summary.csv, Ns 20, L {} {}, {}"
fft = [keep("DBP-X обученный, БПФ", [r.rm_fft_4], [r.test_mean], SRC54.format(r.L, r["mode"], "rm_fft_4"))
       for _, r in d.iterrows()]
direct = [keep("DBP-X обученный, прямой счёт", [r.rm_direct_4], [r.test_mean], SRC54.format(r.L, r["mode"], "rm_direct_4"))
          for _, r in d.iterrows()]

# ---- DBP-X untrained (E050 file_stats: ridge kernel, honest filters, Ns 8 / 12 / 20)
fs = pd.read_csv(ROOT / "E050_dbpx_physical/results/file_stats.csv")
fs = fs[(fs.kernel == "ridge") & (fs.filters == "honest") & fs.n_steps.isin([8, 12, 20])].sort_values("n_steps")
untr = keep("DBP-X без обучения (Ns = 8, 12, 20)", fs.rm_fft_4, fs.reduction_mean_over_files,
            "E050/results/file_stats.csv, ridge, honest, rm_fft_4, reduction_mean_over_files")

# ---- figure
INK, INK2, GRID = "#111111", "#4a4a4a", "#e2e2e2"
C_A, C_B, C_C = "#8c8c8c", "#a6761d", "#5e4b3c"        # triplet arms: grey, brown, dark brown
C_A1, C_T6, C_T2, C_OTH = "#d55e00", "#0072b2", "#56b4e9", "#c4c4c4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 28 * PX, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(25.6, 11.0), dpi=100, facecolor="white")
LW, LWB, MS = 4 * PX * 1.0 + 1, 7 * PX * 1.0 + 2, 16 * PX + 2     # >= 3 px lines, >= 14 px markers


def series(ax, xy, to_q, **kw):
    x, y = xy
    ax.plot(x, [q2_db(RAW * (1 - v / 100)) for v in y] if to_q else y, **kw)


for i, ax in enumerate(axes):
    q = i == 1
    for v, xy in others.items():
        series(ax, xy, q, color=C_OTH, lw=3 * PX + 0.2, zorder=1)
    series(ax, arm[1], q, color=C_A, lw=LW, ls=(0, (6, 3)), marker="o", ms=MS, zorder=2)
    series(ax, arm[2], q, color=C_B, lw=LW, ls=(0, (8, 3, 2, 3)), marker="v", ms=MS, zorder=2)
    series(ax, arm[3], q, color=C_C, lw=LW, ls=(0, (2, 2)), marker="^", ms=MS, zorder=2)
    series(ax, arm[4], q, color=C_A1, lw=LWB, marker="D", ms=MS, zorder=4)
    series(ax, t6xy, q, color=C_T6, lw=LWB, marker="o", ms=MS, zorder=5)
    series(ax, t0xy, q, color=C_T6, lw=0, marker="*", ms=MS * 1.9, mec="white", mew=1.5, zorder=6)
    series(ax, t2xy, q, color=C_T2, lw=0, marker="D", ms=MS * 1.15, mec=C_T6, mew=2.5, zorder=6)
    series(ax, untr, q, color=INK2, lw=LW, ls=(0, (1.5, 2.5)), marker="X", ms=MS * 1.1, zorder=6)
    for f, dd in zip(fft, direct):
        series(ax, ([f[0][0], dd[0][0]], [f[1][0], dd[1][0]]), q, color=INK, lw=3 * PX + 0.2, ls=(0, (1, 2.5)), zorder=6)
        series(ax, f, q, color=INK, lw=0, marker="s", ms=MS * 1.1, zorder=7, clip_on=False)
        series(ax, dd, q, color=INK, lw=0, marker="s", ms=MS * 1.1, mfc="white", mew=3, zorder=7, clip_on=False)
    ax.axvline(1400, color=INK2, lw=3 * PX + 0.2, ls=(0, (4, 4)), zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(30, 20000)
    ax.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000, 3000, 10000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.grid(color=GRID, lw=1.2)
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=28 * PX, width=1.5, length=8)

a, b = axes
a.set_title("Снижение BER, %", fontsize=34 * PX, loc="left", pad=16)
b.set_title("Q², дБ", fontsize=34 * PX, loc="left", pad=16)
a.set_ylim(0, 100)
b.set_ylim(13.5, 18.8)
raw_q2 = q2_db(RAW)
b.axhline(raw_q2, color=INK2, lw=3 * PX + 0.2, ls=(0, (6, 4)))
b.text(19000, raw_q2 + 0.08, "без компенсации, 13.97 дБ", fontsize=28 * PX, color=INK2, va="bottom", ha="right")
for ax, y in ((a, 3), (b, 13.62)):
    ax.text(1400 * 1.06, y, "1.4k RM", fontsize=28 * PX, color=INK2, va="bottom")


def note(text, xy, xytext, color):
    a.annotate(text, xy=xy, xytext=xytext, fontsize=28 * PX, color=color, fontweight="bold",
               arrowprops=dict(arrowstyle="-", color=color, lw=2), va="center",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


a1 = [(x, y) for x, y in zip(*arm[4]) if round(x) == 1242][0]
t6p = [(x, y) for x, y in zip(*t6xy) if round(x) == 1308][0]
note(f"{a1[1]:.1f} % при {a1[0]:.0f} RM", a1, (75, 72), C_A1)
note(f"{t6p[1]:.1f} %", t6p, (2600, 40), C_T6)
note(f"{t2xy[1][0]:.2f} % при {t2xy[0][0]:.0f} RM", (t2xy[0][0], t2xy[1][0]), (3300, 52), C_T6)
dbp63 = (fft[0][0][0], fft[0][1][0])
note(f"{dbp63[1]:.2f} % при {dbp63[0]:.0f} RM", dbp63, (1900, 86), INK)

handles = [
    Line2D([], [], color=C_A, lw=LW, ls=(0, (6, 3)), marker="o", ms=MS, label="Все триплеты"),
    Line2D([], [], color=C_B, lw=LW, ls=(0, (8, 3, 2, 3)), marker="v", ms=MS, label="Керр без памяти + внеосевые триплеты"),
    Line2D([], [], color=C_C, lw=LW, ls=(0, (2, 2)), marker="^", ms=MS, label="Керр с памятью, без внеосевых"),
    Line2D([], [], color=C_A1, lw=LWB, marker="D", ms=MS, label="КИХ7 + Керр с памятью + внеосевые (A1)"),
    Line2D([], [], color=C_T6, lw=LWB, marker="o", ms=MS, label="Связка по m·n + отбор членов (лучшая кривая)"),
    Line2D([], [], color=C_T6, lw=0, marker="*", ms=MS * 1.9, mec="white", label="Свободные коэффициенты, 368"),
    Line2D([], [], color=C_T2, lw=0, marker="D", ms=MS * 1.15, mec=C_T6, mew=2.5, label="Общий коэффициент на группу m·n, 159"),
    Line2D([], [], color=C_OTH, lw=3 * PX + 0.2, label="Другие способы отбора членов (T4, T5)"),
    Line2D([], [], color=INK, lw=0, marker="s", ms=MS * 1.1, label="DBP-X обученный, цена по БПФ"),
    Line2D([], [], color=INK, lw=0, marker="s", ms=MS * 1.1, mfc="white", mew=3, label="DBP-X обученный, прямой счёт"),
    Line2D([], [], color=INK2, lw=LW, ls=(0, (1.5, 2.5)), marker="X", ms=MS * 1.1, label="DBP-X без обучения (Ns = 8, 12, 20)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=28 * PX,
           bbox_to_anchor=(0.5, 0.0), columnspacing=2.0, handlelength=3.2)
fig.supxlabel("RM на символ (триплеты и DBP-X с пустыми маркерами: прямой счёт; DBP-X с закрашенными: БПФ)",
              fontsize=28 * PX, y=0.235)
fig.subplots_adjust(left=0.04, right=0.985, top=0.92, bottom=0.33, wspace=0.12)
fig.savefig(E061 / "figures/E061_slide3.png", dpi=100, facecolor="white")
pd.DataFrame(rows).to_csv(E061 / "results/slide3_points.csv", index=False)
print("ok", len(rows), "points")
