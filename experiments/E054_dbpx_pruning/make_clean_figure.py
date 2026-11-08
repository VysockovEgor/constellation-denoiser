"""E054: clean presentation figure (Ns = 20; the lower part of the left axis shows direct_ld at L = 15 and 7). Run from the repository root:
    .venv/bin/python experiments/E054_dbpx_pruning/make_clean_figure.py
"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

s = pd.read_csv("experiments/E054_dbpx_pruning/results/summary.csv")
s = s[s.n_steps == 20]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LS = [7, 15, 31, 63, 255]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(12, 5.2), facecolor=SURF)
gs = fig.add_gridspec(2, 2, height_ratios=(3, 1.3), hspace=0.08)
a = fig.add_subplot(gs[0, 0])
a2 = fig.add_subplot(gs[1, 0], sharex=a)      # lower part of the broken axis: collapsed configurations
b = fig.add_subplot(gs[:, 1])
for ax in (a, a2, b):
    ax.set_facecolor(SURF)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(LS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlim(5.5, 330)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("длина КИХ шага L, отводов")

# left: BER reduction; upper part = working configurations, lower part = direct_ld at L = 15, 7 (collapsed)
pr = s[(s["mode"] == "progressive") & s.L.isin([31, 63, 255])].sort_values("L")
ld = s[(s["mode"] == "direct_ld") & (s.L == 31)]
ld_low = s[(s["mode"] == "direct_ld") & s.L.isin([7, 15])].sort_values("L")
a.errorbar(pr.L, pr.test_mean, yerr=pr.test_se, color=BLUE, lw=2, marker="o", ms=8, capsize=4,
           mec=SURF, mew=2, label="прогрессивная обрезка (10 seed)")
a.errorbar(ld.L, ld.test_mean, yerr=ld.test_se, color=ORANGE, lw=0, marker="s", ms=9, capsize=4,
           mec=SURF, mew=2, label="совместный МНК каскада, сразу L (5 seed)")
for _, r in pr.iterrows():
    below = r.L == 31
    a.annotate(f"{r.test_mean:.2f}", (r.L, r.test_mean), xytext=(0, -18 if below else 9), textcoords="offset points",
               ha="center", va="top" if below else "bottom", color=INK2, fontsize=10)
r = ld.iloc[0]
a.annotate(f"{r.test_mean:.2f}", (r.L, r.test_mean), xytext=(-12, 0), textcoords="offset points",
           ha="right", va="center", color=INK2, fontsize=10)
a2.errorbar(ld_low.L, ld_low.test_mean, yerr=ld_low.test_se, color=ORANGE, lw=0, marker="s", ms=9, capsize=4,
            mec=SURF, mew=2)
for _, r in ld_low.iterrows():
    a2.annotate(f"{r.test_mean:.0f}".replace("-", "−"), (r.L, r.test_mean), xytext=(12, 0), textcoords="offset points",
                ha="left", va="center", color=INK2, fontsize=10)
a.set_ylim(90, 100)
a2.set_ylim(-3800, 0)
a2.set_yticks([-3000, -1500, 0])
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}".replace("-", "−")))
a.spines["bottom"].set_visible(False)
a.tick_params(axis="x", bottom=False, labelbottom=False)
a.set_xlabel("")
for ax, y in ((a, 0), (a2, 1)):                 # break marks between the two parts of the axis
    ax.plot([0, 0.015], [y - 0.02, y + 0.02], transform=ax.transAxes, color=INK2, lw=1, clip_on=False)
a2.text(0.98, 0.1, "ниже 0: BER хуже, чем без компенсации", transform=a2.transAxes, ha="right", va="bottom",
        color=INK2, fontsize=9)
fig.text(0.015, 0.55, "снижение BER на тесте, %", rotation=90, va="center", ha="left", color=INK)
a.set_title("DBP-X, Ns = 20: качество (среднее ± SE)", loc="left", fontsize=12)
a.legend(frameon=False, loc="lower right", fontsize=9.5)

# right: RM per symbol
rm = s[s["mode"] == "progressive"].set_index("L").reindex(LS)
for col, mk, ls, lab in (("rm_direct_4", "o", "-", "прямая свёртка"), ("rm_fft_4", "s", "--", "БПФ (overlap-save)")):
    b.plot(LS, rm[col], color=INK2, lw=2, ls=ls, marker=mk, ms=7, mec=SURF, mew=2, label=lab)
b.set_yscale("log")
b.set_ylabel("RM на символ (4 RM на компл. умножение)")
b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " ")))
b.set_title("Цена на символ, Ns = 20", loc="left", fontsize=12)
b.legend(frameon=False, loc="upper left", fontsize=9.5)
fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.11, wspace=0.22)
fig.savefig("experiments/E054_dbpx_pruning/figures/E054_clean.png", dpi=200, facecolor=SURF)
print(pr[["L", "test_mean", "test_se"]].to_string(), ld[["L", "test_mean", "test_se"]].to_string(),
      ld_low[["L", "test_mean", "test_se"]].to_string(), rm[["rm_direct_4", "rm_fft_4"]].to_string())
