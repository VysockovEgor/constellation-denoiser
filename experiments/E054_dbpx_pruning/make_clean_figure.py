"""E054: clean presentation figure (Ns = 20, working configurations only). Run from the repository root:
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
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(LS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlim(5.5, 330)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("длина КИХ шага L, отводов")

# left: BER reduction, only working configurations
pr = s[(s["mode"] == "progressive") & s.L.isin([31, 63, 255])].sort_values("L")
ld = s[(s["mode"] == "direct_ld") & (s.L == 31)]
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
a.set_ylim(90, 100)
a.set_ylabel("снижение BER на тесте, %")
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
fig.tight_layout()
fig.savefig("experiments/E054_dbpx_pruning/figures/E054_clean.png", dpi=200, facecolor=SURF)
print(pr[["L", "test_mean", "test_se"]].to_string(), ld[["L", "test_mean", "test_se"]].to_string(), rm[["rm_direct_4", "rm_fft_4"]].to_string())
