"""Shared matplotlib style for E010+ figures (fixed categorical order, recessive axes)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LINESTYLES = ["-", "--", ":", "-."]


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 5, "font.size": 9, "legend.frameon": False,
    })


def color(index: int) -> str:
    """Categorical slot in fixed order; more than 8 identities must be folded or faceted."""
    if index >= len(SERIES):
        raise ValueError("more than 8 categorical series: facet or fold into 'other'")
    return SERIES[index]


def band(ax, x, mean, low, high, index: int, label: str, linestyle: str = "-") -> None:
    ax.plot(x, mean, color=color(index), linestyle=linestyle, label=label)
    ax.fill_between(x, low, high, color=color(index), alpha=0.18, linewidth=0)
