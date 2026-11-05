"""Shared figure style: every graph must have named, unit-labeled axes
(project rule), so this is centralized here rather than repeated per task.
"""

import matplotlib.pyplot as plt

_STYLE = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.frameon": False,
    "font.size": 10,
}


def set_style() -> None:
    """Apply the project's shared matplotlib style. Call once, e.g. at the
    top of a script or notebook."""
    plt.rcParams.update(_STYLE)


def _label(name: str, unit: str | None) -> str:
    return f"{name} ({unit})" if unit else name


def new_figure(
    xlabel: str,
    ylabel: str,
    xunit: str | None = None,
    yunit: str | None = None,
    title: str | None = None,
):
    """Create a (fig, ax) pair with mandatory axis labels (and units when
    the quantity has one, e.g. "time", "s" -> "time (s)"; unitless
    quantities like "epoch" or "batch index" can omit xunit/yunit)."""
    if not xlabel or not ylabel:
        raise ValueError("xlabel and ylabel are mandatory for every figure")
    fig, ax = plt.subplots()
    ax.set_xlabel(_label(xlabel, xunit))
    ax.set_ylabel(_label(ylabel, yunit))
    if title:
        ax.set_title(title)
    return fig, ax
