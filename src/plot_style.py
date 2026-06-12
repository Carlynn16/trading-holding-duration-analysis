"""Shared plot style for all figures in this project."""

import matplotlib.pyplot as plt
import seaborn as sns


ACCENT = "steelblue"
LOSS_COLOR = "#c0392b"
PROFIT_COLOR = "#27ae60"

# Sequential palette for duration bins (7 bins max)
_CREST = sns.color_palette("crest", n_colors=7)
BIN_PALETTE = _CREST

SAVEFIG_KW = dict(dpi=150, bbox_inches="tight")


def apply_style() -> None:
    """Apply the project-wide matplotlib/seaborn style. Call once per script."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "whitesmoke",
            "grid.color": "grey",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 100,
        }
    )
