"""Compute Dunn post-hoc tests and generate fig_dunn_heatmap.png."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scikit_posthocs as sp

from src.plot_style import apply_style, ACCENT, SAVEFIG_KW

FIGURES   = Path("figures")
PROCESSED = Path("data/processed")

HOUR_ORDER = ["0-1h", "1h-3h", "3h-6h", "6h-8h", "8h-9h", "9h-24h", ">1 day"]
DAY_ORDER  = ["0-1d", "1-3d", "3-5d", "5-7d", ">7d"]


def compute_dunn():
    print("Loading data…")
    df = pd.read_parquet(
        PROCESSED / "trades_clean.parquet",
        columns=["duration_bin_hours", "duration_bin_days", "Profit"],
    )
    df["duration_bin_hours"] = df["duration_bin_hours"].astype(str)
    df["duration_bin_days"]  = df["duration_bin_days"].astype(str)

    print(f"  {len(df):,} trades loaded")

    print("Computing Dunn / Holm for hour bins (7×7)…")
    dunn_h = sp.posthoc_dunn(
        df, val_col="Profit", group_col="duration_bin_hours", p_adjust="holm"
    )
    dunn_h = dunn_h.reindex(index=HOUR_ORDER, columns=HOUR_ORDER)
    dunn_h.to_csv(PROCESSED / "dunn_hour.csv")
    print("  saved dunn_hour.csv")

    print("Computing Dunn / Holm for day bins (5×5)…")
    dunn_d = sp.posthoc_dunn(
        df, val_col="Profit", group_col="duration_bin_days", p_adjust="holm"
    )
    dunn_d = dunn_d.reindex(index=DAY_ORDER, columns=DAY_ORDER)
    dunn_d.to_csv(PROCESSED / "dunn_day.csv")
    print("  saved dunn_day.csv")

    return dunn_h, dunn_d


def _annot_and_color(pmat):
    """Return (annot 2-D array, color 2-D array) for a p-value matrix."""
    n = len(pmat)
    annot  = np.empty((n, n), dtype=object)
    color  = np.zeros((n, n), dtype=float)  # 0 = sig, 1 = n.s., nan = diag

    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j]  = ""
                color[i, j]  = np.nan
            else:
                p = float(pmat.iloc[i, j])
                if p < 0.001:
                    annot[i, j] = "< 0.001"
                    color[i, j] = 0.0
                elif p < 0.05:
                    annot[i, j] = f"{p:.3f}"
                    color[i, j] = 0.0
                else:
                    annot[i, j] = "n.s."
                    color[i, j] = 1.0

    return annot, color


def fig_dunn_heatmap(dunn_h, dunn_d):
    apply_style()

    # Two colors: significant = muted steel-blue, n.s. = amber (stands out)
    SIG_COLOR = "#3D6B8E"
    NS_COLOR  = "#FFC857"
    cmap = mcolors.ListedColormap([SIG_COLOR, NS_COLOR])
    mask_diag_h = np.eye(len(dunn_h), dtype=bool)
    mask_diag_d = np.eye(len(dunn_d), dtype=bool)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pmat, mask, title, fsize in [
        (ax1, dunn_h, mask_diag_h, "Profit by hour bin  (7×7)", 8),
        (ax2, dunn_d, mask_diag_d, "Profit by day bin   (5×5)", 9),
    ]:
        annot, color = _annot_and_color(pmat)
        labels = pmat.columns.tolist()

        sns.heatmap(
            color,
            ax=ax,
            mask=mask,
            cmap=cmap,
            vmin=0, vmax=1,
            annot=annot,
            fmt="",
            annot_kws={"size": fsize},
            linewidths=0.5,
            linecolor="white",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            square=True,
        )
        # White diagonal squares
        for k in range(len(labels)):
            ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                                        color="white", lw=0))

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=9)

    # Shared legend patches
    import matplotlib.patches as mpatches
    sig_patch = mpatches.Patch(color=SIG_COLOR, label="p < 0.05 (significant)")
    ns_patch  = mpatches.Patch(color=NS_COLOR,  label="p ≥ 0.05 (n.s.)")
    fig.legend(handles=[sig_patch, ns_patch], loc="lower center",
               ncol=2, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Dunn Post-Hoc Tests — Holm-Adjusted Pairwise p-values",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_dunn_heatmap.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_dunn_heatmap.png")


def report_ns_pairs(dunn_h, dunn_d):
    """Print all non-significant pairs (Holm p >= 0.05)."""
    print("\n--- Non-significant pairs (Holm p >= 0.05) ---")
    for label, pmat in [("HOUR", dunn_h), ("DAY", dunn_d)]:
        cols = pmat.columns.tolist()
        n = len(cols)
        found = []
        for i in range(n):
            for j in range(i + 1, n):
                p = float(pmat.iloc[i, j])
                if p >= 0.05:
                    found.append(f"  {cols[i]} vs {cols[j]}: p={p:.4f}")
        if found:
            print(f"{label} bins:")
            for f in found:
                print(f)
        else:
            print(f"{label} bins: all pairs significant")


if __name__ == "__main__":
    dunn_h, dunn_d = compute_dunn()
    fig_dunn_heatmap(dunn_h, dunn_d)
    report_ns_pairs(dunn_h, dunn_d)
    print("\nDone.")
