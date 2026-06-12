"""Generate all descriptive figures and save to figures/.

Run from repo root:
    python scripts/generate_descriptive_figures.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.descriptive import duration_summary, spearman_duration_profit, save_summaries
from src.plot_style import apply_style, ACCENT, BIN_PALETTE, SAVEFIG_KW, LOSS_COLOR

FIGURES = Path("figures")
PROCESSED = Path("data/processed")


def _load() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "trades_clean.parquet")


# ---------------------------------------------------------------------------
# Fig 1 — Duration distribution (raw hours + log1p)
# ---------------------------------------------------------------------------

def fig_dist_duration(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        df["holding_duration_hours"].clip(upper=df["holding_duration_hours"].quantile(0.99)),
        bins=80, color=ACCENT, edgecolor="white", linewidth=0.3,
    )
    axes[0].set_title("Holding Duration (hours, 99th pct clip)")
    axes[0].set_xlabel("Hours")
    axes[0].set_ylabel("Trade count")

    axes[1].hist(df["log_holding"], bins=80, color=ACCENT, edgecolor="white", linewidth=0.3)
    axes[1].set_title("Holding Duration — log1p(hours)")
    axes[1].set_xlabel("log1p(hours)")
    axes[1].set_ylabel("Trade count")

    fig.suptitle("Distribution of Trade Holding Duration", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_dist_duration.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_dist_duration.png")


# ---------------------------------------------------------------------------
# Fig 2 — Profit distribution (raw + sign-safe log)
# ---------------------------------------------------------------------------

def fig_dist_profit(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    p1, p99 = df["Profit"].quantile(0.01), df["Profit"].quantile(0.99)
    axes[0].hist(
        df["Profit"].clip(lower=p1, upper=p99),
        bins=80, color=ACCENT, edgecolor="white", linewidth=0.3,
    )
    axes[0].axvline(0, color="grey", linestyle="--", linewidth=1)
    axes[0].set_title("Profit per Trade (1–99th pct clip)")
    axes[0].set_xlabel("Profit")
    axes[0].set_ylabel("Trade count")

    axes[1].hist(df["log_profit"], bins=80, color=ACCENT, edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="grey", linestyle="--", linewidth=1)
    axes[1].set_title("Profit — sign-safe log transform")
    axes[1].set_xlabel("sign(P) · log1p(|P|)")
    axes[1].set_ylabel("Trade count")

    fig.suptitle("Distribution of Trade Profit", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_dist_profit.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_dist_profit.png")


# ---------------------------------------------------------------------------
# Fig 3 — Hexbin: Profit vs log holding duration
# ---------------------------------------------------------------------------

def fig_duration_profit_hexbin(df: pd.DataFrame) -> None:
    spear = spearman_duration_profit(df)
    rho, p = spear["rho"], spear["p_value"]

    fig, ax = plt.subplots(figsize=(8, 5))
    p1, p99 = df["Profit"].quantile(0.01), df["Profit"].quantile(0.99)
    hb = ax.hexbin(
        df["log_holding"],
        df["Profit"].clip(lower=p1, upper=p99),
        gridsize=60,
        cmap="crest",
        mincnt=1,
    )
    cb = fig.colorbar(hb, ax=ax, label="Trade count")
    ax.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    p_str = f"p < 0.001" if p < 0.001 else f"p = {p:.4f}"
    ax.annotate(
        f"Spearman ρ = {rho:.3f}  ({p_str})",
        xy=(0.04, 0.93), xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )
    ax.set_xlabel("log1p(holding duration in hours)")
    ax.set_ylabel("Profit (1–99th pct clip)")
    ax.set_title("Profit vs. Holding Duration")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_duration_profit_hexbin.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_duration_profit_hexbin.png")


# ---------------------------------------------------------------------------
# Fig 4 & 5 — Loss probability horizontal barplots
# ---------------------------------------------------------------------------

def _loss_prob_barplot(summary: pd.DataFrame, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [LOSS_COLOR if lp >= 0.35 else ACCENT for lp in summary["loss_probability"]]
    bars = ax.barh(summary["bin"].astype(str), summary["loss_probability"] * 100, color=colors)

    for bar, val in zip(bars, summary["loss_probability"]):
        ax.text(
            val * 100 + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9,
        )

    ax.axvline(50, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Loss probability (%)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, 60)
    plt.tight_layout()
    fig.savefig(FIGURES / fname, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  saved {fname}")


def fig_lossprob_hours(df: pd.DataFrame) -> None:
    s = duration_summary(df, "duration_bin_hours")
    _loss_prob_barplot(s, "Loss Probability by Holding Duration (hour bins)", "fig_lossprob_hours.png")


def fig_lossprob_days(df: pd.DataFrame) -> None:
    s = duration_summary(df, "duration_bin_days")
    _loss_prob_barplot(s, "Loss Probability by Holding Duration (day bins)", "fig_lossprob_days.png")


# ---------------------------------------------------------------------------
# Fig 6 & 7 — Profit boxplots (5–95 profit percentile filter)
# ---------------------------------------------------------------------------

def _profit_boxplot(df: pd.DataFrame, bin_col: str, title: str, fname: str) -> None:
    p5, p95 = df["Profit"].quantile(0.05), df["Profit"].quantile(0.95)
    plot_df = df[(df["Profit"] >= p5) & (df["Profit"] <= p95)]

    bin_order = plot_df[bin_col].cat.categories.tolist()
    palette = {b: c for b, c in zip(bin_order, BIN_PALETTE)}

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(
        data=plot_df, x=bin_col, y="Profit", order=bin_order,
        hue=bin_col, hue_order=bin_order, palette=palette,
        showfliers=False, legend=False, ax=ax,
        medianprops=dict(color="white", linewidth=2),
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Duration bin")
    ax.set_ylabel("Profit (5–95th pct)")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(FIGURES / fname, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  saved {fname}")


def fig_profit_box_hours(df: pd.DataFrame) -> None:
    _profit_boxplot(df, "duration_bin_hours", "Profit by Holding Duration (hour bins)", "fig_profit_box_hours.png")


def fig_profit_box_days(df: pd.DataFrame) -> None:
    _profit_boxplot(df, "duration_bin_days", "Profit by Holding Duration (day bins)", "fig_profit_box_days.png")


# ---------------------------------------------------------------------------
# Fig 8 — 3-panel metrics by hour bin
# ---------------------------------------------------------------------------

def fig_metrics_by_hours(df: pd.DataFrame) -> None:
    s = duration_summary(df, "duration_bin_hours")
    bins = s["bin"].astype(str).tolist()
    colors = BIN_PALETTE[: len(bins)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1 — loss probability
    axes[0].bar(bins, s["loss_probability"] * 100, color=colors)
    axes[0].set_title("Loss Probability (%)")
    axes[0].set_xlabel("Duration bin")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[0].tick_params(axis="x", rotation=30)

    # Panel 2 — avg profit
    axes[1].bar(bins, s["avg_profit"], color=colors)
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=1)
    axes[1].set_title("Average Profit")
    axes[1].set_xlabel("Duration bin")
    axes[1].tick_params(axis="x", rotation=30)

    # Panel 3 — median profit
    axes[2].bar(bins, s["median_profit"], color=colors)
    axes[2].axhline(0, color="grey", linestyle="--", linewidth=1)
    axes[2].set_title("Median Profit")
    axes[2].set_xlabel("Duration bin")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle("Trade Metrics by Holding Duration (hour bins)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_metrics_by_hours.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_metrics_by_hours.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    apply_style()
    print("Loading clean trades…")
    df = _load()

    print("Computing summaries and saving CSVs…")
    hours_df, days_df, stats_df = save_summaries(df)

    print("Generating figures…")
    fig_dist_duration(df)
    fig_dist_profit(df)
    fig_duration_profit_hexbin(df)
    fig_lossprob_hours(df)
    fig_lossprob_days(df)
    fig_profit_box_hours(df)
    fig_profit_box_days(df)
    fig_metrics_by_hours(df)

    print("\nHours summary:")
    print(hours_df.to_string(index=False))
    print("\nDays summary:")
    print(days_df.to_string(index=False))
    print("\nStats tests:")
    print(stats_df.to_string(index=False))
    print("\nAll done.")
