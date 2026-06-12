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
from src.plot_style import apply_style, ACCENT, BIN_PALETTE, SAVEFIG_KW, LOSS_COLOR, PROFIT_COLOR
from src.seasonality import save_seasonality_tables

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

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
        gridsize=40,
        bins="log",
        cmap="crest",
        mincnt=1,
    )
    cb = fig.colorbar(hb, ax=ax, label="Trade count (log scale)")
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
    p5, p95   = df["Profit"].quantile(0.05), df["Profit"].quantile(0.95)
    plot_df   = df[(df["Profit"] >= p5) & (df["Profit"] <= p95)]
    bin_order = plot_df[bin_col].cat.categories.tolist()
    palette   = {b: c for b, c in zip(bin_order, BIN_PALETTE)}

    # Compute whisker extents per bin for y-axis zoom
    all_lo, all_hi = [], []
    for b in bin_order:
        grp = plot_df.loc[plot_df[bin_col] == b, "Profit"]
        if len(grp) < 4:
            continue
        q1, q3 = grp.quantile(0.25), grp.quantile(0.75)
        iqr    = q3 - q1
        all_lo.append(max(float(grp.min()), q1 - 1.5 * iqr))
        all_hi.append(min(float(grp.max()), q3 + 1.5 * iqr))

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=plot_df, x=bin_col, y="Profit", order=bin_order,
        hue=bin_col, hue_order=bin_order, palette=palette,
        showfliers=False, width=0.6, legend=False, ax=ax,
        medianprops=dict(color="white", linewidth=2),
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.8)

    if all_lo and all_hi:
        y_lo, y_hi = min(all_lo), max(all_hi)
        margin     = (y_hi - y_lo) * 0.15
        ax.set_ylim(y_lo - margin, y_hi + margin)
        print(f"    {fname}: y-limits [{y_lo - margin:.2f}, {y_hi + margin:.2f}]")

    ax.set_xlabel("Duration bin")
    ax.set_ylabel("Profit (5–95th pct, zoomed to whisker region)")
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

    # Shortest bin at top → longest at bottom; reverse so top = index 0 in barh
    s = s.iloc[::-1].reset_index(drop=True)
    bins   = s["bin"].astype(str).tolist()
    colors = list(reversed(BIN_PALETTE[: len(bins)]))
    y      = range(len(bins))

    fig, axes = plt.subplots(1, 3, figsize=(14, max(3.5, len(bins) * 0.55)),
                             sharey=True)

    def _annotate_h(ax, values, fmt):
        for i, v in enumerate(values):
            sign   = 1 if v >= 0 else -1
            offset = sign * abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
            ha     = "left" if v >= 0 else "right"
            ax.text(v + offset, i, fmt(v), va="center", ha=ha, fontsize=8)

    # Panel 1 — loss probability
    vals0 = (s["loss_probability"] * 100).tolist()
    axes[0].barh(y, vals0, color=colors)
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter())
    axes[0].set_title("Loss Probability", fontsize=11)
    axes[0].set_xlabel("%")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(bins)
    axes[0].invert_yaxis()           # shortest bin at top
    axes[0].set_xlim(0, max(vals0) * 1.18)
    _annotate_h(axes[0], vals0, lambda v: f"{v:.1f}%")

    # Panel 2 — average profit
    vals1 = s["avg_profit"].tolist()
    axes[1].barh(y, vals1, color=colors)
    axes[1].axvline(0, color="grey", linestyle="--", linewidth=1)
    axes[1].set_title("Average Profit", fontsize=11)
    axes[1].set_xlabel("Profit")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    span1  = max(abs(min(vals1)), abs(max(vals1)))
    axes[1].set_xlim(-span1 * 1.30, span1 * 1.30)
    _annotate_h(axes[1], vals1, lambda v: f"{v:,.0f}")

    # Panel 3 — median profit
    vals2 = s["median_profit"].tolist()
    axes[2].barh(y, vals2, color=colors)
    axes[2].axvline(0, color="grey", linestyle="--", linewidth=1)
    axes[2].set_title("Median Profit", fontsize=11)
    axes[2].set_xlabel("Profit")
    axes[2].tick_params(axis="y", left=False, labelleft=False)
    span2  = max(abs(min(vals2)), abs(max(vals2)))
    axes[2].set_xlim(-span2 * 1.30, span2 * 1.30)
    _annotate_h(axes[2], vals2, lambda v: f"{v:,.2f}")

    fig.suptitle("Trade Metrics by Holding Duration (hour bins)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_metrics_by_hours.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_metrics_by_hours.png")


# ---------------------------------------------------------------------------
# Fig 9 — Loss probability by calendar month
# ---------------------------------------------------------------------------

def fig_loss_by_month(df: pd.DataFrame) -> None:
    from src.seasonality import loss_rate_by_month
    month_df = loss_rate_by_month(df)
    base_rate = float(df["is_loss"].mean())

    months = month_df["month"].tolist()
    rates  = (month_df["loss_rate"] * 100).tolist()
    labels = [MONTH_LABELS[m - 1] for m in months]
    colors = [LOSS_COLOR if r > base_rate * 100 else ACCENT for r in rates]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(base_rate * 100, color="grey", linestyle="--", linewidth=1.5,
               label=f"Overall base rate ({base_rate:.1%})")
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{r:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Entry month")
    ax.set_ylabel("Loss probability (%)")
    ax.set_title("Loss Probability by Calendar Month")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_loss_by_month.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_loss_by_month.png")


# ---------------------------------------------------------------------------
# Fig 10 — Loss probability by entry hour of day
# ---------------------------------------------------------------------------

def fig_loss_by_hour(df: pd.DataFrame) -> None:
    from src.seasonality import loss_rate_by_hour
    hour_df = loss_rate_by_hour(df)
    base_rate = float(df["is_loss"].mean())

    hours  = hour_df["hour"].tolist()
    rates  = (hour_df["loss_rate"] * 100).tolist()
    colors = [LOSS_COLOR if r > base_rate * 100 else ACCENT for r in rates]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hours, rates, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(base_rate * 100, color="grey", linestyle="--", linewidth=1.5,
               label=f"Overall base rate ({base_rate:.1%})")
    ax.set_xlabel("Entry hour of day (server time)")
    ax.set_ylabel("Loss probability (%)")
    ax.set_title("Loss Probability by Entry Hour of Day")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(hours)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_loss_by_hour.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_loss_by_hour.png")


# ---------------------------------------------------------------------------
# Fig 11 — Monthly Sharpe proxy
# ---------------------------------------------------------------------------

def fig_monthly_sharpe(df: pd.DataFrame) -> None:
    from src.seasonality import monthly_sharpe
    sharpe_df = monthly_sharpe(df)

    months = sharpe_df["month"].tolist()
    values = sharpe_df["sharpe_proxy"].tolist()
    labels = [MONTH_LABELS[m - 1] for m in months]
    colors = [PROFIT_COLOR if v > 0 else LOSS_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    for bar, v in zip(bars, values):
        offset = 0.002 if v >= 0 else -0.002
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:.3f}", ha="center", va=va, fontsize=8)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Sharpe proxy (mean / std of per-trade profit)")
    ax.set_title(
        "Monthly Sharpe Proxy\n"
        "(mean / std of raw per-trade profit — not annualised)"
    )
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_monthly_sharpe.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_monthly_sharpe.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    apply_style()
    print("Loading clean trades…")
    df = _load()

    print("Computing summaries and saving CSVs…")
    hours_df, days_df, stats_df = save_summaries(df)
    month_df, hour_df, sharpe_df = save_seasonality_tables(df)

    print("Generating figures…")
    fig_dist_duration(df)
    fig_dist_profit(df)
    fig_duration_profit_hexbin(df)
    fig_lossprob_hours(df)
    fig_lossprob_days(df)
    fig_profit_box_hours(df)
    fig_profit_box_days(df)
    fig_metrics_by_hours(df)
    fig_monthly_sharpe(df)

    print("\nHours summary:")
    print(hours_df.to_string(index=False))
    print("\nDays summary:")
    print(days_df.to_string(index=False))
    print("\nStats tests:")
    print(stats_df.to_string(index=False))
    print("\nAll done.")
