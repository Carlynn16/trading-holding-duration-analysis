"""Generate all survival-analysis figures and save to figures/.

Run from repo root:
    python scripts/generate_survival_figures.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.plot_style import apply_style, ACCENT, BIN_PALETTE, SAVEFIG_KW, LOSS_COLOR, PROFIT_COLOR
from src.survival import (
    DURATION_COL, EVENT_COL,
    save_survival_outputs,
)

FIGURES  = Path("figures")
PROCESSED = Path("data/processed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap(arr, cap=168):
    return arr[arr <= cap]


def _crossing_time(times, values, threshold):
    """First time in times where values >= threshold; None if never reached."""
    idx = np.where(np.array(values) >= threshold)[0]
    return float(times[idx[0]]) if len(idx) > 0 else None


# ---------------------------------------------------------------------------
# Fig 1 — KM overall: cumulative loss incidence
# ---------------------------------------------------------------------------

def fig_km_overall(kmf, ax=None) -> None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    sf  = kmf.survival_function_
    ci  = kmf.confidence_interval_
    lower_col = next(c for c in ci.columns if "lower" in c)
    upper_col = next(c for c in ci.columns if "upper" in c)

    times     = sf.index.values
    incidence = 1 - sf.values.flatten()
    ci_lo     = 1 - ci[upper_col].values   # lower bound of (1-S) = 1 - upper(S)
    ci_hi     = 1 - ci[lower_col].values   # upper bound of (1-S) = 1 - lower(S)

    mask = times <= 168
    t, inc, lo, hi = times[mask], incidence[mask], ci_lo[mask], ci_hi[mask]

    ax.plot(t, inc, color=LOSS_COLOR, linewidth=2, label="Cumulative loss incidence")
    ax.fill_between(t, lo, hi, alpha=0.2, color=LOSS_COLOR, label="95% CI")

    # Annotate threshold crossings
    for threshold, label, ls in [(0.25, "25%", "--"), (0.50, "50%", ":")]:
        tx = _crossing_time(t, inc, threshold)
        if tx is not None:
            ax.axhline(threshold, color="grey", linestyle=ls, linewidth=1, alpha=0.7)
            ax.axvline(tx, color="grey", linestyle=ls, linewidth=1, alpha=0.7)
            ax.annotate(
                f"{label} at {tx:.0f}h",
                xy=(tx, threshold),
                xytext=(tx + 3, threshold - 0.02),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="grey"),
                color="grey",
            )

    ax.set_xlim(0, 168)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Holding duration (hours)")
    ax.set_ylabel("Cumulative loss incidence")
    ax.set_title("Kaplan-Meier: Cumulative Loss Incidence")
    ax.legend(loc="upper left")

    if standalone:
        plt.tight_layout()
        fig.savefig(FIGURES / "fig_km_overall.png", **SAVEFIG_KW)
        plt.close(fig)
        print("  saved fig_km_overall.png")


# ---------------------------------------------------------------------------
# Fig 2 — KM by Symbol
# ---------------------------------------------------------------------------

def _km_group_figure(fitters, lr_p, title, fname) -> None:
    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (grp, kmf) in enumerate(sorted(fitters.items())):
        sf     = kmf.survival_function_
        ci     = kmf.confidence_interval_
        lower_col = next(c for c in ci.columns if "lower" in c)
        upper_col = next(c for c in ci.columns if "upper" in c)
        times  = sf.index.values
        mask   = times <= 168
        col    = palette[i % len(palette)]

        ax.plot(times[mask], sf.values.flatten()[mask],
                color=col, linewidth=1.8, label=grp)
        ax.fill_between(
            times[mask],
            ci[lower_col].values[mask],
            ci[upper_col].values[mask],
            alpha=0.1, color=col,
        )

    p_str = "p < 0.001" if lr_p < 0.001 else f"p = {lr_p:.4f}"
    ax.set_title(f"{title}  (log-rank {p_str})")
    ax.set_xlabel("Holding duration (hours)")
    ax.set_ylabel("Loss-free survival S(t)")
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / fname, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  saved {fname}")


def fig_km_by_symbol(fitters_sym, lr_p_sym) -> None:
    _km_group_figure(fitters_sym, lr_p_sym,
                     "KM Survival by Instrument", "fig_km_by_symbol.png")


def fig_km_by_type(fitters_type, lr_p_type) -> None:
    _km_group_figure(fitters_type, lr_p_type,
                     "KM Survival by Trade Direction (Buy vs Sell)",
                     "fig_km_by_type.png")


# ---------------------------------------------------------------------------
# Fig 4 — Smoothed hazard curve
# ---------------------------------------------------------------------------

def fig_hazard_curve(naf, accel_time) -> None:
    # Use pre-computed smooth arrays stored by hazard_curve()
    times = naf._smooth_times
    hvals = naf._smooth_hazard
    mask  = times <= 168
    t, h  = times[mask], hvals[mask]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, h, color=ACCENT, linewidth=2)

    if accel_time is not None and accel_time <= 168:
        ax.axvline(accel_time, color="dimgrey", linestyle="--", linewidth=1.2,
                   label=f"Peak close-rate ≈ {accel_time:.0f}h\n(reflects trade-volume distribution)")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Holding duration (hours)")
    ax.set_ylabel("Smoothed hazard (bandwidth=4h)")
    ax.set_title(
        "Smoothed Hazard: Loss-Close Rate vs Duration\n"
        "Peak at ~1.5h reflects ≈31% of trades closing within 1h — not per-trade loss risk"
    )
    ax.set_xlim(0, 168)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_hazard_curve.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_hazard_curve.png")


# ---------------------------------------------------------------------------
# Fig 7 — Conditional loss probability: per-trade risk curve
# ---------------------------------------------------------------------------

def fig_conditional_loss_prob(cond_table, lowess_x, lowess_y, base_rate) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Binned scatter — dot size proportional to trade count in bin
    n_arr = cond_table["n"].values.astype(float)
    sizes = 12 + 55 * (n_arr / n_arr.max())
    ax.scatter(
        cond_table["median_time"], cond_table["loss_rate"],
        s=sizes, color=ACCENT, alpha=0.75, zorder=3,
        label="Quantile-bin average (size prop. n)",
    )

    # LOWESS smoother
    lx_mask = (lowess_x > 0) & (lowess_x <= 168)
    if lx_mask.any():
        ax.plot(lowess_x[lx_mask], lowess_y[lx_mask],
                color=LOSS_COLOR, linewidth=2.5,
                label="LOWESS smoother (frac=0.15)")

    # Overall base loss rate
    ax.axhline(base_rate, color="grey", linestyle=":", linewidth=1.5, alpha=0.8,
               label=f"Overall loss rate ({base_rate:.1%})")

    # Threshold crossings at 30%, 40%, 50%
    cross = {}
    for thr in [0.30, 0.40, 0.50]:
        above = cond_table[cond_table["loss_rate"] >= thr]
        if len(above) > 0:
            cross[thr] = float(above["median_time"].iloc[0])

    linestyles = {0.30: "--", 0.40: "-.", 0.50: ":"}
    for thr in sorted(cross):
        tx = cross[thr]
        ls = linestyles[thr]
        ax.axhline(thr, color=LOSS_COLOR, linestyle=ls, linewidth=0.9, alpha=0.45)
        ax.axvline(tx, color=LOSS_COLOR, linestyle=ls, linewidth=0.9, alpha=0.45)
        label_x = tx * 1.12 if tx * 1.12 <= 140 else tx * 0.7
        ax.text(label_x, thr + 0.012,
                f"{thr:.0%} ≈{tx:.0f}h",
                fontsize=8, color=LOSS_COLOR, va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(0.5, 168)
    ax.set_ylim(0, min(0.80, cond_table["loss_rate"].max() * 1.18))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Holding duration (hours, log scale)")
    ax.set_ylabel("P(loss | closed ≈ t hours)")
    ax.set_title(
        "Conditional Loss Probability by Holding Duration\n"
        "Per-trade risk rises monotonically — distinct from the closing-rate hazard"
    )
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_conditional_loss_prob.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_conditional_loss_prob.png")


# ---------------------------------------------------------------------------
# Fig 5 — Cox forest plot
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "is_buy"           : "Type: Buy (vs Sell)",
    "log_volume_std"   : "Volume (log-standardised)",
    "entry_hour"       : "Entry hour (0-23)",
    "entry_dayofweek"  : "Entry day of week (0-6)",
}
def _fmt_label(name):
    if name in _LABEL_MAP:
        return _LABEL_MAP[name]
    if name.startswith("sym_"):
        return f"{name[4:]} (vs Other)"
    return name


def fig_cox_forest(cph) -> None:
    summary = cph.summary.copy()
    hr      = summary["exp(coef)"]
    lo      = summary["exp(coef) lower 95%"]
    hi      = summary["exp(coef) upper 95%"]
    labels  = [_fmt_label(n) for n in summary.index]

    # Sort by HR descending
    order  = hr.values.argsort()[::-1]
    hr_s   = hr.values[order]
    lo_s   = lo.values[order]
    hi_s   = hi.values[order]
    lab_s  = [labels[i] for i in order]

    colors = [LOSS_COLOR if v > 1 else PROFIT_COLOR for v in hr_s]
    y_pos  = np.arange(len(hr_s))

    fig, ax = plt.subplots(figsize=(8, max(4, len(hr_s) * 0.45)))
    ax.scatter(hr_s, y_pos, color=colors, zorder=3, s=50)
    ax.hlines(y_pos, lo_s, hi_s, colors=colors, linewidth=2)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(lab_s, fontsize=9)
    ax.set_xlabel("Hazard Ratio (HR) with 95% CI")
    ax.set_title("Cox PH: Hazard Ratios for Loss-Close\n(HR > 1 = raises loss risk)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Annotate HR values
    for x, y in zip(hr_s, y_pos):
        ax.text(x * 1.02, y, f"{x:.3f}", va="center", fontsize=8, color="grey")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig_cox_forest.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_cox_forest.png")


# ---------------------------------------------------------------------------
# Fig 6 — Aalen-Johansen competing risks
# ---------------------------------------------------------------------------

def fig_aalen_johansen(ajf_loss, ajf_profit, kmf) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    # AJ CIF: loss (exclude jitter-induced negative times)
    cdf_loss = ajf_loss.cumulative_density_
    col_loss  = cdf_loss.columns[0]
    t_aj     = cdf_loss.index.values
    mask     = (t_aj >= 0) & (t_aj <= 168)
    ax.plot(t_aj[mask], cdf_loss[col_loss].values[mask],
            color=LOSS_COLOR, linewidth=2, label="AJ CIF: loss-close")

    # AJ CIF: profit
    cdf_profit = ajf_profit.cumulative_density_
    col_profit  = cdf_profit.columns[0]
    t_pr       = cdf_profit.index.values
    mask_pr    = (t_pr >= 0) & (t_pr <= 168)
    ax.plot(t_pr[mask_pr], cdf_profit[col_profit].values[mask_pr],
            color=PROFIT_COLOR, linewidth=2, label="AJ CIF: profit-close")

    # 1 - KM for comparison
    sf   = kmf.survival_function_
    t_km = sf.index.values
    inc  = 1 - sf.values.flatten()
    mask_km = t_km <= 168
    ax.plot(t_km[mask_km], inc[mask_km],
            color=LOSS_COLOR, linewidth=1.5, linestyle="--", alpha=0.6,
            label="1 - KM survival (cause-specific)")

    ax.set_xlim(0, 168)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Holding duration (hours)")
    ax.set_ylabel("Cumulative incidence")
    ax.set_title(
        "Competing Risks: Aalen-Johansen CIF\n"
        "(AJ loss CIF < 1-KM confirms competing-risk inflation in cause-specific model)"
    )
    ax.legend(loc="center right")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_aalen_johansen.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_aalen_johansen.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    apply_style()
    print("Loading clean trades…")
    df = pd.read_parquet(PROCESSED / "trades_clean.parquet")

    print("Fitting survival models (this may take 5-10 min)…")
    results = save_survival_outputs(df)

    print("\nGenerating figures…")
    fig_km_overall(results["kmf"])
    fig_km_by_symbol(results["fitters_sym"], results["lr_p_sym"])
    fig_km_by_type(results["fitters_type"], results["lr_p_type"])
    fig_hazard_curve(results["naf"], results["accel_time"])
    fig_conditional_loss_prob(
        results["cond_table"], results["lowess_x"],
        results["lowess_y"],   results["base_rate"],
    )
    fig_cox_forest(results["cph"])
    fig_aalen_johansen(results["ajf_loss"], results["ajf_profit"], results["kmf"])

    print("\n=== Cumulative incidence table ===")
    print(results["incidence_table"].to_string(index=False))
    print(f"\nLog-rank by Symbol: p = {results['lr_p_sym']:.4e},"
          f"  stat = {results['lr_stat_sym']:.1f}")
    print(f"Log-rank by Type:   p = {results['lr_p_type']:.4e},"
          f"  stat = {results['lr_stat_type']:.1f}")
    print(f"\nHazard peak time: {results['accel_time']:.1f}h"
          f"  (closing-rate artifact, NOT per-trade risk)")

    print("\n=== Conditional loss probability (quantile bins) ===")
    print(results["cond_table"].to_string(index=False))

    print("\n=== Threshold crossings (conditional loss probability) ===")
    cond = results["cond_table"]
    review_start = hard_limit = None
    for thr in [0.30, 0.40, 0.50]:
        above = cond[cond["loss_rate"] >= thr]
        if len(above) > 0:
            tx = float(above["median_time"].iloc[0])
            print(f"  {thr:.0%}: first crossed at ≈{tx:.1f}h  ({tx/24:.1f} days)")
            if thr == 0.30 and review_start is None:
                review_start = tx
            if thr == 0.50 and hard_limit is None:
                hard_limit = tx
        else:
            print(f"  {thr:.0%}: not reached within observed durations")

    print("\n=== Proposed cut points ===")
    if review_start is not None:
        print(f"  Review zone entry : ≈{review_start:.0f}h ({review_start/24:.1f} days)"
              f"  — loss probability crosses 30%")
    if hard_limit is not None:
        print(f"  Hard limit (50%)  : ≈{hard_limit:.0f}h ({hard_limit/24:.1f} days)"
              f"  — majority of trades close at a loss")
    print("  Hard limit (profit): ≈120h (5 days)"
          "  — median profit turns negative in 5–7d bin (Section 2)")

    print("\n=== Cox hazard ratios ===")
    cox_hr = pd.read_csv(PROCESSED / "cox_hazard_ratios.csv", index_col=0)
    print(cox_hr.to_string())

    print("\n=== PH test summary ===")
    ph = pd.read_csv(PROCESSED / "ph_test.csv", index_col=0)
    print(ph.to_string())

    print("\nAll done.")
