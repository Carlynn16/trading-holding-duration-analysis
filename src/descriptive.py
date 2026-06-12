"""Descriptive statistics and non-parametric tests for trade-level data."""

from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


def duration_summary(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    """Aggregate key metrics per duration bin.

    Returns columns: bin, n_trades, pct_of_total, loss_probability,
    avg_profit, median_profit. Ordered by the categorical bin ordering.
    """
    grp = df.groupby(bin_col, observed=True)
    n_total = len(df)

    summary = (
        grp["Profit"]
        .agg(n_trades="count", avg_profit="mean", median_profit="median")
        .reset_index()
        .rename(columns={bin_col: "bin"})
    )
    summary["pct_of_total"] = (summary["n_trades"] / n_total * 100).round(2)
    summary["loss_probability"] = (
        grp["is_loss"].mean().reset_index()["is_loss"].round(4)
    )
    return summary[
        ["bin", "n_trades", "pct_of_total", "loss_probability", "avg_profit", "median_profit"]
    ]


def spearman_duration_profit(df: pd.DataFrame) -> dict:
    """Spearman correlation between holding_duration_hours and Profit.

    Returns rho, p_value, n.
    """
    clean = df[["holding_duration_hours", "Profit"]].dropna()
    rho, p = stats.spearmanr(clean["holding_duration_hours"], clean["Profit"])
    return {"rho": round(float(rho), 6), "p_value": float(p), "n": len(clean)}


def kruskal_by_bin(df: pd.DataFrame, bin_col: str) -> dict:
    """Kruskal-Wallis H test of Profit across duration bins.

    Returns H, p_value, epsilon_squared, k (number of groups), n (total obs).
    eps2 = (H - k + 1) / (n - k)  — bounded [0, 1] effect size.
    """
    groups = [
        g["Profit"].values
        for _, g in df.groupby(bin_col, observed=True)
        if len(g) > 0
    ]
    k = len(groups)
    n = sum(len(g) for g in groups)
    H, p = stats.kruskal(*groups)
    eps2 = (H - k + 1) / (n - k)
    eps2 = max(0.0, min(1.0, eps2))
    return {
        "H": round(float(H), 4),
        "p_value": float(p),
        "epsilon_squared": round(float(eps2), 6),
        "k": k,
        "n": n,
    }


def dunn_posthoc(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    """Pairwise Dunn test with Holm correction across duration bins.

    Returns a symmetric pairwise p-value matrix (DataFrame).
    Diagonal values are 1.0.
    """
    group_data = {
        str(name): g["Profit"].values
        for name, g in df.groupby(bin_col, observed=True)
    }
    labels = list(group_data.keys())
    data_list = [group_data[l] for l in labels]
    result = sp.posthoc_dunn(data_list, p_adjust="holm")
    result.index = labels
    result.columns = labels
    # posthoc_dunn returns a read-only backing array; copy before writing diagonal
    arr = result.to_numpy().copy()
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=labels, columns=labels)


def save_summaries(
    df: pd.DataFrame, out_dir: str = "data/processed"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute and persist all summary tables. Returns (hours_df, days_df, stats_df)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hours_df = duration_summary(df, "duration_bin_hours")
    days_df = duration_summary(df, "duration_bin_days")

    spear = spearman_duration_profit(df)
    kw_h = kruskal_by_bin(df, "duration_bin_hours")
    kw_d = kruskal_by_bin(df, "duration_bin_days")

    stats_df = pd.DataFrame(
        [
            {"test": "spearman_hours_profit", **spear},
            {"test": "kruskal_hours", **{k: v for k, v in kw_h.items()}},
            {"test": "kruskal_days", **{k: v for k, v in kw_d.items()}},
        ]
    )

    hours_df.to_csv(out / "desc_summary_hours.csv", index=False)
    days_df.to_csv(out / "desc_summary_days.csv", index=False)
    stats_df.to_csv(out / "stats_tests.csv", index=False)

    return hours_df, days_df, stats_df
