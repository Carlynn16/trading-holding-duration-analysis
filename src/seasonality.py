"""Seasonality and time-of-day analysis for trade loss rates."""

from pathlib import Path

import numpy as np
import pandas as pd


def loss_rate_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Loss probability and trade count per calendar month (1–12).

    Returns columns: month, loss_rate, n_trades. Ordered 1→12.
    """
    grp = (
        df.groupby("month", sort=True)["is_loss"]
        .agg(loss_rate="mean", n_trades="count")
        .reset_index()
    )
    grp["loss_rate"] = grp["loss_rate"].round(6)
    return grp


def loss_rate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Loss probability and trade count per entry hour-of-day (0–23).

    Returns columns: hour, loss_rate, n_trades. Ordered 0→23.
    """
    grp = (
        df.groupby("hour", sort=True)["is_loss"]
        .agg(loss_rate="mean", n_trades="count")
        .reset_index()
    )
    grp["loss_rate"] = grp["loss_rate"].round(6)
    return grp


def monthly_sharpe(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly Sharpe proxy = mean(Profit) / std(Profit) per calendar month.

    NOTE: this is a per-trade raw-profit proxy, NOT an annualised Sharpe ratio.
    It measures the signal-to-noise ratio of per-trade returns within each month.
    Returns NaN for any month whose Profit standard deviation is zero.

    Returns columns: month, sharpe_proxy, n_trades. Ordered 1→12.
    """
    def _sharpe(x: pd.Series) -> float:
        s = float(x.std())
        return float(x.mean() / s) if s > 0 else np.nan

    grp = (
        df.groupby("month", sort=True)["Profit"]
        .agg(sharpe_proxy=_sharpe, n_trades="count")
        .reset_index()
    )
    grp["sharpe_proxy"] = grp["sharpe_proxy"].round(6)
    return grp


def save_seasonality_tables(
    df: pd.DataFrame,
    out_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute and persist all seasonality tables.

    Returns (month_df, hour_df, sharpe_df).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    month_df = loss_rate_by_month(df)
    hour_df = loss_rate_by_hour(df)
    sharpe_df = monthly_sharpe(df)

    month_df.to_csv(out / "loss_by_month.csv", index=False)
    hour_df.to_csv(out / "loss_by_hour.csv", index=False)
    sharpe_df.to_csv(out / "monthly_sharpe.csv", index=False)

    return month_df, hour_df, sharpe_df
