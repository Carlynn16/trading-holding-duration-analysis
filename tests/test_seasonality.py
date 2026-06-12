"""Unit tests for src/seasonality.py — synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from src.seasonality import loss_rate_by_month, loss_rate_by_hour, monthly_sharpe


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _make_seasonality_df(seed: int = 0) -> pd.DataFrame:
    """240-row DataFrame with all 12 months (20 each) and all 24 hours (10 each)."""
    rng = np.random.default_rng(seed)
    months = np.tile(np.arange(1, 13), 20)   # 240 values, each month exactly 20 times
    hours  = np.tile(np.arange(24),    10)    # 240 values, each hour exactly 10 times
    profit = rng.normal(1.0, 5.0, 240)
    return pd.DataFrame({
        "month"  : months,
        "hour"   : hours,
        "Profit" : profit,
        "is_loss": (profit < 0).astype(int),
    })


# ---------------------------------------------------------------------------
# loss_rate_by_month
# ---------------------------------------------------------------------------

def test_loss_rate_by_month_has_twelve_rows():
    df = _make_seasonality_df()
    out = loss_rate_by_month(df)
    assert len(out) == 12, f"Expected 12 rows, got {len(out)}"


def test_loss_rate_by_month_covers_all_months():
    df = _make_seasonality_df()
    out = loss_rate_by_month(df)
    assert set(out["month"]) == set(range(1, 13))


def test_loss_rate_by_month_rates_in_unit_interval():
    df = _make_seasonality_df()
    out = loss_rate_by_month(df)
    assert (out["loss_rate"] >= 0).all() and (out["loss_rate"] <= 1).all()


def test_loss_rate_by_month_n_trades_positive():
    df = _make_seasonality_df()
    out = loss_rate_by_month(df)
    assert (out["n_trades"] > 0).all()


# ---------------------------------------------------------------------------
# loss_rate_by_hour
# ---------------------------------------------------------------------------

def test_loss_rate_by_hour_has_twentyfour_rows():
    df = _make_seasonality_df()
    out = loss_rate_by_hour(df)
    assert len(out) == 24, f"Expected 24 rows, got {len(out)}"


def test_loss_rate_by_hour_covers_all_hours():
    df = _make_seasonality_df()
    out = loss_rate_by_hour(df)
    assert set(out["hour"]) == set(range(24))


def test_loss_rate_by_hour_rates_in_unit_interval():
    df = _make_seasonality_df()
    out = loss_rate_by_hour(df)
    assert (out["loss_rate"] >= 0).all() and (out["loss_rate"] <= 1).all()


# ---------------------------------------------------------------------------
# monthly_sharpe
# ---------------------------------------------------------------------------

def test_monthly_sharpe_has_twelve_rows():
    df = _make_seasonality_df()
    out = monthly_sharpe(df)
    assert len(out) == 12, f"Expected 12 rows, got {len(out)}"


def test_monthly_sharpe_values_finite():
    df = _make_seasonality_df()
    out = monthly_sharpe(df)
    assert np.isfinite(out["sharpe_proxy"]).all(), \
        f"Non-finite Sharpe values: {out['sharpe_proxy'].tolist()}"


def test_monthly_sharpe_zero_std_returns_nan():
    """If all profits in a month are identical, sharpe_proxy should be NaN."""
    df = _make_seasonality_df()
    df.loc[df["month"] == 6, "Profit"] = 1.0   # month 6 → constant profit, std = 0
    out = monthly_sharpe(df)
    row = out.loc[out["month"] == 6, "sharpe_proxy"]
    assert row.isna().all(), "Expected NaN for month with zero std"
