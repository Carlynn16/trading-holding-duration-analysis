"""Unit tests for src/descriptive.py using synthetic DataFrames."""

import numpy as np
import pandas as pd
import pytest

from src.descriptive import (
    duration_summary,
    dunn_posthoc,
    kruskal_by_bin,
    spearman_duration_profit,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    """30-row synthetic DataFrame with known properties."""
    rng = np.random.default_rng(42)
    n = 30

    hours = np.array([0.5] * 10 + [5.0] * 10 + [30.0] * 10)
    profit = np.concatenate([
        rng.uniform(5, 50, 10),       # short holds — mostly profit
        rng.uniform(-10, 30, 10),     # medium — mixed
        rng.uniform(-50, -5, 10),     # long — mostly loss
    ])

    hour_bins = pd.Categorical(
        ["0-1h"] * 10 + ["3h-6h"] * 10 + [">1 day"] * 10,
        categories=["0-1h", "1h-3h", "3h-6h", "6h-8h", "8h-9h", "9h-24h", ">1 day"],
        ordered=True,
    )
    day_bins = pd.Categorical(
        ["0-1d"] * 10 + ["0-1d"] * 10 + [">7d"] * 10,
        categories=["0-1d", "1-3d", "3-5d", "5-7d", ">7d"],
        ordered=True,
    )

    return pd.DataFrame(
        {
            "holding_duration_hours": hours,
            "Profit": profit,
            "is_loss": (profit < 0).astype(int),
            "duration_bin_hours": hour_bins,
            "duration_bin_days": day_bins,
        }
    )


# ---------------------------------------------------------------------------
# duration_summary
# ---------------------------------------------------------------------------

def test_duration_summary_loss_probability_matches():
    """loss_probability for the '>1 day' bin must equal hand-computed value."""
    df = _make_df()
    s = duration_summary(df, "duration_bin_hours")
    long_row = s[s["bin"] == ">1 day"].iloc[0]
    expected = df[df["duration_bin_hours"] == ">1 day"]["is_loss"].mean()
    assert abs(long_row["loss_probability"] - expected) < 1e-4


def test_duration_summary_pct_of_total_sums_to_100():
    df = _make_df()
    s = duration_summary(df, "duration_bin_hours")
    assert abs(s["pct_of_total"].sum() - 100.0) < 0.05


def test_duration_summary_n_trades_sums_correctly():
    df = _make_df()
    s = duration_summary(df, "duration_bin_hours")
    assert s["n_trades"].sum() == len(df)


# ---------------------------------------------------------------------------
# spearman_duration_profit
# ---------------------------------------------------------------------------

def test_spearman_rho_in_range():
    df = _make_df()
    result = spearman_duration_profit(df)
    assert -1.0 <= result["rho"] <= 1.0


def test_spearman_p_valid():
    df = _make_df()
    result = spearman_duration_profit(df)
    assert 0.0 <= result["p_value"] <= 1.0


def test_spearman_n_correct():
    df = _make_df()
    result = spearman_duration_profit(df)
    assert result["n"] == len(df)


def test_spearman_negative_for_constructed_data():
    """Synthetic data has negative hours-profit relationship by construction."""
    df = _make_df()
    result = spearman_duration_profit(df)
    assert result["rho"] < 0, f"Expected negative rho, got {result['rho']}"


# ---------------------------------------------------------------------------
# kruskal_by_bin
# ---------------------------------------------------------------------------

def test_kruskal_H_nonnegative():
    df = _make_df()
    result = kruskal_by_bin(df, "duration_bin_hours")
    assert result["H"] >= 0


def test_kruskal_epsilon_squared_in_unit_interval():
    df = _make_df()
    result = kruskal_by_bin(df, "duration_bin_hours")
    assert 0.0 <= result["epsilon_squared"] <= 1.0


def test_kruskal_p_valid():
    df = _make_df()
    result = kruskal_by_bin(df, "duration_bin_hours")
    assert 0.0 <= result["p_value"] <= 1.0


def test_kruskal_significant_for_constructed_data():
    """Groups are clearly separated by construction — expect p < 0.05."""
    df = _make_df()
    result = kruskal_by_bin(df, "duration_bin_hours")
    assert result["p_value"] < 0.05


# ---------------------------------------------------------------------------
# dunn_posthoc
# ---------------------------------------------------------------------------

def test_dunn_returns_square_matrix():
    df = _make_df()
    mat = dunn_posthoc(df, "duration_bin_hours")
    assert mat.shape[0] == mat.shape[1]


def test_dunn_diagonal_is_one():
    df = _make_df()
    mat = dunn_posthoc(df, "duration_bin_hours")
    diag = np.diag(mat.values)
    np.testing.assert_allclose(diag, 1.0, atol=1e-10)


def test_dunn_symmetric():
    df = _make_df()
    mat = dunn_posthoc(df, "duration_bin_hours")
    np.testing.assert_allclose(mat.values, mat.values.T, atol=1e-10)


def test_dunn_pvalues_in_unit_interval():
    df = _make_df()
    mat = dunn_posthoc(df, "duration_bin_hours")
    # Off-diagonal values should be valid p-values
    mask = ~np.eye(len(mat), dtype=bool)
    off_diag = mat.values[mask]
    assert (off_diag >= 0).all() and (off_diag <= 1).all()
