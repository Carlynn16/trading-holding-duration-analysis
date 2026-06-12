"""Unit tests for src/preprocessing.py using synthetic DataFrames."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean_trades, clean_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(**overrides) -> pd.DataFrame:
    """Minimal valid trades DataFrame (5 rows unless overridden)."""
    base = {
        "Time": pd.to_datetime(
            ["2023-01-01 08:00", "2023-01-02 09:00", "2023-01-03 10:00",
             "2023-01-04 11:00", "2023-01-05 12:00"]
        ),
        "Time.1": pd.to_datetime(
            ["2023-01-01 09:00", "2023-01-02 12:00", "2023-01-03 16:00",
             "2023-01-05 11:00", "2023-01-07 12:00"]
        ),
        "Volume": [1.0, 0.5, 2.0, 1.5, 1.0],
        "Symbol": ["EURUSD", "GBPUSD", "XAUUSD", "EURUSD", "AUDUSD"],
        "Price": [1.05, 1.25, 1800.0, 1.06, 0.67],
        "Price.1": [1.06, 1.24, 1810.0, 1.04, 0.68],
        "Profit": [100.0, -50.0, 200.0, -150.0, 75.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_signals(**overrides) -> pd.DataFrame:
    base = {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "title": ["Strat Alpha", "Strat Beta", "Strat Gamma"],
        "Growth": [12.5, -3.2, 8.1],
        "Trades": [200, 150, 300],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# holding_duration_hours — correctness
# ---------------------------------------------------------------------------


def test_holding_duration_hours_known_values():
    """Manually verified durations from synthetic timestamps."""
    df = _make_trades()
    result = clean_trades(df)
    expected_hours = [1.0, 3.0, 6.0, 24.0, 48.0]
    np.testing.assert_allclose(
        result["holding_duration_hours"].values, expected_hours, rtol=1e-6
    )


def test_holding_duration_hours_never_negative():
    """After cleaning, no negative holding durations remain."""
    df = _make_trades()
    # Inject a row where close is before open
    bad_row = {
        "Time": pd.Timestamp("2023-01-10 10:00"),
        "Time.1": pd.Timestamp("2023-01-10 08:00"),  # 2 h before open
        "Volume": 1.0,
        "Symbol": "EURUSD",
        "Price": 1.05,
        "Price.1": 1.06,
        "Profit": 10.0,
    }
    df = pd.concat([df, pd.DataFrame([bad_row])], ignore_index=True)
    result = clean_trades(df)
    assert (result["holding_duration_hours"] >= 0).all()


# ---------------------------------------------------------------------------
# is_loss — correctness
# ---------------------------------------------------------------------------


def test_is_loss_binary():
    """is_loss must contain only 0 and 1."""
    result = clean_trades(_make_trades())
    assert set(result["is_loss"].unique()).issubset({0, 1})


def test_is_loss_matches_profit_sign():
    """is_loss == 1 iff Profit < 0."""
    result = clean_trades(_make_trades())
    assert (result["is_loss"] == (result["Profit"] < 0).astype(int)).all()


# ---------------------------------------------------------------------------
# Duration bins — boundary values
# ---------------------------------------------------------------------------

_HOUR_BIN_CASES = [
    # (holding_hours, expected_label)
    (0.0, "0-1h"),
    (0.5, "0-1h"),
    (1.0, "1h-3h"),   # boundary: 1h is start of [1,3)
    (2.9, "1h-3h"),
    (3.0, "3h-6h"),   # boundary
    (5.9, "3h-6h"),
    (6.0, "6h-8h"),
    (8.0, "8h-9h"),
    (9.0, "9h-24h"),
    (23.9, "9h-24h"),
    (24.0, ">1 day"),  # boundary
    (100.0, ">1 day"),
]


@pytest.mark.parametrize("hours, expected_label", _HOUR_BIN_CASES)
def test_duration_bin_hours_boundaries(hours, expected_label):
    """Bin assignment is correct at all boundary values."""
    open_ts = pd.Timestamp("2023-06-01 00:00")
    close_ts = open_ts + pd.Timedelta(hours=hours)
    df = pd.DataFrame({
        "Time": [open_ts],
        "Time.1": [close_ts],
        "Volume": [1.0],
        "Symbol": ["EURUSD"],
        "Price": [1.05],
        "Price.1": [1.06],
        "Profit": [10.0],
    })
    result = clean_trades(df)
    assert result["duration_bin_hours"].iloc[0] == expected_label, (
        f"hours={hours}: got {result['duration_bin_hours'].iloc[0]!r}, "
        f"expected {expected_label!r}"
    )


# ---------------------------------------------------------------------------
# No NaN in key columns after cleaning
# ---------------------------------------------------------------------------


def test_no_nan_in_key_columns():
    """Key columns must be free of NaN after clean_trades."""
    key_cols = ["Volume", "Symbol", "Price", "Profit", "holding_duration_hours"]
    result = clean_trades(_make_trades())
    for col in key_cols:
        assert result[col].notna().all(), f"NaN found in column: {col}"


def test_missing_required_columns_drops_row():
    """Rows with NaN in required fields are dropped."""
    df = _make_trades()
    df.loc[0, "Volume"] = np.nan
    result = clean_trades(df)
    assert len(result) == len(df) - 1


# ---------------------------------------------------------------------------
# clean_signals — PII removal and anonymization
# ---------------------------------------------------------------------------


def test_clean_signals_removes_name_column():
    result = clean_signals(_make_signals())
    assert "name" not in result.columns


def test_clean_signals_removes_id_column():
    result = clean_signals(_make_signals())
    assert "id" not in result.columns


def test_clean_signals_anonymizes_title():
    result = clean_signals(_make_signals())
    assert "title" in result.columns
    titles = result["title"].tolist()
    assert titles == ["signal_001", "signal_002", "signal_003"]


def test_clean_signals_preserves_numeric_columns():
    result = clean_signals(_make_signals())
    assert "Growth" in result.columns
    assert "Trades" in result.columns


def test_clean_signals_no_real_names_in_title():
    """Title column must not contain any of the original real-world strategy names."""
    original_titles = ["Strat Alpha", "Strat Beta", "Strat Gamma"]
    result = clean_signals(_make_signals())
    for t in original_titles:
        assert t not in result["title"].values
