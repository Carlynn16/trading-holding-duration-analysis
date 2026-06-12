"""Unit tests for src/modeling.py — synthetic data only, no real CSVs."""

import numpy as np
import pandas as pd
import pytest

from src.modeling import FORBIDDEN_COLS, build_features, leakage_check, train_compare


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _make_model_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "is_loss"               : rng.binomial(1, 0.27, n),
        "Type"                  : rng.choice(["Buy", "Sell"], n),
        "Volume"                : rng.exponential(1.0, n) + 0.01,
        "Symbol"                : rng.choice(
            ["EURUSD", "XAUUSD", "GBPUSD", "AUDCAD",
             "USDCAD", "AUDUSD", "NZDCAD", "AUDNZD", "EURJPY"], n,
        ),
        "hour"                  : rng.integers(0, 24, n),
        "dayofweek"             : rng.integers(0, 7, n),
        "month"                 : rng.integers(1, 13, n),
        "holding_duration_hours": rng.exponential(20.0, n) + 0.01,
    })


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

def test_build_features_excludes_forbidden_columns():
    df = _make_model_df()
    X, _ = build_features(df, include_duration=False)
    leakage = FORBIDDEN_COLS & set(X.columns)
    assert len(leakage) == 0, f"Forbidden columns found in X: {leakage}"


def test_build_features_adds_duration_when_flag_set():
    df = _make_model_df()
    X, _ = build_features(df, include_duration=True)
    assert "holding_duration_hours" in X.columns


def test_build_features_y_is_binary():
    df = _make_model_df()
    _, y = build_features(df)
    assert set(y.unique()).issubset({0, 1})


def test_build_features_no_duration_by_default():
    df = _make_model_df()
    X, _ = build_features(df)
    assert "holding_duration_hours" not in X.columns


# ---------------------------------------------------------------------------
# train_compare
# ---------------------------------------------------------------------------

def test_train_compare_returns_four_model_rows():
    df = _make_model_df(n=600)
    results = train_compare(df, sample_n=600)
    assert len(results["comparison"]) == 4


def test_train_compare_aucs_in_unit_interval():
    df = _make_model_df(n=600)
    results = train_compare(df, sample_n=600)
    for col in ["roc_auc", "pr_auc"]:
        vals = results["comparison"][col].values
        assert (vals >= 0).all() and (vals <= 1).all(), \
            f"{col} out of [0, 1]: {vals}"


def test_train_compare_has_required_keys():
    df = _make_model_df(n=600)
    results = train_compare(df, sample_n=600)
    for key in ("best_model_name", "best_model", "X_test", "y_test",
                "probas", "cv_auc", "feature_names"):
        assert key in results, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# leakage_check
# ---------------------------------------------------------------------------

def test_leakage_check_returns_finite_aucs():
    df = _make_model_df(n=600)
    auc_entry, auc_dur = leakage_check(df, sample_n=600)
    assert np.isfinite(auc_entry), f"auc_entry not finite: {auc_entry}"
    assert np.isfinite(auc_dur),   f"auc_dur not finite: {auc_dur}"


def test_leakage_check_aucs_in_unit_interval():
    df = _make_model_df(n=600)
    auc_entry, auc_dur = leakage_check(df, sample_n=600)
    assert 0.0 <= auc_entry <= 1.0
    assert 0.0 <= auc_dur   <= 1.0
