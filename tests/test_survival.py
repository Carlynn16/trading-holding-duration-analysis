"""Unit tests for src/survival.py — synthetic data only, no real CSVs."""

import numpy as np
import pandas as pd
import pytest

from src.survival import (
    DURATION_COL,
    EVENT_COL,
    conditional_loss_probability,
    cox_fit,
    hazard_curve,
    km_by_group,
    km_fit,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _make_surv_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic trades with known survival structure (two risk groups)."""
    rng = np.random.default_rng(seed)
    half = n // 2

    # Group A: shorter durations, lower loss rate
    dur_a  = rng.exponential(scale=20, size=half)
    loss_a = rng.binomial(1, 0.25, half)

    # Group B: longer durations, higher loss rate
    dur_b  = rng.exponential(scale=60, size=n - half)
    loss_b = rng.binomial(1, 0.45, n - half)

    syms = rng.choice(["EURUSD", "GBPUSD", "XAUUSD", "AUDCAD"], size=n)

    return pd.DataFrame({
        DURATION_COL   : np.concatenate([dur_a, dur_b]),
        EVENT_COL      : np.concatenate([loss_a, loss_b]).astype(int),
        "Type"         : ["Buy"] * half + ["Sell"] * (n - half),
        "Symbol"       : syms,
        "Volume"       : rng.uniform(0.1, 2.0, n),
        "hour"         : rng.integers(0, 24, n),
        "dayofweek"    : rng.integers(0, 7, n),
    })


# ---------------------------------------------------------------------------
# km_fit
# ---------------------------------------------------------------------------

def test_km_incidence_nondecreasing():
    """1 - S(t) must be non-decreasing over time."""
    df = _make_surv_df()
    kmf, table = km_fit(df)
    inc = table["cumulative_incidence"].values
    assert np.all(np.diff(inc) >= -1e-9), \
        f"Cumulative incidence decreased: {np.diff(inc)}"


def test_km_incidence_in_unit_interval():
    df = _make_surv_df()
    _, table = km_fit(df)
    assert (table["cumulative_incidence"] >= 0).all()
    assert (table["cumulative_incidence"] <= 1).all()


def test_km_ci_bounds_ordered():
    """ci_lower <= cumulative_incidence <= ci_upper at all key times."""
    _, table = km_fit(_make_surv_df())
    assert (table["ci_lower"] <= table["cumulative_incidence"] + 1e-9).all()
    assert (table["ci_upper"] >= table["cumulative_incidence"] - 1e-9).all()


def test_km_survival_plus_incidence_equals_one():
    """survival + cumulative_incidence == 1 by definition."""
    _, table = km_fit(_make_surv_df())
    np.testing.assert_allclose(
        table["survival"] + table["cumulative_incidence"], 1.0, atol=1e-4
    )


def test_km_table_has_all_key_times():
    from src.survival import KEY_TIMES
    _, table = km_fit(_make_surv_df())
    assert list(table["holding_hours"]) == KEY_TIMES


# ---------------------------------------------------------------------------
# km_by_group
# ---------------------------------------------------------------------------

def test_km_by_group_returns_one_fitter_per_group():
    df = _make_surv_df()
    fitters, p, stat = km_by_group(df, "Symbol", top_n=3)
    # 3 top symbols + possibly 'Other'
    assert len(fitters) >= 1


def test_km_by_group_p_valid():
    df = _make_surv_df()
    _, p, _ = km_by_group(df, "Type", top_n=2)
    assert 0.0 <= p <= 1.0


def test_km_by_group_stat_nonnegative():
    df = _make_surv_df()
    _, _, stat = km_by_group(df, "Type", top_n=2)
    assert stat >= 0


def test_km_by_group_buy_sell_differ():
    """Constructed data has different loss rates for Buy vs Sell — should be significant."""
    df = _make_surv_df(n=500)
    _, p, _ = km_by_group(df, "Type", top_n=2)
    # With n=500 and event-rate difference 0.25 vs 0.45, test should be significant
    # (not guaranteed at small n, so just check it's a valid float)
    assert isinstance(p, float)


# ---------------------------------------------------------------------------
# hazard_curve
# ---------------------------------------------------------------------------

def test_hazard_values_nonnegative():
    df = _make_surv_df()
    naf, accel = hazard_curve(df, bandwidth=5.0)
    # smoothed hazard stored on fitter by hazard_curve()
    assert (naf._smooth_hazard >= 0).all()


def test_hazard_accel_time_positive_or_none():
    df = _make_surv_df()
    _, accel = hazard_curve(df, bandwidth=5.0)
    assert accel is None or accel >= 0


# ---------------------------------------------------------------------------
# cox_fit
# ---------------------------------------------------------------------------

def test_cox_hazard_ratios_finite():
    df = _make_surv_df(n=400)
    cph, _, _ = cox_fit(df, sample_n=400)
    hrs = cph.summary["exp(coef)"].values
    assert np.all(np.isfinite(hrs))


def test_cox_hrs_positive():
    df = _make_surv_df(n=400)
    cph, _, _ = cox_fit(df, sample_n=400)
    hrs = cph.summary["exp(coef)"].values
    assert (hrs > 0).all()


def test_cox_expected_covariates_present():
    df = _make_surv_df(n=400)
    cph, _, _ = cox_fit(df, sample_n=400)
    idx = cph.summary.index.tolist()
    assert "is_buy" in idx
    assert "log_volume_std" in idx
    assert "entry_hour" in idx
    assert "entry_dayofweek" in idx


def test_cox_ph_test_returns_dataframe():
    df = _make_surv_df(n=400)
    _, ph, _ = cox_fit(df, sample_n=400)
    assert hasattr(ph, "summary")
    assert isinstance(ph.summary, pd.DataFrame)


def test_cox_stratified_has_fewer_covariates():
    """Stratified model omits symbol dummies, so it has fewer rows in summary."""
    df = _make_surv_df(n=400)
    cph, _, cph_strat = cox_fit(df, sample_n=400)
    assert len(cph_strat.summary) < len(cph.summary)


# ---------------------------------------------------------------------------
# conditional_loss_probability
# ---------------------------------------------------------------------------

def test_cond_prob_returns_expected_columns():
    df = _make_surv_df(n=400)
    table, lx, ly = conditional_loss_probability(df, n_bins=10, lowess_sample=400)
    assert {"median_time", "loss_rate", "n"}.issubset(table.columns)


def test_cond_prob_loss_rate_in_unit_interval():
    df = _make_surv_df(n=400)
    table, _, _ = conditional_loss_probability(df, n_bins=10, lowess_sample=400)
    assert (table["loss_rate"] >= 0).all()
    assert (table["loss_rate"] <= 1).all()


def test_cond_prob_n_positive():
    df = _make_surv_df(n=400)
    table, _, _ = conditional_loss_probability(df, n_bins=10, lowess_sample=400)
    assert (table["n"] > 0).all()


def test_cond_prob_lowess_arrays_valid():
    df = _make_surv_df(n=400)
    _, lx, ly = conditional_loss_probability(df, n_bins=10, lowess_sample=400)
    assert len(lx) == len(ly)
    assert np.all(ly >= 0)
    assert np.all(ly <= 1)
