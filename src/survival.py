"""
Cause-specific survival analysis for 'loss-close' events.

Framing
-------
duration       = holding_duration_hours   (time axis)
event_observed = is_loss                  (1 = closed at a loss = EVENT;
                                           0 = closed at profit/break-even = CENSORED)

This is a cause-specific hazard model.  Profitable closes leave the risk set
(treated as censoring).  1 - S(t) = cumulative incidence of a loss-close by
holding time t.

Cross-check: AalenJohansenFitter treats loss-close (1) and profit-close (2) as
competing events.  The AJ CIF for loss is systematically lower than 1 - KM_S(t)
because KM assumes censored units (profit-closes) would eventually turn into
losses, which inflates the estimate.  Agreement in shape, divergence in level,
validates the cause-specific framing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import (
    AalenJohansenFitter,
    CoxPHFitter,
    KaplanMeierFitter,
    NelsonAalenFitter,
)
from lifelines.exceptions import ConvergenceError as _LifelinesConvergenceError
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

DURATION_COL = "holding_duration_hours"
EVENT_COL    = "is_loss"
KEY_TIMES    = [1, 3, 6, 12, 24, 48, 72, 168]

KM_SAMPLE          = 300_000
AJ_SAMPLE          = 150_000
COX_SAMPLE         = 300_000
TOP_SYMBOLS        = 8
COND_PROB_SAMPLE   = 200_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf_at(kmf: KaplanMeierFitter, times: list) -> list:
    return [float(kmf.survival_function_at_times(t).values[0]) for t in times]


def _ci_at(kmf: KaplanMeierFitter, times: list) -> tuple[list, list]:
    """Return (S_lower_95, S_upper_95) at each time via step interpolation."""
    ci  = kmf.confidence_interval_
    lower_col = next(c for c in ci.columns if "lower" in c)
    upper_col = next(c for c in ci.columns if "upper" in c)
    idx  = ci.index
    lows, highs = [], []
    for t in times:
        valid = idx[idx <= t]
        if len(valid) == 0:
            lows.append(1.0); highs.append(1.0)
        else:
            lows.append(float(ci[lower_col].loc[valid[-1]]))
            highs.append(float(ci[upper_col].loc[valid[-1]]))
    return lows, highs


# ---------------------------------------------------------------------------
# KM — overall
# ---------------------------------------------------------------------------

def km_fit(df: pd.DataFrame) -> tuple[KaplanMeierFitter, pd.DataFrame]:
    """Fit KM on a random sample; return (fitter, cumulative_incidence_table).

    Table: holding_hours, survival, cumulative_incidence,
           ci_lower (of 1-S), ci_upper (of 1-S).
    """
    sample = df.sample(n=min(KM_SAMPLE, len(df)), random_state=42)
    kmf = KaplanMeierFitter(label="Loss-free survival")
    kmf.fit(sample[DURATION_COL], event_observed=sample[EVENT_COL])

    sf_vals       = _sf_at(kmf, KEY_TIMES)
    s_lows, s_highs = _ci_at(kmf, KEY_TIMES)

    table = pd.DataFrame({
        "holding_hours"        : KEY_TIMES,
        "survival"             : [round(s, 4) for s in sf_vals],
        "cumulative_incidence" : [round(1 - s, 4) for s in sf_vals],
        # CI of (1-S): lower(1-S) = 1 - upper(S), upper(1-S) = 1 - lower(S)
        "ci_lower"             : [round(1 - h, 4) for h in s_highs],
        "ci_upper"             : [round(1 - l, 4) for l in s_lows],
    })
    return kmf, table


# ---------------------------------------------------------------------------
# KM — by group  +  log-rank
# ---------------------------------------------------------------------------

def km_by_group(
    df: pd.DataFrame,
    group_col: str,
    top_n: int = 6,
) -> tuple[dict, float, float]:
    """Fit KM per group (top_n by count, rest = 'Other').

    Returns (fitters_dict, log_rank_p, log_rank_test_statistic).
    """
    top_cats = df[group_col].value_counts().head(top_n).index.tolist()
    sample   = df.sample(n=min(KM_SAMPLE, len(df)), random_state=42)

    sample = sample.copy()
    sample["_group"] = (
        sample[group_col]
        .astype(str)
        .apply(lambda x: x if x in top_cats else "Other")
    )

    fitters: dict = {}
    for grp, gdf in sample.groupby("_group"):
        kmf = KaplanMeierFitter(label=str(grp))
        kmf.fit(gdf[DURATION_COL], event_observed=gdf[EVENT_COL])
        fitters[str(grp)] = kmf

    lr = multivariate_logrank_test(sample[DURATION_COL], sample["_group"], sample[EVENT_COL])
    return fitters, float(lr.p_value), float(lr.test_statistic)


# ---------------------------------------------------------------------------
# Nelson-Aalen smoothed hazard
# ---------------------------------------------------------------------------

def hazard_curve(
    df: pd.DataFrame,
    bandwidth: float = 4.0,
) -> tuple[NelsonAalenFitter, float | None]:
    """Fit Nelson-Aalen; return (fitter, hazard_peak_time_hours).

    hazard_peak_time is the time within 0-168h where the smoothed hazard
    reaches its maximum.

    IMPORTANT: this peak reflects the front-loaded closing-rate distribution
    (≈31% of trades close within 1h), NOT per-trade risk. A high hazard at
    1.5h means many trades are closing at that duration, not that those trades
    are individually more likely to be losses. Use conditional_loss_probability()
    for the actual per-trade risk curve.

    Implementation note: lifelines smoothed_hazard_() allocates an O(m²)
    kernel matrix (m = unique event times) which is infeasible at large m.
    We instead compute the hazard as finite differences of the cumulative
    hazard H(t), then smooth with a rolling Gaussian window.
    """
    sample = df.sample(n=min(KM_SAMPLE, len(df)), random_state=42)
    # Round to 0.25h bins to collapse unique times and keep computation light
    dur_rounded = (sample[DURATION_COL] / 0.25).round() * 0.25

    naf = NelsonAalenFitter()
    naf.fit(dur_rounded, event_observed=sample[EVENT_COL])

    # Compute smoothed hazard via finite differences + rolling window
    cumH = naf.cumulative_hazard_.copy()
    col  = cumH.columns[0]
    times   = cumH.index.values.astype(float)
    cum_val = cumH[col].values

    # Finite differences: h_i ≈ ΔH_i / Δt_i
    dt    = np.diff(times, prepend=times[0])
    dt    = np.where(dt == 0, 1e-9, dt)
    dH    = np.diff(cum_val, prepend=cum_val[0])
    h_raw = dH / dt

    # Smooth with a rolling window of ~bandwidth hours
    bin_width   = 0.25
    window_bins = max(3, int(bandwidth / bin_width))
    # Gaussian-weighted rolling average
    kernel = np.exp(-0.5 * np.linspace(-2, 2, window_bins) ** 2)
    kernel /= kernel.sum()
    h_smooth = np.convolve(h_raw, kernel, mode="same")
    h_smooth = np.maximum(h_smooth, 0)

    # Store as a pseudo-DataFrame on the fitter for the figure to consume
    naf._smooth_times  = times
    naf._smooth_hazard = h_smooth

    # Peak within 0-168h
    mask = times <= 168
    if mask.any():
        peak_time = float(times[mask][h_smooth[mask].argmax()])
    else:
        peak_time = None

    return naf, peak_time


# ---------------------------------------------------------------------------
# Conditional loss probability — per-trade risk curve
# ---------------------------------------------------------------------------

def conditional_loss_probability(
    df: pd.DataFrame,
    n_bins: int = 50,
    lowess_sample: int = COND_PROB_SAMPLE,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """P(is_loss=1 | holding_duration ≈ t) via quantile-binned aggregation
    and a LOWESS smoother on a random sample.

    This is the correct per-trade risk curve: it shows what fraction of trades
    that were actually held for ~t hours closed as losses. Unlike the Nelson-Aalen
    hazard, it is not confounded by the trade-count distribution.

    Returns
    -------
    table    : DataFrame with columns median_time, loss_rate, n
    lowess_x : smoothed x-values in hours (expm1 of log1p-scale fit)
    lowess_y : smoothed loss-probability, clipped to [0, 1]
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess

    # Quantile-binned aggregation on the full dataset
    bins = pd.qcut(df[DURATION_COL], q=n_bins, duplicates="drop")
    table = (
        df.assign(_bin=bins)
          .groupby("_bin", observed=True)
          .agg(
              median_time=(DURATION_COL, "median"),
              loss_rate=(EVENT_COL,      "mean"),
              n=(EVENT_COL,              "count"),
          )
          .reset_index(drop=True)
    )

    # LOWESS on a random sample in log1p space for numerical stability
    sample = df.sample(n=min(lowess_sample, len(df)), random_state=seed)
    x = np.log1p(sample[DURATION_COL].values.astype(float))
    y = sample[EVENT_COL].values.astype(float)
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    x_range = float(x_s[-1] - x_s[0]) if len(x_s) > 1 else 1.0
    # delta skips linear-interpolation between nearby points — makes LOWESS O(n) at large n
    delta = max(x_range * 0.005, 1e-6)

    sm_result = _sm_lowess(
        y_s, x_s,
        frac=0.15, it=0, delta=delta,
        is_sorted=True, return_sorted=True,
    )
    lowess_x = np.expm1(sm_result[:, 0])
    lowess_y = np.clip(sm_result[:, 1], 0.0, 1.0)

    return table, lowess_x, lowess_y


# ---------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------

def cox_fit(
    df: pd.DataFrame,
    sample_n: int = COX_SAMPLE,
    seed: int = 42,
) -> tuple[CoxPHFitter, object, CoxPHFitter]:
    """Fit Cox PH on entry-time features only (no leakage).

    Covariates
    ----------
    is_buy          : 1 = Buy, 0 = Sell
    log_volume_std  : standardised log1p(Volume)
    sym_<X>         : top-8 Symbol dummies (Other = reference, omitted)
    entry_hour      : 0-23
    entry_dayofweek : 0-6

    Returns (cph, ph_test_results, cph_stratified_by_symbol).

    Note: at n=300K the Schoenfeld test has extreme power and will reject PH
    for almost any covariate.  Interpret HR magnitudes and 95% CIs, not p-values
    from the PH test alone.  The stratified model (strata=Symbol) relaxes the
    PH assumption for instrument and is provided as a robustness check.
    """
    sample = (
        df[df["Type"].astype(str).isin(["Buy", "Sell"])]
        .sample(n=min(sample_n, len(df)), random_state=seed)
    )

    # --- design matrix ---
    cph_df = pd.DataFrame(index=sample.index)
    cph_df["duration"]        = sample[DURATION_COL].values
    cph_df["event"]           = sample[EVENT_COL].values
    cph_df["is_buy"]          = (sample["Type"].astype(str) == "Buy").astype(int).values

    log_vol                   = np.log1p(sample["Volume"].astype(float))
    cph_df["log_volume_std"]  = ((log_vol - log_vol.mean()) / log_vol.std()).values

    top_syms = sample["Symbol"].value_counts().head(TOP_SYMBOLS).index.tolist()
    for sym in top_syms:
        cph_df[f"sym_{sym}"] = (sample["Symbol"].astype(str) == sym).astype(int).values

    cph_df["entry_hour"]       = sample["hour"].values
    cph_df["entry_dayofweek"]  = sample["dayofweek"].values
    cph_df["symbol_strata"]    = (
        sample["Symbol"].astype(str)
        .apply(lambda x: x if x in top_syms else "Other")
        .values
    )
    cph_df = cph_df.dropna()

    sym_dummy_cols = [c for c in cph_df.columns if c.startswith("sym_")]
    model_cols = (
        ["duration", "event", "is_buy", "log_volume_std"]
        + sym_dummy_cols
        + ["entry_hour", "entry_dayofweek"]
    )
    model_df = cph_df[model_cols].copy()

    # --- primary Cox; fall back to small ridge penalty only if Hessian singular
    #     (this only occurs at very small n, e.g. test fixtures — not at 300K) ---
    def _fit_cox(df_fit, **kwargs):
        try:
            m = CoxPHFitter()
            m.fit(df_fit, **kwargs)
        except (np.linalg.LinAlgError, _LifelinesConvergenceError):
            m = CoxPHFitter(penalizer=0.1)
            m.fit(df_fit, **kwargs)
        return m

    cph = _fit_cox(model_df, duration_col="duration", event_col="event")

    # --- PH test (Schoenfeld residuals) ---
    ph_results = proportional_hazard_test(cph, model_df, time_transform="rank")

    # --- stratified robustness model (strata=Symbol, no symbol dummies) ---
    strat_df = cph_df[
        ["duration", "event", "is_buy", "log_volume_std",
         "entry_hour", "entry_dayofweek", "symbol_strata"]
    ].copy()
    cph_strat = _fit_cox(
        strat_df,
        duration_col="duration",
        event_col="event",
        strata=["symbol_strata"],
    )

    return cph, ph_results, cph_strat


# ---------------------------------------------------------------------------
# Aalen-Johansen competing risks
# ---------------------------------------------------------------------------

def aj_fit(df: pd.DataFrame) -> tuple[AalenJohansenFitter, AalenJohansenFitter]:
    """Competing-risks CIF: loss (event 1) vs profit (event 2)."""
    sample = df.sample(n=min(AJ_SAMPLE, len(df)), random_state=42)
    # recode: 1 = loss-close, 2 = profit-close (no true censoring in this dataset)
    event_codes = sample[EVENT_COL].map({1: 1, 0: 2})

    ajf_loss = AalenJohansenFitter()
    ajf_loss.fit(sample[DURATION_COL], event_observed=event_codes,
                 event_of_interest=1, label="CIF: loss-close")

    ajf_profit = AalenJohansenFitter()
    ajf_profit.fit(sample[DURATION_COL], event_observed=event_codes,
                   event_of_interest=2, label="CIF: profit-close")

    return ajf_loss, ajf_profit


# ---------------------------------------------------------------------------
# Orchestrator — run all models, save CSVs
# ---------------------------------------------------------------------------

def save_survival_outputs(
    df: pd.DataFrame,
    out_dir: str = "data/processed",
) -> dict:
    """Fit all survival models, persist CSV tables, return fitted objects."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("  KM overall…")
    kmf, incidence_table = km_fit(df)
    incidence_table.to_csv(out / "km_incidence_table.csv", index=False)

    print("  KM by Symbol…")
    fitters_sym, lr_p_sym, lr_stat_sym = km_by_group(df, "Symbol", top_n=6)

    print("  KM by Type…")
    fitters_type, lr_p_type, lr_stat_type = km_by_group(df, "Type", top_n=2)

    logrank_df = pd.DataFrame([
        {"group_col": "Symbol", "test_statistic": round(lr_stat_sym, 2),
         "p_value": lr_p_sym},
        {"group_col": "Type",   "test_statistic": round(lr_stat_type, 2),
         "p_value": lr_p_type},
    ])
    logrank_df.to_csv(out / "logrank_results.csv", index=False)

    print("  Nelson-Aalen hazard curve…")
    naf, accel_time = hazard_curve(df)

    print("  Cox PH (300K sample)…")
    cph, ph_results, cph_strat = cox_fit(df)

    cox_summary = cph.summary[
        ["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    ].copy()
    cox_summary.columns = ["coef", "HR", "HR_lower_95", "HR_upper_95", "p_value"]
    cox_summary = cox_summary.round(4)
    cox_summary.to_csv(out / "cox_hazard_ratios.csv")

    ph_df = ph_results.summary.copy()
    ph_df.to_csv(out / "ph_test.csv")

    print("  Aalen-Johansen competing risks…")
    ajf_loss, ajf_profit = aj_fit(df)

    accel_row = pd.DataFrame([{"bandwidth_h": 4.0, "peak_hazard_time_h": accel_time}])
    accel_row.to_csv(out / "hazard_accel.csv", index=False)

    print("  Conditional loss probability…")
    base_rate = float(df[EVENT_COL].mean())
    cond_table, lowess_x, lowess_y = conditional_loss_probability(df)
    cond_table.to_csv(out / "conditional_loss_prob.csv", index=False)

    return dict(
        kmf=kmf,
        incidence_table=incidence_table,
        fitters_sym=fitters_sym,
        fitters_type=fitters_type,
        lr_p_sym=lr_p_sym,
        lr_p_type=lr_p_type,
        lr_stat_sym=lr_stat_sym,
        lr_stat_type=lr_stat_type,
        naf=naf,
        accel_time=accel_time,
        cph=cph,
        ph_results=ph_results,
        cph_strat=cph_strat,
        ajf_loss=ajf_loss,
        ajf_profit=ajf_profit,
        base_rate=base_rate,
        cond_table=cond_table,
        lowess_x=lowess_x,
        lowess_y=lowess_y,
    )
