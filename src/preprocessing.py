"""Cleaning and feature engineering for trades and signals DataFrames."""

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Duration bin definitions
# ---------------------------------------------------------------------------

_HOUR_BINS = [0, 1, 3, 6, 8, 9, 24, float("inf")]
_HOUR_LABELS = ["0-1h", "1h-3h", "3h-6h", "6h-8h", "8h-9h", "9h-24h", ">1 day"]

_DAY_BINS = [0, 1, 3, 5, 7, float("inf")]
_DAY_LABELS = ["0-1d", "1-3d", "3-5d", "5-7d", ">7d"]

# Columns that must be present and non-null for a trade row to be usable
_REQUIRED_TRADE_COLS = ["Volume", "Symbol", "Price", "Time.1", "Price.1"]

# Columns to drop from trades if present (fees not relevant to duration analysis)
_DROP_TRADE_COLS = ["Commission", "Swap"]

# Columns that identify real people in the signals table
_PII_SIGNAL_COLS = ["name", "id"]


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned, feature-enriched copy of the trade-level DataFrame.

    Steps
    -----
    1. Drop rows with any missing value in required columns.
    2. Drop Commission / Swap columns if present.
    3. Ensure Time and Time.1 are datetime.
    4. Compute holding_duration_hours; keep only non-negative rows.
    5. Compute holding_duration_days.
    6. Compute is_loss binary target.
    7. Compute log_holding and log_profit (sign-safe, viz only).
    8. Bin duration in hours and in days.
    9. Extract time features from open timestamp.
    10. Flag rows above 95th percentile of holding_duration_hours.
    """
    out = df.copy()

    # 1. Drop rows missing required columns
    out = out.dropna(subset=_REQUIRED_TRADE_COLS)

    # 2. Drop fee columns
    out = out.drop(columns=[c for c in _DROP_TRADE_COLS if c in out.columns])

    # 3. Parse datetimes
    for col in ("Time", "Time.1"):
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col])

    # 4. Holding duration in hours, keep non-negative
    out["holding_duration_hours"] = (
        (out["Time.1"] - out["Time"]).dt.total_seconds() / 3600
    )
    out = out[out["holding_duration_hours"] >= 0].copy()

    # 5. Duration in days
    out["holding_duration_days"] = out["holding_duration_hours"] / 24

    # 6. Binary loss target
    out["is_loss"] = (out["Profit"] < 0).astype(int)

    # 7. Log transforms
    out["log_holding"] = np.log1p(out["holding_duration_hours"])
    # Sign-safe log for viz: sign(x) * log1p(|x|)
    out["log_profit"] = np.sign(out["Profit"]) * np.log1p(np.abs(out["Profit"]))

    # 8. Duration bins
    out["duration_bin_hours"] = pd.cut(
        out["holding_duration_hours"],
        bins=_HOUR_BINS,
        labels=_HOUR_LABELS,
        right=False,
    )
    out["duration_bin_days"] = pd.cut(
        out["holding_duration_days"],
        bins=_DAY_BINS,
        labels=_DAY_LABELS,
        right=False,
    )

    # 9. Time features from open timestamp
    if "Time" in out.columns:
        out["month"] = out["Time"].dt.month
        out["dayofweek"] = out["Time"].dt.dayofweek
        out["hour"] = out["Time"].dt.hour

    # 10. Outlier flag (does NOT remove rows — survival analysis needs full range)
    p95 = out["holding_duration_hours"].quantile(0.95)
    out["is_trimmed_95"] = out["holding_duration_hours"] > p95

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def clean_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned, anonymized copy of the signal-level summary DataFrame.

    - Drops PII columns: name, id.
    - Replaces title with anonymized labels signal_001, signal_002, …
    - Keeps all numeric summary columns unchanged.
    """
    out = df.copy()

    # Drop PII columns
    out = out.drop(columns=[c for c in _PII_SIGNAL_COLS if c in out.columns])

    # Anonymize strategy titles
    if "title" in out.columns:
        n = len(out)
        width = len(str(n))
        out["title"] = [f"signal_{str(i + 1).zfill(max(width, 3))}" for i in range(n)]

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trades anonymization
# ---------------------------------------------------------------------------

# Patterns that suggest personal identifiers in column names
# ID is a numeric signal/account identifier present in combined_history.csv
_KNOWN_ID_COLS = ("ID",)

_PII_PATTERNS = ("name", "account", "login", "trader", "owner", "user")


def anonymize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Drop known account/signal identifier columns and any PII-named columns."""
    to_drop = [c for c in df.columns if c in _KNOWN_ID_COLS]
    to_drop += [c for c in df.columns if any(p in c.lower() for p in _PII_PATTERNS)]
    to_drop = list(dict.fromkeys(to_drop))  # deduplicate, preserve order
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_processed(
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    out_dir: str = "data/processed",
) -> None:
    """Write cleaned DataFrames to parquet in out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(out / "trades_clean.parquet", index=False)
    signals_df.to_parquet(out / "signals_clean.parquet", index=False)
