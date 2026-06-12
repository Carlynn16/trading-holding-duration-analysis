"""Load raw CSVs from data/raw/ into DataFrames."""

from pathlib import Path
import pandas as pd


_TRADES_DTYPES = {
    "Type": "category",
    "Symbol": "category",
    "Volume": "float32",
    "Price": "float64",
    "Volume.1": "float32",
    "Price.1": "float64",
    "Profit": "float64",
    "Commission": "float64",
    "Swap": "float64",
}


def load_raw(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read combined_history.csv and general_data.csv from data_dir.

    Returns
    -------
    trades_df : DataFrame
        One row per trade (combined_history.csv).
    signals_df : DataFrame
        One row per signal/strategy (general_data.csv).
    """
    raw = Path(data_dir)

    trades_df = pd.read_csv(
        raw / "combined_history.csv",
        dtype=_TRADES_DTYPES,
        parse_dates=["Time", "Time.1"],
        low_memory=False,
    )

    signals_df = pd.read_csv(raw / "general_data.csv", low_memory=False)

    return trades_df, signals_df
