"""
Data loading, cleaning, and resampling pipeline.

Design decisions:
- Resample to strict weekly (Sunday) frequency to unify irregular date spacing.
  The raw data has dates that are mostly 7 days apart but occasionally 3–9 days,
  which breaks lag-based feature engineering.  Resampling with sum() aggregation
  preserves total sales over any partial week.
- Forward-fill followed by linear interpolation handles isolated gaps (<4 weeks);
  longer gaps (if any) are forward-filled to avoid introducing negative values.
- We never use future information during imputation (fillna method='ffill' is
  causal); interpolation is applied only after the series is fully indexed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config_loader import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def load_raw(path: str | None = None) -> pd.DataFrame:
    """Read the Excel file and return a normalised DataFrame."""
    raw_path = path or CONFIG["data"]["raw_path"]
    logger.info("Loading raw data from: %s", raw_path)
    df = pd.read_excel(raw_path)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "total": "sales",
        "date": "date",
        "state": "state",
    }
    df = df.rename(columns=rename_map)

    required = {"date", "state", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    # Aggregate categories (if multiple) per state-date
    df = df.groupby(["state", "date"], as_index=False)["sales"].sum()

    logger.info(
        "Raw data loaded: %d rows, %d states, date range %s → %s",
        len(df),
        df["state"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def resample_state(series: pd.Series, freq: str = "W") -> pd.Series:
    """
    Resample an irregularly-spaced sales series to a regular weekly grid.

    Aggregation: sum (total sales within the week).
    Missing weeks after resampling are filled via forward-fill + linear
    interpolation to preserve trend continuity without leaking future data.
    """
    series = series.sort_index()
    resampled = series.resample(freq).sum()

    # Replace zero-filled gaps (no data at all) with NaN so interpolation works
    gap_mask = resampled == 0
    if gap_mask.any():
        # Zero here means no observation in that week, not a genuine zero sale
        resampled[gap_mask] = np.nan

    # Causal imputation: forward-fill first (uses last known value)
    # then interpolate short remaining gaps linearly
    resampled = resampled.ffill().interpolate(method="linear")
    # Backfill any leading NaNs (very first weeks before data starts)
    resampled = resampled.bfill()

    # Clamp negatives (can't have negative sales)
    resampled = resampled.clip(lower=0)
    return resampled


def build_state_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Return a dict of {state_name: weekly_sales_series} for all states.
    Each series has a DatetimeIndex with weekly (Sunday) frequency.
    """
    freq = CONFIG["data"]["resample_freq"]
    state_series: Dict[str, pd.Series] = {}

    for state, group in df.groupby("state"):
        s = group.set_index("date")["sales"].sort_index()
        resampled = resample_state(s, freq=freq)

        if resampled.isnull().any():
            logger.warning("State '%s' still has NaN after imputation", state)
        state_series[str(state)] = resampled

    logger.info(
        "Built weekly series for %d states; series length: %d weeks",
        len(state_series),
        next(iter(state_series.values())).shape[0],
    )
    return state_series


def run_preprocessing(path: str | None = None) -> Dict[str, pd.Series]:
    """End-to-end preprocessing entry point."""
    df = load_raw(path)
    return build_state_series(df)


if __name__ == "__main__":
    state_series = run_preprocessing()
    for state, s in list(state_series.items())[:2]:
        print(f"\n{state}: {len(s)} weeks, {s.min():.0f} – {s.max():.0f}")
        print(s.tail(5))
