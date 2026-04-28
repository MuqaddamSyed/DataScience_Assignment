"""
Feature engineering for ML-based forecasting (XGBoost).

Key design decisions:
- All features are computed with a strict shift() so that feature[t] only
  uses information available at time t-1 or earlier → NO DATA LEAKAGE.
- Rolling statistics are computed on the already-shifted series; the window
  is placed entirely in the past before the target timestep.
- Holiday flags use the `holidays` library for US federal holidays mapped
  to the nearest Monday-anchored week, ensuring alignment with our weekly index.
- Features returned in two groups:
    1. `build_features(series)` → full DataFrame for train/val
    2. `build_future_features(series, horizon)` → future rows for inference
"""

from __future__ import annotations

from typing import List, Tuple

import holidays as hd
import numpy as np
import pandas as pd

from src.config_loader import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

LAG_WEEKS: List[int] = CONFIG["features"]["lags"]
ROLLING_WINDOWS: List[int] = CONFIG["features"]["rolling_windows"]
HOLIDAY_COUNTRY: str = CONFIG["features"]["holiday_country"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _us_holidays_series(index: pd.DatetimeIndex) -> pd.Series:
    """
    Binary flag: 1 if any US federal holiday falls within the 7-day week
    ending on that Sunday index date.
    """
    years = index.year.unique().tolist()
    us_hols = hd.country_holidays(HOLIDAY_COUNTRY, years=years)
    hol_dates = set(us_hols.keys())

    flags = []
    for dt in index:
        week_dates = pd.date_range(end=dt, periods=7, freq="D").date
        flags.append(int(any(d in hol_dates for d in week_dates)))
    return pd.Series(flags, index=index, name="holiday_flag")


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def build_features(series: pd.Series, state: str = "") -> pd.DataFrame:
    """
    Build supervised ML features from a weekly sales series.

    Parameters
    ----------
    series : weekly pd.Series with DatetimeIndex
    state  : label for logging only

    Returns
    -------
    DataFrame with features + 'target' column, NaN rows (due to lags) dropped.
    """
    df = pd.DataFrame({"sales": series})

    # --- Lag features (strictly in the past) --------------------------------
    for lag in LAG_WEEKS:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # --- Rolling features (window ends at t-1, not t) -----------------------
    # shift(1) first so the window looks into the past only
    shifted = df["sales"].shift(1)
    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w, min_periods=max(1, w // 2)).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w, min_periods=max(1, w // 2)).std()

    # --- Date-based features ------------------------------------------------
    df["day_of_week"] = df.index.dayofweek          # 0=Mon … 6=Sun
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year

    # --- Holiday flag -------------------------------------------------------
    df["holiday_flag"] = _us_holidays_series(df.index)

    # --- Target (current week sales) ----------------------------------------
    df["target"] = df["sales"]

    # Drop rows with NaN caused by lags / rolling windows
    df = df.drop(columns=["sales"])
    before = len(df)
    df = df.dropna()
    logger.debug(
        "State '%s': built %d feature rows (dropped %d NaN rows)",
        state, len(df), before - len(df),
    )
    return df


def train_val_split(
    df: pd.DataFrame, test_size: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/validation split — never random.
    Chronological integrity is critical for time series: the model must
    only see data from the past, not the future.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    return train, val


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature column names (excludes 'target')."""
    return [c for c in df.columns if c != "target"]


# Note: future-feature generation is intentionally NOT implemented as a
# standalone helper here because it cannot work in isolation — feature
# values for step t+k depend on the model's prediction for t+k-1.
# The correct recursive procedure is implemented in `src/predict.py`
# (`_forecast_xgboost`), which interleaves prediction and feature
# rebuilding step by step.
