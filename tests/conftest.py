"""
Shared pytest fixtures.

A small synthetic weekly series is preferred over the real Excel data so
tests are fast (<1s each) and don't require model files to exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_weekly_series() -> pd.Series:
    """
    52-week-long deterministic weekly sales series with a trend, a yearly
    sinusoidal cycle, and gaussian noise. Used to keep feature/eval tests
    independent of the production dataset.
    """
    rng = np.random.default_rng(42)
    n = 156  # 3 years of weeks
    dates = pd.date_range(start="2020-01-05", periods=n, freq="W")
    trend = np.linspace(100, 200, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 52)
    noise = rng.normal(0, 5, size=n)
    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name="sales")


@pytest.fixture
def irregular_raw_df() -> pd.DataFrame:
    """Raw-style DataFrame mimicking the messy Excel input (irregular dates)."""
    rows = [
        ("CA", "2021-01-03", 100.0),
        ("CA", "2021-01-10", 110.0),
        ("CA", "2021-01-25", 130.0),  # gap of 15 days
        ("CA", "2021-02-01", 125.0),
        ("CA", "2021-02-08", 140.0),
        ("TX", "2021-01-03", 200.0),
        ("TX", "2021-01-10", 210.0),
        ("TX", "2021-01-17", 220.0),
        ("TX", "2021-01-24", 215.0),
    ]
    df = pd.DataFrame(rows, columns=["state", "date", "sales"])
    df["date"] = pd.to_datetime(df["date"])
    return df
