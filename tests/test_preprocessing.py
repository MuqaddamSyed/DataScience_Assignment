"""Unit tests for preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import build_state_series, resample_state


class TestResampleState:
    def test_regular_weekly_series_unchanged(self, synthetic_weekly_series):
        out = resample_state(synthetic_weekly_series, freq="W")
        assert len(out) == len(synthetic_weekly_series)
        assert not out.isnull().any()

    def test_no_negative_values(self, synthetic_weekly_series):
        s = synthetic_weekly_series.copy()
        s.iloc[10] = -50  # corrupt one point
        out = resample_state(s, freq="W")
        assert (out >= 0).all(), "resample_state must clamp negatives"

    def test_gap_imputation(self):
        idx = pd.to_datetime([
            "2021-01-03", "2021-01-10", "2021-01-31", "2021-02-07"
        ])
        s = pd.Series([100, 110, 130, 140], index=idx)
        out = resample_state(s, freq="W")
        assert not out.isnull().any(), "All gaps should be filled"
        assert len(out) >= len(s), "Resampled series must include filled weeks"
        assert (out >= 0).all()

    def test_imputation_is_causal(self):
        """Imputed values must not exceed neighbouring known values' bounds."""
        idx = pd.to_datetime(["2021-01-03", "2021-01-31"])  # 4-week gap
        s = pd.Series([100, 200], index=idx)
        out = resample_state(s, freq="W")
        assert out.min() >= 100 - 1e-6
        assert out.max() <= 200 + 1e-6


class TestBuildStateSeries:
    def test_groups_by_state(self, irregular_raw_df):
        out = build_state_series(irregular_raw_df)
        assert set(out.keys()) == {"CA", "TX"}

    def test_each_series_has_datetimeindex(self, irregular_raw_df):
        out = build_state_series(irregular_raw_df)
        for state, series in out.items():
            assert isinstance(series.index, pd.DatetimeIndex)
            assert series.index.is_monotonic_increasing

    def test_all_series_have_no_nans(self, irregular_raw_df):
        out = build_state_series(irregular_raw_df)
        for state, series in out.items():
            assert not series.isnull().any(), f"NaN found in {state}"
