"""
Feature engineering tests.

The most important test in this whole project is `test_no_data_leakage`:
if any feature at time t uses the value at time t (or later), our model
evaluation is invalid and forecasts will look great in val and disastrous
in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    build_features,
    get_feature_columns,
    train_val_split,
)


class TestBuildFeatures:
    def test_returns_dataframe_with_target_column(self, synthetic_weekly_series):
        df = build_features(synthetic_weekly_series)
        assert "target" in df.columns
        assert len(df) > 0

    def test_no_data_leakage_lag_features(self, synthetic_weekly_series):
        """lag_k at index t MUST equal sales at index t-k (never t)."""
        df = build_features(synthetic_weekly_series)
        s = synthetic_weekly_series

        for ts in df.index[:5]:  # check first 5 rows
            for lag in [1, 2, 4, 8]:
                lag_col = f"lag_{lag}"
                if lag_col not in df.columns:
                    continue
                expected = s.loc[ts - pd.Timedelta(weeks=lag)]
                actual = df.loc[ts, lag_col]
                assert actual == pytest.approx(expected, rel=1e-9), (
                    f"Leakage detected: {lag_col} at {ts} = {actual}, "
                    f"expected {expected}"
                )

    def test_no_data_leakage_target(self, synthetic_weekly_series):
        """Target at t must equal sales at t (the value we want to predict)."""
        df = build_features(synthetic_weekly_series)
        for ts in df.index[:5]:
            assert df.loc[ts, "target"] == pytest.approx(
                synthetic_weekly_series.loc[ts], rel=1e-9
            )

    def test_rolling_mean_uses_past_only(self, synthetic_weekly_series):
        """rolling_mean_4 at t must NOT include sales[t]."""
        df = build_features(synthetic_weekly_series)
        s = synthetic_weekly_series
        for ts in df.index[5:10]:  # skip warm-up window
            window = [s.loc[ts - pd.Timedelta(weeks=k)] for k in range(1, 5)]
            expected = np.mean(window)
            actual = df.loc[ts, "rolling_mean_4"]
            assert actual == pytest.approx(expected, rel=1e-6), (
                f"Rolling mean leak at {ts}: got {actual}, expected {expected}"
            )

    def test_calendar_features_present(self, synthetic_weekly_series):
        df = build_features(synthetic_weekly_series)
        assert {"day_of_week", "month", "week_of_year", "quarter", "year"}.issubset(
            df.columns
        )

    def test_holiday_flag_is_binary(self, synthetic_weekly_series):
        df = build_features(synthetic_weekly_series)
        assert df["holiday_flag"].isin([0, 1]).all()

    def test_no_nan_after_build(self, synthetic_weekly_series):
        df = build_features(synthetic_weekly_series)
        assert df.isnull().sum().sum() == 0


class TestTrainValSplit:
    def test_chronological(self, synthetic_weekly_series):
        df = build_features(synthetic_weekly_series)
        train, val = train_val_split(df, test_size=0.2)
        assert train.index.max() < val.index.min(), "Split must be chronological"

    def test_split_proportions(self, synthetic_weekly_series):
        df = build_features(synthetic_weekly_series)
        train, val = train_val_split(df, test_size=0.2)
        assert len(train) + len(val) == len(df)
        assert abs(len(val) / len(df) - 0.2) < 0.05  # within 5pp tolerance

    def test_no_overlap(self, synthetic_weekly_series):
        df = build_features(synthetic_weekly_series)
        train, val = train_val_split(df, test_size=0.2)
        assert len(set(train.index) & set(val.index)) == 0


class TestGetFeatureColumns:
    def test_excludes_target(self, synthetic_weekly_series):
        df = build_features(synthetic_weekly_series)
        cols = get_feature_columns(df)
        assert "target" not in cols
        assert len(cols) == len(df.columns) - 1
