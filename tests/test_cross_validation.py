"""Tests for walk-forward CV splitter."""

from __future__ import annotations

import pandas as pd
import pytest

from src.cross_validation import expanding_window_splits


class TestExpandingWindowSplits:
    def test_yields_correct_number_of_folds(self, synthetic_weekly_series):
        folds = list(expanding_window_splits(
            synthetic_weekly_series, n_splits=5, val_horizon=8,
        ))
        assert len(folds) == 5

    def test_train_grows_each_fold(self, synthetic_weekly_series):
        folds = list(expanding_window_splits(
            synthetic_weekly_series, n_splits=5, val_horizon=8,
        ))
        train_sizes = [len(t) for t, _ in folds]
        assert train_sizes == sorted(train_sizes), "Train must expand"
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] - train_sizes[i - 1] == 8

    def test_val_size_equals_horizon(self, synthetic_weekly_series):
        for train, val in expanding_window_splits(
            synthetic_weekly_series, n_splits=5, val_horizon=8,
        ):
            assert len(val) == 8

    def test_train_and_val_are_chronological(self, synthetic_weekly_series):
        for train, val in expanding_window_splits(
            synthetic_weekly_series, n_splits=5, val_horizon=8,
        ):
            assert train.index.max() < val.index.min()

    def test_raises_on_too_short_series(self):
        s = pd.Series(range(20), index=pd.date_range("2020-01-05", periods=20, freq="W"))
        with pytest.raises(ValueError):
            list(expanding_window_splits(s, n_splits=5, val_horizon=8))
