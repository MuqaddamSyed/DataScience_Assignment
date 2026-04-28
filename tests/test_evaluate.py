"""Evaluation metric tests with hand-computed expected values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_metrics, mae, mape, rmse


class TestMetricsAreCorrect:
    def test_perfect_prediction(self):
        actual = pd.Series([100.0, 200.0, 300.0])
        pred = pd.Series([100.0, 200.0, 300.0])
        m = compute_metrics(actual, pred)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["mape"] == pytest.approx(0.0)

    def test_constant_offset_prediction(self):
        actual = pd.Series([100.0, 200.0, 300.0])
        pred = pd.Series([110.0, 210.0, 310.0])  # +10 each
        m = compute_metrics(actual, pred)
        assert m["mae"] == pytest.approx(10.0)
        assert m["rmse"] == pytest.approx(10.0)

    def test_mae_robust_to_outlier_compared_to_rmse(self):
        """RMSE should be larger than MAE in presence of large outlier."""
        actual = pd.Series([100.0, 100.0, 100.0])
        pred = pd.Series([100.0, 100.0, 1000.0])  # one bad outlier
        m = compute_metrics(actual, pred)
        assert m["rmse"] > m["mae"]

    def test_mape_units_are_percent(self):
        actual = pd.Series([100.0, 100.0])
        pred = pd.Series([110.0, 90.0])  # +10% and -10%
        m = compute_metrics(actual, pred)
        assert m["mape"] == pytest.approx(10.0, abs=0.01)

    def test_metrics_align_lengths(self):
        actual = pd.Series([100.0, 200.0, 300.0])
        pred = pd.Series([100.0, 200.0])  # shorter
        m = compute_metrics(actual, pred)
        assert m["rmse"] == pytest.approx(0.0)


class TestStandaloneFunctions:
    def test_rmse(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([2.0, 3.0, 4.0])  # all off by 1
        assert rmse(a, b) == pytest.approx(1.0)

    def test_mae(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([2.0, 3.0, 4.0])
        assert mae(a, b) == pytest.approx(1.0)

    def test_mape_zero_safe(self):
        """Should not divide-by-zero when actual is 0."""
        a = pd.Series([0.0, 100.0])
        b = pd.Series([0.0, 100.0])
        result = mape(a, b)
        assert np.isfinite(result)
