"""
Model evaluation metrics.

RMSE  — primary metric for model selection (penalises large errors heavily,
        aligned with supply-chain cost: large forecast misses hurt most).
MAE   — interpretable in the same units as sales; robust to outliers.
MAPE  — percentage-based; useful for comparing states with different scales.
        A small epsilon prevents division-by-zero when actuals are near 0.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


_EPS = 1e-8


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual.values - predicted.values) ** 2)))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual.values - predicted.values)))


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    act = actual.values
    pct = np.abs((act - predicted.values) / (np.abs(act) + _EPS))
    return float(np.mean(pct) * 100)


def compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> Dict[str, float]:
    """Return all three metrics in a single dict."""
    actual = actual.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)

    # Align lengths
    n = min(len(actual), len(predicted))
    actual = actual.iloc[:n]
    predicted = predicted.iloc[:n]

    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
    }


def metrics_summary(all_results: dict) -> pd.DataFrame:
    """
    Convert nested results dict to a flat DataFrame for reporting.

    Parameters
    ----------
    all_results : {state: {"metrics": {model: {rmse, mae, mape}}, "best_model": str}}
    """
    rows = []
    for state, res in all_results.items():
        for model, m in res["metrics"].items():
            rows.append({
                "state": state,
                "model": model,
                "rmse": round(m["rmse"], 2),
                "mae": round(m["mae"], 2),
                "mape_pct": round(m["mape"], 2),
                "is_best": model == res["best_model"],
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["state", "rmse"])
