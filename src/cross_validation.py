"""
Walk-forward (rolling-origin) cross-validation for time-series models.

Why walk-forward instead of a single hold-out
─────────────────────────────────────────────
A single chronological 80/20 split gives one validation score, but the
variance of that score depends entirely on which 20% you happened to pick.
If the last 20% of the series happens to coincide with an unusually quiet
or noisy period, the "best model" you select is just a coincidence.

Walk-forward CV trains and evaluates the same model on N successive
splits, where each split's training set ends just before its own
validation window. This produces a distribution of scores per model
which is what we should be comparing — median + IQR is far more robust
than a single number.

This module provides:
- `expanding_window_splits` : generates (train, val) index pairs
- `evaluate_walk_forward`   : runs a model factory across all folds
- `compare_models_cv`       : applies it to ARIMA / SARIMA / Prophet
                              for one state and returns a tidy DataFrame.

The default (5 folds, 8-week validation horizon per fold) matches the
production setting of forecasting 8 weeks ahead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from src.evaluate import compute_metrics
from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Splitter
# ══════════════════════════════════════════════════════════════════════════════

def expanding_window_splits(
    series: pd.Series,
    n_splits: int = 5,
    val_horizon: int = 8,
    min_train_size: int = 52,
) -> Iterator[Tuple[pd.Series, pd.Series]]:
    """
    Yield `n_splits` (train, val) pairs using an expanding training window.

    Layout (h = val_horizon):

        |─── train ──── | val |
        |─── train ──────── | val |
        |─── train ─────────────── | val |
        ...

    Each successive train window is `val_horizon` weeks longer than the
    previous one, ending just before that fold's validation slice.

    The earliest fold's training set is forced to be at least
    `min_train_size` to avoid models being fit on absurdly short data.
    """
    n = len(series)
    total_val = n_splits * val_horizon
    if n - total_val < min_train_size:
        raise ValueError(
            f"Series too short for {n_splits} folds × {val_horizon} weeks "
            f"with min_train_size={min_train_size} (have n={n})"
        )

    first_val_start = n - total_val
    for k in range(n_splits):
        train_end = first_val_start + k * val_horizon
        val_end = train_end + val_horizon
        yield series.iloc[:train_end], series.iloc[train_end:val_end]


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

ModelFitFn = Callable[[pd.Series, int], pd.Series]
"""A function that takes (train_series, horizon) and returns a Series of
predictions of length `horizon`."""


@dataclass
class FoldResult:
    fold: int
    train_size: int
    rmse: float
    mae: float
    mape: float


def evaluate_walk_forward(
    series: pd.Series,
    fit_predict: ModelFitFn,
    n_splits: int = 5,
    val_horizon: int = 8,
) -> pd.DataFrame:
    """
    Run a model across `n_splits` expanding-window folds and return per-fold
    metrics.

    Parameters
    ----------
    series      : full weekly series (with DatetimeIndex)
    fit_predict : function (train, h) → predicted Series of length h
    """
    rows: List[FoldResult] = []
    for k, (train, val) in enumerate(
        expanding_window_splits(series, n_splits, val_horizon)
    ):
        try:
            preds = fit_predict(train, val_horizon)
            preds = pd.Series(preds.values, index=val.index)
            m = compute_metrics(val, preds)
            rows.append(FoldResult(
                fold=k,
                train_size=len(train),
                rmse=m["rmse"],
                mae=m["mae"],
                mape=m["mape"],
            ))
        except Exception as exc:
            logger.warning("Fold %d failed: %s", k, exc)
            rows.append(FoldResult(
                fold=k, train_size=len(train),
                rmse=np.inf, mae=np.inf, mape=np.inf,
            ))
    return pd.DataFrame([r.__dict__ for r in rows])


# ══════════════════════════════════════════════════════════════════════════════
# Model factories — each returns a horizon-length forecast given a train series
# ══════════════════════════════════════════════════════════════════════════════

def _fit_arima(train: pd.Series, horizon: int) -> pd.Series:
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train, order=(1, 1, 1)).fit()
    return pd.Series(model.forecast(steps=horizon).values)


def _fit_sarima(train: pd.Series, horizon: int) -> pd.Series:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    return pd.Series(model.forecast(steps=horizon).values)


def _fit_prophet(train: pd.Series, horizon: int) -> pd.Series:
    from prophet import Prophet
    df = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False)
    m.fit(df)
    future_idx = pd.date_range(
        start=train.index[-1], periods=horizon + 1, freq="W"
    )[1:]
    fut = pd.DataFrame({"ds": future_idx})
    return pd.Series(m.predict(fut)["yhat"].values).clip(lower=0)


def compare_models_cv(
    series: pd.Series,
    state: str = "",
    n_splits: int = 5,
    val_horizon: int = 8,
) -> pd.DataFrame:
    """
    Walk-forward comparison of ARIMA / SARIMA / Prophet on one series.

    Returns a tidy DataFrame with columns:
        ['model', 'fold', 'train_size', 'rmse', 'mae', 'mape']
    """
    factories = {
        "arima": _fit_arima,
        "sarima": _fit_sarima,
        "prophet": _fit_prophet,
    }
    frames = []
    for name, fn in factories.items():
        logger.info("[%s] CV evaluating %s …", state, name)
        df = evaluate_walk_forward(series, fn, n_splits, val_horizon)
        df["model"] = name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[["model", "fold", "train_size", "rmse", "mae", "mape"]]


def cv_summary(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold metrics: median, mean, IQR (q1/q3), std."""
    g = cv_df.groupby("model")
    summary = pd.DataFrame({
        "rmse_median": g["rmse"].median(),
        "rmse_mean": g["rmse"].mean(),
        "rmse_q1": g["rmse"].quantile(0.25),
        "rmse_q3": g["rmse"].quantile(0.75),
        "mae_median": g["mae"].median(),
        "mape_median": g["mape"].median(),
    }).round(2)
    return summary.sort_values("rmse_median")
