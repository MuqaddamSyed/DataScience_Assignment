"""
Inference / forecasting module.

Loads the saved best model for each state and generates 8-week forecasts.

Recursive forecasting strategy (XGBoost):
  At each future step t+k:
    1. Append the previous prediction to the series.
    2. Recompute lag and rolling features on the extended series.
    3. Predict step t+k.
  This is the standard recursive multi-step approach.  It accumulates
  error over the horizon but is the correct way to avoid leakage.

ARIMA / SARIMA:  forecast(steps=8) natively provides multi-step ahead
Prophet:         extend future dataframe and call predict()
LSTM:            unroll 8 steps using the model's sequence, feeding
                 each prediction back as the next input.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

from src.config_loader import CONFIG
from src.feature_engineering import build_features, get_feature_columns
from src.logger import get_logger
from src.train import SalesLSTM

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

MODELS_DIR = Path(CONFIG["paths"]["models_dir"])
HORIZON = CONFIG["data"]["forecast_horizon"]
FREQ = CONFIG["data"]["resample_freq"]


# ══════════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════════

def _safe_name(state: str) -> str:
    return state.replace(" ", "_").lower()


_VALID_MODELS = {"arima", "sarima", "xgboost", "lstm", "prophet"}


def load_best_model(state: str, model_name: str) -> Any:
    """
    Load a trained model from disk with strict path-traversal protection.

    Security note
    -------------
    pickle/joblib both deserialise arbitrary Python objects, which means
    loading an untrusted file is equivalent to running its code.  We
    mitigate this by:
      1. validating `model_name` against a hard-coded allow-list,
      2. sanitising `state` to a safe slug (alphanumerics + underscore),
      3. requiring the resolved path stays inside `MODELS_DIR`.
    The `models/` directory is therefore part of the trust boundary —
    only authorised CI/training jobs should be able to write to it.
    """
    if model_name not in _VALID_MODELS:
        raise ValueError(f"Unknown model name: {model_name!r}")

    safe = _safe_name(state)
    if not safe.replace("_", "").isalnum():
        raise ValueError(f"Invalid state name: {state!r}")

    ext_map = {
        "arima": "pkl",
        "sarima": "pkl",
        "xgboost": "joblib",
        "lstm": "joblib",
        "prophet": "joblib",
    }
    ext = ext_map[model_name]
    path = (MODELS_DIR / f"{safe}_{model_name}.{ext}").resolve()

    # Path-traversal guard
    try:
        path.relative_to(MODELS_DIR.resolve())
    except ValueError:
        raise ValueError(f"Resolved path escapes models dir: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if ext == "pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# Per-model forecasters
# ══════════════════════════════════════════════════════════════════════════════

def _forecast_arima(model: Any, series: pd.Series, horizon: int) -> List[float]:
    preds = model.forecast(steps=horizon)
    return [max(0.0, float(v)) for v in preds]


def _forecast_sarima(model: Any, series: pd.Series, horizon: int) -> List[float]:
    preds = model.forecast(steps=horizon)
    return [max(0.0, float(v)) for v in preds]


def _forecast_xgboost(model: Any, series: pd.Series, horizon: int) -> List[float]:
    """Recursive multi-step forecasting with XGBoost."""
    extended = series.copy()
    future_dates = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=FREQ)[1:]
    predictions: List[float] = []

    for fdate in future_dates:
        # Add placeholder NaN for the target date
        placeholder = pd.Series([np.nan], index=pd.DatetimeIndex([fdate]))
        temp = pd.concat([extended, placeholder])

        feat_df = build_features(temp)
        if fdate not in feat_df.index:
            # Fallback: use last known value (cast to native float to avoid numpy scalar surprises)
            fallback = float(max(0.0, float(extended.iloc[-1])))
            predictions.append(fallback)
            extended = pd.concat([extended, pd.Series([fallback], index=pd.DatetimeIndex([fdate]))])
            continue

        row = feat_df.loc[[fdate]]
        feat_cols = get_feature_columns(row)
        pred = float(model.predict(row[feat_cols].values)[0])
        pred = max(0.0, pred)
        predictions.append(pred)

        extended = pd.concat([
            extended,
            pd.Series([pred], index=pd.DatetimeIndex([fdate])),
        ])

    return predictions


def _forecast_lstm(artifact: Dict[str, Any], series: pd.Series, horizon: int) -> List[float]:
    """Recursive LSTM forecasting: unroll horizon steps."""
    scaler = artifact["scaler"]
    seq_len = artifact["seq_len"]
    device = torch.device("cpu")

    if artifact.get("model_state") is None:
        return [float(series.mean())] * horizon

    model = SalesLSTM(
        input_size=artifact["input_size"],
        hidden_size=artifact["hidden_size"],
        num_layers=artifact["num_layers"],
        dropout=artifact["dropout"],
    ).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()

    # Seed sequence with last `seq_len` known values
    seed = series.values[-seq_len:].reshape(-1, 1)
    seed_scaled = scaler.transform(seed).flatten()
    sequence = list(seed_scaled)

    predictions: List[float] = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.FloatTensor(sequence[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
            pred_scaled = model(x).item()
            sequence.append(pred_scaled)
            pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            predictions.append(max(0.0, pred))

    return predictions


def _forecast_prophet(model: Any, series: pd.Series, horizon: int) -> List[float]:
    future_dates = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=FREQ)[1:]
    future_df = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future_df)
    return [max(0.0, float(v)) for v in forecast["yhat"].values]


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def forecast_state(
    state: str,
    series: pd.Series,
    best_model_name: str,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Generate a forecast for a single state using its best model.

    Returns a DataFrame with columns: ['date', 'forecast', 'model'].
    """
    logger.info("Forecasting '%s' with %s (horizon=%d)", state, best_model_name, horizon)
    model = load_best_model(state, best_model_name)

    dispatchers = {
        "arima": _forecast_arima,
        "sarima": _forecast_sarima,
        "xgboost": _forecast_xgboost,
        "lstm": _forecast_lstm,
        "prophet": _forecast_prophet,
    }

    fn = dispatchers.get(best_model_name)
    if fn is None:
        raise ValueError(f"Unknown model: {best_model_name}")

    preds = fn(model, series, horizon)

    future_dates = pd.date_range(
        start=series.index[-1], periods=horizon + 1, freq=FREQ
    )[1:]

    result = pd.DataFrame({
        "date": future_dates,
        "forecast": preds,
        "model": best_model_name,
        "state": state,
    })
    return result


def forecast_all_states(
    state_series: Dict[str, pd.Series],
    model_registry: Dict[str, Dict[str, Any]],
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Generate forecasts for all states.

    Parameters
    ----------
    state_series    : {state: weekly pd.Series}
    model_registry  : {state: {"best_model": str, "metrics": {...}}}

    Returns a combined DataFrame for all states.
    """
    all_forecasts = []
    for state, series in state_series.items():
        try:
            best = model_registry[state]["best_model"]
            df = forecast_state(state, series, best, horizon)
            all_forecasts.append(df)
        except Exception as exc:
            logger.error("Forecast failed for '%s': %s", state, exc)

    if all_forecasts:
        return pd.concat(all_forecasts, ignore_index=True)
    return pd.DataFrame(columns=["date", "forecast", "model", "state"])
