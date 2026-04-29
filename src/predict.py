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
from typing import Any, Dict, List, Tuple

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
# Per-model forecasters — each returns (point, low, high) lists of length `horizon`
#
# ARIMA / SARIMA / Prophet use native confidence intervals from their
# distributional assumptions. XGBoost / LSTM use bootstrap CIs from
# validation residuals (point ± 1.96·σ for 95%).
# ══════════════════════════════════════════════════════════════════════════════

ForecastTriplet = Tuple[List[float], List[float], List[float]]
_Z_95 = 1.96  # standard normal critical value for 95% CI


def _expand_horizon_steps(steps: int) -> List[float]:
    """Multiplier so uncertainty grows with horizon (sqrt(t) random walk)."""
    return [float(np.sqrt(i)) for i in range(1, steps + 1)]


def _forecast_arima(model: Any, series: pd.Series, horizon: int) -> ForecastTriplet:
    res = model.get_forecast(steps=horizon)
    point = res.predicted_mean.values
    ci = res.conf_int(alpha=0.05).values  # [low, high]
    low, high = ci[:, 0], ci[:, 1]
    return (
        [max(0.0, float(v)) for v in point],
        [max(0.0, float(v)) for v in low],
        [max(0.0, float(v)) for v in high],
    )


def _forecast_sarima(model: Any, series: pd.Series, horizon: int) -> ForecastTriplet:
    return _forecast_arima(model, series, horizon)


def _forecast_xgboost(artifact: Any, series: pd.Series, horizon: int) -> ForecastTriplet:
    """Recursive multi-step forecasting with XGBoost. Returns (point, low, high)."""
    # Backward compatibility: artifact can be the bare model (old) or a dict (new)
    if isinstance(artifact, dict):
        model = artifact["model"]
        residual_std = artifact.get("residual_std", 0.0)
    else:
        model = artifact
        residual_std = 0.0

    extended = series.copy()
    future_dates = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=FREQ)[1:]
    point: List[float] = []

    for fdate in future_dates:
        placeholder = pd.Series([np.nan], index=pd.DatetimeIndex([fdate]))
        temp = pd.concat([extended, placeholder])
        feat_df = build_features(temp)

        if fdate not in feat_df.index:
            fallback = float(max(0.0, float(extended.iloc[-1])))
            point.append(fallback)
            extended = pd.concat([extended, pd.Series([fallback], index=pd.DatetimeIndex([fdate]))])
            continue

        row = feat_df.loc[[fdate]]
        feat_cols = get_feature_columns(row)
        pred = float(model.predict(row[feat_cols].values)[0])
        pred = max(0.0, pred)
        point.append(pred)
        extended = pd.concat([extended, pd.Series([pred], index=pd.DatetimeIndex([fdate]))])

    expand = _expand_horizon_steps(horizon)
    low = [max(0.0, p - _Z_95 * residual_std * e) for p, e in zip(point, expand)]
    high = [p + _Z_95 * residual_std * e for p, e in zip(point, expand)]
    return point, low, high


def _forecast_lstm(artifact: Dict[str, Any], series: pd.Series, horizon: int) -> ForecastTriplet:
    """Recursive LSTM forecasting with bootstrap CIs."""
    scaler = artifact["scaler"]
    seq_len = artifact["seq_len"]
    residual_std = artifact.get("residual_std", 0.0)
    device = torch.device("cpu")

    if artifact.get("model_state") is None:
        flat = [float(series.mean())] * horizon
        return flat, flat, flat

    model = SalesLSTM(
        input_size=artifact["input_size"],
        hidden_size=artifact["hidden_size"],
        num_layers=artifact["num_layers"],
        dropout=artifact["dropout"],
    ).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()

    seed = series.values[-seq_len:].reshape(-1, 1)
    seed_scaled = scaler.transform(seed).flatten()
    sequence = list(seed_scaled)

    point: List[float] = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.FloatTensor(sequence[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
            pred_scaled = model(x).item()
            sequence.append(pred_scaled)
            pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            point.append(max(0.0, pred))

    expand = _expand_horizon_steps(horizon)
    low = [max(0.0, p - _Z_95 * residual_std * e) for p, e in zip(point, expand)]
    high = [p + _Z_95 * residual_std * e for p, e in zip(point, expand)]
    return point, low, high


def _forecast_prophet(model: Any, series: pd.Series, horizon: int) -> ForecastTriplet:
    future_dates = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=FREQ)[1:]
    future_df = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future_df)
    point = [max(0.0, float(v)) for v in forecast["yhat"].values]
    low = [max(0.0, float(v)) for v in forecast["yhat_lower"].values]
    high = [max(0.0, float(v)) for v in forecast["yhat_upper"].values]
    return point, low, high


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

_DISPATCHERS = {
    "arima": _forecast_arima,
    "sarima": _forecast_sarima,
    "xgboost": _forecast_xgboost,
    "lstm": _forecast_lstm,
    "prophet": _forecast_prophet,
}


def _forecast_single(
    state: str, series: pd.Series, model_name: str, horizon: int,
) -> ForecastTriplet:
    """Load one base model and produce (point, low, high)."""
    if model_name not in _DISPATCHERS:
        raise ValueError(f"Unknown model: {model_name}")
    model = load_best_model(state, model_name)
    return _DISPATCHERS[model_name](model, series, horizon)


def _forecast_ensemble(
    state: str,
    series: pd.Series,
    components: List[str],
    weights: List[float],
    horizon: int,
) -> ForecastTriplet:
    """Weighted average of component forecasts."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()  # safety normalisation
    pts, lows, highs = [], [], []
    for c in components:
        p, l, h = _forecast_single(state, series, c, horizon)
        pts.append(p); lows.append(l); highs.append(h)
    pts = np.average(np.array(pts), axis=0, weights=weights)
    lows = np.average(np.array(lows), axis=0, weights=weights)
    highs = np.average(np.array(highs), axis=0, weights=weights)
    return pts.tolist(), lows.tolist(), highs.tolist()


def forecast_state(
    state: str,
    series: pd.Series,
    best_model_name: str,
    horizon: int = HORIZON,
    ensemble_meta: dict | None = None,
) -> pd.DataFrame:
    """
    Generate a forecast for a single state using its selected best model.

    If `best_model_name == "ensemble"`, `ensemble_meta` must be supplied
    with keys 'components' (list of model names) and 'weights' (list of floats).

    Returns DataFrame with columns:
        ['date', 'forecast', 'forecast_low', 'forecast_high', 'model', 'state']
    """
    logger.info("Forecasting '%s' with %s (horizon=%d)", state, best_model_name, horizon)

    if best_model_name == "ensemble":
        if not ensemble_meta:
            raise ValueError("ensemble_meta is required when best_model_name=='ensemble'")
        point, low, high = _forecast_ensemble(
            state, series,
            ensemble_meta["components"], ensemble_meta["weights"],
            horizon,
        )
    else:
        point, low, high = _forecast_single(state, series, best_model_name, horizon)

    future_dates = pd.date_range(
        start=series.index[-1], periods=horizon + 1, freq=FREQ
    )[1:]
    return pd.DataFrame({
        "date": future_dates,
        "forecast": point,
        "forecast_low": low,
        "forecast_high": high,
        "model": best_model_name,
        "state": state,
    })


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
