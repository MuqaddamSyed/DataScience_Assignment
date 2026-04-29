"""
Model training pipeline.

Models implemented:
─────────────────────────────────────────────────────────────────
A. ARIMA   — captures autoregressive + moving-average structure in
             stationary time series.  Fast, interpretable.
             Limitation: univariate, no external features.

B. SARIMA  — extends ARIMA with seasonal (52-week) components,
             important for annual beverage sales cycles.
             Limitation: slow to fit; can over-parameterise on small data.

C. XGBoost — gradient-boosted trees on engineered features (lags,
             rolling stats, date features, holidays).  Handles
             non-linearity and feature interactions very well.
             Limitation: requires feature engineering; no native
             sequence modelling.

D. LSTM    — recurrent deep network that learns temporal dependencies
             from raw sequences.  Good for capturing long-range patterns.
             Limitation: needs substantial data; prone to overfitting
             on short series; requires scaling.

E. Prophet — Facebook's additive decomposition model with built-in
             trend changepoints, yearly/weekly seasonality and holidays.
             Extremely robust to missing data and outliers.
             Limitation: limited feature extensibility.

Best-model selection:
  Each model is scored on the hold-out validation set (last 20 %).
  The model with the lowest RMSE on that set is chosen as the
  production model for that state.  RMSE is used as the primary
  metric because large errors are penalised quadratically, which
  aligns with business impact (large stock shortfalls are costly).
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.config_loader import CONFIG
from src.evaluate import compute_metrics
from src.feature_engineering import build_features, get_feature_columns, train_val_split
from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

MODELS_DIR = Path(CONFIG["paths"]["models_dir"])
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SEED = CONFIG["project"]["random_seed"]
torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# LSTM model definition
# ══════════════════════════════════════════════════════════════════════════════

class SalesLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _state_path(state: str, model_name: str, ext: str) -> Path:
    safe = state.replace(" ", "_").lower()
    return MODELS_DIR / f"{safe}_{model_name}.{ext}"


def _make_sequences(
    values: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding (X, y) sequences for LSTM training."""
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len: i])
        y.append(values[i])
    return np.array(X), np.array(y)


# ══════════════════════════════════════════════════════════════════════════════
# ARIMA
# ══════════════════════════════════════════════════════════════════════════════

def train_arima(
    train_series: pd.Series,
    val_series: pd.Series,
    state: str,
) -> Tuple[Any, Dict[str, float]]:
    cfg = CONFIG["models"]["arima"]
    best_aic, best_order, best_model = np.inf, (1, 1, 1), None

    for p in range(cfg["max_p"] + 1):
        for d in range(cfg["max_d"] + 1):
            for q in range(cfg["max_q"] + 1):
                try:
                    m = ARIMA(train_series, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_order, best_model = m.aic, (p, d, q), m
                except Exception:
                    continue

    if best_model is None:
        logger.warning("ARIMA failed for state '%s'; using (1,1,1)", state)
        best_model = ARIMA(train_series, order=(1, 1, 1)).fit()

    logger.info("  ARIMA best order %s (AIC=%.1f) for '%s'", best_order, best_aic, state)

    preds = best_model.forecast(steps=len(val_series))
    preds = pd.Series(preds.values, index=val_series.index)
    metrics = compute_metrics(val_series, preds)

    path = _state_path(state, "arima", "pkl")
    with open(path, "wb") as f:
        pickle.dump(best_model, f)

    return best_model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# SARIMA
# ══════════════════════════════════════════════════════════════════════════════

def train_sarima(
    train_series: pd.Series,
    val_series: pd.Series,
    state: str,
) -> Tuple[Any, Dict[str, float]]:
    cfg = CONFIG["models"]["sarima"]
    order = tuple(cfg["order"])
    seas_order = tuple(cfg["seasonal_order"])

    # Reduce seasonal period if data is too short
    n = len(train_series)
    m = seas_order[3]
    if n < 2 * m:
        m = 4   # fall back to quarterly
        seas_order = (seas_order[0], seas_order[1], seas_order[2], m)
        logger.info("  SARIMA: short series for '%s', seasonal period → %d", state, m)

    try:
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seas_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    except Exception as exc:
        logger.warning("  SARIMA fallback (1,1,1)(0,0,0,0) for '%s': %s", state, exc)
        model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=False)

    preds = model.forecast(steps=len(val_series))
    preds = pd.Series(preds.values, index=val_series.index)
    metrics = compute_metrics(val_series, preds)

    path = _state_path(state, "sarima", "pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    return model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    state: str,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    cfg = CONFIG["models"]["xgboost"]
    feat_cols = get_feature_columns(train_df)

    X_train = train_df[feat_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feat_cols].values
    y_val = val_df["target"].values

    # Pass all configured params (including any added by run_tuning.py).
    # Defaults are merged in for safety.
    xgb_kwargs = {
        "verbosity": 0,
        "eval_metric": "rmse",
        "early_stopping_rounds": 20,
        **{k: v for k, v in cfg.items() if k not in ("verbosity",)},
    }
    model = XGBRegressor(**xgb_kwargs)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_val)
    preds = pd.Series(preds, index=val_df.index)
    metrics = compute_metrics(pd.Series(y_val, index=val_df.index), preds)

    # Persist residual std so predict.py can construct 95% CIs.
    # The bootstrap assumption: future errors ~ N(0, residual_std²).
    residuals = y_val - preds.values
    residual_std = float(np.std(residuals))

    artifact = {"model": model, "residual_std": residual_std}
    path = _state_path(state, "xgboost", "joblib")
    joblib.dump(artifact, path)

    return artifact, metrics


# ══════════════════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(
    train_series: pd.Series,
    val_series: pd.Series,
    state: str,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    cfg = CONFIG["models"]["lstm"]
    seq_len = cfg["sequence_length"]
    device = torch.device("cpu")

    # Fit scaler ONLY on training data to prevent validation leakage.
    # If validation values exceed train range they will be clipped slightly
    # outside [0, 1] when transformed — that is acceptable and realistic.
    scaler = MinMaxScaler()
    scaler.fit(train_series.values.reshape(-1, 1))

    train_scaled = scaler.transform(train_series.values.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_series.values.reshape(-1, 1)).flatten()

    combined = np.concatenate([train_scaled, val_scaled])

    X_train, y_train = _make_sequences(train_scaled, seq_len)
    # For validation: sequences come from train tail + val
    full_combined = combined
    X_val, y_val = [], []
    for i in range(len(train_scaled), len(full_combined)):
        if i >= seq_len:
            X_val.append(full_combined[i - seq_len: i])
            y_val.append(full_combined[i])
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    if len(X_train) == 0:
        logger.warning("  LSTM: not enough data for '%s'", state)
        dummy = {"model_state": None, "scaler": scaler, "seq_len": seq_len, "input_size": 1}
        return dummy, {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(-1).to(device) if len(X_val) > 0 else None
    y_val_t = torch.FloatTensor(y_val).unsqueeze(-1).to(device) if len(y_val) > 0 else None

    model = SalesLSTM(
        input_size=1,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if X_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= cfg["patience"]:
                logger.debug("  LSTM early stop @ epoch %d for '%s'", epoch, state)
                break

    if best_state:
        model.load_state_dict(best_state)

    # Compute metrics + residual std on validation
    residual_std = 0.0
    if X_val_t is not None and len(X_val_t) > 0:
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_val_t).cpu().numpy().flatten()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        actual = val_series.values[-len(preds):]
        metrics = compute_metrics(
            pd.Series(actual), pd.Series(preds)
        )
        residual_std = float(np.std(actual - preds))
    else:
        metrics = {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    artifact = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "seq_len": seq_len,
        "input_size": 1,
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg["num_layers"],
        "dropout": cfg["dropout"],
        "train_tail": train_series.values[-seq_len:],
        "residual_std": residual_std,
    }
    path = _state_path(state, "lstm", "joblib")
    joblib.dump(artifact, path)

    return artifact, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Prophet
# ══════════════════════════════════════════════════════════════════════════════

def train_prophet(
    train_series: pd.Series,
    val_series: pd.Series,
    state: str,
) -> Tuple[Any, Dict[str, float]]:
    try:
        from prophet import Prophet  # lazy import to avoid slow startup
    except ImportError:
        logger.error("Prophet not installed; skipping for state '%s'", state)
        return None, {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    cfg = CONFIG["models"]["prophet"]
    country = CONFIG["features"]["holiday_country"]

    train_df = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values,
    })

    m = Prophet(
        yearly_seasonality=cfg["yearly_seasonality"],
        weekly_seasonality=cfg["weekly_seasonality"],
        daily_seasonality=cfg["daily_seasonality"],
        changepoint_prior_scale=cfg["changepoint_prior_scale"],
        seasonality_prior_scale=cfg["seasonality_prior_scale"],
    )
    try:
        m.add_country_holidays(country_name=country)
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train_df)

    future = pd.DataFrame({"ds": val_series.index})
    forecast = m.predict(future)
    preds = pd.Series(forecast["yhat"].values, index=val_series.index)
    preds = preds.clip(lower=0)
    metrics = compute_metrics(val_series, preds)

    path = _state_path(state, "prophet", "joblib")
    joblib.dump(m, path)

    return m, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Master training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    state: str,
    series: pd.Series,
) -> Dict[str, Any]:
    """
    Train all models for a single state and return a results dict.

    Returns
    -------
    {
        "state": str,
        "metrics": {model_name: {rmse, mae, mape}},
        "best_model": str,
        "series_length": int,
    }
    """
    test_size = CONFIG["train"]["test_size"]
    min_train = CONFIG["train"]["min_train_size"]

    # Build feature frame for ML models
    feature_df = build_features(series, state=state)
    train_df, val_df = train_val_split(feature_df, test_size=test_size)

    if len(train_df) < min_train:
        logger.warning("State '%s': insufficient train data (%d rows)", state, len(train_df))

    # Raw series splits (for ARIMA / SARIMA / LSTM / Prophet)
    split_idx = int(len(series) * (1 - test_size))
    train_series = series.iloc[:split_idx]
    val_series = series.iloc[split_idx:]

    results: Dict[str, Dict[str, float]] = {}

    # ── A. ARIMA ─────────────────────────────────────────────────────────────
    logger.info("[%s] Training ARIMA …", state)
    try:
        _, arima_metrics = train_arima(train_series, val_series, state)
        results["arima"] = arima_metrics
        logger.info("[%s] ARIMA  RMSE=%.2f", state, arima_metrics["rmse"])
    except Exception as exc:
        logger.error("[%s] ARIMA failed: %s", state, exc)
        results["arima"] = {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    # ── B. SARIMA ─────────────────────────────────────────────────────────────
    logger.info("[%s] Training SARIMA …", state)
    try:
        _, sarima_metrics = train_sarima(train_series, val_series, state)
        results["sarima"] = sarima_metrics
        logger.info("[%s] SARIMA RMSE=%.2f", state, sarima_metrics["rmse"])
    except Exception as exc:
        logger.error("[%s] SARIMA failed: %s", state, exc)
        results["sarima"] = {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    # ── C. XGBoost ────────────────────────────────────────────────────────────
    logger.info("[%s] Training XGBoost …", state)
    try:
        _, xgb_metrics = train_xgboost(train_df, val_df, state)
        results["xgboost"] = xgb_metrics
        logger.info("[%s] XGBoost RMSE=%.2f", state, xgb_metrics["rmse"])
    except Exception as exc:
        logger.error("[%s] XGBoost failed: %s", state, exc)
        results["xgboost"] = {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    # ── D. LSTM ───────────────────────────────────────────────────────────────
    logger.info("[%s] Training LSTM …", state)
    try:
        _, lstm_metrics = train_lstm(train_series, val_series, state)
        results["lstm"] = lstm_metrics
        logger.info("[%s] LSTM   RMSE=%.2f", state, lstm_metrics["rmse"])
    except Exception as exc:
        logger.error("[%s] LSTM failed: %s", state, exc)
        results["lstm"] = {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    # ── E. Prophet ────────────────────────────────────────────────────────────
    logger.info("[%s] Training Prophet …", state)
    try:
        _, prophet_metrics = train_prophet(train_series, val_series, state)
        results["prophet"] = prophet_metrics
        logger.info("[%s] Prophet RMSE=%.2f", state, prophet_metrics["rmse"])
    except Exception as exc:
        logger.error("[%s] Prophet failed: %s", state, exc)
        results["prophet"] = {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    # ── Build top-2 ensemble (inverse-RMSE-weighted) ─────────────────────────
    valid = {k: v for k, v in results.items() if np.isfinite(v["rmse"])}
    ensemble_meta = None
    if len(valid) >= 2:
        top2 = sorted(valid.keys(), key=lambda k: valid[k]["rmse"])[:2]
        weights = [1.0 / valid[m]["rmse"] for m in top2]
        weights = [w / sum(weights) for w in weights]
        ens_metrics = _evaluate_ensemble(top2, val_series, val_df, state, train_series, train_df)
        if np.isfinite(ens_metrics["rmse"]):
            results["ensemble"] = ens_metrics
            ensemble_meta = {"components": top2, "weights": weights}
            logger.info(
                "[%s] Ensemble(%s) RMSE=%.2f", state, "+".join(top2), ens_metrics["rmse"],
            )

    # ── Select best by RMSE (now possibly the ensemble) ───────────────────────
    if not valid and "ensemble" not in results:
        best_model = "arima"
        logger.error("[%s] All models failed! Defaulting to ARIMA.", state)
    else:
        candidates = {k: v for k, v in results.items() if np.isfinite(v["rmse"])}
        best_model = min(candidates, key=lambda k: candidates[k]["rmse"])

    logger.info(
        "[%s] ✓ Best model: %s (RMSE=%.2f)",
        state, best_model, results[best_model]["rmse"],
    )

    payload = {
        "state": state,
        "metrics": results,
        "best_model": best_model,
        "series_length": len(series),
    }
    if ensemble_meta and best_model == "ensemble":
        payload["ensemble"] = ensemble_meta
    elif ensemble_meta:
        # Keep ensemble metadata even if not best — useful for /models inspection
        payload["ensemble"] = ensemble_meta
    return payload


def _evaluate_ensemble(
    component_names: list,
    val_series: pd.Series,
    val_df: pd.DataFrame,
    state: str,
    train_series: pd.Series,
    train_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Re-derive validation predictions for each component and average them
    using inverse-RMSE weights from the just-finished training pass.

    This is "in-sample" relative to the components (each was already fit on
    the same train slice), so we read each component's predictions cheaply.
    """
    preds_per_component = []
    for name in component_names:
        try:
            preds = _component_val_predictions(name, train_series, val_series, train_df, val_df, state)
            if preds is None:
                continue
            preds_per_component.append(np.asarray(preds))
        except Exception as exc:
            logger.warning("[%s] ensemble component %s failed: %s", state, name, exc)
            return {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    if len(preds_per_component) < 2:
        return {"rmse": np.inf, "mae": np.inf, "mape": np.inf}

    # Equal weights here; weights are stored in registry for inference time.
    avg = np.mean(np.stack(preds_per_component), axis=0)
    return compute_metrics(
        pd.Series(val_series.values), pd.Series(avg)
    )


def _component_val_predictions(
    name: str,
    train_series: pd.Series,
    val_series: pd.Series,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    state: str,
) -> np.ndarray | None:
    """Load the just-saved component model and produce val predictions."""
    if name == "arima" or name == "sarima":
        path = _state_path(state, name, "pkl")
        with open(path, "rb") as f:
            m = pickle.load(f)
        return np.asarray(m.forecast(steps=len(val_series)))
    if name == "xgboost":
        path = _state_path(state, "xgboost", "joblib")
        artifact = joblib.load(path)
        m = artifact["model"] if isinstance(artifact, dict) else artifact
        feat_cols = get_feature_columns(train_df)
        return m.predict(val_df[feat_cols].values)
    if name == "lstm":
        path = _state_path(state, "lstm", "joblib")
        artifact = joblib.load(path)
        if artifact.get("model_state") is None:
            return None
        scaler = artifact["scaler"]
        seq_len = artifact["seq_len"]
        m = SalesLSTM(
            input_size=artifact["input_size"], hidden_size=artifact["hidden_size"],
            num_layers=artifact["num_layers"], dropout=artifact["dropout"],
        )
        m.load_state_dict(artifact["model_state"])
        m.eval()
        train_scaled = scaler.transform(train_series.values.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val_series.values.reshape(-1, 1)).flatten()
        full = np.concatenate([train_scaled, val_scaled])
        Xv = []
        for i in range(len(train_scaled), len(full)):
            if i >= seq_len:
                Xv.append(full[i - seq_len: i])
        if not Xv:
            return None
        with torch.no_grad():
            pred_scaled = m(torch.FloatTensor(np.array(Xv)).unsqueeze(-1)).numpy().flatten()
        preds = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        # If sequence-window cuts initial val rows, pad with the first prediction
        if len(preds) < len(val_series):
            preds = np.concatenate([np.full(len(val_series) - len(preds), preds[0]), preds])
        return preds[: len(val_series)]
    if name == "prophet":
        path = _state_path(state, "prophet", "joblib")
        m = joblib.load(path)
        future = pd.DataFrame({"ds": val_series.index})
        return np.clip(m.predict(future)["yhat"].values, 0, None)
    return None
