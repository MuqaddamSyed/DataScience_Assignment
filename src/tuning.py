"""
Hyperparameter tuning for XGBoost using Optuna with walk-forward CV.

Why Optuna + walk-forward:
- A single train/val split would overfit hyper-parameters to one specific
  validation window. Walk-forward CV averages metric across folds and is
  the right way to tune time-series models.
- Optuna's Tree-structured Parzen Estimator (TPE) sampler is more efficient
  than grid or random search and converges in 30–50 trials.

We tune three states (one large, one median, one small) and use the average
of their best params as the new defaults written into config.yaml. Tuning
all 43 states would multiply training time by 43× without much marginal
gain — model defaults are already reasonable.

Output:
- models/tuned_xgboost.json   — best params + study summary
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from src.config_loader import CONFIG
from src.cross_validation import expanding_window_splits
from src.evaluate import compute_metrics
from src.feature_engineering import build_features, get_feature_columns
from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _xgb_cv_score(
    series: pd.Series,
    params: Dict[str, Any],
    n_splits: int = 4,
    val_horizon: int = 8,
) -> float:
    """
    Average RMSE across `n_splits` walk-forward folds for one parameter set.

    For each fold:
      1. Build features on the train+val concatenation (no leakage — features
         only reach back into the past via shift).
      2. Train XGBoost on the train portion.
      3. Predict the val portion (8 weeks).
    """
    full_features = build_features(series)
    rmses = []

    for train_series, val_series in expanding_window_splits(
        series, n_splits=n_splits, val_horizon=val_horizon
    ):
        train_df = full_features.loc[full_features.index.isin(train_series.index)]
        val_df = full_features.loc[full_features.index.isin(val_series.index)]
        if len(train_df) < 30 or len(val_df) == 0:
            continue

        feat_cols = get_feature_columns(train_df)
        X_tr, y_tr = train_df[feat_cols].values, train_df["target"].values
        X_va, y_va = val_df[feat_cols].values, val_df["target"].values

        model = XGBRegressor(**params, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        rmses.append(np.sqrt(np.mean((y_va - preds) ** 2)))

    if not rmses:
        return float("inf")
    return float(np.mean(rmses))


def _objective_xgboost(trial: optuna.Trial, series: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "random_state": 42,
        "early_stopping_rounds": 20,
        "eval_metric": "rmse",
    }
    return _xgb_cv_score(series, params)


def tune_xgboost(
    series_dict: Dict[str, pd.Series],
    n_trials: int = 30,
    states: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Tune XGBoost hyper-parameters across `states` (default: pick 3 by size).
    Returns a dict with per-state best params and the averaged params.
    """
    if states is None:
        sizes = {s: ss.sum() for s, ss in series_dict.items()}
        ranked = sorted(sizes, key=sizes.get, reverse=True)
        states = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]

    per_state = {}
    for state in states:
        if state not in series_dict:
            logger.warning("Tune: skipping unknown state %s", state)
            continue
        logger.info("[%s] Optuna tuning XGBoost (%d trials)", state, n_trials)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda t, s=state: _objective_xgboost(t, series_dict[s]),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        per_state[state] = {
            "best_params": study.best_params,
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
        }
        logger.info("[%s] best RMSE=%.2f, params=%s",
                    state, study.best_value, study.best_params)

    # Average numeric params (median) across states for a portable default
    keys = next(iter(per_state.values()))["best_params"].keys()
    averaged = {}
    for k in keys:
        vals = [per_state[s]["best_params"][k] for s in per_state]
        if isinstance(vals[0], int):
            averaged[k] = int(np.median(vals))
        else:
            averaged[k] = float(np.median(vals))
    averaged["random_state"] = 42

    return {"per_state": per_state, "averaged_params": averaged, "tuned_states": list(per_state.keys())}


def save_tuning_result(result: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved tuning result → %s", path)
