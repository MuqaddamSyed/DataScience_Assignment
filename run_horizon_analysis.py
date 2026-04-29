"""
Per-step horizon error analysis.

Multi-step forecasts have a property single-step ones don't: error grows
with horizon. The 1-week-ahead prediction is usually much better than the
8-week-ahead one because:
- Recursive ML models accumulate prediction error.
- Stochastic models (ARIMA/SARIMA/Prophet) explicitly widen their CIs
  with horizon under the random-walk assumption.

This script:
1. For each (state, model), fits on train and predicts the 8-week val window.
2. Records per-step absolute error (|actual - predicted|).
3. Aggregates across all states to show median per-step error per model.
4. Saves:
   - reports/horizon_error.csv
   - reports/horizon_error.png  (lines: median error vs forecast step,
                                  shaded IQR, one line per model)

Usage:
    python run_horizon_analysis.py [--states CA TX ...]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import CONFIG
from src.cross_validation import _fit_arima, _fit_prophet, _fit_sarima
from src.feature_engineering import build_features, get_feature_columns
from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
HORIZON = CONFIG["data"]["forecast_horizon"]
TEST_SIZE = CONFIG["train"]["test_size"]


def _fit_xgb(train: pd.Series, horizon: int, full_series: pd.Series) -> pd.Series:
    """Fit XGBoost on train slice and forecast `horizon` steps via recursive prediction."""
    from xgboost import XGBRegressor

    # Build features on the full series so val features match production
    feats = build_features(full_series)
    train_feats = feats.loc[feats.index.isin(train.index)]
    feat_cols = get_feature_columns(train_feats)

    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.85, random_state=42,
        verbosity=0, eval_metric="rmse",
    )
    model.fit(train_feats[feat_cols].values, train_feats["target"].values)

    # Recursive
    extended = train.copy()
    preds = []
    future_dates = pd.date_range(start=train.index[-1], periods=horizon + 1, freq="W")[1:]
    for fdate in future_dates:
        placeholder = pd.Series([np.nan], index=pd.DatetimeIndex([fdate]))
        temp = pd.concat([extended, placeholder])
        f = build_features(temp)
        if fdate not in f.index:
            preds.append(extended.iloc[-1])
        else:
            row = f.loc[[fdate]][feat_cols].values
            preds.append(max(0.0, float(model.predict(row)[0])))
        extended = pd.concat([extended, pd.Series([preds[-1]], index=pd.DatetimeIndex([fdate]))])
    return pd.Series(preds)


def per_step_errors_for_state(state: str, series: pd.Series) -> pd.DataFrame:
    """Return DataFrame [model, step, abs_error, pct_error] for one state."""
    split = int(len(series) * (1 - TEST_SIZE))
    train, val = series.iloc[:split], series.iloc[split: split + HORIZON]
    if len(val) < HORIZON:
        return pd.DataFrame()

    rows = []
    fits = {
        "arima": lambda t, h: _fit_arima(t, h),
        "sarima": lambda t, h: _fit_sarima(t, h),
        "prophet": lambda t, h: _fit_prophet(t, h),
        "xgboost": lambda t, h: _fit_xgb(t, h, series),
    }
    for name, fn in fits.items():
        try:
            preds = fn(train, HORIZON).values
            actual = val.values
            for step in range(1, HORIZON + 1):
                err = abs(actual[step - 1] - preds[step - 1])
                pct = err / max(actual[step - 1], 1e-8) * 100
                rows.append({
                    "state": state, "model": name, "step": step,
                    "abs_error": err, "pct_error": pct,
                })
        except Exception as exc:
            logger.warning("[%s] %s failed: %s", state, name, exc)

    return pd.DataFrame(rows)


def plot_horizon_error(all_errors: pd.DataFrame) -> None:
    """Median pct_error per step per model, shaded IQR band."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"arima": "#1f77b4", "sarima": "#ff7f0e",
              "xgboost": "#2ca02c", "prophet": "#9467bd"}

    for model, grp in all_errors.groupby("model"):
        per_step = grp.groupby("step")["pct_error"]
        med = per_step.median()
        q1, q3 = per_step.quantile(0.25), per_step.quantile(0.75)
        c = colors.get(model, "#888")
        ax.plot(med.index, med.values, marker="o", color=c, label=f"{model}", lw=2)
        ax.fill_between(med.index, q1.values, q3.values, color=c, alpha=0.12)

    ax.set_xlabel("Forecast step (week ahead)")
    ax.set_ylabel("Median absolute % error (across states, IQR shaded)")
    ax.set_title("Forecast accuracy degradation over horizon", fontsize=13)
    ax.legend(title="Model")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_xticks(range(1, HORIZON + 1))
    plt.tight_layout()
    out = REPORTS_DIR / "horizon_error.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("Saved: %s", out)


def main(states: list[str] | None) -> None:
    series_map = joblib.load(Path(CONFIG["paths"]["models_dir"]) / "state_series.joblib")
    if not states:
        # 8 representative states (largest, smallest, mid, plus volatile)
        sizes = {s: ss.sum() for s, ss in series_map.items()}
        ranked = sorted(sizes, key=sizes.get, reverse=True)
        states = ranked[:3] + ranked[len(ranked)//2 - 1: len(ranked)//2 + 1] + ranked[-3:]
    logger.info("Horizon analysis on %d states", len(states))

    all_errors = []
    for i, state in enumerate(states, 1):
        if state not in series_map:
            continue
        logger.info("[%d/%d] %s", i, len(states), state)
        df = per_step_errors_for_state(state, series_map[state])
        if not df.empty:
            all_errors.append(df)

    errors = pd.concat(all_errors, ignore_index=True)
    errors.to_csv(REPORTS_DIR / "horizon_error.csv", index=False)
    plot_horizon_error(errors)

    print("\n" + "=" * 70)
    print("MEDIAN % ERROR BY HORIZON STEP")
    print("=" * 70)
    pivot = errors.groupby(["model", "step"])["pct_error"].median().unstack().round(2)
    print(pivot.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--states", nargs="*", default=None)
    args = p.parse_args()
    main(states=args.states)
