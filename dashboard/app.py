"""
Streamlit dashboard — interactive forecast explorer.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import CONFIG
from src.predict import forecast_state

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)


# ── Cached loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_registry():
    path = Path(CONFIG["paths"]["models_dir"]) / "model_registry.joblib"
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_series():
    path = Path(CONFIG["paths"]["models_dir"]) / "state_series.joblib"
    return joblib.load(path)


@st.cache_data(show_spinner="Generating forecast …")
def cached_forecast(state: str, model: str, horizon: int):
    series = load_series()[state]
    registry = load_registry()
    meta = registry.get(state, {}).get("ensemble") if model == "ensemble" else None
    df = forecast_state(state, series, model, horizon=horizon, ensemble_meta=meta)
    return df


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("Forecast settings")

try:
    registry = load_registry()
    series_map = load_series()
except FileNotFoundError as e:
    st.error(f"No trained models found. Run `make train` first.\n\n{e}")
    st.stop()

states = sorted(registry.keys())
default_idx = states.index("California") if "California" in states else 0
state = st.sidebar.selectbox("State", states, index=default_idx)

best = registry[state]["best_model"]
all_models = sorted(set(registry[state]["metrics"].keys()))
model_choice = st.sidebar.selectbox(
    "Model",
    options=[best] + [m for m in all_models if m != best],
    help=f"Default = best model for this state ({best}).",
)
horizon = st.sidebar.slider("Horizon (weeks)", 1, 16, value=8)

# ── Main panel ──────────────────────────────────────────────────────────────
st.title("Sales Forecasting Dashboard")
st.caption(
    "Interactive explorer for the per-state, multi-model forecasting pipeline. "
    "Each state has its best model auto-selected by validation RMSE; you can "
    "override it from the sidebar to compare."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("State", state)
col2.metric("Best model", best)
col3.metric("Series length (weeks)", registry[state]["series_length"])
metrics = registry[state]["metrics"].get(best, {})
if metrics.get("rmse") is not None:
    col4.metric("Best RMSE (val)", f"{metrics['rmse']:,.0f}")

st.divider()

# ── Forecast plot ──────────────────────────────────────────────────────────
series = series_map[state]
test_size = CONFIG["train"]["test_size"]
split = int(len(series) * (1 - test_size))
train, val = series.iloc[:split], series.iloc[split:]

forecast_df = cached_forecast(state, model_choice, horizon)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(train.index, train.values, color="#1f77b4", label="Train (history)", lw=1.2)
ax.plot(val.index, val.values, color="#2ca02c", label="Val (held-out)", lw=1.5)
ax.plot(
    forecast_df["date"], forecast_df["forecast"],
    color="#d62728", ls="--", marker="o", lw=2, label=f"Forecast ({model_choice})",
)
ax.fill_between(
    forecast_df["date"],
    forecast_df["forecast_low"],
    forecast_df["forecast_high"],
    color="#d62728", alpha=0.18, label="95% CI",
)
ax.axvline(val.index[0], color="grey", ls=":", alpha=0.6)
ax.set_title(f"{state} — {horizon}-week forecast (model: {model_choice})")
ax.set_ylabel("Weekly sales")
ax.legend(loc="upper left")
ax.grid(linestyle=":", alpha=0.4)
st.pyplot(fig)

# ── Forecast table ─────────────────────────────────────────────────────────
st.subheader("Forecast values")
display_df = forecast_df.copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
display_df["forecast"] = display_df["forecast"].round(2)
display_df["forecast_low"] = display_df["forecast_low"].round(2)
display_df["forecast_high"] = display_df["forecast_high"].round(2)
display_df.insert(0, "week", range(1, len(display_df) + 1))
st.dataframe(
    display_df[["week", "date", "forecast", "forecast_low", "forecast_high"]],
    width="stretch",
    hide_index=True,
)

# ── Validation metrics ─────────────────────────────────────────────────────
st.subheader("All models — validation metrics for this state")
mdf = pd.DataFrame(registry[state]["metrics"]).T.round(2)
mdf = mdf.sort_values("rmse")
mdf["selected"] = mdf.index == best
st.dataframe(mdf, width="stretch")

# ── Global view ────────────────────────────────────────────────────────────
with st.expander("Best-model distribution across all states"):
    counts = pd.Series([r["best_model"] for r in registry.values()]).value_counts()
    st.bar_chart(counts)

st.caption(
    "Powered by FastAPI + Prophet + statsmodels + XGBoost + PyTorch. "
    "Source: `src/`, `api/`, `dashboard/app.py`."
)
