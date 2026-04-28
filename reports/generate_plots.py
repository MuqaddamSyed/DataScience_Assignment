"""
Generate diagnostic plots from the trained model registry.

Outputs (saved to reports/):
  • model_distribution.png   — bar chart of best-model counts across states
  • mape_by_state.png        — sorted MAPE for each state's best model
  • forecast_<state>.png     — train + val + 8-week forecast for 4 hand-picked states
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config_loader import CONFIG
from src.logger import get_logger
from src.predict import forecast_state

logger = get_logger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(CONFIG["paths"]["models_dir"])
TEST_SIZE = CONFIG["train"]["test_size"]


# ── Load artefacts ──────────────────────────────────────────────────────────
def _load() -> tuple[Dict, Dict[str, pd.Series]]:
    registry = joblib.load(MODELS_DIR / "model_registry.joblib")
    series = joblib.load(MODELS_DIR / "state_series.joblib")
    return registry, series


# ── 1. Best-model distribution ──────────────────────────────────────────────
def plot_model_distribution(registry: Dict) -> None:
    counts = pd.Series(
        [r["best_model"] for r in registry.values()]
    ).value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index, counts.values,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    )
    ax.set_title("Best Model Selected per State (lowest validation RMSE)", fontsize=13)
    ax.set_ylabel("Number of states")
    ax.set_xlabel("Model")
    for bar, n in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(n),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = REPORTS_DIR / "model_distribution.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("Saved: %s", out)


# ── 2. MAPE per state ───────────────────────────────────────────────────────
def plot_mape_by_state(registry: Dict) -> None:
    rows = []
    for state, r in registry.items():
        best = r["best_model"]
        m = r["metrics"].get(best, {})
        if "mape" in m and np.isfinite(m["mape"]):
            rows.append((state, best, m["mape"]))
    df = pd.DataFrame(rows, columns=["state", "model", "mape"])
    df = df.sort_values("mape")

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = {
        "arima": "#1f77b4",
        "sarima": "#ff7f0e",
        "xgboost": "#2ca02c",
        "lstm": "#d62728",
        "prophet": "#9467bd",
    }
    bar_colors = [colors.get(m, "#888") for m in df["model"]]
    ax.barh(df["state"], df["mape"], color=bar_colors)
    ax.set_xlabel("Validation MAPE (%) — lower is better")
    ax.set_title("Best-Model MAPE per State", fontsize=13)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=colors[m], label=m)
        for m in df["model"].unique()
    ]
    ax.legend(handles=legend_handles, loc="lower right", title="Model")

    plt.tight_layout()
    out = REPORTS_DIR / "mape_by_state.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("Saved: %s", out)


# ── 3. Per-state forecast plots ─────────────────────────────────────────────
def plot_forecast_for_state(
    state: str,
    series: pd.Series,
    registry_entry: Dict,
) -> None:
    best_model = registry_entry["best_model"]
    split_idx = int(len(series) * (1 - TEST_SIZE))
    train, val = series.iloc[:split_idx], series.iloc[split_idx:]

    # Generate the 8-week future forecast from the trained best model
    try:
        future_df = forecast_state(state, series, best_model, horizon=8)
    except Exception as exc:
        logger.error("Forecast failed for %s: %s", state, exc)
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train.index, train.values, color="#1f77b4", label="Train (history)", lw=1.4)
    ax.plot(val.index, val.values, color="#2ca02c", label="Val (held-out)", lw=1.6)
    ax.plot(
        future_df["date"], future_df["forecast"],
        color="#d62728", ls="--", marker="o", lw=2, label=f"Forecast ({best_model})",
    )
    # Vertical line at train/val boundary
    ax.axvline(val.index[0], color="grey", ls=":", alpha=0.6)
    ax.set_title(f"{state} — 8-Week Sales Forecast (best model: {best_model})", fontsize=13)
    ax.set_ylabel("Weekly sales")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    safe = state.lower().replace(" ", "_")
    out = REPORTS_DIR / f"forecast_{safe}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("Saved: %s", out)


# ── Main ────────────────────────────────────────────────────────────────────
def pick_demo_states(registry: Dict, series_map: Dict[str, pd.Series]) -> List[str]:
    """Pick 4 representative states: largest, smallest, and one per non-prophet model."""
    by_size = sorted(series_map.items(), key=lambda kv: kv[1].sum(), reverse=True)
    largest = by_size[0][0]
    smallest = by_size[-1][0]

    # Pick first state using LSTM and first using SARIMA (if any)
    by_model: Dict[str, str] = {}
    for state, r in registry.items():
        m = r["best_model"]
        by_model.setdefault(m, state)

    picks = [largest, smallest]
    for m in ("lstm", "sarima", "xgboost", "arima"):
        if m in by_model and by_model[m] not in picks:
            picks.append(by_model[m])
        if len(picks) >= 4:
            break
    return picks[:4]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry, series_map = _load()

    plot_model_distribution(registry)
    plot_mape_by_state(registry)

    demo_states = pick_demo_states(registry, series_map)
    logger.info("Demo states: %s", demo_states)
    for state in demo_states:
        plot_forecast_for_state(state, series_map[state], registry[state])


if __name__ == "__main__":
    main()
