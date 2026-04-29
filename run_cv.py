"""
Walk-forward cross-validation runner.

Validates that the single-split RMSE numbers in `models/training_report.json`
aren't a fluke. Runs 5-fold expanding-window CV on a representative sample
of states for ARIMA / SARIMA / Prophet, and saves:
- reports/cv_results.csv         — per-fold raw scores
- reports/cv_summary.csv         — median/mean/IQR per (state, model)
- reports/cv_box_<state>.png     — RMSE box-plot per state

Usage:
    python run_cv.py [--states S1 S2 ...] [--folds 5]
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import CONFIG
from src.cross_validation import compare_models_cv, cv_summary
from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _default_states(state_series: dict) -> list[str]:
    """3 representative: largest, median-sized, smallest."""
    sizes = {s: ss.sum() for s, ss in state_series.items()}
    by_size = sorted(sizes, key=sizes.get, reverse=True)
    return [by_size[0], by_size[len(by_size) // 2], by_size[-1]]


def plot_cv_box(cv_df: pd.DataFrame, state: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = cv_df.pivot(index="fold", columns="model", values="rmse")
    pivot.boxplot(ax=ax, grid=False)
    # Overlay individual fold points
    for i, model in enumerate(pivot.columns, start=1):
        ax.scatter(
            [i] * len(pivot), pivot[model],
            color="black", alpha=0.6, zorder=3, s=30,
        )
    ax.set_title(f"{state} — RMSE distribution across {len(pivot)} CV folds")
    ax.set_ylabel("RMSE (lower is better)")
    plt.tight_layout()
    safe = state.lower().replace(" ", "_")
    out = REPORTS_DIR / f"cv_box_{safe}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("Saved: %s", out)


def main(states: list[str] | None, n_folds: int) -> None:
    state_series = joblib.load(Path(CONFIG["paths"]["models_dir"]) / "state_series.joblib")

    if not states:
        states = _default_states(state_series)
    logger.info("CV on %d states: %s", len(states), states)

    all_results = []
    all_summaries = []
    for state in states:
        if state not in state_series:
            logger.warning("Skipping unknown state: %s", state)
            continue
        s = state_series[state]
        cv_df = compare_models_cv(s, state=state, n_splits=n_folds, val_horizon=8)
        cv_df["state"] = state
        all_results.append(cv_df)

        summary = cv_summary(cv_df)
        summary["state"] = state
        summary = summary.reset_index().rename(columns={"index": "model"})
        all_summaries.append(summary)

        plot_cv_box(cv_df, state)

    if all_results:
        results = pd.concat(all_results, ignore_index=True)
        summary = pd.concat(all_summaries, ignore_index=True)
        results.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
        summary.to_csv(REPORTS_DIR / "cv_summary.csv", index=False)
        logger.info("Saved CV results to reports/cv_results.csv and cv_summary.csv")

        print("\n" + "=" * 70)
        print("WALK-FORWARD CV SUMMARY (median RMSE across 5 folds)")
        print("=" * 70)
        print(
            summary[["state", "model", "rmse_median", "rmse_q1", "rmse_q3"]]
            .sort_values(["state", "rmse_median"])
            .to_string(index=False)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(states=args.states, n_folds=args.folds)
