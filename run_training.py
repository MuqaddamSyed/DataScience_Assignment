"""
Full training pipeline orchestrator.

Run:
    python run_training.py [--states STATE1 STATE2 ...]

Trains all models for every state, selects best per state,
saves models to models/, and persists the model registry to
models/model_registry.joblib.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from src.config_loader import CONFIG
from src.evaluate import metrics_summary
from src.logger import get_logger
from src.preprocessing import run_preprocessing
from src.train import train_all_models

logger = get_logger(__name__)

REGISTRY_PATH = Path(CONFIG["paths"]["models_dir"]) / "model_registry.joblib"
SERIES_PATH = Path(CONFIG["paths"]["models_dir"]) / "state_series.joblib"


def main(states: Optional[List[str]] = None) -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("Sales Forecasting — Training Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    logger.info("Step 1/3 — Preprocessing …")
    state_series = run_preprocessing()

    if states:
        state_series = {k: v for k, v in state_series.items() if k in states}
        logger.info("Filtered to %d states: %s", len(state_series), states)

    # Persist series for API use (avoid re-loading Excel at request time)
    joblib.dump(state_series, SERIES_PATH)
    logger.info("State series saved → %s", SERIES_PATH)

    # ── Step 2: Train all models ──────────────────────────────────────────────
    logger.info("Step 2/3 — Training models for %d states …", len(state_series))
    registry: Dict[str, dict] = {}

    for i, (state, series) in enumerate(state_series.items(), 1):
        logger.info("─" * 50)
        logger.info("[%d/%d] State: %s", i, len(state_series), state)
        try:
            result = train_all_models(state, series)
            entry = {
                "best_model": result["best_model"],
                "metrics": result["metrics"],
                "series_length": result["series_length"],
            }
            if "ensemble" in result:
                entry["ensemble"] = result["ensemble"]
            registry[state] = entry
        except Exception as exc:
            logger.error("Training failed for '%s': %s", state, exc)
            registry[state] = {
                "best_model": "arima",
                "metrics": {},
                "series_length": len(series),
            }

    # ── Step 3: Save registry + report ───────────────────────────────────────
    logger.info("Step 3/3 — Saving registry …")
    joblib.dump(registry, REGISTRY_PATH)
    logger.info("Model registry saved → %s", REGISTRY_PATH)

    # Print summary table
    summary = metrics_summary(registry)
    best_only = summary[summary["is_best"]].sort_values("state")
    print("\n" + "=" * 70)
    print("BEST MODEL PER STATE (validation metrics)")
    print("=" * 70)
    print(best_only[["state", "model", "rmse", "mae", "mape_pct"]].to_string(index=False))

    elapsed = time.time() - start
    logger.info("Training complete in %.1f seconds.", elapsed)

    # Save human-readable JSON report
    report_path = Path(CONFIG["paths"]["models_dir"]) / "training_report.json"
    report: Dict = {}
    for state, r in registry.items():
        report[state] = {
            "best_model": r["best_model"],
            "metrics": {
                model: {k: (None if not pd.api.types.is_float(v) else round(v, 4)) for k, v in m.items()}
                for model, m in r["metrics"].items()
            },
        }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Training report → %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument(
        "--states", nargs="*", default=None,
        help="List of states to train (default: all)",
    )
    args = parser.parse_args()
    main(states=args.states)
