"""
Hyperparameter tuning runner.

Tunes XGBoost on 3 representative states using Optuna + walk-forward CV,
saves best params to models/tuned_xgboost.json, and updates config.yaml
in-place so the next training run uses the tuned values.

Usage:
    python run_tuning.py [--trials 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import CONFIG, PROJECT_ROOT
from src.logger import get_logger
from src.tuning import save_tuning_result, tune_xgboost

logger = get_logger(__name__)


def update_config_with_tuned(tuned: dict, config_path: Path) -> None:
    """Patch config.yaml::models.xgboost with the tuned averaged params."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    xgb_section = cfg.setdefault("models", {}).setdefault("xgboost", {})
    for k, v in tuned.items():
        xgb_section[k] = v

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info("Updated config.yaml::models.xgboost with tuned values")


def main(n_trials: int) -> None:
    series = joblib.load(Path(CONFIG["paths"]["models_dir"]) / "state_series.joblib")
    logger.info("Tuning across 3 representative states with %d trials each", n_trials)

    result = tune_xgboost(series, n_trials=n_trials)

    out = Path(CONFIG["paths"]["models_dir"]) / "tuned_xgboost.json"
    save_tuning_result(result, out)

    print("\n" + "=" * 70)
    print("XGBOOST TUNING RESULTS")
    print("=" * 70)
    for state, info in result["per_state"].items():
        print(f"\n{state} — best RMSE: {info['best_value']:,.0f}")
        for k, v in info["best_params"].items():
            print(f"  {k}: {v}")

    print("\nAveraged params written to config.yaml:")
    for k, v in result["averaged_params"].items():
        print(f"  {k}: {v}")

    update_config_with_tuned(result["averaged_params"], PROJECT_ROOT / "config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    main(n_trials=args.trials)
