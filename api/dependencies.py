"""
FastAPI dependency injection: loads state series and model registry once at
startup and caches them for the lifetime of the process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.config_loader import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(CONFIG["paths"]["models_dir"])
REGISTRY_PATH = MODELS_DIR / "model_registry.joblib"
SERIES_PATH = MODELS_DIR / "state_series.joblib"


class AppState:
    """Singleton that holds all loaded artefacts."""

    _registry: Optional[Dict[str, Any]] = None
    _series: Optional[Dict[str, pd.Series]] = None

    @classmethod
    def load(cls) -> None:
        if not REGISTRY_PATH.exists():
            raise RuntimeError(
                f"Model registry not found at {REGISTRY_PATH}. "
                "Run `python run_training.py` first."
            )
        if not SERIES_PATH.exists():
            raise RuntimeError(
                f"State series not found at {SERIES_PATH}. "
                "Run `python run_training.py` first."
            )
        cls._registry = joblib.load(REGISTRY_PATH)
        cls._series = joblib.load(SERIES_PATH)
        logger.info(
            "Loaded registry (%d states) and series (%d states)",
            len(cls._registry),
            len(cls._series),
        )

    @classmethod
    def get_registry(cls) -> Dict[str, Any]:
        if cls._registry is None:
            raise RuntimeError("AppState not loaded. Call AppState.load() first.")
        return cls._registry

    @classmethod
    def get_series(cls) -> Dict[str, pd.Series]:
        if cls._series is None:
            raise RuntimeError("AppState not loaded. Call AppState.load() first.")
        return cls._series

    @classmethod
    def is_ready(cls) -> bool:
        return cls._registry is not None and cls._series is not None
