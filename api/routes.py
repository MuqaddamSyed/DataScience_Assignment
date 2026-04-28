"""
FastAPI route handlers.

Endpoints
─────────
GET /health          — liveness / readiness check
GET /models          — list best model per state with metrics
GET /predict         — 8-week forecast for a specific state
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import AppState
from api.schemas import (
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from src.config_loader import CONFIG
from src.logger import get_logger
from src.predict import forecast_state

logger = get_logger(__name__)
router = APIRouter()

VERSION = CONFIG["api"]["version"]
HORIZON = CONFIG["data"]["forecast_horizon"]


# ══════════════════════════════════════════════════════════════════════════════
# /health
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Returns API readiness status and loaded model counts."""
    ready = AppState.is_ready()
    registry = AppState.get_registry() if ready else {}
    series = AppState.get_series() if ready else {}

    return HealthResponse(
        status="healthy" if ready else "degraded",
        version=VERSION,
        states_loaded=len(series),
        models_loaded=len(registry),
        message="All systems operational." if ready else "Models not loaded.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# /models
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List best model per state",
    tags=["Models"],
)
def list_models(
    state: Optional[str] = Query(
        None,
        description="Filter by state name (case-insensitive)",
        min_length=2,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z .\-]*$",
    ),
) -> ModelsResponse:
    """Returns the best selected model and validation metrics for each state."""
    registry = AppState.get_registry()

    if state:
        # Case-insensitive match
        matched = {
            k: v for k, v in registry.items()
            if k.lower() == state.lower()
        }
        if not matched:
            raise HTTPException(
                status_code=404,
                detail=f"State '{state}' not found. See /models for the full list.",
            )
        registry = matched

    models = []
    for s, r in sorted(registry.items()):
        # Clean metrics: replace inf/nan with None for JSON serialisation
        cleaned_metrics = {}
        for model_name, m in r.get("metrics", {}).items():
            cleaned_metrics[model_name] = {
                k: (None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else round(v, 4))
                for k, v in m.items()
            }
        models.append(ModelInfo(
            state=s,
            best_model=r["best_model"],
            validation_metrics=cleaned_metrics,
            series_length=r.get("series_length", 0),
        ))

    return ModelsResponse(total_states=len(models), models=models)


# ══════════════════════════════════════════════════════════════════════════════
# /predict
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/predict",
    response_model=ForecastResponse,
    summary="8-week sales forecast for a state",
    tags=["Forecasting"],
)
def predict(
    state: str = Query(
        ...,
        description="US state name (e.g. California)",
        min_length=2,
        max_length=64,
        # Conservative: alphanumeric, spaces, hyphens, periods only
        pattern=r"^[A-Za-z][A-Za-z .\-]*$",
    ),
) -> ForecastResponse:
    """
    Returns 8-week ahead weekly sales forecast for the requested state.

    The best model (lowest RMSE on held-out validation) is used automatically.
    """
    registry = AppState.get_registry()
    series_map = AppState.get_series()

    # Case-insensitive lookup
    state_key = next(
        (k for k in registry if k.lower() == state.lower()), None
    )
    if state_key is None:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found. See /models for the full list.",
        )

    series = series_map.get(state_key)
    if series is None:
        raise HTTPException(status_code=500, detail=f"Series missing for state '{state_key}'")

    best_model = registry[state_key]["best_model"]

    try:
        forecast_df = forecast_state(
            state=state_key,
            series=series,
            best_model_name=best_model,
            horizon=HORIZON,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model file missing for '{state_key}'. Re-run training. Details: {exc}",
        )
    except Exception as exc:
        logger.error("Forecast error for '%s': %s", state_key, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed for '{state_key}': {str(exc)}",
        )

    forecast_points = [
        ForecastPoint(
            week=i + 1,
            date=str(pd_ts.date()),
            forecast_sales=round(float(forecast), 2),
        )
        for i, (pd_ts, forecast) in enumerate(
            zip(forecast_df["date"].tolist(), forecast_df["forecast"].tolist())
        )
    ]

    return ForecastResponse(
        state=state_key,
        model_used=best_model,
        forecast_horizon_weeks=HORIZON,
        forecast=forecast_points,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
