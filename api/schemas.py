"""Pydantic request/response schemas for the Forecasting API."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    week: int = Field(..., description="Week number in forecast (1–8)")
    date: str = Field(..., description="Forecast date (ISO 8601)")
    forecast_sales: float = Field(..., description="Predicted sales (point estimate)")
    forecast_low: float = Field(..., description="Lower bound of 95% confidence interval")
    forecast_high: float = Field(..., description="Upper bound of 95% confidence interval")


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast_horizon_weeks: int = 8
    forecast: List[ForecastPoint]
    generated_at: str


class ModelInfo(BaseModel):
    state: str
    best_model: str
    validation_metrics: Dict[str, Dict[str, Optional[float]]]
    series_length: int


class ModelsResponse(BaseModel):
    total_states: int
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    version: str
    states_loaded: int
    models_loaded: int
    message: str
