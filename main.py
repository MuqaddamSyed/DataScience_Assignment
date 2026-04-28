"""
FastAPI application entry point.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.dependencies import AppState
from api.routes import router
from src.config_loader import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)
api_cfg = CONFIG["api"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models once at startup; release resources on shutdown."""
    logger.info("Starting up — loading models and state series …")
    try:
        AppState.load()
        logger.info("Startup complete. API is ready.")
    except RuntimeError as exc:
        logger.error("Startup error: %s", exc)
        logger.error("Run `python run_training.py` to train models first.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=api_cfg["title"],
    description=api_cfg["description"],
    version=api_cfg["version"],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS — driven by config; default open for demo, tighten for production.
# IMPORTANT: do not combine allow_origins=["*"] with allow_credentials=True
# (the browser will reject the response). Defaults below are safe.
_cors_origins = api_cfg.get("cors_origins", ["*"])
_cors_creds = api_cfg.get("cors_allow_credentials", False)
if "*" in _cors_origins:
    _cors_creds = False  # enforce safe combo

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_creds,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=False,
        log_level="info",
    )
