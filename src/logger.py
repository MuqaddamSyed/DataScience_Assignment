"""Centralized logging configuration for the forecasting system."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """Return a configured logger that writes to console and a rotating file."""
    if log_dir is None:
        log_dir = str(_PROJECT_ROOT / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "forecasting.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
