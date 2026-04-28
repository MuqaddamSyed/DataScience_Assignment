"""Load and expose typed project configuration from config.yaml."""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(value: str) -> str:
    """Convert a config path to an absolute path under the project root."""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config and resolve all paths to absolute paths."""
    config_path = PROJECT_ROOT / path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Make all relative paths absolute so the app can be launched from any cwd
    if "paths" in cfg:
        for key, value in cfg["paths"].items():
            cfg["paths"][key] = _resolve_path(value)

    if "data" in cfg and "raw_path" in cfg["data"]:
        cfg["data"]["raw_path"] = _resolve_path(cfg["data"]["raw_path"])

    return cfg


CONFIG = load_config()

