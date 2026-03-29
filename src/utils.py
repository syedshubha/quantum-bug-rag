"""
utils.py – Shared utilities: logging, config loading, and file helpers.

I keep all cross-cutting helpers here to avoid duplication across modules.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    I configure the root logger and return it.

    Parameters
    ----------
    level:
        Logging level string, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    log_file:
        Optional path to write logs to in addition to stdout.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("quantum_bug_rag")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    I load a YAML configuration file and return it as a plain dictionary.

    Parameters
    ----------
    config_path:
        Path to a YAML file (e.g. ``config.yaml``).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Any:
    """I load and return a JSON file."""
    with open(path, "r") as fh:
        return json.load(fh)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """I serialise *data* to a JSON file at *path*, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=indent)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    I retrieve an environment variable by *key*.

    I intentionally never embed secrets in source code; callers should
    set API keys via environment variables or a ``.env`` file.
    """
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """I return the repository root directory."""
    return Path(__file__).resolve().parent.parent
