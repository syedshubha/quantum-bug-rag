"""
utils.py – Shared utility functions.

Covers configuration loading, logging setup, and common file helpers.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import yaml


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger for *name*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """
    Load a YAML configuration file and return it as a nested dict.

    Falls back to an empty dict (and logs a warning) if the file does not exist,
    allowing components to use their own defaults.
    """
    logger = get_logger(__name__)
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file '%s' not found; using default settings.", path)
        return {}
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return data or {}


def get_nested(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested value from a config dict."""
    node = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is default:
            return default
    return node


# ── File helpers ──────────────────────────────────────────────────────────────

def load_json(path: str | Path) -> Any:
    """Load and return JSON from *path*."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise *obj* as pretty-printed JSON to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)


def append_jsonl(record: dict, path: str | Path) -> None:
    """Append a single JSON object as one line to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read all lines from a JSONL file and return as a list of dicts."""
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Identifiers ───────────────────────────────────────────────────────────────

def new_run_id() -> str:
    """Generate a short, unique run identifier."""
    return uuid.uuid4().hex[:12]


# ── Environment helpers ───────────────────────────────────────────────────────

def require_env(var: str) -> str:
    """
    Return the value of environment variable *var*.

    Raises RuntimeError if the variable is not set, to avoid silent credential
    failures.
    """
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{var}' is not set. "
            "Please export it before running this script."
        )
    return value
