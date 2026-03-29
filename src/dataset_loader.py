"""
dataset_loader.py – Loading and preprocessing Bugs4Q benchmark programs.

I read the Bugs4Q dataset from the local ``data/bugs4q/`` directory (which
the user must populate via ``scripts/prepare_bugs4q.py``).  Each item is
returned as a plain dictionary that maps cleanly to the rest of the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .utils import load_json, repo_root

logger = logging.getLogger(__name__)

# I expect the dataset to live here relative to the repository root.
_DEFAULT_DATA_DIR = repo_root() / "data" / "bugs4q"


# ---------------------------------------------------------------------------
# Data model (plain dict to keep dependencies minimal)
# ---------------------------------------------------------------------------

# Each program record has the following keys:
#   id          – str   unique identifier
#   source_code – str   raw Python/Qiskit source
#   has_bug     – bool  ground-truth label (True = contains a bug)
#   bug_class   – str | None  taxonomy label (may be None for unlabelled items)
#   location    – str | None  "file:line" hint if known
#   description – str | None  optional natural-language description


def _record_from_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """I normalise a raw JSON object into a canonical program record."""
    return {
        "id": str(raw.get("id", "unknown")),
        "source_code": raw.get("source_code", raw.get("code", "")),
        "has_bug": bool(raw.get("has_bug", raw.get("buggy", False))),
        "bug_class": raw.get("bug_class", raw.get("taxonomy_class", None)),
        "location": raw.get("location", None),
        "description": raw.get("description", None),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_bugs4q(
    data_dir: Optional[str | Path] = None,
    split: str = "all",
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    I load the Bugs4Q benchmark from ``data_dir`` and return a list of
    normalised program records.

    Parameters
    ----------
    data_dir:
        Path to the directory containing the Bugs4Q JSON files.
        Defaults to ``data/bugs4q/`` in the repository root.
    split:
        Which subset to load.  Accepted values: ``"all"``, ``"buggy"``,
        ``"clean"``.
    max_items:
        If set, I truncate the list to the first *max_items* entries.
        Useful for quick subset experiments.
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    logger.info("Loading Bugs4Q from %s (split=%s)", data_dir, split)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Bugs4Q data directory not found: {data_dir}\n"
            "Run `python scripts/prepare_bugs4q.py` first."
        )

    records: List[Dict[str, Any]] = []

    # I support two layouts:
    # 1. A single ``bugs4q.json`` file containing a list of records.
    # 2. Individual JSON files, one per program.
    index_file = data_dir / "bugs4q.json"
    if index_file.exists():
        raw_list = load_json(index_file)
        if not isinstance(raw_list, list):
            raw_list = raw_list.get("programs", raw_list.get("data", []))
        for raw in raw_list:
            records.append(_record_from_dict(raw))
    else:
        for json_file in sorted(data_dir.glob("*.json")):
            raw = load_json(json_file)
            if isinstance(raw, list):
                for item in raw:
                    records.append(_record_from_dict(item))
            else:
                records.append(_record_from_dict(raw))

    # I apply the requested split filter.
    if split == "buggy":
        records = [r for r in records if r["has_bug"]]
    elif split == "clean":
        records = [r for r in records if not r["has_bug"]]
    elif split != "all":
        raise ValueError(f"Unknown split '{split}'.  Choose 'all', 'buggy', or 'clean'.")

    if max_items is not None:
        records = records[:max_items]

    logger.info("Loaded %d records (split=%s)", len(records), split)
    return records


def stream_bugs4q(
    data_dir: Optional[str | Path] = None,
    split: str = "all",
) -> Generator[Dict[str, Any], None, None]:
    """
    I yield program records one at a time to keep memory usage low when
    processing large subsets of the benchmark.
    """
    for record in load_bugs4q(data_dir=data_dir, split=split):
        yield record
