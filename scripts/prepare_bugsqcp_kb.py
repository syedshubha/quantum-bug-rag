#!/usr/bin/env python3
"""
prepare_bugsqcp_kb.py – Ingest Bugs-QCP entries into the local knowledge base.

Bugs-QCP (Zenodo 5834281) is the secondary corpus we use for taxonomy grounding
and bug-pattern retrieval.  The raw dataset is NOT bundled in this repository.

Download the archive from https://zenodo.org/records/5834281, extract it
locally, then run this script to normalise and merge entries into the
knowledge base.

Usage
-----
    python scripts/prepare_bugsqcp_kb.py \
        --input-dir /path/to/bugsqcp/ \
        --output-dir knowledge_base/

The script reads CSV/JSON files from *input-dir*, converts each entry to a
BugPattern object, and appends any new patterns to knowledge_base/bug_patterns.json.
Existing entries (same pattern_id) are updated in-place.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_ingest import KnowledgeBase
from src.schemas import BugPattern
from src.utils import get_logger

logger = get_logger("prepare_bugsqcp_kb")


def _load_bugsqcp_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "bugs" in data:
        return data["bugs"]
    logger.warning("Unexpected JSON structure in %s; skipping.", path)
    return []


def _load_bugsqcp_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def _row_to_pattern(row: dict, index: int, source: str = "bugsqcp") -> BugPattern | None:
    """
    Convert a single Bugs-QCP row dict to a BugPattern.

    Field names are mapped from common Bugs-QCP conventions; adjust as needed
    for the actual column names in the downloaded dataset.
    """
    # Try common field name variants.
    pattern_id = (
        row.get("id") or row.get("bug_id") or row.get("ID") or f"bugsqcp_{index:05d}"
    )
    name = row.get("name") or row.get("title") or row.get("bug_name") or f"BugsQCP-{pattern_id}"
    taxonomy_class = (
        row.get("taxonomy_class")
        or row.get("category")
        or row.get("bug_type")
        or "unknown"
    )
    description = row.get("description") or row.get("desc") or row.get("summary") or ""
    fix_hint = row.get("fix") or row.get("fix_hint") or row.get("fix_description") or ""
    example_code = row.get("code") or row.get("buggy_code") or row.get("example") or ""
    tags_raw = row.get("tags") or row.get("keywords") or ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if isinstance(tags_raw, str) else []

    if not description and not name:
        return None

    return BugPattern(
        pattern_id=str(pattern_id),
        name=str(name),
        taxonomy_class=str(taxonomy_class),
        description=str(description),
        example_code=str(example_code),
        fix_hint=str(fix_hint),
        source=source,
        tags=tags,
    )


def ingest_directory(input_dir: Path, kb: KnowledgeBase) -> int:
    """
    Walk *input_dir* for JSON and CSV files and add patterns to *kb*.

    Returns the number of new or updated patterns ingested.
    """
    ingested = 0
    for fpath in sorted(input_dir.rglob("*.json")):
        rows = _load_bugsqcp_json(fpath)
        for i, row in enumerate(rows):
            pattern = _row_to_pattern(row, index=ingested + i, source=str(fpath.stem))
            if pattern:
                kb.add_pattern(pattern)
                ingested += 1

    for fpath in sorted(input_dir.rglob("*.csv")):
        rows = _load_bugsqcp_csv(fpath)
        for i, row in enumerate(rows):
            pattern = _row_to_pattern(row, index=ingested + i, source=str(fpath.stem))
            if pattern:
                kb.add_pattern(pattern)
                ingested += 1

    return ingested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Bugs-QCP entries into the local knowledge base."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing extracted Bugs-QCP files.",
    )
    parser.add_argument(
        "--output-dir",
        default="knowledge_base/",
        help="Knowledge-base directory to update (default: knowledge_base/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse entries but do not write to disk.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("Input directory '%s' does not exist.", input_dir)
        sys.exit(1)

    kb = KnowledgeBase(args.output_dir)
    before = kb.num_patterns if hasattr(kb, "num_patterns") else len(kb.patterns)
    ingested = ingest_directory(input_dir, kb)

    if args.dry_run:
        logger.info("Dry run: would have ingested %d patterns (not written).", ingested)
    else:
        kb.save_patterns()
        after = len(kb.patterns)
        logger.info(
            "Knowledge base updated: %d → %d patterns (%d ingested from Bugs-QCP).",
            before,
            after,
            ingested,
        )

    print(f"Done. {ingested} patterns ingested.")


if __name__ == "__main__":
    main()
