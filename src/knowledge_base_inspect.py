"""Inspection helpers for knowledge base quality checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import BugPattern
from .utils import load_json


@dataclass(frozen=True)
class KnowledgeBaseInspection:
    total_entries: int
    valid_entries: int
    duplicate_ids: list[str]
    invalid_entry_indices: list[int]
    by_source: dict[str, int]
    by_taxonomy_class: dict[str, int]


def inspect_knowledge_base(kb_dir: str | Path) -> KnowledgeBaseInspection:
    kb_path = Path(kb_dir) / "bug_patterns.json"
    rows = load_json(kb_path)
    if not isinstance(rows, list):
        raise ValueError("knowledge_base/bug_patterns.json must contain a JSON list.")

    ids: list[str] = []
    valid_patterns: list[BugPattern] = []
    invalid_indices: list[int] = []

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid_indices.append(idx)
            continue
        pattern_id = row.get("pattern_id")
        if isinstance(pattern_id, str):
            ids.append(pattern_id)
        try:
            valid_patterns.append(BugPattern(**row))
        except Exception:  # noqa: BLE001
            invalid_indices.append(idx)

    id_counts = Counter(ids)
    duplicate_ids = sorted([pid for pid, count in id_counts.items() if count > 1])

    source_counts = Counter((pattern.source or "unknown") for pattern in valid_patterns)
    class_counts = Counter(pattern.taxonomy_class for pattern in valid_patterns)

    return KnowledgeBaseInspection(
        total_entries=len(rows),
        valid_entries=len(valid_patterns),
        duplicate_ids=duplicate_ids,
        invalid_entry_indices=sorted(set(invalid_indices)),
        by_source=dict(sorted(source_counts.items())),
        by_taxonomy_class=dict(sorted(class_counts.items())),
    )


def summarize_inspection(inspection: KnowledgeBaseInspection) -> dict[str, Any]:
    return {
        "total_entries": inspection.total_entries,
        "valid_entries": inspection.valid_entries,
        "invalid_entries": inspection.total_entries - inspection.valid_entries,
        "duplicate_ids": inspection.duplicate_ids,
        "invalid_entry_indices": inspection.invalid_entry_indices,
        "by_source": inspection.by_source,
        "by_taxonomy_class": inspection.by_taxonomy_class,
    }
