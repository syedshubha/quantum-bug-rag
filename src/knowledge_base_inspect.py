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
    missing_required_fields: dict[str, int]
    source_role_counts: dict[str, int]
    by_source: dict[str, int]
    by_taxonomy_class: dict[str, int]
    taxonomy_defined_classes: list[str]
    taxonomy_covered_classes: list[str]
    taxonomy_missing_classes: list[str]
    sample_imported_entries: list[dict[str, str]]


def inspect_knowledge_base(kb_dir: str | Path) -> KnowledgeBaseInspection:
    kb_path = Path(kb_dir) / "bug_patterns.json"
    rows = load_json(kb_path)
    if not isinstance(rows, list):
        raise ValueError("knowledge_base/bug_patterns.json must contain a JSON list.")

    taxonomy_path = Path(kb_dir) / "taxonomy.json"
    taxonomy_rows = []
    if taxonomy_path.exists():
        raw_taxonomy = load_json(taxonomy_path)
        if isinstance(raw_taxonomy, list):
            taxonomy_rows = [row for row in raw_taxonomy if isinstance(row, dict)]

    required_fields = ("pattern_id", "name", "taxonomy_class", "description")

    ids: list[str] = []
    valid_patterns: list[BugPattern] = []
    invalid_indices: list[int] = []
    missing_required_fields = Counter()

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid_indices.append(idx)
            continue
        for field in required_fields:
            value = row.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_required_fields[field] += 1
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
    role_counts = {
        "manual": source_counts.get("manual", 0),
        "bugsqcp": source_counts.get("bugsqcp", 0),
    }

    defined_classes = sorted(
        {
            str(row.get("class_id")).strip()
            for row in taxonomy_rows
            if row.get("class_id")
        }
    )
    covered_classes = sorted(set(class_counts))
    if defined_classes:
        missing_classes = sorted(set(defined_classes) - set(covered_classes))
    else:
        missing_classes = []

    sample_imported_entries: list[dict[str, str]] = []
    for pattern in valid_patterns:
        if pattern.source != "bugsqcp":
            continue
        sample_imported_entries.append(
            {
                "pattern_id": pattern.pattern_id,
                "name": pattern.name,
                "taxonomy_class": pattern.taxonomy_class,
                "source": pattern.source,
            }
        )
        if len(sample_imported_entries) >= 5:
            break

    return KnowledgeBaseInspection(
        total_entries=len(rows),
        valid_entries=len(valid_patterns),
        duplicate_ids=duplicate_ids,
        invalid_entry_indices=sorted(set(invalid_indices)),
        missing_required_fields=dict(sorted(missing_required_fields.items())),
        source_role_counts=role_counts,
        by_source=dict(sorted(source_counts.items())),
        by_taxonomy_class=dict(sorted(class_counts.items())),
        taxonomy_defined_classes=defined_classes,
        taxonomy_covered_classes=covered_classes,
        taxonomy_missing_classes=missing_classes,
        sample_imported_entries=sample_imported_entries,
    )


def summarize_inspection(inspection: KnowledgeBaseInspection) -> dict[str, Any]:
    return {
        "total_entries": inspection.total_entries,
        "valid_entries": inspection.valid_entries,
        "invalid_entries": inspection.total_entries - inspection.valid_entries,
        "duplicate_ids": inspection.duplicate_ids,
        "invalid_entry_indices": inspection.invalid_entry_indices,
        "missing_required_fields": inspection.missing_required_fields,
        "source_role_counts": inspection.source_role_counts,
        "by_source": inspection.by_source,
        "by_taxonomy_class": inspection.by_taxonomy_class,
        "taxonomy_defined_classes": inspection.taxonomy_defined_classes,
        "taxonomy_covered_classes": inspection.taxonomy_covered_classes,
        "taxonomy_missing_classes": inspection.taxonomy_missing_classes,
        "sample_imported_entries": inspection.sample_imported_entries,
    }
