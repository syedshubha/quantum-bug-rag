"""
bugsqcp_ingest.py – Reusable Bugs-QCP ingestion and normalization helpers.

We use Bugs-QCP as a secondary corpus for taxonomy grounding and retrieval
context enrichment. We do not use it as the evaluation benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import BugPattern, TaxonomyEntry
from .utils import get_logger, load_json, save_json

logger = get_logger(__name__)

_ROW_KEYS_ID = ("id", "bug_id", "pattern_id", "ID", "uid")
_ROW_KEYS_NAME = ("name", "title", "bug_name", "pattern_name", "error_name", "bug_pattern")
_ROW_KEYS_DESC = (
    "description", "desc", "summary", "problem", "bug_description",
    "comment", "annotator_comment", "symptom",
)
_ROW_KEYS_FIX = ("fix", "fix_hint", "fix_description", "repair", "solution", "commit_msg")
_ROW_KEYS_CODE = (
    "code",
    "example",
    "example_code",
    "buggy_code",
    "bug_code",
    "snippet",
)
# Note: bug_pattern comes before type so Bugs-QCP rows use the specific bug pattern
# (e.g. "Barrier Related") rather than the Quantum/Classical type label.
_ROW_KEYS_TYPE = ("taxonomy_class", "category", "bug_type", "bug_pattern", "type", "label", "class")
_ROW_KEYS_TAGS = ("tags", "keywords", "keyword", "topics")

_TYPE_ALIAS_MAP: dict[str, str] = {
    # Generic / inferred terms
    "qubit mapping": "incorrect_qubit_mapping",
    "qubit index": "incorrect_qubit_mapping",
    "index": "incorrect_qubit_mapping",
    "operator": "incorrect_operator",
    "gate": "incorrect_operator",
    "wrong gate": "incorrect_operator",
    "barrier": "missing_barrier",
    "initial": "wrong_initial_state",
    "statevector": "wrong_initial_state",
    "measurement": "measurement_error",
    "measure": "measurement_error",
    # Real upstream Bugs-QCP bug_pattern vocabulary
    "barrier related": "missing_barrier",
    "overlooked qubit order": "incorrect_qubit_mapping",
    "msb-lsb convention": "incorrect_qubit_mapping",
    "wrong identifier": "incorrect_qubit_mapping",
    "incorrect numerical computation": "incorrect_operator",
    "incorrect ir - wrong information": "incorrect_operator",
    "incorrect ir - missing information": "incorrect_operator",
    "incorrect circuit": "incorrect_operator",
    "api misuse - internal": "incorrect_operator",
    "api misuse - external": "incorrect_operator",
    "api misuse": "incorrect_operator",
    "typo": "incorrect_operator",
    "type problem": "incorrect_operator",
    "incorrect final measurement": "measurement_error",
    "incorrect randomness handling": "wrong_initial_state",
    "wrong concept": "unknown",
    "incorrect application logic": "unknown",
    "missing error handling": "unknown",
    "overlooked corner case": "unknown",
}


@dataclass(frozen=True)
class BugsQCPIngestReport:
    discovered_files: int
    discovered_records: int
    imported_records: int
    skipped_records: int
    duplicate_in_input: int
    duplicate_with_existing: int
    manual_preserved_on_collision: int
    final_pattern_count: int
    taxonomy_examples_added: int
    source_counts: dict[str, int]


def ingest_bugsqcp_into_kb(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
) -> BugsQCPIngestReport:
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Bugs-QCP input directory '{input_dir}' does not exist or is not a directory.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns_path = output_dir / "bug_patterns.json"
    taxonomy_path = output_dir / "taxonomy.json"

    existing_patterns = _load_patterns(patterns_path)
    taxonomy_entries = _load_taxonomy(taxonomy_path)
    taxonomy_classes = {entry.class_id for entry in taxonomy_entries}

    discovered_files, discovered_rows = _discover_bugsqcp_rows(input_dir)
    discovered_count = len(discovered_rows)

    imported_patterns: list[BugPattern] = []
    skipped_records = 0
    duplicate_in_input = 0
    seen_import_ids: set[str] = set()
    type_counter: Counter[str] = Counter()

    for idx, row_info in enumerate(discovered_rows):
        raw_row = row_info["row"]
        source_file = row_info["source_file"]
        pattern = _normalize_row_to_pattern(
            row=raw_row,
            index=idx,
            source_file=source_file,
            taxonomy_classes=taxonomy_classes,
            type_counter=type_counter,
        )
        if pattern is None:
            skipped_records += 1
            continue
        if pattern.pattern_id in seen_import_ids:
            duplicate_in_input += 1
            continue
        seen_import_ids.add(pattern.pattern_id)
        imported_patterns.append(pattern)

    if not imported_patterns:
        raise ValueError(
            "No usable Bugs-QCP records were found. "
            f"Discovered records={discovered_count}, skipped={skipped_records}, duplicates={duplicate_in_input}."
        )

    merged_patterns, duplicate_with_existing, manual_preserved = _merge_patterns(
        existing_patterns,
        imported_patterns,
    )

    taxonomy_examples_added = _apply_taxonomy_grounding(
        taxonomy_entries,
        imported_patterns,
    )

    mapping_hints_path = output_dir / "taxonomy_mapping_hints.json"
    mapping_hints = {
        "alias_map": _TYPE_ALIAS_MAP,
        "observed_upstream_type_counts": dict(sorted(type_counter.items())),
    }

    if not dry_run:
        save_json([pattern.model_dump() for pattern in merged_patterns], patterns_path)
        save_json([entry.model_dump() for entry in taxonomy_entries], taxonomy_path)
        save_json(mapping_hints, mapping_hints_path)

    source_counts = Counter(pattern.source or "unknown" for pattern in merged_patterns)
    return BugsQCPIngestReport(
        discovered_files=discovered_files,
        discovered_records=discovered_count,
        imported_records=len(imported_patterns),
        skipped_records=skipped_records,
        duplicate_in_input=duplicate_in_input,
        duplicate_with_existing=duplicate_with_existing,
        manual_preserved_on_collision=manual_preserved,
        final_pattern_count=len(merged_patterns),
        taxonomy_examples_added=taxonomy_examples_added,
        source_counts=dict(sorted(source_counts.items())),
    )


def _discover_rows(input_dir: Path) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    files = 0
    for json_path in sorted(input_dir.rglob("*.json")):
        files += 1
        for row in _load_json_rows(json_path):
            rows.append({"row": row, "source_file": str(json_path.relative_to(input_dir))})
    for csv_path in sorted(input_dir.rglob("*.csv")):
        files += 1
        for row in _load_csv_rows(csv_path):
            rows.append({"row": row, "source_file": str(csv_path.relative_to(input_dir))})
    return files, rows


# ---------------------------------------------------------------------------
# Bugs-QCP canonical-layout discovery helpers
# ---------------------------------------------------------------------------

def _discover_bugsqcp_rows(input_dir: Path) -> tuple[int, list[dict[str, Any]]]:
    """Discover rows using the canonical Bugs-QCP archive layout.

    Primary source is ``artifacts/annotation_bugs.csv``.  Each row is enriched
    with per-case ``commit_msg`` and ``annotator_comment`` from the companion
    ``artifacts/minimal_bugfixes/{project}/{issue}/metadata.json`` files.
    Rows where ``real != 'bug'`` (false positives) are excluded.

    Falls back to :func:`_discover_rows` if no ``annotation_bugs.csv`` is
    found, so the function works for custom / non-canonical directory trees.
    """
    annotation_csv = _find_annotation_csv(input_dir)
    if annotation_csv is None:
        logger.warning(
            "No annotation_bugs.csv found under '%s'. "
            "Falling back to generic JSON/CSV discovery (may import noisy records).",
            input_dir,
        )
        return _discover_rows(input_dir)

    logger.info("Found Bugs-QCP annotation CSV at: %s", annotation_csv)

    # artifacts/ directory is the parent of annotation_bugs.csv
    artifacts_dir = annotation_csv.parent
    meta_map = _build_metadata_enrichment_map(artifacts_dir)
    logger.info("Loaded %d metadata enrichment records.", len(meta_map))

    rows: list[dict[str, Any]] = []
    rel_csv = str(annotation_csv.relative_to(input_dir))
    total_rows = 0
    excluded_fp = 0

    try:
        with annotation_csv.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for csv_row in reader:
                total_rows += 1
                if csv_row.get("real", "").strip().lower() != "bug":
                    excluded_fp += 1
                    continue

                enriched: dict[str, Any] = dict(csv_row)
                bug_id = csv_row.get("id", "").strip()
                if bug_id and bug_id in meta_map:
                    meta = meta_map[bug_id]
                    enriched.setdefault("commit_msg", meta.get("commit_msg", ""))
                    enriched.setdefault("annotator_comment", meta.get("annotator_comment", ""))

                rows.append({"row": enriched, "source_file": rel_csv})
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse annotation CSV %s: %s", annotation_csv, exc)
        return 1, []

    logger.info(
        "Annotation CSV: %d total rows, %d false-positive rows excluded, %d bug rows loaded.",
        total_rows,
        excluded_fp,
        len(rows),
    )
    return 1, rows


def _find_annotation_csv(input_dir: Path) -> Path | None:
    """Return the first ``annotation_bugs.csv`` found under *input_dir*.

    Files located inside ``old_dataset_versions/`` are skipped because that
    folder contains upstream test artefacts rather than canonical annotations.
    """
    for candidate in sorted(input_dir.rglob("annotation_bugs.csv")):
        if "old_dataset_versions" not in candidate.parts:
            return candidate
    return None


def _build_metadata_enrichment_map(artifacts_dir: Path) -> dict[str, dict]:
    """Build a ``{bug_id: metadata_dict}`` map from ``minimal_bugfixes/`` JSON files."""
    meta_map: dict[str, dict] = {}
    bugfixes_dir = artifacts_dir / "minimal_bugfixes"
    if not bugfixes_dir.exists():
        return meta_map

    for meta_path in sorted(bugfixes_dir.rglob("metadata.json")):
        if "old_dataset_versions" in meta_path.parts:
            continue
        try:
            meta = load_json(meta_path)
            if isinstance(meta, dict):
                bug_id = str(meta.get("id", "")).strip()
                if bug_id:
                    meta_map[bug_id] = meta
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping unreadable metadata file %s: %s", meta_path, exc)

    return meta_map


def _load_json_rows(path: Path) -> list[dict]:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping unreadable JSON file %s: %s", path, exc)
        return []

    return _coerce_payload_to_rows(payload)


def _coerce_payload_to_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("bugs", "data", "records", "entries", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if any(isinstance(value, str) for value in payload.values()):
            return [payload]
    return []


def _load_csv_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping unreadable CSV file %s: %s", path, exc)
    return rows


def _normalize_row_to_pattern(
    *,
    row: dict,
    index: int,
    source_file: str,
    taxonomy_classes: set[str],
    type_counter: Counter[str],
) -> BugPattern | None:
    raw_id = _first_value(row, _ROW_KEYS_ID)
    name = _first_value(row, _ROW_KEYS_NAME) or ""
    description = _first_value(row, _ROW_KEYS_DESC) or ""
    fix_hint = _first_value(row, _ROW_KEYS_FIX) or ""
    example_code = _first_value(row, _ROW_KEYS_CODE) or ""
    raw_type = _first_value(row, _ROW_KEYS_TYPE) or ""
    tags = _normalize_tags(_first_value(row, _ROW_KEYS_TAGS))

    # Enrich tags with native Bugs-QCP fields when present
    repo_val = row.get("repo", "").strip()
    if repo_val and repo_val.lower() not in {"nan", "none", "null"}:
        project = repo_val.rstrip("/").split("/")[-1].lower()
        if project and len(project) > 1:
            tags.append(f"project:{project[:50]}")

    bug_type_val = row.get("type", "").strip().lower()
    if bug_type_val in {"quantum", "classical"}:
        tags.append(f"bug_type:{bug_type_val}")

    component_val = row.get("component", "").strip()
    if component_val and component_val.lower() not in {"nan", "none", "null"}:
        tags.append(f"component:{component_val[:50].lower()}")

    if raw_type:
        type_counter[raw_type.strip()] += 1

    if not (name or description or fix_hint or example_code):
        return None

    taxonomy_class = _map_taxonomy_class(raw_type, name, description, tags, taxonomy_classes)
    pattern_id = _build_pattern_id(raw_id, source_file, name, description, index)

    normalized_name = name.strip() or f"BugsQCP Pattern {pattern_id}"
    normalized_description = description.strip() or "Bugs-QCP imported pattern."
    provenance_tag = _build_provenance_tag(source_file)
    normalized_tags = sorted({*tags, taxonomy_class, "bugsqcp", provenance_tag})

    return BugPattern(
        pattern_id=pattern_id,
        name=normalized_name,
        taxonomy_class=taxonomy_class,
        description=normalized_description,
        example_code=example_code.strip(),
        fix_hint=fix_hint.strip(),
        source="bugsqcp",
        tags=normalized_tags,
    )


def _build_pattern_id(raw_id: str | None, source_file: str, name: str, description: str, index: int) -> str:
    if raw_id:
        cleaned = _slug(raw_id)
        if cleaned:
            if cleaned.upper().startswith("BP"):
                return cleaned
            if cleaned.upper().startswith("BQCP_"):
                return cleaned
            return f"BQCP_{cleaned}"

    fingerprint = f"{source_file}|{name}|{description}|{index}"
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"BQCP_{digest}"


def _merge_patterns(
    existing_patterns: list[BugPattern],
    imported_patterns: list[BugPattern],
) -> tuple[list[BugPattern], int, int]:
    merged: dict[str, BugPattern] = {pattern.pattern_id: pattern for pattern in existing_patterns}
    duplicate_with_existing = 0
    manual_preserved = 0

    for imported in imported_patterns:
        existing = merged.get(imported.pattern_id)
        if existing is not None:
            duplicate_with_existing += 1
            if existing.source == "manual":
                manual_preserved += 1
                continue
        merged[imported.pattern_id] = imported

    merged_patterns = [merged[key] for key in sorted(merged)]
    return merged_patterns, duplicate_with_existing, manual_preserved


def _apply_taxonomy_grounding(
    taxonomy_entries: list[TaxonomyEntry],
    imported_patterns: list[BugPattern],
    max_examples_per_class: int = 8,
) -> int:
    examples_by_class: dict[str, list[str]] = defaultdict(list)
    for pattern in imported_patterns:
        snippet = f"Bugs-QCP: {pattern.name}"
        examples_by_class[pattern.taxonomy_class].append(snippet)

    added = 0
    for entry in taxonomy_entries:
        extra_examples = examples_by_class.get(entry.class_id, [])
        if not extra_examples:
            continue
        existing = list(entry.examples)
        existing_set = set(existing)
        for example in extra_examples:
            if example in existing_set:
                continue
            if len(existing) >= max_examples_per_class:
                break
            existing.append(example)
            existing_set.add(example)
            added += 1
        entry.examples = existing
    return added


def _map_taxonomy_class(
    raw_type: str,
    name: str,
    description: str,
    tags: list[str],
    taxonomy_classes: set[str],
) -> str:
    for candidate in (raw_type, name, description, " ".join(tags)):
        if not candidate:
            continue
        norm = _normalize_text(candidate)
        if norm in taxonomy_classes:
            return norm
        for alias, class_id in _TYPE_ALIAS_MAP.items():
            if alias in norm and class_id in taxonomy_classes:
                return class_id
    return "unknown" if "unknown" in taxonomy_classes else next(iter(sorted(taxonomy_classes)))


def _load_patterns(path: Path) -> list[BugPattern]:
    if not path.exists():
        return []
    payload = load_json(path)
    return [BugPattern(**row) for row in payload]


def _load_taxonomy(path: Path) -> list[TaxonomyEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found at '{path}'.")
    payload = load_json(path)
    return [TaxonomyEntry(**row) for row in payload]


def _first_value(row: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return None


def _normalize_tags(raw_tags: str | None) -> list[str]:
    if raw_tags is None:
        return []
    text = str(raw_tags)
    parts = re.split(r"[,;|]", text)
    tags = [part.strip().lower() for part in parts if part and part.strip()]
    return sorted(set(tags))


def _normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:80]


def _build_provenance_tag(source_file: str) -> str:
    safe = _slug(source_file.replace("/", "_"))
    return f"origin_file:{safe}" if safe else "origin_file:unknown"
