from __future__ import annotations

import json
from pathlib import Path

from src.bugsqcp_ingest import ingest_bugsqcp_into_kb
from src.knowledge_base_inspect import inspect_knowledge_base
from src.retriever import BugPatternRetriever
from src.schemas import BugPattern


def _write_kb(kb_dir: Path) -> None:
    bug_patterns = [
        {
            "pattern_id": "BP001",
            "name": "Manual Mapping Entry",
            "taxonomy_class": "incorrect_qubit_mapping",
            "description": "Manually curated pattern",
            "source": "manual",
            "tags": ["manual"],
        }
    ]
    taxonomy = [
        {
            "class_id": "incorrect_qubit_mapping",
            "name": "Incorrect Qubit Mapping",
            "description": "Bad qubit index or mapping.",
            "parent_class": None,
            "examples": [],
        },
        {
            "class_id": "measurement_error",
            "name": "Measurement Error",
            "description": "Measurement logic issue.",
            "parent_class": None,
            "examples": [],
        },
        {
            "class_id": "unknown",
            "name": "Unknown",
            "description": "Unknown class.",
            "parent_class": None,
            "examples": [],
        },
    ]
    (kb_dir / "bug_patterns.json").write_text(json.dumps(bug_patterns), encoding="utf-8")
    (kb_dir / "taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")


def test_ingest_preserves_manual_entry_on_id_collision(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    kb_dir = tmp_path / "kb"
    input_dir.mkdir()
    kb_dir.mkdir()
    _write_kb(kb_dir)

    rows = [
        {
            "id": "BP001",
            "title": "Imported Duplicate Should Not Override Manual",
            "description": "duplicate id collision from Bugs-QCP",
            "type": "qubit mapping",
        }
    ]
    (input_dir / "sample.json").write_text(json.dumps(rows), encoding="utf-8")

    report = ingest_bugsqcp_into_kb(input_dir=input_dir, output_dir=kb_dir)
    updated = json.loads((kb_dir / "bug_patterns.json").read_text(encoding="utf-8"))

    assert report.duplicate_with_existing == 1
    assert report.manual_preserved_on_collision == 1
    assert any(p["pattern_id"] == "BP001" and p["source"] == "manual" for p in updated)


def test_ingest_skips_empty_rows_and_tracks_duplicates(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    kb_dir = tmp_path / "kb"
    input_dir.mkdir()
    kb_dir.mkdir()
    _write_kb(kb_dir)

    json_rows = [
        {
            "id": "x1",
            "title": "Wrong target qubit",
            "description": "control qubit and target index mismatch",
            "type": "qubit mapping",
        },
        {
            "id": "x1",
            "title": "Duplicate entry",
            "description": "same id in same import set",
            "type": "qubit mapping",
        },
        {
            "id": "x2",
            "title": "",
            "description": "",
            "fix": "",
            "code": "",
        },
    ]
    (input_dir / "bugs.json").write_text(json.dumps(json_rows), encoding="utf-8")

    report = ingest_bugsqcp_into_kb(input_dir=input_dir, output_dir=kb_dir)

    assert report.discovered_records == 3
    assert report.imported_records == 1
    assert report.duplicate_in_input == 1
    assert report.skipped_records == 1


def test_retriever_can_match_imported_taxonomy_and_fix_text() -> None:
    patterns = [
        BugPattern(
            pattern_id="BQCP_A",
            name="Phase kickback control misuse",
            taxonomy_class="incorrect_operator",
            description="Oracle applies wrong controlled phase gate",
            fix_hint="Use CZ with correct control qubit assignment",
            tags=["oracle", "phase", "control"],
            source="bugsqcp",
        ),
        BugPattern(
            pattern_id="BQCP_B",
            name="Measurement omitted",
            taxonomy_class="measurement_error",
            description="Program reads classical bit before measurement",
            source="bugsqcp",
        ),
    ]

    retriever = BugPatternRetriever(patterns)
    results = retriever.retrieve("incorrect_operator control phase oracle fix", top_k=1)
    assert results
    assert results[0].pattern_id == "BQCP_A"


def test_inspect_knowledge_base_detects_duplicates_and_invalid_rows(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()

    rows = [
        {
            "pattern_id": "DUP_1",
            "name": "A",
            "taxonomy_class": "unknown",
            "description": "ok",
            "source": "bugsqcp",
        },
        {
            "pattern_id": "DUP_1",
            "name": "B",
            "taxonomy_class": "unknown",
            "description": "ok",
            "source": "bugsqcp",
        },
        {"pattern_id": "BAD_ONLY"},
    ]

    (kb_dir / "bug_patterns.json").write_text(json.dumps(rows), encoding="utf-8")

    inspection = inspect_knowledge_base(kb_dir)
    assert inspection.total_entries == 3
    assert inspection.valid_entries == 2
    assert inspection.duplicate_ids == ["DUP_1"]
    assert inspection.invalid_entry_indices == [2]
