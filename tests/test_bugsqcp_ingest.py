from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_ingest_fails_when_no_usable_records_found(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    kb_dir = tmp_path / "kb"
    input_dir.mkdir()
    kb_dir.mkdir()
    _write_kb(kb_dir)

    rows = [
        {"id": "u1", "title": "", "description": "", "fix": "", "code": ""},
        {"id": "u2", "title": "", "description": "", "fix": "", "code": ""},
    ]
    (input_dir / "empty.json").write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="No usable Bugs-QCP records were found"):
        ingest_bugsqcp_into_kb(input_dir=input_dir, output_dir=kb_dir)


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

    taxonomy = [
        {
            "class_id": "unknown",
            "name": "Unknown",
            "description": "Unknown class",
            "parent_class": None,
            "examples": [],
        },
        {
            "class_id": "measurement_error",
            "name": "Measurement Error",
            "description": "Measurement issue",
            "parent_class": None,
            "examples": [],
        },
    ]

    (kb_dir / "bug_patterns.json").write_text(json.dumps(rows), encoding="utf-8")
    (kb_dir / "taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")

    inspection = inspect_knowledge_base(kb_dir)
    assert inspection.total_entries == 3
    assert inspection.valid_entries == 2
    assert inspection.duplicate_ids == ["DUP_1"]
    assert inspection.invalid_entry_indices == [2]
    assert inspection.source_role_counts["bugsqcp"] == 2
    assert inspection.missing_required_fields["name"] == 1
    assert inspection.taxonomy_defined_classes == ["measurement_error", "unknown"]
    assert inspection.taxonomy_covered_classes == ["unknown"]
    assert inspection.taxonomy_missing_classes == ["measurement_error"]
    assert len(inspection.sample_imported_entries) == 2


# ---------------------------------------------------------------------------
# Real Bugs-QCP canonical-layout tests
# ---------------------------------------------------------------------------

_FULL_TAXONOMY = [
    {"class_id": "incorrect_operator",     "name": "Incorrect Operator",     "description": "Wrong gate or operator.", "parent_class": None, "examples": []},
    {"class_id": "incorrect_qubit_mapping","name": "Incorrect Qubit Mapping","description": "Bad qubit index.",        "parent_class": None, "examples": []},
    {"class_id": "missing_barrier",        "name": "Missing Barrier",        "description": "Barrier omitted.",       "parent_class": None, "examples": []},
    {"class_id": "wrong_initial_state",    "name": "Wrong Initial State",    "description": "Bad initial state.",     "parent_class": None, "examples": []},
    {"class_id": "measurement_error",      "name": "Measurement Error",      "description": "Measurement issue.",     "parent_class": None, "examples": []},
    {"class_id": "no_bug_detected",        "name": "No Bug Detected",        "description": "No bug found.",         "parent_class": None, "examples": []},
    {"class_id": "unknown",                "name": "Unknown",                "description": "Unknown class.",         "parent_class": None, "examples": []},
]


def _write_full_kb(kb_dir: Path) -> None:
    (kb_dir / "bug_patterns.json").write_text("[]", encoding="utf-8")
    (kb_dir / "taxonomy.json").write_text(json.dumps(_FULL_TAXONOMY), encoding="utf-8")


_ANNOTATION_CSV_HEADER = "id,real,type,repo,commit_hash,component,symptom,bug_pattern,comment,localization\n"

_ANNOTATION_CSV_ROWS = (
    # real bugs
    "qiskit-aer#767,bug,Quantum,https://github.com/Qiskit/qiskit-aer,abc123,"
    "Transpiler,Simulation gives wrong result,Barrier Related,"
    "Missing barrier between entanglement and measurement,statevector_simulator.py\n"
    "qiskit-terra#3410,bug,Classical,https://github.com/Qiskit/qiskit-terra,def456,"
    "Compiler,Incorrect matrix applied,Incorrect Numerical Computation,"
    "Wrong rotation angle in U3 gate,compiler.py\n"
    "qiskit-terra#2009,bug,Quantum,https://github.com/Qiskit/qiskit-terra,ghi789,"
    "Circuit,Output measurement off by one,Overlooked Qubit Order,"
    "MSB/LSB convention not applied correctly,circuit.py\n"
    # false positive – should be excluded
    "qiskit-aer#10,fp,Quantum,https://github.com/Qiskit/qiskit-aer,xyz000,"
    "Backend,Not a real bug,API Misuse - External,"
    "Calling deprecated API but behaviour unchanged,aer_backend.py\n"
)


def _make_real_layout(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal Bugs-QCP canonical archive under *tmp_path*."""
    root = tmp_path / "MattePalte-Bugs-Quantum-Computing-Platforms-0c1c805"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)

    (artifacts / "annotation_bugs.csv").write_text(
        _ANNOTATION_CSV_HEADER + _ANNOTATION_CSV_ROWS, encoding="utf-8"
    )

    # per-case metadata enrichment for the first real bug
    case_dir = artifacts / "minimal_bugfixes" / "qiskit-aer" / "qiskit-aer#767"
    case_dir.mkdir(parents=True)
    meta = {
        "commit_hash": "abc123",
        "commit_msg": "Fix barrier ordering in circuit execution",
        "project_name": "qiskit-aer",
        "id": "qiskit-aer#767",
        "human_id": "aer-767",
        "annotator_comment": "Developer forgot to add barrier between CNOT and measurement gates.",
    }
    (case_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # a distraction directory that should be ignored
    old_vers = artifacts / "old_dataset_versions" / "v1"
    old_vers.mkdir(parents=True)
    (old_vers / "annotation_bugs.csv").write_text("junk,data\n1,2\n", encoding="utf-8")

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    _write_full_kb(kb_dir)

    return root, kb_dir


def test_ingest_from_real_annotation_csv_layout(tmp_path: Path) -> None:
    """Only the three real-bug rows should be imported; the false-positive is excluded."""
    root, kb_dir = _make_real_layout(tmp_path)

    report = ingest_bugsqcp_into_kb(input_dir=root, output_dir=kb_dir)

    assert report.discovered_files == 1, "Only the canonical CSV should be counted"
    assert report.discovered_records == 3, "Three rows have real=='bug'"
    assert report.imported_records == 3, "All three real bugs should be imported"
    assert report.skipped_records == 0

    patterns = json.loads((kb_dir / "bug_patterns.json").read_text(encoding="utf-8"))
    ids = {p["pattern_id"] for p in patterns}
    assert "BQCP_qiskit-aer_767" in ids
    assert "BQCP_qiskit-terra_3410" in ids
    assert "BQCP_qiskit-terra_2009" in ids

    tax_classes = {p["taxonomy_class"] for p in patterns}
    assert "missing_barrier" in tax_classes
    assert "incorrect_operator" in tax_classes
    assert "incorrect_qubit_mapping" in tax_classes


def test_ingest_metadata_enrichment_populates_fix_hint(tmp_path: Path) -> None:
    """fix_hint should be populated from metadata.json commit_msg when available."""
    root, kb_dir = _make_real_layout(tmp_path)

    ingest_bugsqcp_into_kb(input_dir=root, output_dir=kb_dir)

    patterns = json.loads((kb_dir / "bug_patterns.json").read_text(encoding="utf-8"))
    enriched = next(p for p in patterns if p["pattern_id"] == "BQCP_qiskit-aer_767")
    # commit_msg from metadata.json should land in fix_hint
    assert "barrier" in enriched.get("fix_hint", "").lower()


def test_ingest_false_positives_are_excluded(tmp_path: Path) -> None:
    """Rows with real=='fp' must never appear in the output KB."""
    root, kb_dir = _make_real_layout(tmp_path)

    ingest_bugsqcp_into_kb(input_dir=root, output_dir=kb_dir)

    patterns = json.loads((kb_dir / "bug_patterns.json").read_text(encoding="utf-8"))
    ids = {p["pattern_id"] for p in patterns}
    assert "BQCP_qiskit-aer_10" not in ids, "False-positive row must not be imported"


def test_ingest_ignores_old_dataset_versions(tmp_path: Path) -> None:
    """The annotation_bugs.csv under old_dataset_versions/ must not be parsed."""
    root, kb_dir = _make_real_layout(tmp_path)

    report = ingest_bugsqcp_into_kb(input_dir=root, output_dir=kb_dir)
    # If old_dataset_versions/ were parsed the junk CSV would add no useful rows
    # but the discovered_files count would increase to >1.
    assert report.discovered_files == 1
