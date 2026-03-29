"""
test_schemas.py – Unit tests for src/schemas.py.

We verify that Pydantic model validation, field defaults, and helper methods
behave as specified.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    BugDiagnostic,
    BugPattern,
    BugSample,
    EvalSummary,
    TaxonomyEntry,
)


# ── BugSample ─────────────────────────────────────────────────────────────────

class TestBugSample:
    def test_valid_construction(self) -> None:
        s = BugSample(sample_id="s1", source="bugs4q", code="qc = QuantumCircuit(1)")
        assert s.sample_id == "s1"
        assert s.ground_truth is None
        assert s.metadata == {}

    def test_empty_code_raises(self) -> None:
        with pytest.raises(ValidationError):
            BugSample(sample_id="s1", source="bugs4q", code="   ")

    def test_with_ground_truth(self) -> None:
        s = BugSample(
            sample_id="s2",
            source="bugs4q",
            code="qc.h(0)",
            ground_truth="incorrect_operator",
        )
        assert s.ground_truth == "incorrect_operator"

    def test_metadata_stored(self) -> None:
        s = BugSample(
            sample_id="s3",
            source="bugs4q",
            code="qc = QuantumCircuit(2)",
            metadata={"synthetic": True},
        )
        assert s.metadata["synthetic"] is True


# ── BugDiagnostic ─────────────────────────────────────────────────────────────

class TestBugDiagnostic:
    def test_basic_construction(self) -> None:
        d = BugDiagnostic(
            sample_id="s1",
            mode="rag",
            bug_likelihood=0.85,
            taxonomy_class="measurement_error",
        )
        assert d.correct is None
        assert d.retrieved_patterns == []

    def test_bug_likelihood_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            BugDiagnostic(
                sample_id="s1",
                mode="rag",
                bug_likelihood=1.5,  # > 1.0
                taxonomy_class="unknown",
            )

    def test_compute_correctness_match(self) -> None:
        d = BugDiagnostic(
            sample_id="s1",
            mode="static",
            bug_likelihood=0.7,
            taxonomy_class="incorrect_operator",
            ground_truth="incorrect_operator",
        )
        d.compute_correctness()
        assert d.correct is True

    def test_compute_correctness_mismatch(self) -> None:
        d = BugDiagnostic(
            sample_id="s1",
            mode="prompt_only",
            bug_likelihood=0.7,
            taxonomy_class="unknown",
            ground_truth="incorrect_operator",
        )
        d.compute_correctness()
        assert d.correct is False

    def test_compute_correctness_no_ground_truth(self) -> None:
        d = BugDiagnostic(
            sample_id="s1",
            mode="rag",
            bug_likelihood=0.5,
            taxonomy_class="unknown",
        )
        d.compute_correctness()
        assert d.correct is None


# ── BugPattern ────────────────────────────────────────────────────────────────

class TestBugPattern:
    def test_minimal_construction(self) -> None:
        p = BugPattern(
            pattern_id="BP001",
            name="Test Pattern",
            taxonomy_class="incorrect_operator",
            description="A test pattern.",
        )
        assert p.tags == []
        assert p.source == ""

    def test_full_construction(self) -> None:
        p = BugPattern(
            pattern_id="BP001",
            name="Test Pattern",
            taxonomy_class="incorrect_operator",
            description="A test pattern.",
            example_code="qc.h(0)",
            fix_hint="Use the correct gate.",
            source="manual",
            tags=["h_gate", "test"],
        )
        assert p.tags == ["h_gate", "test"]


# ── TaxonomyEntry ─────────────────────────────────────────────────────────────

class TestTaxonomyEntry:
    def test_construction(self) -> None:
        te = TaxonomyEntry(
            class_id="incorrect_operator",
            name="Incorrect Operator",
            description="Wrong gate applied.",
        )
        assert te.parent_class is None
        assert te.examples == []


# ── EvalSummary ───────────────────────────────────────────────────────────────

class TestEvalSummary:
    def test_construction(self) -> None:
        s = EvalSummary(
            run_id="abc123",
            mode="rag",
            num_samples=50,
            accuracy=0.74,
            f1_macro=0.71,
            precision_macro=0.73,
            recall_macro=0.69,
        )
        assert s.per_class_f1 == {}
        assert s.notes == ""
