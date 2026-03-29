"""
test_schemas.py – Unit tests for Pydantic schemas.
"""

import pytest
from pydantic import ValidationError

from src.schemas import BugTaxonomyClass, DiagnosticResult, EvaluationSummary


class TestDiagnosticResult:
    def test_valid_result(self):
        result = DiagnosticResult(
            program_id="test_001",
            bug_likelihood=0.8,
            taxonomy_class=BugTaxonomyClass.MISSING_MEASUREMENT,
            justification="No measure call found.",
            mode="prompt_only",
        )
        assert result.program_id == "test_001"
        assert result.bug_likelihood == 0.8
        assert result.taxonomy_class == BugTaxonomyClass.MISSING_MEASUREMENT.value  # use_enum_values=True → stored as string

    def test_likelihood_bounds(self):
        with pytest.raises(ValidationError):
            DiagnosticResult(
                program_id="test_002",
                bug_likelihood=1.5,  # out of range
                taxonomy_class=BugTaxonomyClass.UNKNOWN,
                justification="test",
                mode="mock",
            )

    def test_negative_likelihood(self):
        with pytest.raises(ValidationError):
            DiagnosticResult(
                program_id="test_003",
                bug_likelihood=-0.1,
                taxonomy_class=BugTaxonomyClass.UNKNOWN,
                justification="test",
                mode="mock",
            )

    def test_optional_fields_default(self):
        result = DiagnosticResult(
            program_id="test_004",
            bug_likelihood=0.0,
            taxonomy_class=BugTaxonomyClass.UNKNOWN,
            justification="",
            mode="static",
        )
        assert result.suspected_location is None
        assert result.retrieved_patterns == []

    def test_json_round_trip(self):
        result = DiagnosticResult(
            program_id="test_005",
            bug_likelihood=0.65,
            taxonomy_class=BugTaxonomyClass.INCORRECT_GATE,
            suspected_location="circuit.py:10",
            justification="Deprecated u1 gate used.",
            retrieved_patterns=["BP004"],
            mode="rag",
        )
        serialised = result.model_dump()
        restored = DiagnosticResult(**serialised)
        assert restored.program_id == result.program_id
        assert restored.bug_likelihood == result.bug_likelihood


class TestEvaluationSummary:
    def test_valid_summary(self):
        summary = EvaluationSummary(
            mode="mock",
            n_samples=10,
            detection_precision=0.8,
            detection_recall=0.7,
            detection_f1=0.75,
            classification_macro_f1=0.6,
        )
        assert summary.n_samples == 10

    def test_taxonomy_enum_values(self):
        """I verify that all expected taxonomy classes are present."""
        classes = {c.value for c in BugTaxonomyClass}
        assert "missing_measurement" in classes
        assert "wrong_qubit_order" in classes
        assert "incorrect_gate" in classes
        assert "unknown" in classes
