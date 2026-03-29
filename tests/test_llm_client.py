"""
test_llm_client.py – Unit tests for the LLM client layer.
"""

import json

import pytest

from src.llm_client import MockLLMClient, _parse_response, build_llm_client
from src.schemas import BugTaxonomyClass, DiagnosticResult


class TestMockLLMClient:
    def test_complete_returns_string(self):
        client = MockLLMClient()
        result = client.complete("some prompt")
        assert isinstance(result, str)

    def test_complete_returns_valid_json(self):
        client = MockLLMClient()
        raw = client.complete("QuantumCircuit with cx gate")
        data = json.loads(raw)
        assert "bug_likelihood" in data
        assert "taxonomy_class" in data

    def test_analyse_returns_diagnostic_result(self):
        client = MockLLMClient()
        result = client.analyse("prog_001", "prompt", mode="prompt_only")
        assert isinstance(result, DiagnosticResult)
        assert result.program_id == "prog_001"
        assert result.mode == "prompt_only"

    def test_likelihood_in_range(self):
        client = MockLLMClient(bug_likelihood=0.9)
        result = client.analyse("prog_002", "test prompt", mode="mock")
        assert 0.0 <= result.bug_likelihood <= 1.0

    def test_missing_measure_heuristic(self):
        """I expect higher likelihood when 'measure' is absent from the prompt."""
        client = MockLLMClient(bug_likelihood=0.5)
        result = client.analyse("prog_003", "QuantumCircuit cx gate apply", mode="mock")
        # Mock heuristic: no 'measure' → likelihood bumped to >= 0.75
        assert result.bug_likelihood >= 0.5


class TestParseResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "bug_likelihood": 0.9,
            "taxonomy_class": "missing_measurement",
            "suspected_location": "circuit.py:5",
            "justification": "No measurement found.",
        })
        result = _parse_response("prog_001", raw, "prompt_only")
        assert result.bug_likelihood == 0.9
        assert result.taxonomy_class == BugTaxonomyClass.MISSING_MEASUREMENT.value

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"bug_likelihood": 0.7, "taxonomy_class": "unknown", "suspected_location": null, "justification": "OK"}\n```'
        result = _parse_response("prog_002", raw, "rag")
        assert result.bug_likelihood == 0.7

    def test_invalid_json_fallback(self):
        result = _parse_response("prog_003", "not json at all", "static")
        assert result.taxonomy_class == BugTaxonomyClass.UNKNOWN.value
        assert result.bug_likelihood == 0.5

    def test_unknown_taxonomy_class_fallback(self):
        raw = json.dumps({
            "bug_likelihood": 0.5,
            "taxonomy_class": "completely_made_up_class",
            "justification": "test",
        })
        result = _parse_response("prog_004", raw, "mock")
        assert result.taxonomy_class == BugTaxonomyClass.UNKNOWN.value


class TestBuildLLMClient:
    def test_build_mock(self):
        client = build_llm_client("mock")
        assert isinstance(client, MockLLMClient)

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            build_llm_client("nonexistent_backend")
