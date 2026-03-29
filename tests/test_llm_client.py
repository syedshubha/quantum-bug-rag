"""
test_llm_client.py – Unit tests for src/llm_client.py.

We test the MockLLMClient (deterministic, no API calls), the JSON parsing
helper, and the factory function.  OpenAI and Gemini clients are integration-
tested separately (requires credentials); they are excluded here.
"""

from __future__ import annotations

import json

import pytest

from src.llm_client import MockLLMClient, build_llm_client


class TestMockLLMClient:
    def test_complete_returns_string(self) -> None:
        client = MockLLMClient(seed=0)
        messages = [{"role": "user", "content": "Analyse this code."}]
        result = client.complete(messages)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_complete_returns_valid_json(self) -> None:
        client = MockLLMClient(seed=1)
        result = client.complete([{"role": "user", "content": "test"}])
        parsed = json.loads(result)
        assert "bug_likelihood" in parsed
        assert "taxonomy_class" in parsed
        assert "justification" in parsed

    def test_complete_and_parse_returns_dict(self) -> None:
        client = MockLLMClient(seed=2)
        result = client.complete_and_parse([{"role": "user", "content": "test"}])
        assert isinstance(result, dict)
        assert "bug_likelihood" in result

    def test_bug_likelihood_in_range(self) -> None:
        client = MockLLMClient(seed=3)
        for _ in range(10):
            result = json.loads(client.complete([{"role": "user", "content": "x"}]))
            assert 0.0 <= result["bug_likelihood"] <= 1.0

    def test_deterministic_with_same_seed(self) -> None:
        c1 = MockLLMClient(seed=42)
        c2 = MockLLMClient(seed=42)
        msg = [{"role": "user", "content": "test"}]
        assert c1.complete(msg) == c2.complete(msg)

    def test_different_seeds_may_differ(self) -> None:
        results = set()
        for seed in range(20):
            client = MockLLMClient(seed=seed)
            r = json.loads(client.complete([{"role": "user", "content": "test"}]))
            results.add(r["taxonomy_class"])
        # With 20 different seeds we expect more than one distinct class.
        assert len(results) > 1

    def test_complete_and_parse_handles_bad_json(self) -> None:
        """complete_and_parse should degrade gracefully on non-JSON output."""
        client = MockLLMClient(seed=0)
        # Monkey-patch complete to return garbage.
        client.complete = lambda msgs, **kw: "not json at all {{"  # type: ignore[method-assign]
        result = client.complete_and_parse([{"role": "user", "content": "x"}])
        assert "_parse_error" in result
        assert "_raw" in result

    def test_complete_and_parse_strips_markdown_fences(self) -> None:
        """JSON wrapped in ```json … ``` fences should parse correctly."""
        client = MockLLMClient(seed=0)
        payload = {"bug_likelihood": 0.5, "taxonomy_class": "unknown", "suspected_location": "", "justification": ""}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        client.complete = lambda msgs, **kw: fenced  # type: ignore[method-assign]
        result = client.complete_and_parse([{"role": "user", "content": "x"}])
        assert result["taxonomy_class"] == "unknown"


class TestBuildLLMClient:
    def test_mock_backend(self) -> None:
        client = build_llm_client({"llm": {"backend": "mock"}})
        assert isinstance(client, MockLLMClient)

    def test_default_backend_is_mock(self) -> None:
        client = build_llm_client({})
        assert isinstance(client, MockLLMClient)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            build_llm_client({"llm": {"backend": "nonexistent"}})
