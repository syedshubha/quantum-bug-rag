"""
test_retriever.py – Unit tests for the knowledge-base retriever.
"""

from pathlib import Path

import pytest

from src.retriever import KnowledgeBaseRetriever
from src.schemas import BugPatternEntry

# I use the actual knowledge base file from the repo.
KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "bug_patterns.json"


@pytest.fixture()
def retriever():
    return KnowledgeBaseRetriever(patterns_path=KB_PATH, top_k=3)


class TestKnowledgeBaseRetriever:
    def test_loads_patterns(self, retriever):
        assert len(retriever._patterns) > 0

    def test_retrieve_returns_list(self, retriever):
        results = retriever.retrieve("QuantumCircuit measure qubit")
        assert isinstance(results, list)

    def test_retrieve_top_k(self, retriever):
        results = retriever.retrieve("circuit", top_k=2)
        assert len(results) <= 2

    def test_retrieve_returns_pattern_entries(self, retriever):
        results = retriever.retrieve("measure")
        for item in results:
            assert isinstance(item, BugPatternEntry)

    def test_retrieve_measurement_query(self, retriever):
        """I expect the missing-measurement pattern to rank highly for a measurement query."""
        results = retriever.retrieve("QuantumCircuit execute missing measure counts empty")
        ids = [r.id for r in results]
        assert "BP001" in ids

    def test_retrieve_empty_query(self, retriever):
        results = retriever.retrieve("")
        # I accept either empty or all patterns returned (implementation-defined).
        assert isinstance(results, list)

    def test_retrieve_by_class(self, retriever):
        patterns = retriever.retrieve_by_class("missing_measurement")
        assert all(p.taxonomy_class == "missing_measurement" for p in patterns)

    def test_missing_file_returns_empty(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.json"
        r = KnowledgeBaseRetriever(patterns_path=nonexistent, top_k=3)
        assert r.retrieve("anything") == []
