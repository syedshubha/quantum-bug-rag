"""
test_retriever.py – Unit tests for src/retriever.py.

We verify TF-IDF index construction, retrieval ordering, and edge-case
behaviour (empty pattern list, no-match query).
"""

from __future__ import annotations

import pytest

from src.retriever import BugPatternRetriever
from src.schemas import BugPattern


def _make_pattern(pid: str, name: str, desc: str, tax: str = "unknown") -> BugPattern:
    return BugPattern(
        pattern_id=pid,
        name=name,
        taxonomy_class=tax,
        description=desc,
    )


PATTERNS = [
    _make_pattern("P1", "CNOT Self-Loop", "CNOT applied with same control and target qubit", "incorrect_qubit_mapping"),
    _make_pattern("P2", "Missing Measurement", "Classical register read without prior measurement gate", "measurement_error"),
    _make_pattern("P3", "Wrong Angle", "Rotation gate given wrong angle in degrees instead of radians", "incorrect_operator"),
    _make_pattern("P4", "Barrier Missing", "No barrier before measurement allows optimiser to reorder gates", "missing_barrier"),
    _make_pattern("P5", "Qubit Index OOB", "Qubit index exceeds circuit register size", "incorrect_qubit_mapping"),
]


class TestBugPatternRetriever:
    def test_empty_patterns(self) -> None:
        ret = BugPatternRetriever([])
        assert ret.num_patterns == 0
        result = ret.retrieve("CNOT qubit", top_k=3)
        assert result == []

    def test_num_patterns(self) -> None:
        ret = BugPatternRetriever(PATTERNS)
        assert ret.num_patterns == len(PATTERNS)

    def test_retrieve_returns_list(self) -> None:
        ret = BugPatternRetriever(PATTERNS)
        result = ret.retrieve("CNOT qubit control target", top_k=3)
        assert isinstance(result, list)

    def test_retrieve_top_k_limit(self) -> None:
        ret = BugPatternRetriever(PATTERNS)
        result = ret.retrieve("qubit measurement gate", top_k=2)
        assert len(result) <= 2

    def test_retrieve_relevant_pattern(self) -> None:
        ret = BugPatternRetriever(PATTERNS)
        result = ret.retrieve("CNOT same control target qubit loop")
        ids = [p.pattern_id for p in result]
        # CNOT self-loop pattern should be top-ranked.
        assert ids[0] == "P1"

    def test_retrieve_no_match_returns_empty(self) -> None:
        ret = BugPatternRetriever(PATTERNS)
        result = ret.retrieve("xyzzy quux frobnicate")
        assert result == []

    def test_pattern_ids(self) -> None:
        ret = BugPatternRetriever(PATTERNS)
        ids = ret.pattern_ids()
        assert set(ids) == {"P1", "P2", "P3", "P4", "P5"}

    def test_retrieve_respects_top_k_all(self) -> None:
        ret = BugPatternRetriever(PATTERNS)
        result = ret.retrieve("qubit gate circuit measurement", top_k=10)
        # Can't exceed total patterns.
        assert len(result) <= len(PATTERNS)
