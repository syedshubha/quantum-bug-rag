"""Framework-aware diversified BM25 retriever for the v6 taxonomy track.

Pipeline at retrieval time:

1. Detect the framework of the query (Qiskit / PennyLane / Cirq / Q#) via
   syntactic signatures.
2. Compute BM25 scores over the full KB.
3. Boost scores by 1.5x for KB entries tagged with the detected framework.
4. From a top-pool of size ``top_pool``, pick at least one entry per
   distinct taxonomy class (diversification).
5. Fill remaining slots with the next-best regardless of class.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import BugPattern


def _tokenize(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9_]+", s.lower()) if t and len(t) > 1]


_FRAMEWORK_TAGS: set[str] = {"qiskit", "pennylane", "cirq", "qsharp"}


def detect_framework(code: str) -> str:
    """Identify which quantum framework a code snippet uses.

    Returns one of {"qiskit", "pennylane", "cirq", "qsharp", "other"}.
    Threshold is 3: if no framework reaches 3 signature points, returns
    ``"other"``.
    """
    code_l = code.lower()
    score = {"qiskit": 0, "cirq": 0, "pennylane": 0, "qsharp": 0}
    if "qiskit" in code_l:
        score["qiskit"] += 5
    if "quantumcircuit" in code_l:
        score["qiskit"] += 3
    if "qiskit_aer" in code_l or "qiskit-aer" in code_l:
        score["qiskit"] += 3
    if "aer.get_backend" in code_l or "aer_simulator" in code_l:
        score["qiskit"] += 2
    if "qasm" in code_l:
        score["qiskit"] += 1
    if re.search(r"\bcirq\.", code_l):
        score["cirq"] += 5
    if "gridqubit" in code_l or "lineqbit" in code_l:
        score["cirq"] += 3
    if "pennylane" in code_l or "qml." in code_l:
        score["pennylane"] += 5
    if "@qml.qnode" in code_l or "qml.device" in code_l:
        score["pennylane"] += 3
    if ".qs ===" in code_l[:300] or "operation " in code_l[:400] or "using (" in code_l[:400]:
        score["qsharp"] += 3
    best_fw, best_score = max(score.items(), key=lambda x: x[1])
    return best_fw if best_score >= 3 else "other"


class FrameworkAwareRetriever:
    """BM25 retriever with framework boosting and class diversification."""

    def __init__(self, patterns: Iterable[BugPattern], top_pool: int = 20) -> None:
        self.patterns: list[BugPattern] = list(patterns)
        self.top_pool = top_pool
        if self.patterns:
            corpus = [_tokenize(self._to_text(p)) for p in self.patterns]
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None
        # Pre-index patterns by framework tag for fast lookup
        self._fw_index: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(self.patterns):
            for fw in _FRAMEWORK_TAGS & set(p.tags):
                self._fw_index[fw].append(i)

    @staticmethod
    def _to_text(p: BugPattern) -> str:
        return " ".join(filter(None, [
            p.name, p.taxonomy_class, p.taxonomy_class,
            p.description, " ".join(p.tags),
        ]))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        framework: str = "other",
    ) -> list[BugPattern]:
        if self.bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)

        if framework in self._fw_index:
            boost_indices = set(self._fw_index[framework])
            scores = scores.copy()
            for i in boost_indices:
                scores[i] *= 1.5

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        pool = [(i, s) for i, s in ranked[: self.top_pool] if s > 0]
        if not pool:
            return []

        result: list[BugPattern] = []
        seen_classes: set[str] = set()
        # First pass: best per distinct taxonomy class
        for i, _ in pool:
            p = self.patterns[i]
            if p.taxonomy_class not in seen_classes:
                result.append(p)
                seen_classes.add(p.taxonomy_class)
                if len(result) >= top_k:
                    break
        # Second pass: fill remaining slots
        for i, _ in pool:
            if len(result) >= top_k:
                break
            if self.patterns[i] not in result:
                result.append(self.patterns[i])
        return result[:top_k]
