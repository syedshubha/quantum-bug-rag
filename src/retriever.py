"""
retriever.py – Simple keyword-based retrieval over a local knowledge base.

I implement a first version of retrieval that scores each knowledge-base
entry against the query using keyword overlap (TF-style).  This is
intentionally simple and can be swapped for a dense vector retriever
(e.g. FAISS + sentence-transformers) later without changing the public API.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import BugPatternEntry
from .utils import load_json, repo_root

logger = logging.getLogger(__name__)

_DEFAULT_KB_DIR = repo_root() / "knowledge_base"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class KnowledgeBaseRetriever:
    """
    I wrap the local ``knowledge_base/bug_patterns.json`` and provide a
    ``retrieve`` method that returns the most relevant entries for a query.
    """

    def __init__(
        self,
        patterns_path: Optional[str | Path] = None,
        top_k: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        patterns_path:
            Path to ``bug_patterns.json``.  Defaults to
            ``knowledge_base/bug_patterns.json`` in the repo root.
        top_k:
            Number of entries to return per query.
        """
        self.top_k = top_k
        patterns_path = Path(patterns_path) if patterns_path else _DEFAULT_KB_DIR / "bug_patterns.json"
        self._patterns: List[BugPatternEntry] = self._load_patterns(patterns_path)
        logger.info("KnowledgeBaseRetriever loaded %d patterns", len(self._patterns))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_patterns(path: Path) -> List[BugPatternEntry]:
        """I load and validate the bug-pattern knowledge base."""
        if not path.exists():
            logger.warning("Bug patterns file not found at %s; retrieval will return empty results.", path)
            return []
        raw: List[Dict[str, Any]] = load_json(path)
        return [BugPatternEntry(**entry) for entry in raw]

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """I split text into lowercase word tokens."""
        return re.findall(r"\w+", text.lower())

    def _score(self, pattern: BugPatternEntry, query_tokens: List[str]) -> float:
        """
        I compute a relevance score for *pattern* given *query_tokens*.

        The score is the fraction of query tokens that appear in the
        combined token set of the pattern's title, description, and keywords.
        """
        pattern_text = " ".join(
            [pattern.title, pattern.description] + pattern.keywords + ([pattern.code_snippet] if pattern.code_snippet else [])
        )
        pattern_tokens = set(self._tokenise(pattern_text))
        if not query_tokens:
            return 0.0
        matches = sum(1 for t in query_tokens if t in pattern_tokens)
        return matches / len(query_tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[BugPatternEntry]:
        """
        I return the top-*k* most relevant bug-pattern entries for *query*.

        Parameters
        ----------
        query:
            Free-text query, typically the Qiskit source code or a
            natural-language description of a potential bug.
        top_k:
            Overrides the instance-level ``top_k`` for this call.
        """
        k = top_k if top_k is not None else self.top_k
        if not self._patterns:
            return []

        query_tokens = self._tokenise(query)
        scored = [(self._score(p, query_tokens), p) for p in self._patterns]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:k]]

    def retrieve_by_class(self, taxonomy_class: str) -> List[BugPatternEntry]:
        """I return all patterns that belong to a specific taxonomy class."""
        return [p for p in self._patterns if p.taxonomy_class == taxonomy_class]
