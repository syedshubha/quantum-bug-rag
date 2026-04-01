"""
retriever.py – Local retrieval module for bug-pattern context.

We implement a BM25 (Okapi BM25) retriever as the default local option.
BM25 handles code-token overlap and short documents better than vanilla
TF-IDF cosine similarity.  A configurable minimum-score threshold filters
out low-relevance patterns to avoid polluting the LLM context window.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import BugPattern
from .utils import get_logger

logger = get_logger(__name__)

# Default minimum BM25 score — patterns scoring below this are dropped.
_DEFAULT_MIN_SCORE: float = 1.0


class BugPatternRetriever:
    """
    BM25-based retriever over a collection of BugPattern entries.

    Usage::

        retriever = BugPatternRetriever(patterns, min_score=1.0)
        top_patterns = retriever.retrieve(code_snippet, top_k=5)
    """

    def __init__(
        self,
        patterns: list[BugPattern],
        min_score: float = _DEFAULT_MIN_SCORE,
        exclude_classes: list[str] | None = None,
    ) -> None:
        if exclude_classes:
            before = len(patterns)
            patterns = [
                p for p in patterns if p.taxonomy_class not in exclude_classes
            ]
            logger.info(
                "Filtered retrieval index: %d → %d patterns (excluded classes: %s).",
                before,
                len(patterns),
                exclude_classes,
            )
        if not patterns:
            logger.warning("Retriever initialised with an empty pattern list.")
        self._patterns = patterns
        self._bm25: Optional[BM25Okapi] = None
        self._min_score = min_score
        if patterns:
            self._build_index()

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self) -> None:
        tokenised_corpus = [
            self._pattern_to_text(p).lower().split() for p in self._patterns
        ]
        self._bm25 = BM25Okapi(tokenised_corpus)
        logger.debug(
            "BM25 index built over %d patterns (min_score=%.2f).",
            len(self._patterns),
            self._min_score,
        )

    @staticmethod
    def _pattern_to_text(p: BugPattern) -> str:
        """Concatenate pattern fields into a retrieval document."""
        tags_text = " ".join(p.tags)
        fields = [
            p.name,
            p.name,
            p.taxonomy_class,
            p.taxonomy_class,
            p.description,
            p.fix_hint,
            tags_text,
            tags_text,
            p.example_code,
        ]
        return " ".join(filter(None, fields))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[BugPattern]:
        """
        Return the *top_k* most relevant patterns for *query*.

        Patterns whose BM25 score falls below ``self._min_score`` are
        excluded even if fewer than *top_k* results remain.  Returns an
        empty list if the index is uninitialised (no patterns loaded).
        """
        if self._bm25 is None:
            return []

        tokenised_query = query.lower().split()
        scores = self._bm25.get_scores(tokenised_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            self._patterns[i]
            for i in top_indices
            if scores[i] >= self._min_score
        ]
        if logger.isEnabledFor(10):  # DEBUG
            kept = len(results)
            logger.debug(
                "BM25 retrieved %d/%d patterns (threshold %.2f, top score %.3f).",
                kept,
                top_k,
                self._min_score,
                float(scores[top_indices[0]]) if len(top_indices) else 0.0,
            )
        return results

    # ── Metadata ─────────────────────────────────────────────────────────────

    @property
    def num_patterns(self) -> int:
        return len(self._patterns)

    def pattern_ids(self) -> list[str]:
        return [p.pattern_id for p in self._patterns]
