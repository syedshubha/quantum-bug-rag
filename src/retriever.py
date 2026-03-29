"""
retriever.py – Local retrieval module for bug-pattern context.

We implement a lightweight TF-IDF + cosine-similarity retriever as the default
local option.  Replacing this with a dense retriever (e.g., Sentence-Transformers
+ FAISS) requires only implementing the same interface.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import BugPattern
from .utils import get_logger

logger = get_logger(__name__)


class BugPatternRetriever:
    """
    TF-IDF based retriever over a collection of BugPattern entries.

    Usage::

        retriever = BugPatternRetriever(patterns)
        top_patterns = retriever.retrieve(code_snippet, top_k=5)
    """

    def __init__(self, patterns: list[BugPattern]) -> None:
        if not patterns:
            logger.warning("Retriever initialised with an empty pattern list.")
        self._patterns = patterns
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._corpus_matrix: Optional[np.ndarray] = None
        if patterns:
            self._build_index()

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self) -> None:
        corpus = [self._pattern_to_text(p) for p in self._patterns]
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=8192,
            sublinear_tf=True,
        )
        self._corpus_matrix = self._vectorizer.fit_transform(corpus).toarray()
        logger.debug("TF-IDF index built over %d patterns.", len(self._patterns))

    @staticmethod
    def _pattern_to_text(p: BugPattern) -> str:
        """Concatenate pattern fields into a single retrieval document."""
        return " ".join(
            filter(None, [p.name, p.description, p.fix_hint, p.example_code, " ".join(p.tags)])
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[BugPattern]:
        """
        Return the *top_k* most similar patterns for *query*.

        Returns an empty list if the index is uninitialised (no patterns loaded).
        """
        if self._vectorizer is None or self._corpus_matrix is None:
            return []

        q_vec = self._vectorizer.transform([query]).toarray()
        sims = cosine_similarity(q_vec, self._corpus_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = [self._patterns[i] for i in top_indices if sims[i] > 0.0]
        return results

    # ── Metadata ─────────────────────────────────────────────────────────────

    @property
    def num_patterns(self) -> int:
        return len(self._patterns)

    def pattern_ids(self) -> list[str]:
        return [p.pattern_id for p in self._patterns]
