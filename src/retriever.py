"""
retriever.py – Local retrieval module for bug-pattern context.

We support three retrieval modes:

1. **BM25** (Okapi BM25) – lexical retrieval over tokenised pattern text.
2. **Dense** – semantic retrieval via sentence-transformer embeddings + FAISS.
3. **Hybrid** (default when dense model is provided) – a weighted combination
   of normalised BM25 and dense cosine-similarity scores.

A configurable minimum-score threshold filters out low-relevance patterns
to avoid polluting the LLM context window.
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
    Hybrid BM25 + Dense retriever over a collection of BugPattern entries.

    When *dense_model* is provided the retriever encodes patterns with
    sentence-transformers, builds a FAISS inner-product index, and returns
    results ranked by a weighted combination of normalised BM25 and dense
    cosine-similarity scores.

    Usage::

        retriever = BugPatternRetriever(
            patterns, min_score=1.0,
            dense_model="all-MiniLM-L6-v2", dense_weight=0.5,
        )
        top_patterns = retriever.retrieve(code_snippet, top_k=5)
    """

    def __init__(
        self,
        patterns: list[BugPattern],
        min_score: float = _DEFAULT_MIN_SCORE,
        exclude_classes: list[str] | None = None,
        dense_model: str | None = None,
        dense_weight: float = 0.5,
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

        # Dense retrieval state
        self._dense_model_name = dense_model
        self._dense_weight = max(0.0, min(1.0, dense_weight))
        self._st_model = None  # lazy-loaded SentenceTransformer
        self._faiss_index = None
        self._dense_enabled = dense_model is not None

        if patterns:
            self._build_bm25_index()
            if self._dense_enabled:
                self._build_dense_index()

    # ── Index construction ────────────────────────────────────────────────────

    def _build_bm25_index(self) -> None:
        tokenised_corpus = [
            self._pattern_to_text(p).lower().split() for p in self._patterns
        ]
        self._bm25 = BM25Okapi(tokenised_corpus)
        logger.debug(
            "BM25 index built over %d patterns (min_score=%.2f).",
            len(self._patterns),
            self._min_score,
        )

    def _build_dense_index(self) -> None:
        """Encode all patterns and build a FAISS inner-product index."""
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Building dense index with model '%s' over %d patterns …",
            self._dense_model_name,
            len(self._patterns),
        )
        self._st_model = SentenceTransformer(self._dense_model_name)
        texts = [self._pattern_to_text(p) for p in self._patterns]
        embeddings = self._st_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)  # inner-product on L2-normalised vecs = cosine
        self._faiss_index.add(embeddings)
        logger.info(
            "Dense FAISS index ready (%d vectors, dim=%d).", len(self._patterns), dim,
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

        When a dense index is present, scores are computed as a weighted
        combination of min-max-normalised BM25 and dense cosine similarity.
        Otherwise falls back to pure BM25 scoring with ``self._min_score``
        as the floor threshold.
        """
        if self._bm25 is None:
            return []

        tokenised_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenised_query)

        if self._dense_enabled and self._faiss_index is not None:
            return self._retrieve_hybrid(query, bm25_scores, top_k)

        # Pure BM25 path (unchanged from Phase 2).
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        results = [
            self._patterns[i]
            for i in top_indices
            if bm25_scores[i] >= self._min_score
        ]
        if logger.isEnabledFor(10):  # DEBUG
            kept = len(results)
            logger.debug(
                "BM25 retrieved %d/%d patterns (threshold %.2f, top score %.3f).",
                kept,
                top_k,
                self._min_score,
                float(bm25_scores[top_indices[0]]) if len(top_indices) else 0.0,
            )
        return results

    def _retrieve_hybrid(
        self,
        query: str,
        bm25_scores: np.ndarray,
        top_k: int,
    ) -> list[BugPattern]:
        """
        Combine normalised BM25 and dense cosine-similarity scores.

        hybrid_score = (1 − α) × BM25_norm  +  α × dense_cosine
        where α = self._dense_weight.
        """
        assert self._st_model is not None and self._faiss_index is not None

        # Dense scores via FAISS (all patterns).
        q_emb = self._st_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)
        dense_scores_arr, dense_indices = self._faiss_index.search(
            q_emb, len(self._patterns),
        )
        # FAISS returns results in descending-score order; map back to pattern order.
        dense_scores = np.zeros(len(self._patterns), dtype=np.float32)
        for score, idx in zip(dense_scores_arr[0], dense_indices[0]):
            if idx >= 0:
                dense_scores[idx] = score

        # Min-max normalise BM25 to [0, 1].
        bm25_min, bm25_max = float(bm25_scores.min()), float(bm25_scores.max())
        if bm25_max - bm25_min > 1e-9:
            bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm = np.zeros_like(bm25_scores)

        # Dense cosine similarities from normalised embeddings are in [-1, 1];
        # clamp to [0, 1] (negative similarity means highly dissimilar).
        dense_scores_clamped = np.clip(dense_scores, 0.0, 1.0)

        alpha = self._dense_weight
        hybrid = (1.0 - alpha) * bm25_norm + alpha * dense_scores_clamped

        top_indices = np.argsort(hybrid)[::-1][:top_k]
        results = [self._patterns[int(i)] for i in top_indices]

        if logger.isEnabledFor(10):  # DEBUG
            logger.debug(
                "Hybrid retrieved %d patterns (α=%.2f, top hybrid=%.3f, "
                "top BM25_norm=%.3f, top dense=%.3f).",
                len(results),
                alpha,
                float(hybrid[top_indices[0]]),
                float(bm25_norm[top_indices[0]]),
                float(dense_scores_clamped[top_indices[0]]),
            )
        else:
            logger.info(
                "Hybrid retrieval: top score=%.3f (α=%.2f).",
                float(hybrid[top_indices[0]]),
                alpha,
            )
        return results

    # ── Metadata ─────────────────────────────────────────────────────────────

    @property
    def num_patterns(self) -> int:
        return len(self._patterns)

    def pattern_ids(self) -> list[str]:
        return [p.pattern_id for p in self._patterns]
