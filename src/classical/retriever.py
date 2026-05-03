"""BM25 retrievers for the classical-vs-quantum binary track.

The classical notebook used two retrieval conditions:

  - ``BM25Retriever`` over the quantum-only KB to expose the asymmetric-KB
    confound.
  - ``BalancedRetriever`` that runs one BM25 per domain and concatenates the
    top results from each side of the symmetric KB.
"""

from __future__ import annotations

import re
from typing import Iterable

from rank_bm25 import BM25Okapi

from .schemas import KBEntry


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9_]+", text.lower()) if tok and len(tok) > 1]


class BM25Retriever:
    """Plain BM25 over an arbitrary KB pool."""

    def __init__(self, pool: Iterable[KBEntry]) -> None:
        self.pool: list[KBEntry] = list(pool)
        self.bm25 = BM25Okapi([_tokenize(entry.description) for entry in self.pool]) if self.pool else None

    def top_k(self, query: str, k: int = 4) -> list[KBEntry]:
        if self.bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        idxs = sorted(range(len(self.pool)), key=lambda idx: -scores[idx])[:k]
        return [self.pool[idx] for idx in idxs if scores[idx] > 0]


class BalancedRetriever:
    """Run one BM25 per domain and concatenate top-k from each."""

    def __init__(
        self,
        pool_quantum: Iterable[KBEntry],
        pool_classical: Iterable[KBEntry],
    ) -> None:
        self.quantum = BM25Retriever(pool_quantum)
        self.classical = BM25Retriever(pool_classical)

    def top_k(self, query: str, per_domain: int = 2) -> list[KBEntry]:
        return self.quantum.top_k(query, k=per_domain) + self.classical.top_k(query, k=per_domain)
