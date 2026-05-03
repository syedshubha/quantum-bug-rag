"""Per-mode evaluation orchestrator for the classical-vs-quantum track.

Three modes are supported:
  - ``prompt_only``    : code only, no retrieval.
  - ``biased_rag``     : ``BM25Retriever`` over the quantum-only KB.
  - ``balanced_rag``   : ``BalancedRetriever`` over the symmetric KB.

Each mode emits one ``Diagnostic`` per sample. The threshold for the
discrete prediction is fixed at 0.5 on ``score_quantum``.
"""

from __future__ import annotations

from typing import Optional

from .llm import BaseLLM
from .prompts import build_prompt_only, build_rag
from .retriever import BM25Retriever, BalancedRetriever
from .schemas import BugSample, Diagnostic

RAG_K_BIASED = 4    # 4 entries from quantum KB
RAG_K_BALANCED = 2  # 2 quantum + 2 classical


def predict(
    sample: BugSample,
    mode: str,
    llm: BaseLLM,
    retriever_biased: Optional[BM25Retriever] = None,
    retriever_balanced: Optional[BalancedRetriever] = None,
    k_biased: int = RAG_K_BIASED,
    k_balanced: int = RAG_K_BALANCED,
) -> Diagnostic:
    if mode == "prompt_only":
        retrieved = []
        msgs = build_prompt_only(sample)
    elif mode == "biased_rag":
        if retriever_biased is None:
            raise ValueError("biased_rag requires retriever_biased")
        retrieved = retriever_biased.top_k(sample.code[:5000], k=k_biased)
        msgs = build_rag(sample, retrieved)
    elif mode == "balanced_rag":
        if retriever_balanced is None:
            raise ValueError("balanced_rag requires retriever_balanced")
        retrieved = retriever_balanced.top_k(sample.code[:5000], per_domain=k_balanced)
        msgs = build_rag(sample, retrieved)
    else:
        raise ValueError(f"unknown mode: {mode}")

    raw = llm.complete(msgs)
    parsed = llm.parse(raw)

    score = parsed.get("score_quantum")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))
    pred = "quantum" if score >= 0.5 else "classical"

    return Diagnostic(
        sample_id=sample.sample_id, mode=mode,
        predicted=pred, score_quantum=score,
        ground_truth=sample.ground_truth, correct=(pred == sample.ground_truth),
        retrieved_ids=[e.entry_id for e in retrieved],
        reasoning=str(parsed.get("reasoning", ""))[:300],
    )


def run_dataset(
    name: str,
    samples: list[BugSample],
    llm: BaseLLM,
    retriever_biased: BM25Retriever,
    retriever_balanced: BalancedRetriever,
    modes: tuple[str, ...] = ("prompt_only", "biased_rag", "balanced_rag"),
    progress_every: int = 25,
) -> dict[str, list[Diagnostic]]:
    """Run every (sample, mode) combination; return diagnostics keyed by mode."""
    diags: dict[str, list[Diagnostic]] = {m: [] for m in modes}
    for i, s in enumerate(samples, 1):
        for m in modes:
            diags[m].append(predict(
                s, m, llm,
                retriever_biased=retriever_biased,
                retriever_balanced=retriever_balanced,
            ))
        if i % progress_every == 0 or i == len(samples):
            print(f"  [{name}] {i}/{len(samples)}", flush=True)
    return diags
