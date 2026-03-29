#!/usr/bin/env python3
"""
run_subset_eval.py – Run all three pipeline modes on a small subset and compare metrics.

I prepare the data (or use existing synthetic data), run prompt-only, RAG,
and static pipelines over a configurable subset, compute evaluation metrics,
and print a comparison table.

Usage
-----
    python scripts/run_subset_eval.py [--n 10] [--backend mock] [--out outputs/subset_eval]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import analyse_static
from src.benchmark_runner import run_benchmark
from src.dataset_loader import load_bugs4q
from src.evaluate import EvaluationSummary, evaluate_results, print_summary
from src.llm_client import build_llm_client
from src.prompt_builder import build_prompt_only, build_rag_prompt
from src.retriever import KnowledgeBaseRetriever
from src.schemas import DiagnosticResult
from src.utils import repo_root, save_json, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUT = repo_root() / "outputs" / "subset_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="Subset size (default: 10)")
    parser.add_argument("--backend", default="mock", choices=["mock", "openai", "gemini"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("=== Subset evaluation  n=%d  backend=%s ===", args.n, args.backend)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_bugs4q(data_dir=args.data_dir, max_items=args.n)
    llm = build_llm_client(args.backend)
    retriever = KnowledgeBaseRetriever(top_k=args.top_k)

    # ------------------------------------------------------------------
    # 1. Prompt-only
    # ------------------------------------------------------------------
    def prompt_only_analyser(record: dict) -> DiagnosticResult:
        prompt = build_prompt_only(record["id"], record["source_code"], record.get("description"))
        return llm.analyse(record["id"], prompt, mode="prompt_only")

    prompt_results = run_benchmark(records, prompt_only_analyser)
    save_json([r.model_dump() for r in prompt_results], out_dir / "prompt_only_results.json")

    # ------------------------------------------------------------------
    # 2. RAG
    # ------------------------------------------------------------------
    def rag_analyser(record: dict) -> DiagnosticResult:
        patterns = retriever.retrieve(record["source_code"])
        prompt = build_rag_prompt(record["id"], record["source_code"], patterns, record.get("description"))
        result = llm.analyse(record["id"], prompt, mode="rag")
        result.retrieved_patterns = [p.id for p in patterns]
        return result

    rag_results = run_benchmark(records, rag_analyser)
    save_json([r.model_dump() for r in rag_results], out_dir / "rag_results.json")

    # ------------------------------------------------------------------
    # 3. Static
    # ------------------------------------------------------------------
    def static_analyser(record: dict) -> DiagnosticResult:
        return analyse_static(record["id"], record["source_code"])

    static_results = run_benchmark(records, static_analyser)
    save_json([r.model_dump() for r in static_results], out_dir / "static_results.json")

    # ------------------------------------------------------------------
    # Evaluate all three
    # ------------------------------------------------------------------
    summaries: list[EvaluationSummary] = []
    for mode_results, mode_label in [
        (prompt_results, "prompt_only"),
        (rag_results, "rag"),
        (static_results, "static"),
    ]:
        summary = evaluate_results(mode_results, records, mode=mode_label)
        summaries.append(summary)
        print_summary(summary)

    save_json([s.model_dump() for s in summaries], out_dir / "comparison.json")
    logger.info("Comparison saved to %s/comparison.json", out_dir)


if __name__ == "__main__":
    main()
