#!/usr/bin/env python3
"""
run_rag.py – Retrieval-augmented LLM pipeline.

I load Bugs4Q, retrieve relevant bug patterns from the local knowledge base
for each program, build an augmented prompt, and call the LLM.

Usage
-----
    python scripts/run_rag.py [--config config.yaml] [--backend mock] \
        [--top-k 3] [--max-items 10] [--out outputs/rag]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_runner import run_benchmark
from src.dataset_loader import load_bugs4q
from src.llm_client import build_llm_client
from src.prompt_builder import build_rag_prompt
from src.retriever import KnowledgeBaseRetriever
from src.schemas import DiagnosticResult
from src.utils import load_config, repo_root, save_json, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUT = repo_root() / "outputs" / "rag"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="mock", choices=["mock", "openai", "gemini"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--kb-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=str(Path(args.out) / "run.log"))

    cfg: dict = {}
    if args.config:
        cfg = load_config(args.config)

    backend = args.backend or cfg.get("llm", {}).get("backend", "mock")
    llm_kwargs = cfg.get("llm", {}).get("kwargs", {})
    top_k = args.top_k or cfg.get("retrieval", {}).get("top_k", 3)

    logger.info("=== RAG pipeline  backend=%s  top_k=%d ===", backend, top_k)

    llm = build_llm_client(backend, **llm_kwargs)
    kb_patterns_path = (Path(args.kb_dir) / "bug_patterns.json") if args.kb_dir else None
    retriever = KnowledgeBaseRetriever(patterns_path=kb_patterns_path, top_k=top_k)
    records = load_bugs4q(data_dir=args.data_dir, max_items=args.max_items)

    def analyser(record: dict) -> DiagnosticResult:
        # I retrieve relevant patterns using the source code as the query.
        patterns = retriever.retrieve(record["source_code"])
        prompt = build_rag_prompt(
            program_id=record["id"],
            source_code=record["source_code"],
            retrieved_patterns=patterns,
            description=record.get("description"),
        )
        result = llm.analyse(record["id"], prompt, mode="rag")
        result.retrieved_patterns = [p.id for p in patterns]
        return result

    results = run_benchmark(records, analyser)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    save_json([r.model_dump() for r in results], out_file)
    logger.info("Saved %d results to %s", len(results), out_file)


if __name__ == "__main__":
    main()
