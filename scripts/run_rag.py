#!/usr/bin/env python3
"""
run_rag.py – Run the RAG pipeline on Bugs4Q.

This mode augments the LLM prompt with retrieved bug-pattern context from the
local knowledge base.  The knowledge base must be populated before running
(see scripts/prepare_bugsqcp_kb.py).

Usage
-----
    python scripts/run_rag.py \
        --data-dir data/bugs4q/ \
        --kb-dir knowledge_base/ \
        --output-dir outputs/ \
        --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_runner import BenchmarkRunner
from src.dataset_loader import describe_dataset, load_bugs4q_dataset
from src.utils import get_logger, load_config

logger = get_logger("run_rag")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG LLM pipeline.")
    parser.add_argument("--data-dir", default="data/bugs4q/", help="Bugs4Q data directory.")
    parser.add_argument("--kb-dir", default="knowledge_base/", help="Knowledge-base directory.")
    parser.add_argument("--output-dir", default="outputs/", help="Output directory.")
    parser.add_argument("--config", default="config.yaml", help="Configuration file.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of samples (dev use).")
    args = parser.parse_args()

    config = load_config(args.config)
    # Allow CLI override of KB path.
    config.setdefault("paths", {})["knowledge_base"] = args.kb_dir

    try:
        dataset = load_bugs4q_dataset(args.data_dir, dataset="active")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    logger.info("RAG dataset: %s", describe_dataset(dataset))
    samples = dataset.samples

    if args.limit:
        samples = samples[: args.limit]

    runner = BenchmarkRunner(mode="rag", config=config, output_dir=args.output_dir)
    diagnostics, summary = runner.run(samples)
    logger.info("RAG run complete. Run ID: %s", summary.run_id)


if __name__ == "__main__":
    main()
