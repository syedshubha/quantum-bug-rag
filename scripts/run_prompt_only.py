#!/usr/bin/env python3
"""
run_prompt_only.py – Run the prompt-only LLM baseline on Bugs4Q.

This mode sends the raw code snippet to the configured LLM backend without
any retrieved context.  Results are written to the output directory.

Usage
-----
    python scripts/run_prompt_only.py \
        --data-dir data/bugs4q/ \
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

logger = get_logger("run_prompt_only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prompt-only LLM baseline.")
    parser.add_argument("--data-dir", default="data/bugs4q/", help="Bugs4Q data directory.")
    parser.add_argument("--output-dir", default="outputs/", help="Output directory.")
    parser.add_argument("--config", default="config.yaml", help="Configuration file.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of samples (dev use).")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        dataset = load_bugs4q_dataset(args.data_dir, dataset="active")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    logger.info("Prompt-only dataset: %s", describe_dataset(dataset))
    samples = dataset.samples

    if args.limit:
        samples = samples[: args.limit]
        logger.info("Capped to %d samples.", len(samples))

    runner = BenchmarkRunner(mode="prompt_only", config=config, output_dir=args.output_dir)
    diagnostics, summary = runner.run(samples)
    logger.info("Prompt-only run complete. Run ID: %s", summary.run_id)


if __name__ == "__main__":
    main()
