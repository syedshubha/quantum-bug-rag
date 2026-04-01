#!/usr/bin/env python3
"""
reproducible_runner.py – Run a small reproducible benchmark across multiple modes.

This script wraps the existing BenchmarkRunner so that prompt-only, RAG,
and optionally static baseline can be executed on the same subset of data
with one command.

Example
-------
    python scripts/reproducible_runner.py --limit 10
    python scripts/reproducible_runner --limit 20 --modes prompt_only rag
    python scripts/reproducible_runner --modes static prompt_only rag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_runner import BenchmarkRunner
from src.dataset_loader import describe_dataset, load_bugs4q_dataset
from src.utils import get_logger, load_config

logger = get_logger("reproducible_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible benchmark across multiple modes."
    )
    parser.add_argument(
        "--data-dir",
        default="data/bugs4q/",
        help="Bugs4Q data directory.",
    )
    parser.add_argument(
        "--kb-dir",
        default="knowledge_base/",
        help="Knowledge-base directory (used for rag mode).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/",
        help="Output directory.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to run in the benchmark.",
    )
    parser.add_argument(
        "--dataset",
        choices=["active", "real", "synthetic"],
        default="active",
        help="Dataset artifact to evaluate.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["static", "prompt_only", "rag"],
        default=["prompt_only", "rag"],
        help="Benchmark modes to execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config.setdefault("paths", {})["knowledge_base"] = args.kb_dir

    try:
        dataset = load_bugs4q_dataset(args.data_dir, dataset=args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    logger.info("Dataset: %s", describe_dataset(dataset))
    samples = [s for s in dataset.samples if s.ground_truth is not None]

    if args.limit:
        samples = samples[: args.limit]
    logger.info("Benchmark capped to %d samples.", len(samples))

    run_summaries: list[tuple[str, str]] = []

    for mode in args.modes:
        logger.info("Starting run for mode='%s'...", mode)
        runner = BenchmarkRunner(mode=mode, config=config, output_dir=args.output_dir)
        _, summary = runner.run(samples)
        run_summaries.append((mode, summary.run_id))
        logger.info("Finished mode='%s' with run_id=%s", mode, summary.run_id)

    print("\n" + "=" * 60)
    print("Reproducible benchmark complete")
    print("=" * 60)
    for mode, run_id in run_summaries:
        print(f"Mode: {mode:<12} Run ID: {run_id}")
    print(f"Outputs written to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()