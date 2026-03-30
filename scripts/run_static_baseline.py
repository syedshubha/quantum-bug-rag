#!/usr/bin/env python3
"""
run_static_baseline.py – Run the rule-based static baseline on Bugs4Q.

The static baseline uses lightweight textual heuristics.  It does not call any
LLM and requires no API keys.  It is an intentional placeholder rather than a
faithful re-implementation of any published quantum static analyser.

Usage
-----
    python scripts/run_static_baseline.py \
        --data-dir data/bugs4q/ \
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_runner import BenchmarkRunner
from src.dataset_loader import describe_dataset, load_bugs4q_dataset
from src.utils import get_logger

logger = get_logger("run_static_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-based static baseline.")
    parser.add_argument("--data-dir", default="data/bugs4q/", help="Bugs4Q data directory.")
    parser.add_argument("--output-dir", default="outputs/", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of samples (dev use).")
    args = parser.parse_args()

    try:
        dataset = load_bugs4q_dataset(args.data_dir, dataset="active")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    logger.info("Static baseline dataset: %s", describe_dataset(dataset))
    samples = dataset.samples

    if args.limit:
        samples = samples[: args.limit]

    runner = BenchmarkRunner(mode="static", config={}, output_dir=args.output_dir)
    diagnostics, summary = runner.run(samples)
    logger.info("Static baseline run complete. Run ID: %s", summary.run_id)


if __name__ == "__main__":
    main()
