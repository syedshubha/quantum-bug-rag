#!/usr/bin/env python3
"""
run_subset_eval.py – Quick subset evaluation across all three pipeline modes.

This script runs static, prompt-only, and RAG modes on a small subset of
Bugs4Q samples and prints a side-by-side comparison.  It is intended for rapid
iteration and does not replace a full evaluation run.

⚠️  When using smoke-test synthetic data, results are for infrastructure
validation only and must NOT be reported as benchmark results.

Usage
-----
    # With real Bugs4Q data:
    python scripts/run_subset_eval.py \
        --data-dir data/bugs4q/ \
        --subset-size 20 \
        --config config.yaml

    # With synthetic smoke-test data:
    python scripts/run_subset_eval.py \
        --smoke-test \
        --subset-size 10 \
        --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_runner import BenchmarkRunner
from src.dataset_loader import generate_smoke_samples, load_bugs4q
from src.evaluate import print_summary
from src.utils import get_logger, load_config

logger = get_logger("run_subset_eval")

MODES = ["static", "prompt_only", "rag"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick subset evaluation.")
    parser.add_argument("--data-dir", default="data/bugs4q/", help="Bugs4Q data directory.")
    parser.add_argument("--kb-dir", default="knowledge_base/", help="Knowledge-base directory.")
    parser.add_argument("--output-dir", default="outputs/", help="Output directory.")
    parser.add_argument("--config", default="config.yaml", help="Configuration file.")
    parser.add_argument("--subset-size", type=int, default=20, help="Number of samples.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use synthetic smoke-test data (infrastructure validation only).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
        help="Modes to run (default: all three).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("paths", {})["knowledge_base"] = args.kb_dir

    if args.smoke_test:
        logger.warning(
            "⚠  Using SYNTHETIC smoke-test data. "
            "Results are for pipeline validation only."
        )
        samples = generate_smoke_samples(n=args.subset_size)
    else:
        all_samples = load_bugs4q(args.data_dir)
        if not all_samples:
            logger.error(
                "No samples found in '%s'. Run prepare_bugs4q.py first.", args.data_dir
            )
            sys.exit(1)
        samples = all_samples[: args.subset_size]

    summaries = {}
    for mode in args.modes:
        logger.info("Running mode: %s …", mode)
        runner = BenchmarkRunner(mode=mode, config=config, output_dir=args.output_dir)
        _, summary = runner.run(samples)
        summaries[mode] = summary

    # Print comparison table.
    print("\n" + "=" * 70)
    print("  Subset Evaluation Summary")
    print("=" * 70)
    header = f"{'Mode':<15} {'N':>5} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}"
    print(header)
    print("-" * 70)
    for mode, s in summaries.items():
        print(
            f"{mode:<15} {s.num_samples:>5} {s.accuracy:>8.4f} "
            f"{s.f1_macro:>8.4f} {s.precision_macro:>8.4f} {s.recall_macro:>8.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
