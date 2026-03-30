#!/usr/bin/env python3
"""
create_bugs4q_splits.py – Create deterministic Bugs4Q train/dev/eval splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_splits import create_bugs4q_splits
from src.utils import get_logger

logger = get_logger("create_bugs4q_splits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic Bugs4Q splits.")
    parser.add_argument("--data-dir", default="data/bugs4q/", help="Bugs4Q data directory.")
    parser.add_argument(
        "--dataset",
        choices=["active", "real", "synthetic"],
        default="active",
        help="Prepared dataset selection (default: active).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split ratio.")
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio.")
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="Eval split ratio.")
    parser.add_argument(
        "--include-unlabelled",
        action="store_true",
        help="Include unlabelled samples when creating splits.",
    )
    parser.add_argument("--output-file", default="splits.json", help="Output split file name.")
    args = parser.parse_args()

    try:
        result = create_bugs4q_splits(
            args.data_dir,
            dataset=args.dataset,
            seed=args.seed,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            eval_ratio=args.eval_ratio,
            labelled_only=not args.include_unlabelled,
            output_file=args.output_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to create splits: %s", exc)
        sys.exit(1)

    logger.info(
        "Created splits at %s | candidates=%d excluded_unlabelled=%d counts=%s",
        result.splits_path,
        result.total_candidates,
        result.excluded_unlabelled,
        result.split_counts,
    )
    print(f"Done. Wrote splits to {result.splits_path}.")


if __name__ == "__main__":
    main()
