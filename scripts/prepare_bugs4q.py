#!/usr/bin/env python3
"""
prepare_bugs4q.py – Fetch and normalise the Bugs4Q benchmark dataset.

Bugs4Q is the primary evaluation dataset for this project. Raw dataset files
are not bundled in this repository; this script clones the upstream repository
and converts its contents into the BugSample JSON schema used by our pipeline.

Usage
-----
    # Full preparation (requires internet access):
    python scripts/prepare_bugs4q.py --output-dir data/bugs4q/

    # Smoke-test mode (generates synthetic samples; for pipeline testing only):
    python scripts/prepare_bugs4q.py --smoke-test --output-dir data/bugs4q/

Smoke-test data must never be used for reporting benchmark results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_prep import prepare_bugs4q_dataset
from src.utils import get_logger

logger = get_logger("prepare_bugs4q")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Bugs4Q dataset.")
    parser.add_argument("--output-dir", default="data/bugs4q/", help="Output directory.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Write synthetic smoke-test samples instead of real data.",
    )
    parser.add_argument(
        "--smoke-n", type=int, default=20, help="Number of synthetic samples (smoke-test only)."
    )
    parser.add_argument(
        "--bugs4q-dir",
        default=None,
        help="Path to an existing Bugs4Q clone (skips git clone/pull).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    bugs4q_dir = Path(args.bugs4q_dir) if args.bugs4q_dir else None
    dataset_path, count, dataset_type = prepare_bugs4q_dataset(
        output_dir,
        smoke_test=args.smoke_test,
        smoke_n=args.smoke_n,
        bugs4q_dir=bugs4q_dir,
    )

    logger.info(
        "Active Bugs4Q dataset updated: output=%s records=%d dataset_type=%s",
        dataset_path,
        count,
        dataset_type,
    )
    print(f"Done. {count} samples written to {dataset_path}.")


if __name__ == "__main__":
    main()
