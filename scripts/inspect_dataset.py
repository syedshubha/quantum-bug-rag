#!/usr/bin/env python3
"""
inspect_dataset.py – Inspect a prepared Bugs4Q dataset.

We use this script to verify which prepared dataset is active, whether it is
synthetic or real, and what label coverage is available before running an
evaluation.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import load_bugs4q_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a prepared Bugs4Q dataset.")
    parser.add_argument("--data-dir", default="data/bugs4q/", help="Bugs4Q data directory.")
    parser.add_argument(
        "--dataset",
        choices=["active", "real", "synthetic"],
        default="active",
        help="Prepared dataset selection (default: active).",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=0,
        help="Number of random sample previews to print.",
    )
    parser.add_argument(
        "--id-count",
        type=int,
        default=5,
        help="Number of leading sample IDs to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for previews.",
    )
    args = parser.parse_args()

    dataset = load_bugs4q_dataset(args.data_dir, dataset=args.dataset)
    label_distribution = Counter(sample.ground_truth or "<missing>" for sample in dataset.samples)
    first_ids = [sample.sample_id for sample in dataset.samples[: args.id_count]]
    synthetic_present = any(bool(sample.metadata.get("synthetic")) for sample in dataset.samples)
    unlabelled_count = dataset.record_count - dataset.labelled_count

    if dataset.synthetic:
        dataset_profile = "synthetic data"
    elif unlabelled_count == 0:
        dataset_profile = "real labelled data"
    elif dataset.labelled_count > 0:
        dataset_profile = "real partially unlabeled data"
    else:
        dataset_profile = "real unlabeled data"

    print("Bugs4Q Dataset Inspection")
    print("=" * 32)
    print(f"Selection: {args.dataset}")
    print(f"Dataset path: {dataset.dataset_path}")
    print(f"Dataset type: {dataset.dataset_type}")
    print(f"Dataset profile: {dataset_profile}")
    print(f"Sample source: {dataset.sample_source}")
    print(f"Synthetic samples: {'yes' if synthetic_present else 'no'}")
    print(f"Total samples: {dataset.record_count}")
    print(f"Labelled samples: {dataset.labelled_count}")
    print(f"Unlabelled samples: {unlabelled_count}")
    print("Label distribution:")
    for label, count in sorted(label_distribution.items()):
        print(f"  {label}: {count}")
    print("First sample IDs:")
    for sample_id in first_ids:
        print(f"  {sample_id}")

    if args.preview_count > 0:
        rng = random.Random(args.seed)
        preview_samples = rng.sample(dataset.samples, k=min(args.preview_count, len(dataset.samples)))
        print("Random previews:")
        for sample in preview_samples:
            preview = sample.code.splitlines()[0][:100] if sample.code else ""
            label = sample.ground_truth or "<missing>"
            print(f"  {sample.sample_id} | label={label} | preview={preview}")


if __name__ == "__main__":
    main()
