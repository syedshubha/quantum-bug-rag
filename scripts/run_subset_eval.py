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
from src.benchmark_splits import load_split_ids
from src.dataset_loader import describe_dataset, load_bugs4q_dataset
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
        help="Load the prepared synthetic smoke-test dataset instead of the active dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=["active", "real", "synthetic"],
        default="active",
        help="Prepared dataset selection (default: active).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
        help="Modes to run (default: all three).",
    )
    parser.add_argument(
        "--labelled-only",
        action="store_true",
        help="Deprecated alias; benchmark mode defaults to labelled-only.",
    )
    parser.add_argument(
        "--include-unlabelled",
        action="store_true",
        help="Include unlabelled samples in subset selection (non-benchmark exploratory mode).",
    )
    parser.add_argument(
        "--split",
        choices=["none", "train", "dev", "eval"],
        default="none",
        help="Optional split to sample from (requires splits file).",
    )
    parser.add_argument(
        "--splits-file",
        default="splits.json",
        help="Split file path relative to --data-dir unless absolute.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("paths", {})["knowledge_base"] = args.kb_dir

    dataset_selection = "synthetic" if args.smoke_test else args.dataset
    try:
        dataset = load_bugs4q_dataset(args.data_dir, dataset=dataset_selection)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    logger.info("Subset evaluation dataset: %s", describe_dataset(dataset))
    run_data_profile = "synthetic data"
    if dataset.synthetic:
        logger.warning(
            "Using synthetic smoke-test data. Results are for pipeline validation only."
        )
    else:
        if dataset.labelled_count == dataset.record_count:
            run_data_profile = "real labelled data"
        elif dataset.labelled_count > 0:
            run_data_profile = "real partially unlabeled data"
        else:
            run_data_profile = "real unlabeled data"
        logger.info("Using %s from %s", run_data_profile, dataset.dataset_path)

    samples = dataset.samples
    if args.split != "none":
        splits_path = Path(args.splits_file)
        if not splits_path.is_absolute():
            splits_path = Path(args.data_dir) / splits_path
        try:
            split_ids = load_split_ids(splits_path, args.split)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to load split '%s': %s", args.split, exc)
            sys.exit(1)
        samples = [sample for sample in samples if sample.sample_id in split_ids]
        logger.info("Applied split=%s from %s -> %d samples", args.split, splits_path, len(samples))

    labelled_only = not args.include_unlabelled
    if args.labelled_only and args.include_unlabelled:
        logger.warning("Both --labelled-only and --include-unlabelled were passed; including unlabelled samples.")

    if labelled_only:
        before = len(samples)
        samples = [sample for sample in samples if sample.ground_truth is not None]
        logger.info(
            "Default benchmark filter to labelled samples: %d -> %d",
            before,
            len(samples),
        )

    samples = samples[: args.subset_size]
    if not samples:
        logger.error("No samples available after filtering for subset evaluation.")
        sys.exit(1)
    if len(samples) < args.subset_size:
        logger.warning(
            "Requested subset size %d but dataset only has %d samples.",
            args.subset_size,
            len(samples),
        )

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
