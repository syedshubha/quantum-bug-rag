"""
benchmark_splits.py – Deterministic Bugs4Q split generation helpers.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .dataset_loader import load_bugs4q_dataset


@dataclass(frozen=True)
class SplitResult:
    splits_path: Path
    split_counts: dict[str, int]
    total_candidates: int
    excluded_unlabelled: int


def create_bugs4q_splits(
    data_dir: str | Path,
    *,
    dataset: str = "active",
    seed: int = 42,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    eval_ratio: float = 0.2,
    labelled_only: bool = True,
    output_file: str = "splits.json",
) -> SplitResult:
    if abs((train_ratio + dev_ratio + eval_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    dataset_result = load_bugs4q_dataset(data_dir, dataset=dataset)  # type: ignore[arg-type]
    samples = dataset_result.samples
    candidate_samples = samples
    excluded_unlabelled = 0

    if labelled_only:
        candidate_samples = [sample for sample in samples if sample.ground_truth is not None]
        excluded_unlabelled = len(samples) - len(candidate_samples)

    if not candidate_samples:
        raise ValueError("No candidate samples available for split generation.")

    sample_ids = sorted(sample.sample_id for sample in candidate_samples)
    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    total = len(sample_ids)
    train_cut = int(total * train_ratio)
    dev_cut = train_cut + int(total * dev_ratio)

    split_ids = {
        "train": sample_ids[:train_cut],
        "dev": sample_ids[train_cut:dev_cut],
        "eval": sample_ids[dev_cut:],
    }

    split_counts = {name: len(ids) for name, ids in split_ids.items()}
    manifest = {
        "dataset_path": str(dataset_result.dataset_path),
        "dataset_type": dataset_result.dataset_type,
        "synthetic": dataset_result.synthetic,
        "seed": seed,
        "ratios": {"train": train_ratio, "dev": dev_ratio, "eval": eval_ratio},
        "labelled_only": labelled_only,
        "excluded_unlabelled": excluded_unlabelled,
        "split_counts": split_counts,
        "leakage_control": {
            "policy": "Do not use eval split samples as retrieval-source candidates.",
            "eval_ids_must_be_excluded_from_retrieval": True,
        },
        "splits": split_ids,
    }

    output_path = Path(data_dir) / output_file
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return SplitResult(
        splits_path=output_path,
        split_counts=split_counts,
        total_candidates=len(candidate_samples),
        excluded_unlabelled=excluded_unlabelled,
    )


def load_split_ids(splits_path: str | Path, split_name: str) -> set[str]:
    path = Path(splits_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if split_name not in payload.get("splits", {}):
        raise ValueError(f"Split '{split_name}' is not present in '{path}'.")
    return set(payload["splits"][split_name])
