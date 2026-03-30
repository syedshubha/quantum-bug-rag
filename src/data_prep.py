"""
data_prep.py – Reusable Bugs4Q dataset preparation helpers.

This module contains reusable preparation logic used by CLI entry points and
tests. Scripts under scripts/ should remain thin wrappers.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

from .bugs4q_labels import apply_labels_to_samples, extract_cases_from_readme, write_type_mapping_manifest
from .dataset_loader import (
    LEGACY_DATASET_FILENAME,
    REAL_DATASET_FILENAME,
    SYNTHETIC_DATASET_FILENAME,
    generate_smoke_samples,
    write_active_dataset_manifest,
)
from .schemas import BugSample
from .utils import get_logger

logger = get_logger(__name__)

BUGS4Q_REPO_URL = "https://github.com/Z-928/Bugs4Q.git"
LEGACY_REAL_GLOB = "bugs4q_*.json"
_SKIP_FILENAMES = {"fixed.py", "fix.py", "fixed_version.py", "modify.py", "mod.py"}


def clone_or_update_bugs4q(target_dir: Path, repo_url: str = BUGS4Q_REPO_URL) -> None:
    if (target_dir / ".git").exists():
        logger.info("Updating existing Bugs4Q clone at %s", target_dir)
        subprocess.run(["git", "-C", str(target_dir), "pull", "--ff-only"], check=True)
    else:
        logger.info("Cloning Bugs4Q into %s", target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)


def prepare_real_dataset(source_dir: Path, output_dir: Path) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / REAL_DATASET_FILENAME
    samples = _normalise_bugs4q(source_dir)
    readme_path = source_dir / "README.md"
    cases = extract_cases_from_readme(readme_path) if readme_path.exists() else []
    samples, label_stats, label_rows = apply_labels_to_samples(samples, cases)

    if not samples:
        raise ValueError(f"No real Bugs4Q samples were found under '{source_dir}'.")

    _write_dataset_jsonl(dataset_path, samples)
    _write_label_artifacts(output_dir, cases, label_rows, label_stats)
    _remove_legacy_dataset_artifacts(output_dir)
    manifest_path = write_active_dataset_manifest(
        output_dir,
        active_file=dataset_path.name,
        dataset_type="real",
        synthetic=False,
        sample_source="bugs4q",
        record_count=len(samples),
    )
    logger.info(
        "Prepared Bugs4Q dataset: output=%s records=%d dataset_type=real synthetic=False active_manifest=%s",
        dataset_path,
        len(samples),
        manifest_path,
    )
    logger.info(
        "Real label coverage: total=%d labelled=%d unlabelled=%d",
        label_stats.total_samples,
        label_stats.labelled_samples,
        label_stats.unlabelled_samples,
    )
    logger.info("Real label distribution: %s", label_stats.label_distribution)
    if label_stats.unmapped_types:
        logger.warning("Unmapped upstream types: %s", label_stats.unmapped_types)

    return dataset_path, len(samples)


def prepare_smoke_dataset(output_dir: Path, n: int, seed: int = 42) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / SYNTHETIC_DATASET_FILENAME
    samples = generate_smoke_samples(n=n, seed=seed)
    _write_dataset_jsonl(dataset_path, samples)
    _remove_legacy_dataset_artifacts(output_dir)
    manifest_path = write_active_dataset_manifest(
        output_dir,
        active_file=dataset_path.name,
        dataset_type="synthetic",
        synthetic=True,
        sample_source="synthetic_smoke_test",
        record_count=len(samples),
    )
    logger.warning(
        "Prepared Bugs4Q smoke-test dataset: output=%s records=%d dataset_type=synthetic synthetic=True active_manifest=%s",
        dataset_path,
        len(samples),
        manifest_path,
    )
    return dataset_path, len(samples)


def prepare_bugs4q_dataset(
    output_dir: Path,
    *,
    smoke_test: bool,
    smoke_n: int,
    bugs4q_dir: Path | None,
) -> tuple[Path, int, Literal["real", "synthetic"]]:
    if smoke_test:
        dataset_path, count = prepare_smoke_dataset(output_dir, smoke_n)
        return dataset_path, count, "synthetic"

    source_dir = bugs4q_dir if bugs4q_dir is not None else Path("_tmp_bugs4q_clone")
    if bugs4q_dir is None:
        clone_or_update_bugs4q(source_dir)
    dataset_path, count = prepare_real_dataset(source_dir, output_dir)
    return dataset_path, count, "real"


def _normalise_bugs4q(source_dir: Path) -> list[BugSample]:
    samples: list[BugSample] = []

    for index, py_file in enumerate(sorted(source_dir.rglob("*.py"))):
        if not _should_include_real_sample(py_file, source_dir):
            continue
        code = py_file.read_text(encoding="utf-8", errors="replace")
        if not code.strip():
            continue

        relative_path = py_file.relative_to(source_dir)
        samples.append(
            BugSample(
                sample_id=f"bugs4q_{index:04d}",
                source="bugs4q",
                code=code,
                ground_truth=None,
                metadata={
                    "synthetic": False,
                    "path": str(relative_path),
                    "collection": relative_path.parts[0],
                    "variant": _infer_variant(relative_path),
                },
            )
        )

    return samples


def _should_include_real_sample(py_file: Path, source_dir: Path) -> bool:
    name_lower = py_file.name.lower()
    if py_file.name.startswith("_") or py_file.stem == "conftest":
        return False
    if name_lower in _SKIP_FILENAMES:
        return False

    relative_parts_lower = [part.lower() for part in py_file.relative_to(source_dir).parts]
    if name_lower.startswith("bug"):
        return True
    if "buggy" in relative_parts_lower:
        return True
    if py_file.parent.name == "Program":
        return True
    return False


def _infer_variant(relative_path: Path) -> str:
    name_lower = relative_path.name.lower()
    if name_lower == "buggy.py":
        return "buggy_file"
    if name_lower == "bug_version.py":
        return "bug_version_file"
    if "buggy" in {part.lower() for part in relative_path.parts[:-1]}:
        return "buggy_directory_file"
    if relative_path.parts[0] == "Program":
        return "standalone_program"
    return "real_benchmark_candidate"


def _write_dataset_jsonl(dataset_path: Path, samples: list[BugSample]) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")


def _remove_legacy_dataset_artifacts(output_dir: Path) -> None:
    legacy_jsonl = output_dir / LEGACY_DATASET_FILENAME
    if legacy_jsonl.exists():
        legacy_jsonl.unlink()
        logger.info("Removed legacy dataset file %s", legacy_jsonl)

    removed_json_files = 0
    for old_json in output_dir.glob(LEGACY_REAL_GLOB):
        old_json.unlink()
        removed_json_files += 1
    if removed_json_files:
        logger.info("Removed %d legacy per-sample JSON files from %s", removed_json_files, output_dir)


def _write_label_artifacts(output_dir: Path, cases: list, label_rows: list[dict], label_stats) -> None:
    write_type_mapping_manifest(output_dir)

    (output_dir / "labels.case_manifest.json").write_text(
        json.dumps(
            [
                {"buggy_path": case.buggy_path, "upstream_type": case.upstream_type}
                for case in cases
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    with (output_dir / "labels.sample_map.jsonl").open("w", encoding="utf-8") as fh:
        for row in label_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "labels.summary.json").write_text(
        json.dumps(
            {
                "total_samples": label_stats.total_samples,
                "labelled_samples": label_stats.labelled_samples,
                "unlabelled_samples": label_stats.unlabelled_samples,
                "label_distribution": label_stats.label_distribution,
                "unmapped_types": label_stats.unmapped_types,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
