"""
dataset_loader.py – Loaders and normalisers for the Bugs4Q benchmark dataset.

We treat Bugs4Q as the primary evaluation corpus. Raw dataset files are not
bundled in this repository; use scripts/prepare_bugs4q.py to fetch and place
normalised files under data/bugs4q/.

For infrastructure smoke-testing only, we provide a synthetic sample generator
that must never be used when reporting benchmark results.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from .schemas import BugSample
from .utils import get_logger

logger = get_logger(__name__)

ACTIVE_DATASET_FILENAME = "active_dataset.json"
REAL_DATASET_FILENAME = "samples.real.jsonl"
SYNTHETIC_DATASET_FILENAME = "samples.synthetic.jsonl"
LEGACY_DATASET_FILENAME = "samples.jsonl"
LEGACY_REAL_GLOB = "bugs4q_*.json"

DatasetSelection = Literal["active", "real", "synthetic"]
DatasetStorage = Literal["jsonl", "json-directory"]

_REQUIRED_SAMPLE_FIELDS = frozenset({"sample_id", "source", "code"})


@dataclass(frozen=True)
class DatasetReference:
    path: Path
    dataset_type: Literal["real", "synthetic"]
    storage: DatasetStorage


@dataclass(frozen=True)
class DatasetLoadResult:
    samples: list[BugSample]
    data_dir: Path
    dataset_path: Path
    dataset_type: Literal["real", "synthetic"]
    sample_source: str
    synthetic: bool
    record_count: int
    labelled_count: int


def dataset_file_for_type(
    data_dir: str | Path,
    dataset_type: Literal["real", "synthetic"],
) -> Path:
    data_dir = Path(data_dir)
    filename = REAL_DATASET_FILENAME if dataset_type == "real" else SYNTHETIC_DATASET_FILENAME
    return data_dir / filename


def write_active_dataset_manifest(
    data_dir: str | Path,
    *,
    active_file: str | Path,
    dataset_type: Literal["real", "synthetic"],
    synthetic: bool,
    sample_source: str,
    record_count: int,
) -> Path:
    data_dir = Path(data_dir)
    active_path = Path(active_file)
    manifest_path = data_dir / ACTIVE_DATASET_FILENAME
    manifest = {
        "active_file": str(active_path),
        "dataset_type": dataset_type,
        "synthetic": synthetic,
        "sample_source": sample_source,
        "record_count": record_count,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def read_active_dataset_manifest(data_dir: str | Path) -> dict | None:
    manifest_path = Path(data_dir) / ACTIVE_DATASET_FILENAME
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    _validate_manifest(manifest, manifest_path)
    return manifest


def describe_dataset(result: DatasetLoadResult) -> str:
    return (
        f"dataset_path={result.dataset_path} "
        f"dataset_type={result.dataset_type} "
        f"synthetic={result.synthetic} "
        f"source={result.sample_source} "
        f"records={result.record_count} "
        f"labelled={result.labelled_count}"
    )


def load_bugs4q_dataset(
    data_dir: str | Path,
    dataset: DatasetSelection = "active",
) -> DatasetLoadResult:
    """
    Load a prepared Bugs4Q dataset from *data_dir*.

    The loader supports explicit selection of the active, real, or synthetic
    prepared dataset. It validates the file layout, sample schema, and basic
    dataset consistency before returning samples.
    """
    data_dir = Path(data_dir)
    reference = _resolve_dataset_reference(data_dir, dataset)
    raw_records = _read_dataset_records(reference)

    samples: list[BugSample] = []
    seen_ids: set[str] = set()
    sources: set[str] = set()
    synthetic_flags: set[bool] = set()
    labelled_count = 0

    for index, record in enumerate(raw_records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"Dataset record {index} in '{reference.path}' is not a JSON object."
            )
        missing_fields = sorted(_REQUIRED_SAMPLE_FIELDS - set(record))
        if missing_fields:
            raise ValueError(
                f"Dataset record {index} in '{reference.path}' is missing required fields: "
                f"{', '.join(missing_fields)}"
            )
        sample = BugSample(**record)
        if sample.sample_id in seen_ids:
            raise ValueError(
                f"Dataset '{reference.path}' contains duplicate sample_id '{sample.sample_id}'."
            )
        seen_ids.add(sample.sample_id)
        sources.add(sample.source)
        synthetic_flags.add(_is_synthetic_sample(sample))
        if sample.ground_truth is not None:
            labelled_count += 1
        samples.append(sample)

    if not samples:
        raise ValueError(f"Dataset '{reference.path}' contains no records.")
    if len(sources) != 1:
        raise ValueError(
            f"Dataset '{reference.path}' mixes multiple sample sources: {sorted(sources)}"
        )
    if len(synthetic_flags) != 1:
        raise ValueError(
            f"Dataset '{reference.path}' mixes synthetic and real samples."
        )

    synthetic = next(iter(synthetic_flags))
    inferred_type: Literal["real", "synthetic"] = "synthetic" if synthetic else "real"
    if inferred_type != reference.dataset_type:
        raise ValueError(
            f"Dataset '{reference.path}' is marked as '{reference.dataset_type}' but contains "
            f"{inferred_type} records."
        )

    result = DatasetLoadResult(
        samples=samples,
        data_dir=data_dir,
        dataset_path=reference.path,
        dataset_type=reference.dataset_type,
        sample_source=next(iter(sources)),
        synthetic=synthetic,
        record_count=len(samples),
        labelled_count=labelled_count,
    )
    logger.info("Loaded Bugs4Q dataset: %s", describe_dataset(result))
    return result


def load_bugs4q(
    data_dir: str | Path,
    dataset: DatasetSelection = "active",
) -> list[BugSample]:
    """Load Bugs4Q samples only, preserving the legacy list-based API."""
    return load_bugs4q_dataset(data_dir, dataset=dataset).samples


def iter_bugs4q(
    data_dir: str | Path,
    dataset: DatasetSelection = "active",
) -> Iterator[BugSample]:
    """Yield BugSample objects one at a time."""
    yield from load_bugs4q(data_dir, dataset=dataset)


def _resolve_dataset_reference(data_dir: Path, dataset: DatasetSelection) -> DatasetReference:
    named_real = dataset_file_for_type(data_dir, "real")
    named_synthetic = dataset_file_for_type(data_dir, "synthetic")
    legacy_jsonl = data_dir / LEGACY_DATASET_FILENAME
    legacy_json_dir_has_records = any(data_dir.glob(LEGACY_REAL_GLOB))

    if dataset == "real":
        if named_real.exists():
            return DatasetReference(named_real, "real", "jsonl")
        if legacy_json_dir_has_records:
            return DatasetReference(data_dir, "real", "json-directory")
        raise FileNotFoundError(
            f"Real Bugs4Q dataset not found in '{data_dir}'. Run prepare_bugs4q.py first."
        )

    if dataset == "synthetic":
        if named_synthetic.exists():
            return DatasetReference(named_synthetic, "synthetic", "jsonl")
        if legacy_jsonl.exists():
            return DatasetReference(legacy_jsonl, "synthetic", "jsonl")
        raise FileNotFoundError(
            f"Synthetic smoke-test dataset not found in '{data_dir}'. "
            "Run prepare_bugs4q.py --smoke-test first."
        )

    manifest = read_active_dataset_manifest(data_dir)
    if manifest is not None:
        active_path = data_dir / manifest["active_file"]
        if not active_path.exists():
            raise FileNotFoundError(
                f"Active dataset file '{active_path}' declared in '{data_dir / ACTIVE_DATASET_FILENAME}' "
                "does not exist."
            )
        return DatasetReference(active_path, manifest["dataset_type"], "jsonl")

    inferred_candidates: list[DatasetReference] = []
    if named_real.exists():
        inferred_candidates.append(DatasetReference(named_real, "real", "jsonl"))
    if named_synthetic.exists():
        inferred_candidates.append(DatasetReference(named_synthetic, "synthetic", "jsonl"))
    if legacy_jsonl.exists():
        inferred_candidates.append(DatasetReference(legacy_jsonl, "synthetic", "jsonl"))
    if legacy_json_dir_has_records:
        inferred_candidates.append(DatasetReference(data_dir, "real", "json-directory"))

    if len(inferred_candidates) == 1:
        logger.warning(
            "No active dataset manifest found in '%s'. Falling back to the only available dataset: %s",
            data_dir,
            inferred_candidates[0].path,
        )
        return inferred_candidates[0]

    raise ValueError(
        f"Dataset selection is ambiguous in '{data_dir}'. "
        f"Run prepare_bugs4q.py to create '{ACTIVE_DATASET_FILENAME}', or select 'real' or 'synthetic' explicitly."
    )


def _read_dataset_records(reference: DatasetReference) -> list[dict]:
    if reference.storage == "jsonl":
        return _read_jsonl_records(reference.path)
    return _read_json_directory_records(reference.path)


def _read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in '{path}' at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"Dataset file '{path}' contains no records.")
    return records


def _read_json_directory_records(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    json_files = sorted(data_dir.glob(LEGACY_REAL_GLOB))
    if not json_files:
        raise ValueError(f"No dataset records found in '{data_dir}'.")
    for json_file in json_files:
        with json_file.open(encoding="utf-8") as fh:
            try:
                records.append(json.load(fh))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in '{json_file}': {exc}") from exc
    return records


def _validate_manifest(manifest: dict, manifest_path: Path) -> None:
    required_fields = {"active_file", "dataset_type", "synthetic", "sample_source", "record_count"}
    missing_fields = sorted(required_fields - set(manifest))
    if missing_fields:
        raise ValueError(
            f"Dataset manifest '{manifest_path}' is missing required fields: {', '.join(missing_fields)}"
        )
    if manifest["dataset_type"] not in {"real", "synthetic"}:
        raise ValueError(
            f"Dataset manifest '{manifest_path}' has invalid dataset_type '{manifest['dataset_type']}'."
        )
    if not isinstance(manifest["record_count"], int) or manifest["record_count"] < 0:
        raise ValueError(
            f"Dataset manifest '{manifest_path}' has invalid record_count '{manifest['record_count']}'."
        )


def _is_synthetic_sample(sample: BugSample) -> bool:
    if bool(sample.metadata.get("synthetic")):
        return True
    if sample.source == "synthetic_smoke_test":
        return True
    return sample.sample_id.startswith("smoke_")


_SMOKE_TAXONOMY_CLASSES = [
    "incorrect_operator",
    "incorrect_qubit_mapping",
    "missing_barrier",
    "wrong_initial_state",
    "measurement_error",
]

_SMOKE_CODE_TEMPLATES = [
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\n",
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(1)\nqc.x(0)\nqc.measure_all()\n",
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(3)\nqc.ccx(0, 1, 2)\n",
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.swap(0, 1)\n",
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.ry(1.57, 0)\nqc.cx(0, 1)\n",
]


def generate_smoke_samples(n: int = 10, seed: int = 42) -> list[BugSample]:
    """
    Generate *n* synthetic BugSample objects for smoke-testing only.

    These samples are not real bugs and must never be used for reporting
    benchmark results. They exist solely to validate pipeline infrastructure.
    """
    rng = random.Random(seed)
    samples: list[BugSample] = []
    for index in range(n):
        label = rng.choice(_SMOKE_TAXONOMY_CLASSES)
        code = rng.choice(_SMOKE_CODE_TEMPLATES)
        samples.append(
            BugSample(
                sample_id=f"smoke_{index:04d}",
                source="synthetic_smoke_test",
                code=code,
                ground_truth=label,
                metadata={"synthetic": True},
            )
        )
    return samples
