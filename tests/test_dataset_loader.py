from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.data_prep import prepare_real_dataset, prepare_smoke_dataset
from src.dataset_loader import (
    ACTIVE_DATASET_FILENAME,
    REAL_DATASET_FILENAME,
    SYNTHETIC_DATASET_FILENAME,
    load_bugs4q_dataset,
)
from src.schemas import BugSample


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def test_loader_handles_explicit_real_and_synthetic_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "bugs4q"
    real_record = BugSample(
        sample_id="bugs4q_0000",
        source="bugs4q",
        code="qc = QuantumCircuit(1)",
        metadata={"synthetic": False},
    ).model_dump()
    synthetic_record = BugSample(
        sample_id="smoke_0000",
        source="synthetic_smoke_test",
        code="qc = QuantumCircuit(1)",
        ground_truth="incorrect_operator",
        metadata={"synthetic": True},
    ).model_dump()
    _write_jsonl(data_dir / REAL_DATASET_FILENAME, [real_record])
    _write_jsonl(data_dir / SYNTHETIC_DATASET_FILENAME, [synthetic_record])

    real_dataset = load_bugs4q_dataset(data_dir, dataset="real")
    synthetic_dataset = load_bugs4q_dataset(data_dir, dataset="synthetic")

    assert real_dataset.dataset_path == data_dir / REAL_DATASET_FILENAME
    assert real_dataset.dataset_type == "real"
    assert real_dataset.synthetic is False
    assert real_dataset.sample_source == "bugs4q"

    assert synthetic_dataset.dataset_path == data_dir / SYNTHETIC_DATASET_FILENAME
    assert synthetic_dataset.dataset_type == "synthetic"
    assert synthetic_dataset.synthetic is True
    assert synthetic_dataset.sample_source == "synthetic_smoke_test"


def test_loader_rejects_mixed_synthetic_and_real_records(tmp_path: Path) -> None:
    data_dir = tmp_path / "bugs4q"
    _write_jsonl(
        data_dir / SYNTHETIC_DATASET_FILENAME,
        [
            {
                "sample_id": "smoke_0000",
                "source": "synthetic_smoke_test",
                "code": "qc = QuantumCircuit(1)",
                "ground_truth": "incorrect_operator",
                "metadata": {"synthetic": True},
            },
            {
                "sample_id": "bugs4q_0001",
                "source": "bugs4q",
                "code": "qc = QuantumCircuit(2)",
                "metadata": {"synthetic": False},
            },
        ],
    )

    with pytest.raises(ValueError, match="mixes multiple sample sources|mixes synthetic and real samples"):
        load_bugs4q_dataset(data_dir, dataset="synthetic")


def test_prepare_real_dataset_replaces_active_smoke_dataset(tmp_path: Path) -> None:
    output_dir = tmp_path / "data" / "bugs4q"
    source_dir = tmp_path / "source"
    (source_dir / "Aer" / "bug_1").mkdir(parents=True)
    (source_dir / "Aer" / "bug_1" / "buggy.py").write_text("qc.cx(0, 0)\n", encoding="utf-8")
    (source_dir / "Aer" / "bug_1" / "fixed.py").write_text("qc.cx(0, 1)\n", encoding="utf-8")
    (source_dir / "Program").mkdir(parents=True)
    (source_dir / "Program" / "1999.py").write_text("qc.measure_all()\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bugs4q_0000.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "samples.jsonl").write_text("{}\n", encoding="utf-8")

    smoke_path, smoke_count = prepare_smoke_dataset(output_dir, n=3)
    assert smoke_path == output_dir / SYNTHETIC_DATASET_FILENAME
    assert smoke_count == 3

    real_path, real_count = prepare_real_dataset(source_dir, output_dir)

    assert real_path == output_dir / REAL_DATASET_FILENAME
    assert real_count == 2
    assert not (output_dir / "samples.jsonl").exists()
    assert not (output_dir / "bugs4q_0000.json").exists()

    active_manifest = json.loads((output_dir / ACTIVE_DATASET_FILENAME).read_text(encoding="utf-8"))
    assert active_manifest["active_file"] == REAL_DATASET_FILENAME
    assert active_manifest["dataset_type"] == "real"
    assert active_manifest["synthetic"] is False

    active_dataset = load_bugs4q_dataset(output_dir, dataset="active")
    assert active_dataset.dataset_type == "real"
    assert active_dataset.synthetic is False
    assert active_dataset.record_count == 2


def test_inspect_dataset_utility_prints_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "data" / "bugs4q"
    prepare_smoke_dataset(output_dir, n=4)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "inspect_dataset.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir",
            str(output_dir),
            "--dataset",
            "active",
            "--id-count",
            "3",
            "--preview-count",
            "1",
            "--seed",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Dataset type: synthetic" in result.stdout
    assert "Synthetic samples: yes" in result.stdout
    assert "Total samples: 4" in result.stdout
    assert "First sample IDs:" in result.stdout
    assert "Random previews:" in result.stdout


def test_prepare_cli_is_thin_wrapper() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_bugs4q.py"
    content = script_path.read_text(encoding="utf-8")
    assert "from src.data_prep import prepare_bugs4q_dataset" in content
    assert "def prepare_real_dataset(" not in content
    assert "def prepare_smoke_dataset(" not in content
