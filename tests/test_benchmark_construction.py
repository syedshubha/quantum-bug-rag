from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from src.benchmark_splits import create_bugs4q_splits
from src.data_prep import prepare_real_dataset
from src.dataset_loader import (
    REAL_DATASET_FILENAME,
    load_bugs4q_dataset,
    write_active_dataset_manifest,
)
from src.schemas import BugSample


def _write_minimal_readme(readme_path: Path) -> None:
    readme_path.write_text(
        "\n".join(
            [
                "| Bug Id | Buggy | Fixed | Modify | Status | Version | Type | Registered | Resolved |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                "| 1 | [Buggy](./Aer/bug_1/buggy.py) | [Fixed](./Aer/bug_1/fixed.py) | [Mod](./Aer/bug_1/modify.py) | Closed | --- | output wrong | now | now |",
                "| 2 | [Buggy](./Terra-0-4000/6/buggy.py) | [Fixed](./Terra-0-4000/6/Fixed.py) | [Mod](./Terra-0-4000/6/modify.py) | Closed | --- | --- | now | now |",
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_real_dataset_assigns_labels_and_reports_unlabelled(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"

    (source_dir / "Aer" / "bug_1").mkdir(parents=True)
    (source_dir / "Aer" / "bug_1" / "buggy.py").write_text("qc.measure_all()\n", encoding="utf-8")
    (source_dir / "Terra-0-4000" / "6").mkdir(parents=True)
    (source_dir / "Terra-0-4000" / "6" / "buggy.py").write_text("qc.cx(0, 1)\n", encoding="utf-8")
    (source_dir / "Program").mkdir(parents=True)
    (source_dir / "Program" / "1999.py").write_text("qc.h(0)\n", encoding="utf-8")
    _write_minimal_readme(source_dir / "README.md")

    prepare_real_dataset(source_dir, output_dir)
    dataset = load_bugs4q_dataset(output_dir, dataset="real")

    labels_by_path = {sample.metadata["path"]: sample.ground_truth for sample in dataset.samples}
    assert labels_by_path["Aer/bug_1/buggy.py"] == "measurement_error"
    assert labels_by_path["Terra-0-4000/6/buggy.py"] is None
    assert labels_by_path["Program/1999.py"] is None

    summary = json.loads((output_dir / "labels.summary.json").read_text(encoding="utf-8"))
    assert summary["total_samples"] == 3
    assert summary["labelled_samples"] == 1
    assert summary["unlabelled_samples"] == 2
    assert summary["label_distribution"] == {"measurement_error": 1}
    assert (output_dir / "labels.type_mapping.json").exists()


def test_prepare_real_dataset_is_deterministic(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"

    (source_dir / "Aer" / "bug_1").mkdir(parents=True)
    (source_dir / "Aer" / "bug_1" / "buggy.py").write_text("qc.measure_all()\n", encoding="utf-8")
    (source_dir / "StackExchange" / "1").mkdir(parents=True)
    (source_dir / "StackExchange" / "1" / "buggy.py").write_text("qc.cx(0, 1)\n", encoding="utf-8")
    _write_minimal_readme(source_dir / "README.md")

    prepare_real_dataset(source_dir, output_a)
    prepare_real_dataset(source_dir, output_b)

    ds_a = load_bugs4q_dataset(output_a, dataset="real")
    ds_b = load_bugs4q_dataset(output_b, dataset="real")

    seq_a = [(sample.sample_id, sample.metadata["path"], sample.ground_truth) for sample in ds_a.samples]
    seq_b = [(sample.sample_id, sample.metadata["path"], sample.ground_truth) for sample in ds_b.samples]
    assert seq_a == seq_b


def test_split_generation_is_deterministic(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        BugSample(
            sample_id=f"bugs4q_{idx:04d}",
            source="bugs4q",
            code="qc.h(0)\n",
            ground_truth="incorrect_operator",
            metadata={"synthetic": False, "path": f"Aer/bug_{idx}/buggy.py"},
        )
        for idx in range(10)
    ]
    with (data_dir / REAL_DATASET_FILENAME).open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.model_dump()) + "\n")
    write_active_dataset_manifest(
        data_dir,
        active_file=REAL_DATASET_FILENAME,
        dataset_type="real",
        synthetic=False,
        sample_source="bugs4q",
        record_count=len(samples),
    )

    first = create_bugs4q_splits(data_dir, seed=7, output_file="splits_a.json")
    second = create_bugs4q_splits(data_dir, seed=7, output_file="splits_b.json")

    payload_a = json.loads(first.splits_path.read_text(encoding="utf-8"))
    payload_b = json.loads(second.splits_path.read_text(encoding="utf-8"))
    assert payload_a["splits"] == payload_b["splits"]
    assert payload_a["split_counts"] == {"train": 6, "dev": 2, "eval": 2}


def test_run_subset_eval_can_skip_unlabelled(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)

    records = [
        BugSample(
            sample_id="bugs4q_0000",
            source="bugs4q",
            code="qc.cx(0, 0)\n",
            ground_truth="incorrect_qubit_mapping",
            metadata={"synthetic": False, "path": "Aer/bug_1/buggy.py"},
        ).model_dump(),
        BugSample(
            sample_id="bugs4q_0001",
            source="bugs4q",
            code="qc.measure_all()\n",
            ground_truth="measurement_error",
            metadata={"synthetic": False, "path": "Aer/bug_2/buggy.py"},
        ).model_dump(),
        BugSample(
            sample_id="bugs4q_0002",
            source="bugs4q",
            code="qc.h(0)\n",
            ground_truth=None,
            metadata={"synthetic": False, "path": "Program/1999.py"},
        ).model_dump(),
    ]

    with (data_dir / REAL_DATASET_FILENAME).open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    write_active_dataset_manifest(
        data_dir,
        active_file=REAL_DATASET_FILENAME,
        dataset_type="real",
        synthetic=False,
        sample_source="bugs4q",
        record_count=len(records),
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_subset_eval.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir",
            str(data_dir),
            "--dataset",
            "active",
            "--subset-size",
            "3",
            "--modes",
            "static",
            "--labelled-only",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    merged_output = result.stdout + "\n" + result.stderr
    assert "Filtered to labelled samples: 3 -> 2" in merged_output
    assert re.search(r"static\s+2\s+", result.stdout)
