from __future__ import annotations

import json
from pathlib import Path
import random

from src.study_i.analysis import aggregate_results
from src.study_i.dataset import dataset_summary, load_labeled_bug_reports, to_training_arrays
from src.study_i.schemas import EpochMetric, FoldRunResult, StudyISample
from src.study_i.training import oversample_to_balance


def test_load_labeled_bug_reports_filters_and_maps(tmp_path: Path) -> None:
    data = [
        {
            "id": "a1",
            "bug_category": "classical",
            "name": "Build failure",
            "description": "CMake typo",
            "example_code": "set(VAR 1)",
        },
        {
            "id": "a2",
            "bug_category": "quantum",
            "name": "Wrong qubit",
            "description": "Gate applied to wrong register",
            "code": "qc.cx(0, 0)",
        },
        {
            "id": "a3",
            "bug_category": "other",
            "name": "ignored",
        },
    ]
    path = tmp_path / "study_i.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    samples = load_labeled_bug_reports(path)
    assert [sample.sample_id for sample in samples] == ["a1", "a2"]
    assert samples[0].to_text() == "Build failure\nCMake typo\nset(VAR 1)"
    texts, labels = to_training_arrays(samples)
    assert texts.shape == (2,)
    assert labels.tolist() == [0, 1]


def test_dataset_summary_counts_classes() -> None:
    samples = [
        StudyISample("s1", "n", "d", "c", "classical"),
        StudyISample("s2", "n", "d", "c", "quantum"),
        StudyISample("s3", "n", "d", "c", "classical"),
    ]
    summary = dataset_summary(samples)
    assert summary["n_samples"] == 3
    assert summary["class_distribution"] == {"classical": 2, "quantum": 1}
    assert summary["imbalance_ratio"] == 2.0


def test_oversample_to_balance_equalizes_class_sizes() -> None:
    texts, labels = oversample_to_balance(
        ["a", "b", "c"],
        [0, 0, 1],
        random.Random(0),
    )
    assert len(texts) == len(labels) == 4
    assert labels.count(0) == labels.count(1) == 2


def test_aggregate_results_computes_summary() -> None:
    samples = [
        StudyISample("s1", "n1", "d1", "c1", "classical"),
        StudyISample("s2", "n2", "d2", "c2", "quantum"),
    ]
    results = [
        FoldRunResult(
            repeat=1,
            fold=1,
            cv_seed=42,
            train_seed=4200,
            n_train=1,
            n_validation=1,
            n_test=2,
            accuracy=1.0,
            f1_macro=1.0,
            f1_weighted=1.0,
            roc_auc=1.0,
            y_true=[0, 1],
            y_pred=[0, 1],
            probs=[[0.9, 0.1], [0.1, 0.9]],
            epoch_log=[EpochMetric(epoch=1.0, eval_loss=0.5, eval_accuracy=1.0, eval_f1_macro=1.0)],
        )
    ]
    summary = aggregate_results(samples, results, {"model_name": "microsoft/codebert-base", "n_folds": 5, "cv_seeds": [42]})
    assert summary["mean_accuracy"] == 1.0
    assert summary["pooled_accuracy"] == 1.0
    assert summary["class_distribution"] == {"classical": 1, "quantum": 1}
    assert summary["pooled_confusion_matrix"] == [[1, 0], [0, 1]]
