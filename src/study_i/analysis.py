"""Aggregation and reporting helpers for Study I."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from .schemas import EpochMetric, FoldRunResult, ID_TO_LABEL, LABELS, StudyISample


def _std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def fold_result_rows(results: list[FoldRunResult]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        rows.append({
            "repeat": result.repeat,
            "fold": result.fold,
            "cv_seed": result.cv_seed,
            "train_seed": result.train_seed,
            "n_train": result.n_train,
            "n_validation": result.n_validation,
            "n_test": result.n_test,
            "accuracy": result.accuracy,
            "f1_macro": result.f1_macro,
            "f1_weighted": result.f1_weighted,
            "roc_auc": result.roc_auc,
        })
    return rows


def aggregate_results(
    samples: list[StudyISample],
    results: list[FoldRunResult],
    config: dict,
) -> dict:
    if not results:
        return {"n_fold_runs": 0}

    accuracy = np.array([result.accuracy for result in results], dtype=float)
    f1_macro = np.array([result.f1_macro for result in results], dtype=float)
    f1_weighted = np.array([result.f1_weighted for result in results], dtype=float)
    roc_auc = np.array([result.roc_auc for result in results], dtype=float)

    y_true = np.concatenate([np.asarray(result.y_true, dtype=int) for result in results])
    y_pred = np.concatenate([np.asarray(result.y_pred, dtype=int) for result in results])
    probs = np.concatenate([np.asarray(result.probs, dtype=float) for result in results], axis=0)

    sample_count = len(samples)
    classical_count = sum(1 for sample in samples if sample.label == "classical")
    quantum_count = sum(1 for sample in samples if sample.label == "quantum")
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "task": "study_i_binary_quantum_vs_classical",
        "study": "Study I",
        "model": config["model_name"],
        "n_samples": sample_count,
        "class_distribution": {
            "classical": classical_count,
            "quantum": quantum_count,
        },
        "cv_setup": {
            "n_folds": config["n_folds"],
            "cv_seeds": config["cv_seeds"],
            "n_fold_runs": len(results),
        },
        "hyperparameters": config,
        "mean_accuracy": float(accuracy.mean()),
        "std_accuracy": _std(accuracy),
        "mean_f1_macro": float(f1_macro.mean()),
        "std_f1_macro": _std(f1_macro),
        "mean_f1_weighted": float(f1_weighted.mean()),
        "std_f1_weighted": _std(f1_weighted),
        "mean_roc_auc": float(roc_auc.mean()),
        "std_roc_auc": _std(roc_auc),
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pooled_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "pooled_roc_auc": float(roc_auc_score(y_true, probs[:, 1])),
        "pooled_classification_report": report,
        "pooled_confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "fold_runs": [result.to_dict() for result in results],
    }


def write_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_per_fold_csv(path: Path, results: list[FoldRunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = fold_result_rows(results)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_epoch_logs_json(path: Path, results: list[FoldRunResult]) -> None:
    payload = {
        f"repeat_{result.repeat}_fold_{result.fold}": [
            asdict(metric) for metric in result.epoch_log
        ]
        for result in results
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
