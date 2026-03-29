"""
evaluate.py – Detection and classification metrics.

I compute:
  * Binary detection metrics (precision, recall, F1) – is there a bug?
  * Multi-class classification Macro-F1 – which bug class is it?

I use scikit-learn under the hood and keep the interface simple so that
pipeline scripts only need to call ``evaluate_results``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .schemas import DiagnosticResult, EvaluationSummary

logger = logging.getLogger(__name__)

# I consider a program to be predicted as buggy when bug_likelihood ≥ this.
DEFAULT_THRESHOLD = 0.5


def evaluate_results(
    results: List[DiagnosticResult],
    ground_truth: List[Dict[str, Any]],
    mode: str = "unknown",
    threshold: float = DEFAULT_THRESHOLD,
) -> EvaluationSummary:
    """
    I compute detection and classification metrics for a set of results.

    Parameters
    ----------
    results:
        ``DiagnosticResult`` objects from a benchmark run.
    ground_truth:
        Program records from ``dataset_loader.load_bugs4q``; each must have
        ``id``, ``has_bug``, and optionally ``bug_class`` fields.
    mode:
        Pipeline mode label to embed in the summary.
    threshold:
        Bug-likelihood threshold above which I predict "buggy".
    """
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score  # type: ignore
    except ImportError as exc:
        raise ImportError("Install scikit-learn: `pip install scikit-learn`") from exc

    # I build look-up maps keyed by program_id.
    gt_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in ground_truth}

    y_true_det: List[int] = []
    y_pred_det: List[int] = []
    y_true_cls: List[str] = []
    y_pred_cls: List[str] = []

    for res in results:
        gt = gt_map.get(res.program_id)
        if gt is None:
            logger.warning("No ground truth for %s; skipping.", res.program_id)
            continue

        y_true_det.append(1 if gt["has_bug"] else 0)
        y_pred_det.append(1 if res.bug_likelihood >= threshold else 0)

        # I only include in classification eval if ground truth has a class.
        if gt.get("bug_class"):
            y_true_cls.append(str(gt["bug_class"]))
            y_pred_cls.append(str(res.taxonomy_class))

    if not y_true_det:
        logger.warning("No overlapping IDs between results and ground truth.")
        return EvaluationSummary(
            mode=mode,
            n_samples=0,
            detection_precision=0.0,
            detection_recall=0.0,
            detection_f1=0.0,
            classification_macro_f1=0.0,
        )

    det_precision = float(
        precision_score(y_true_det, y_pred_det, zero_division=0)
    )
    det_recall = float(
        recall_score(y_true_det, y_pred_det, zero_division=0)
    )
    det_f1 = float(
        f1_score(y_true_det, y_pred_det, zero_division=0)
    )

    cls_macro_f1 = 0.0
    per_class: Dict[str, float] = {}
    if y_true_cls:
        labels = sorted(set(y_true_cls + y_pred_cls))
        raw_f1s = f1_score(
            y_true_cls, y_pred_cls, labels=labels, average=None, zero_division=0
        )
        per_class = {label: float(f) for label, f in zip(labels, raw_f1s)}
        cls_macro_f1 = float(
            f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
        )

    summary = EvaluationSummary(
        mode=mode,
        n_samples=len(y_true_det),
        detection_precision=det_precision,
        detection_recall=det_recall,
        detection_f1=det_f1,
        classification_macro_f1=cls_macro_f1,
        per_class_f1=per_class,
    )

    logger.info(
        "Evaluation (%s) – Det-F1=%.3f  Cls-MacroF1=%.3f  n=%d",
        mode, det_f1, cls_macro_f1, len(y_true_det),
    )
    return summary


def print_summary(summary: EvaluationSummary) -> None:
    """I pretty-print an evaluation summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Summary  –  mode: {summary.mode}")
    print(f"{'='*60}")
    print(f"  Samples evaluated    : {summary.n_samples}")
    print(f"  Detection Precision  : {summary.detection_precision:.4f}")
    print(f"  Detection Recall     : {summary.detection_recall:.4f}")
    print(f"  Detection F1         : {summary.detection_f1:.4f}")
    print(f"  Classification Macro-F1 : {summary.classification_macro_f1:.4f}")
    if summary.per_class_f1:
        print("\n  Per-class F1:")
        for cls, score in sorted(summary.per_class_f1.items()):
            print(f"    {cls:<30s} {score:.4f}")
    print(f"{'='*60}\n")
