"""
evaluate.py – Metrics computation for diagnostic outputs.

We compute standard classification metrics over a set of BugDiagnostic objects
that carry ground-truth labels.  All computation uses scikit-learn so that
results are reproducible and compatible with standard reporting conventions.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from .schemas import BugDiagnostic, EvalSummary
from .utils import get_logger, new_run_id

logger = get_logger(__name__)


def _safe_metric(fn, *args, **kwargs) -> float:  # type: ignore[no-untyped-def]
    """Call a sklearn metric function; return 0.0 on failure."""
    try:
        return float(fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Metric computation failed: %s", exc)
        return 0.0


def compute_metrics(
    diagnostics: list[BugDiagnostic],
    run_id: Optional[str] = None,
    mode: Optional[str] = None,
    notes: str = "",
) -> EvalSummary:
    """
    Compute aggregate classification metrics for a list of BugDiagnostic objects.

    Only diagnostics with non-None ground_truth are included in the computation.
    """
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    labelled = [d for d in diagnostics if d.ground_truth is not None]
    if not labelled:
        logger.warning("No labelled diagnostics found; returning zero metrics.")
        return EvalSummary(
            run_id=run_id or new_run_id(),
            mode=mode or "unknown",
            num_samples=0,
            accuracy=0.0,
            f1_macro=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            notes="No labelled samples.",
        )

    y_true = [d.ground_truth for d in labelled]
    y_pred = [d.taxonomy_class for d in labelled]

    accuracy = _safe_metric(accuracy_score, y_true, y_pred)
    f1_macro = _safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0)
    precision_macro = _safe_metric(
        precision_score, y_true, y_pred, average="macro", zero_division=0
    )
    recall_macro = _safe_metric(
        recall_score, y_true, y_pred, average="macro", zero_division=0
    )

    # Per-class F1
    classes = sorted(set(y_true))
    per_class_f1_vals = _get_per_class_f1(y_true, y_pred, classes)

    summary = EvalSummary(
        run_id=run_id or new_run_id(),
        mode=mode or (labelled[0].mode if labelled else "unknown"),
        num_samples=len(labelled),
        accuracy=round(accuracy, 4),
        f1_macro=round(f1_macro, 4),
        precision_macro=round(precision_macro, 4),
        recall_macro=round(recall_macro, 4),
        per_class_f1={k: round(v, 4) for k, v in per_class_f1_vals.items()},
        notes=notes,
    )
    return summary


def _get_per_class_f1(
    y_true: list[str], y_pred: list[str], classes: list[str]
) -> dict[str, float]:
    from sklearn.metrics import f1_score  # noqa: PLC0415

    try:
        scores = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        return dict(zip(classes, [float(s) for s in scores]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Per-class F1 computation failed: %s", exc)
        return {}


def print_summary(summary: EvalSummary) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  Run ID  : {summary.run_id}")
    print(f"  Mode    : {summary.mode}")
    print(f"  Samples : {summary.num_samples}")
    print(f"  Accuracy     : {summary.accuracy:.4f}")
    print(f"  F1 (macro)   : {summary.f1_macro:.4f}")
    print(f"  Prec (macro) : {summary.precision_macro:.4f}")
    print(f"  Recall (macro): {summary.recall_macro:.4f}")
    if summary.per_class_f1:
        print("  Per-class F1:")
        for cls, val in sorted(summary.per_class_f1.items()):
            print(f"    {cls:<35} {val:.4f}")
    if summary.notes:
        print(f"  Notes: {summary.notes}")
    print("=" * 60)


def count_label_distribution(diagnostics: list[BugDiagnostic]) -> Counter:
    """Return a Counter of taxonomy_class predictions."""
    return Counter(d.taxonomy_class for d in diagnostics)
