"""Analysis utilities for the classical-vs-quantum binary track.

* Bootstrap 95 % CI on accuracy.
* Brier score for calibration.
* Reliability bins (mean predicted score vs empirical rate per bucket).
* Per-class recall.
* Confusion matrices (via sklearn) — exposed via a thin wrapper.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .schemas import CLASSES, Diagnostic


# ── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_ci(
    correct_vec: list[bool] | np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of a 0/1 correctness vector."""
    cv = np.asarray(correct_vec, dtype=float)
    if len(cv) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    samples = rng.choice(cv, size=(B, len(cv)), replace=True).mean(axis=1)
    return (
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


# ── Headline metrics ────────────────────────────────────────────────────────

def headline_metrics(diags: list[Diagnostic]) -> dict:
    """Accuracy, macro-F1, and bootstrap-CI on accuracy."""
    if not diags:
        return {"n": 0}
    y_true = [d.ground_truth for d in diags]
    y_pred = [d.predicted for d in diags]
    correct = [d.correct for d in diags]
    lo, hi = bootstrap_ci(correct)
    return {
        "n": len(diags),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=CLASSES,
                                   average="macro", zero_division=0), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
    }


# ── Per-class recall ────────────────────────────────────────────────────────

def per_class_recall(diags: list[Diagnostic], cls: str) -> float:
    truth = [d for d in diags if d.ground_truth == cls]
    if not truth:
        return 0.0
    return sum(1 for d in truth if d.predicted == cls) / len(truth)


def confusion(diags: list[Diagnostic]) -> np.ndarray:
    y_true = [d.ground_truth for d in diags]
    y_pred = [d.predicted for d in diags]
    return confusion_matrix(y_true, y_pred, labels=CLASSES)


# ── Calibration: Brier and reliability bins ─────────────────────────────────

def brier(diags: list[Diagnostic]) -> float:
    s = np.array([d.score_quantum for d in diags])
    y = np.array([1 if d.ground_truth == "quantum" else 0 for d in diags])
    return float(np.mean((s - y) ** 2))


def reliability(
    diags: list[Diagnostic],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean-score, empirical-rate, bin-count) for each non-empty bin."""
    s = np.array([d.score_quantum for d in diags])
    y = np.array([1 if d.ground_truth == "quantum" else 0 for d in diags])
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (s >= lo) & (s < hi if i < n_bins - 1 else s <= hi)
        if mask.sum() > 0:
            xs.append(s[mask].mean())
            ys.append(y[mask].mean())
            ns.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


# ── External-holdout purity ─────────────────────────────────────────────────

def quantum_rate(diags: list[Diagnostic]) -> float:
    """Fraction predicted ``quantum``. For Bugs4Q (all-quantum) the ideal is 1.0."""
    if not diags:
        return 0.0
    return sum(1 for d in diags if d.predicted == "quantum") / len(diags)


def mean_score(diags: list[Diagnostic]) -> float:
    if not diags:
        return 0.0
    return float(np.mean([d.score_quantum for d in diags]))
