"""Analysis utilities for the taxonomy_v6 track."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .schemas import BugDiagnostic, TAXONOMY_FORCED

try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
except Exception:  # pragma: no cover - optional dependency
    sm_mcnemar = None


PRIOR_EPSILON = 0.05


def stratified_dev_test_split(
    diags: list[BugDiagnostic],
    dev_ratio: float = 0.6,
    seed: int = 0,
) -> tuple[list[BugDiagnostic], list[BugDiagnostic]]:
    """Return deterministic stratified dev/test splits over labelled diagnostics."""
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be in (0, 1)")
    labelled = [d for d in diags if d.ground_truth is not None]
    by_label: dict[str, list[BugDiagnostic]] = {}
    for diag in labelled:
        by_label.setdefault(diag.ground_truth or "unknown", []).append(diag)

    rng = np.random.default_rng(seed)
    dev: list[BugDiagnostic] = []
    test: list[BugDiagnostic] = []
    for label in sorted(by_label):
        bucket = list(by_label[label])
        order = rng.permutation(len(bucket))
        shuffled = [bucket[i] for i in order]
        cut = int(round(len(shuffled) * dev_ratio))
        if cut <= 0:
            cut = 1 if len(shuffled) > 1 else len(shuffled)
        if cut >= len(shuffled) and len(shuffled) > 1:
            cut = len(shuffled) - 1
        dev.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    dev.sort(key=lambda d: d.sample_id)
    test.sort(key=lambda d: d.sample_id)
    return dev, test


def _label_arrays(diags: list[BugDiagnostic]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([d.ground_truth for d in diags], dtype=object)
    y_pred = np.array([d.taxonomy_class for d in diags], dtype=object)
    return y_true, y_pred


def _score_matrix(diags: list[BugDiagnostic]) -> np.ndarray:
    if not diags:
        return np.empty((0, len(TAXONOMY_FORCED)), dtype=float)
    rows = []
    for diag in diags:
        raw = np.array(
            [float(diag.class_scores.get(cls, 0.0)) for cls in TAXONOMY_FORCED],
            dtype=float,
        )
        raw = np.clip(raw, 1e-6, 1.0)
        raw = raw / raw.sum()
        rows.append(raw)
    return np.vstack(rows)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=TAXONOMY_FORCED, average="macro", zero_division=0))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def paired_bootstrap_cis(
    diags: list[BugDiagnostic],
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    if not diags:
        zero = (0.0, 0.0)
        return {"accuracy": zero, "macro_f1": zero}
    y_true, y_pred = _label_arrays(diags)
    rng = np.random.default_rng(seed)
    boot_acc: list[float] = []
    boot_f1: list[float] = []
    n = len(diags)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        boot_acc.append(_accuracy(y_true[idx], y_pred[idx]))
        boot_f1.append(_macro_f1(y_true[idx], y_pred[idx]))
    return {
        "accuracy": (
            float(np.percentile(boot_acc, alpha / 2 * 100)),
            float(np.percentile(boot_acc, (1 - alpha / 2) * 100)),
        ),
        "macro_f1": (
            float(np.percentile(boot_f1, alpha / 2 * 100)),
            float(np.percentile(boot_f1, (1 - alpha / 2) * 100)),
        ),
    }


def _top2_accuracy(diags: list[BugDiagnostic]) -> float:
    if not diags:
        return 0.0
    correct = 0
    for diag in diags:
        if not diag.class_scores or diag.ground_truth is None:
            continue
        top2 = sorted(diag.class_scores.items(), key=lambda kv: -float(kv[1]))[:2]
        if diag.ground_truth in {label for label, _ in top2}:
            correct += 1
    return correct / len(diags)


def compute_dev_class_prior(diags: list[BugDiagnostic]) -> dict[str, float]:
    """Estimate the model prior pi-hat(c) from raw dev class_scores."""
    if not diags:
        return {cls: 1.0 / len(TAXONOMY_FORCED) for cls in TAXONOMY_FORCED}
    mat = _score_matrix(diags)
    prior = mat.mean(axis=0)
    return {
        cls: float(prior[idx])
        for idx, cls in enumerate(TAXONOMY_FORCED)
    }


def temperature_scaled_diagnostics(
    diags: list[BugDiagnostic],
    temperature: float,
) -> list[BugDiagnostic]:
    if not diags:
        return []
    probs = _score_matrix(diags)
    scaled = apply_temperature_to_probabilities(probs, temperature)
    scaled_diags: list[BugDiagnostic] = []
    for idx, diag in enumerate(diags):
        score_map = {
            cls: float(scaled[idx, j])
            for j, cls in enumerate(TAXONOMY_FORCED)
        }
        pred = TAXONOMY_FORCED[int(np.argmax(scaled[idx]))]
        scaled_diags.append(replace(
            diag,
            class_scores=score_map,
            taxonomy_class=pred,
            bug_likelihood=float(score_map[pred]),
            correct=(pred == diag.ground_truth) if diag.ground_truth is not None else None,
        ))
    return scaled_diags


def apply_bayesian_prior_correction(
    diags: list[BugDiagnostic],
    dev_prior: dict[str, float],
    epsilon: float = PRIOR_EPSILON,
) -> list[BugDiagnostic]:
    """Apply pi-hat correction to already temperature-scaled diagnostics."""
    if not diags:
        return []
    prior_arr = np.array(
        [max(float(dev_prior.get(cls, 0.0)), float(epsilon)) for cls in TAXONOMY_FORCED],
        dtype=float,
    )
    corrected_diags: list[BugDiagnostic] = []
    for diag in diags:
        scaled = np.array([float(diag.class_scores.get(cls, 0.0)) for cls in TAXONOMY_FORCED], dtype=float)
        corrected = scaled / prior_arr
        corrected = corrected / corrected.sum()
        score_map = {
            cls: float(corrected[idx])
            for idx, cls in enumerate(TAXONOMY_FORCED)
        }
        pred = TAXONOMY_FORCED[int(np.argmax(corrected))]
        corrected_diags.append(replace(
            diag,
            class_scores=score_map,
            taxonomy_class=pred,
            bug_likelihood=float(score_map[pred]),
            correct=(pred == diag.ground_truth) if diag.ground_truth is not None else None,
        ))
    return corrected_diags


def headline_metrics(diags: list[BugDiagnostic], seed: int = 0) -> dict:
    if not diags:
        return {"n": 0}
    y_true, y_pred = _label_arrays(diags)
    per_class = f1_score(
        y_true,
        y_pred,
        labels=TAXONOMY_FORCED,
        average=None,
        zero_division=0,
    )
    cis = paired_bootstrap_cis(diags, seed=seed)
    return {
        "n": len(diags),
        "accuracy": round(_accuracy(y_true, y_pred), 4),
        "accuracy_ci95": [round(cis["accuracy"][0], 4), round(cis["accuracy"][1], 4)],
        "top2_accuracy": round(_top2_accuracy(diags), 4),
        "macro_f1": round(_macro_f1(y_true, y_pred), 4),
        "macro_f1_ci95": [round(cis["macro_f1"][0], 4), round(cis["macro_f1"][1], 4)],
        "macro_p": round(
            precision_score(y_true, y_pred, labels=TAXONOMY_FORCED, average="macro", zero_division=0),
            4,
        ),
        "macro_r": round(
            recall_score(y_true, y_pred, labels=TAXONOMY_FORCED, average="macro", zero_division=0),
            4,
        ),
        "per_class_f1": {
            cls: round(float(score), 4)
            for cls, score in zip(TAXONOMY_FORCED, per_class)
        },
        "label_distribution": dict(Counter(y_true.tolist())),
        "prediction_distribution": dict(Counter(y_pred.tolist())),
    }


def mcnemar_prompt_only_vs_rag(
    prompt_only: list[BugDiagnostic],
    rag: list[BugDiagnostic],
) -> dict:
    po = {d.sample_id: d for d in prompt_only}
    rg = {d.sample_id: d for d in rag}
    common = sorted(set(po) & set(rg))
    b = 0
    c = 0
    both_correct = 0
    both_wrong = 0
    for sample_id in common:
        po_ok = bool(po[sample_id].correct)
        rg_ok = bool(rg[sample_id].correct)
        if po_ok and rg_ok:
            both_correct += 1
        elif not po_ok and not rg_ok:
            both_wrong += 1
        elif po_ok and not rg_ok:
            b += 1
        elif rg_ok and not po_ok:
            c += 1

    table = [[both_correct, b], [c, both_wrong]]
    if sm_mcnemar is not None:
        exact = (b + c) < 25
        result = sm_mcnemar(table, exact=exact, correction=not exact)
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)
        method = "exact" if exact else "continuity_corrected"
    else:
        if (b + c) == 0:
            statistic = 0.0
            pvalue = 1.0
        else:
            statistic = float(((abs(b - c) - 1) ** 2) / (b + c))
            pvalue = float(chi2.sf(statistic, 1))
        method = "continuity_corrected_fallback"

    return {
        "table": {"both_correct": both_correct, "both_wrong": both_wrong, "po_only": b, "rag_only": c},
        "statistic": round(statistic, 6),
        "p_value": round(pvalue, 6),
        "method": method,
    }


def _prob_matrix(diags: list[BugDiagnostic]) -> tuple[np.ndarray, np.ndarray]:
    if not diags:
        return np.empty((0, len(TAXONOMY_FORCED))), np.empty((0,), dtype=int)
    mat = _score_matrix(diags)
    labels = []
    for diag in diags:
        labels.append(TAXONOMY_FORCED.index(diag.ground_truth or TAXONOMY_FORCED[0]))
    return mat, np.array(labels, dtype=int)


def apply_temperature_to_probabilities(prob_matrix: np.ndarray, temperature: float) -> np.ndarray:
    if prob_matrix.size == 0:
        return prob_matrix
    logits = np.log(np.clip(prob_matrix, 1e-12, 1.0))
    return softmax(logits / max(float(temperature), 1e-6), axis=1)


def fit_temperature(diags: list[BugDiagnostic]) -> float:
    probs, y_true = _prob_matrix(diags)
    if len(y_true) == 0:
        return 1.0

    def objective(temp: float) -> float:
        scaled = apply_temperature_to_probabilities(probs, temp)
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(y_true)), y_true], 1e-12, 1.0))))

    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    if not result.success:
        return 1.0
    return float(result.x)


def expected_calibration_error(
    diags: list[BugDiagnostic],
    temperature: float = 1.0,
    n_bins: int = 10,
) -> dict:
    probs, y_true = _prob_matrix(diags)
    if len(y_true) == 0:
        return {"ece": 0.0, "bin_counts": [], "bins": []}
    scaled = apply_temperature_to_probabilities(probs, temperature)
    confidences = scaled.max(axis=1)
    predictions = scaled.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    sorted_idx = np.argsort(confidences)
    buckets = [idx for idx in np.array_split(sorted_idx, n_bins) if len(idx) > 0]

    ece = 0.0
    bins = []
    for bucket in buckets:
        conf = float(confidences[bucket].mean())
        acc = float(correctness[bucket].mean())
        count = int(len(bucket))
        ece += (count / len(diags)) * abs(acc - conf)
        bins.append({
            "count": count,
            "avg_confidence": round(conf, 4),
            "accuracy": round(acc, 4),
        })
    return {
        "ece": round(float(ece), 4),
        "bin_counts": [b["count"] for b in bins],
        "bins": bins,
    }


def tune_tau(
    prompt_only_dev: list[BugDiagnostic],
    rag_dev: list[BugDiagnostic],
) -> dict:
    if not prompt_only_dev or not rag_dev:
        return {"tau": 0.0, "candidates": [], "selected_accuracy": 0.0}
    po = {d.sample_id: d for d in prompt_only_dev}
    rg = {d.sample_id: d for d in rag_dev}
    common = [rg[sid] for sid in sorted(set(po) & set(rg))]
    scores = np.array([float(d.top1_bm25_score or 0.0) for d in common], dtype=float)
    order = np.argsort(scores)
    buckets = [order_bucket for order_bucket in np.array_split(order, 5) if len(order_bucket) > 0]

    candidates = [-np.inf]
    for bucket in buckets:
        candidates.append(float(scores[bucket[0]]))
    candidates.append(np.inf)
    candidates = sorted(set(candidates))

    best_tau = candidates[0]
    best_accuracy = -1.0
    candidate_rows = []
    for tau in candidates:
        correct = 0
        for sample_id in sorted(set(po) & set(rg)):
            rag_diag = rg[sample_id]
            chosen = po[sample_id] if float(rag_diag.top1_bm25_score or 0.0) < tau else rag_diag
            correct += int(bool(chosen.correct))
        accuracy = correct / len(common)
        tau_label: float | str
        if np.isneginf(tau):
            tau_label = "-inf"
        elif np.isposinf(tau):
            tau_label = "inf"
        else:
            tau_label = round(float(tau), 6)
        candidate_rows.append({"tau": tau_label, "accuracy": round(accuracy, 4)})
        if accuracy > best_accuracy:
            best_tau = tau
            best_accuracy = accuracy
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        stored_tau = 0.0
    elif np.isneginf(best_tau):
        stored_tau = float(finite_scores.min() - 1e-6)
    elif np.isposinf(best_tau):
        stored_tau = float(finite_scores.max() + 1e-6)
    else:
        stored_tau = float(best_tau)
    return {
        "tau": stored_tau,
        "tau_display": "-inf" if np.isneginf(best_tau) else ("inf" if np.isposinf(best_tau) else round(float(best_tau), 6)),
        "candidates": candidate_rows,
        "selected_accuracy": round(best_accuracy, 4),
    }


def build_hybrid_diagnostics(
    prompt_only: list[BugDiagnostic],
    rag: list[BugDiagnostic],
    tau: float,
) -> list[BugDiagnostic]:
    po = {d.sample_id: d for d in prompt_only}
    rg = {d.sample_id: d for d in rag}
    hybrid: list[BugDiagnostic] = []
    for sample_id in sorted(set(po) & set(rg)):
        rag_diag = rg[sample_id]
        use_prompt_only = float(rag_diag.top1_bm25_score or 0.0) < tau
        base = po[sample_id] if use_prompt_only else rag_diag
        hybrid.append(replace(
            base,
            mode="hybrid",
            routed_mode="prompt_only" if use_prompt_only else "rag",
            abstained_to_prompt_only=use_prompt_only,
        ))
    return hybrid


def abstention_grounding_report(
    hybrid: list[BugDiagnostic],
    pure_rag: list[BugDiagnostic],
) -> dict:
    if not hybrid:
        return {
            "attribution_failure_rate": 0.0,
            "accuracy_grounded": None,
            "accuracy_ungrounded": None,
            "abstention_rate": 0.0,
            "accuracy_rag_routed": None,
            "accuracy_prompt_only_routed": None,
            "combined_accuracy": 0.0,
            "pure_rag_accuracy_same_samples": 0.0,
        }

    flagged = [d for d in hybrid if d.attribution_failure]
    rag_routed = [d for d in hybrid if d.routed_mode == "rag"]
    po_routed = [d for d in hybrid if d.routed_mode == "prompt_only"]
    grounded = [d for d in rag_routed if d.grounded is True]
    ungrounded = [d for d in rag_routed if d.grounded is False]

    def _acc(diags: list[BugDiagnostic]) -> Optional[float]:
        if not diags:
            return None
        return round(sum(int(bool(d.correct)) for d in diags) / len(diags), 4)

    rag_by_id = {d.sample_id: d for d in pure_rag}
    rag_on_same_samples = [rag_by_id[d.sample_id] for d in hybrid if d.sample_id in rag_by_id]

    return {
        "attribution_failure_rate": round(len(flagged) / len(hybrid), 4),
        "accuracy_grounded": _acc(grounded),
        "accuracy_ungrounded": _acc(ungrounded),
        "abstention_rate": round(len(po_routed) / len(hybrid), 4),
        "accuracy_rag_routed": _acc(rag_routed),
        "accuracy_prompt_only_routed": _acc(po_routed),
        "combined_accuracy": round(sum(int(bool(d.correct)) for d in hybrid) / len(hybrid), 4),
        "pure_rag_accuracy_same_samples": _acc(rag_on_same_samples),
    }
