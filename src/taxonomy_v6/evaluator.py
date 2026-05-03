"""Evaluation orchestrator for the v6 forced-choice taxonomy track.

Outputs:
  - per-sample diagnostics JSONL
  - per-mode metrics JSON (accuracy, top-2 accuracy, macro-F1, macro-P/R,
    per-class F1, label and prediction distributions)

The orchestrator decides the predicted class via ``argmax(class_scores)``
falling back to the LLM-emitted ``taxonomy_class`` field, and only as a
last resort to ``"unknown"``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .llm import BaseLLM
from .prompts import build_prompt_only, build_rag_prompt
from .retriever import FrameworkAwareRetriever, detect_framework
from .schemas import BugDiagnostic, BugSample, TAXONOMY_FORCED


def _argmax_class(scores_dict: dict) -> Optional[str]:
    if not isinstance(scores_dict, dict):
        return None
    valid = {k: v for k, v in scores_dict.items() if k in TAXONOMY_FORCED}
    if not valid:
        return None
    try:
        return max(valid.items(), key=lambda kv: float(kv[1]))[0]
    except (TypeError, ValueError):
        return None


def run_one_sample(
    sample: BugSample,
    mode: str,
    llm: BaseLLM,
    retriever: Optional[FrameworkAwareRetriever] = None,
    top_k: int = 5,
) -> BugDiagnostic:
    """Run a single sample under ``mode`` ∈ {"prompt_only", "rag"}."""
    retrieved = []
    if mode == "rag":
        if retriever is None:
            raise ValueError("rag mode requires a retriever")
        framework = detect_framework(sample.code)
        retrieved = retriever.retrieve(sample.code, top_k=top_k, framework=framework)
        msgs = build_rag_prompt(sample, retrieved)
    else:
        msgs = build_prompt_only(sample)

    try:
        raw = llm.complete(msgs)
        parsed = llm.parse(raw)
    except Exception as exc:
        parsed = {"_parse_error": f"{type(exc).__name__}: {exc}"}

    scores_dict = parsed.get("scores", {})
    tax_from_argmax = _argmax_class(scores_dict)
    tax_from_llm = parsed.get("taxonomy_class")
    if tax_from_argmax:
        tax = tax_from_argmax
    elif tax_from_llm in TAXONOMY_FORCED:
        tax = tax_from_llm
    else:
        tax = "unknown"

    bug_likelihood = (
        float(scores_dict.get(tax, 0.5)) if isinstance(scores_dict, dict) else 0.5
    )
    bug_likelihood = max(0.0, min(1.0, bug_likelihood))

    diag = BugDiagnostic(
        sample_id=sample.sample_id, mode=mode,
        bug_likelihood=bug_likelihood, taxonomy_class=tax,
        class_scores={
            k: float(v) for k, v in scores_dict.items()
            if k in TAXONOMY_FORCED and isinstance(v, (int, float))
        },
        suspected_location=str(parsed.get("suspected_location", ""))[:200],
        justification=str(parsed.get("justification", ""))[:600],
        ground_truth=sample.ground_truth,
        retrieved_patterns=[p.pattern_id for p in retrieved],
    )
    if diag.ground_truth is not None:
        diag.correct = (diag.taxonomy_class == diag.ground_truth)
    return diag


def compute_metrics(diags: list[BugDiagnostic]) -> dict:
    labelled = [d for d in diags if d.ground_truth is not None]
    if not labelled:
        return {"n": 0}
    y_true = [d.ground_truth for d in labelled]
    y_pred = [d.taxonomy_class for d in labelled]
    classes = sorted(set(y_true))
    per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    top2_correct = 0
    for d in labelled:
        if d.class_scores:
            top2 = sorted(d.class_scores.items(), key=lambda kv: -kv[1])[:2]
            top2_classes = [c for c, _ in top2]
            if d.ground_truth in top2_classes:
                top2_correct += 1
    return {
        "n": len(labelled),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "top2_accuracy": round(top2_correct / len(labelled), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_p": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_r": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "per_class_f1": {c: round(float(s), 4) for c, s in zip(classes, per_class)},
        "label_distribution": dict(Counter(y_true)),
        "prediction_distribution": dict(Counter(y_pred)),
    }


def evaluate(
    dataset_name: str,
    samples: list[BugSample],
    mode: str,
    llm: BaseLLM,
    retriever: Optional[FrameworkAwareRetriever],
    results_dir: Path,
    top_k: int = 5,
    progress_every: int = 10,
) -> tuple[list[BugDiagnostic], dict]:
    """Run a full pass over labelled samples and persist per-sample outputs."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    target = [s for s in samples if s.ground_truth]
    diags: list[BugDiagnostic] = []
    print(f"\n[{dataset_name} / {mode}] running on {len(target)} samples ...")
    for i, s in enumerate(target, 1):
        diag = run_one_sample(s, mode, llm, retriever, top_k=top_k)
        diags.append(diag)
        if i % progress_every == 0 or i == len(target):
            correct = sum(1 for d in diags if d.correct)
            print(f"  [{i:3d}/{len(target)}] accuracy: {correct}/{i} = {correct / i:.3f}")

    diag_path = results_dir / f"diagnostics_{dataset_name}_{mode}.jsonl"
    with diag_path.open("w") as fh:
        for d in diags:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    metrics = compute_metrics(diags)
    (results_dir / f"metrics_{dataset_name}_{mode}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return diags, metrics


def paired_comparison(
    diags_prompt_only: list[BugDiagnostic],
    diags_rag: list[BugDiagnostic],
) -> dict:
    """Per-sample paired counts of {both_correct, both_wrong, rag_only, po_only}."""
    po = {d.sample_id: d for d in diags_prompt_only}
    rg = {d.sample_id: d for d in diags_rag}
    common = set(po) & set(rg)
    table = {"both_correct": 0, "both_wrong": 0, "rag_only": 0, "po_only": 0}
    for sid in common:
        p, r = po[sid].correct, rg[sid].correct
        if p is None or r is None:
            continue
        if p and r:
            table["both_correct"] += 1
        elif not p and not r:
            table["both_wrong"] += 1
        elif r and not p:
            table["rag_only"] += 1
        elif p and not r:
            table["po_only"] += 1
    table["net_rag_advantage"] = table["rag_only"] - table["po_only"]
    return table
