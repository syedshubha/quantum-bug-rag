"""Evaluation helpers for the v6 forced-choice taxonomy track."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .analysis import headline_metrics
from .llm import BaseLLM, build_response_format, parsed_response_is_complete
from .prompts import build_prompt_only, build_rag_prompt
from .retriever import FrameworkAwareRetriever, detect_framework
from .schemas import BugDiagnostic, BugPattern, BugSample, TAXONOMY_FORCED

RETRY_TEMPERATURE_DELTA = 0.15


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


def _invoke_strict_json(
    llm: BaseLLM,
    messages: list[dict],
    retrieved_ids: list[str],
    base_temperature: float,
) -> tuple[dict, int, Optional[str]]:
    response_format = build_response_format(retrieved_ids)
    last_error: Optional[str] = None
    for attempt, temperature in enumerate(
        (base_temperature, base_temperature + RETRY_TEMPERATURE_DELTA)
    ):
        try:
            raw = llm.complete(
                messages,
                temperature=temperature,
                response_format=response_format,
                retrieved_ids=retrieved_ids,
            )
            parsed = llm.parse(raw)
        except Exception as exc:  # pragma: no cover - network/client failure
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if parsed_response_is_complete(parsed):
            return parsed, attempt, None
        last_error = parsed.get("_parse_error", "structured output incomplete")
    return {}, 1, last_error


def _build_diag(
    sample: BugSample,
    experiment_mode: str,
    final_mode: str,
    parsed: dict,
    retrieved: list[BugPattern],
    top1_score: Optional[float],
    routed_mode: str,
    parse_retry_count: int,
    prompt_only_fallback_used: bool = False,
    fallback_reason: str = "",
) -> BugDiagnostic:
    scores_dict = parsed.get("scores", {}) if isinstance(parsed, dict) else {}
    tax_from_argmax = _argmax_class(scores_dict)
    tax_from_llm = parsed.get("taxonomy_class") if isinstance(parsed, dict) else None
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

    retrieved_ids = [p.pattern_id for p in retrieved]
    evidence_ids_raw = parsed.get("evidence_ids", []) if isinstance(parsed, dict) else []
    if not isinstance(evidence_ids_raw, list):
        evidence_ids_raw = []
    evidence_ids = [str(v) for v in evidence_ids_raw]
    attribution_failure = any(eid not in retrieved_ids for eid in evidence_ids)

    grounded: Optional[bool]
    if routed_mode != "rag":
        grounded = None
    else:
        grounded = bool(evidence_ids) and not attribution_failure

    diag = BugDiagnostic(
        sample_id=sample.sample_id,
        mode=experiment_mode,
        bug_likelihood=bug_likelihood,
        taxonomy_class=tax,
        class_scores={
            k: float(v)
            for k, v in scores_dict.items()
            if k in TAXONOMY_FORCED and isinstance(v, (int, float))
        },
        evidence_ids=evidence_ids,
        suspected_location=str(parsed.get("suspected_location", ""))[:200],
        justification=str(parsed.get("justification", ""))[:600],
        ground_truth=sample.ground_truth,
        retrieved_patterns=retrieved_ids,
        top1_bm25_score=top1_score,
        routed_mode=routed_mode,
        final_mode=final_mode,
        prompt_only_fallback_used=prompt_only_fallback_used,
        fallback_reason=fallback_reason,
        attribution_failure=attribution_failure,
        grounded=grounded,
        parse_retry_count=parse_retry_count,
    )
    if diag.ground_truth is not None:
        diag.correct = diag.taxonomy_class == diag.ground_truth
    return diag


def run_one_sample(
    sample: BugSample,
    mode: str,
    llm: BaseLLM,
    retriever: Optional[FrameworkAwareRetriever] = None,
    top_k: int = 5,
    bm25_floor: float = 0.0,
    base_temperature: float = 0.0,
) -> BugDiagnostic:
    if mode not in {"prompt_only", "rag"}:
        raise ValueError(f"unsupported mode: {mode}")

    if mode == "prompt_only":
        parsed, retry_count, parse_error = _invoke_strict_json(
            llm,
            build_prompt_only(sample),
            retrieved_ids=[],
            base_temperature=base_temperature,
        )
        if parse_error is not None:
            parsed = {"_parse_error": parse_error, "scores": {}, "evidence_ids": []}
        return _build_diag(
            sample,
            experiment_mode="prompt_only",
            final_mode="prompt_only",
            parsed=parsed,
            retrieved=[],
            top1_score=None,
            routed_mode="prompt_only",
            parse_retry_count=retry_count,
            fallback_reason=parse_error or "",
        )

    if retriever is None:
        raise ValueError("rag mode requires a retriever")

    framework = detect_framework(sample.code)
    retrieved_hits = retriever.retrieve_with_scores(
        sample.code,
        top_k=top_k,
        framework=framework,
        floor=bm25_floor,
    )
    retrieved = [pattern for pattern, _ in retrieved_hits]
    top1_score = retriever.top_score(sample.code, framework=framework)
    parsed, retry_count, parse_error = _invoke_strict_json(
        llm,
        build_rag_prompt(sample, retrieved),
        retrieved_ids=[p.pattern_id for p in retrieved],
        base_temperature=base_temperature,
    )
    if parse_error is None:
        return _build_diag(
            sample,
            experiment_mode="rag",
            final_mode="rag",
            parsed=parsed,
            retrieved=retrieved,
            top1_score=top1_score,
            routed_mode="rag",
            parse_retry_count=retry_count,
        )

    fallback_parsed, fallback_retry_count, fallback_error = _invoke_strict_json(
        llm,
        build_prompt_only(sample),
        retrieved_ids=[],
        base_temperature=base_temperature,
    )
    if fallback_error is not None:
        fallback_parsed = {
            "_parse_error": fallback_error,
            "scores": {},
            "evidence_ids": [],
        }
    return _build_diag(
        sample,
        experiment_mode="rag",
        final_mode="prompt_only",
        parsed=fallback_parsed,
        retrieved=retrieved,
        top1_score=top1_score,
        routed_mode="rag",
        parse_retry_count=retry_count + fallback_retry_count + 1,
        prompt_only_fallback_used=True,
        fallback_reason=parse_error,
    )


def evaluate(
    dataset_name: str,
    samples: list[BugSample],
    mode: str,
    llm: BaseLLM,
    retriever: Optional[FrameworkAwareRetriever],
    top_k: int = 5,
    bm25_floor: float = 0.0,
    progress_every: int = 10,
) -> tuple[list[BugDiagnostic], dict]:
    target = [s for s in samples if s.ground_truth]
    diags: list[BugDiagnostic] = []
    print(f"\n[{dataset_name} / {mode}] running on {len(target)} samples ...")
    for i, sample in enumerate(target, 1):
        diag = run_one_sample(
            sample,
            mode,
            llm,
            retriever,
            top_k=top_k,
            bm25_floor=bm25_floor,
        )
        diags.append(diag)
        if i % progress_every == 0 or i == len(target):
            correct = sum(1 for d in diags if d.correct)
            print(f"  [{i:3d}/{len(target)}] accuracy: {correct}/{i} = {correct / i:.3f}")
    return diags, headline_metrics(diags)


def write_diagnostics(path: Path, diags: list[BugDiagnostic]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for diag in diags:
            fh.write(json.dumps(asdict(diag), ensure_ascii=False) + "\n")


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
