#!/usr/bin/env python3

"""Run the v6 forced-choice taxonomy evaluation with strict dev/test discipline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.taxonomy_v6.analysis import (
    PRIOR_EPSILON,
    apply_bayesian_prior_correction,
    abstention_grounding_report,
    build_hybrid_diagnostics,
    compute_dev_class_prior,
    expected_calibration_error,
    fit_temperature,
    headline_metrics,
    mcnemar_prompt_only_vs_rag,
    stratified_dev_test_split,
    temperature_scaled_diagnostics,
    tune_tau,
)
from src.taxonomy_v6.dataset import build_bugs4q, build_bugsqcp
from src.taxonomy_v6.evaluator import evaluate, write_diagnostics, write_metrics
from src.taxonomy_v6.kb import build_validated_kb
from src.taxonomy_v6.llm import build_llm
from src.taxonomy_v6.retriever import FrameworkAwareRetriever
from src.taxonomy_v6.schemas import BugDiagnostic, BugSample


def _split_samples(
    samples: list[BugSample],
    dev_ratio: float,
    seed: int,
    limit: int | None = None,
) -> tuple[list[BugSample], list[BugSample]]:
    labelled = [sample for sample in samples if sample.ground_truth]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        labelled = sorted(labelled, key=lambda sample: sample.sample_id)[:limit]
    split_diags = [
        BugDiagnostic(
            sample_id=sample.sample_id,
            mode="split_seed",
            bug_likelihood=0.0,
            taxonomy_class=sample.ground_truth or "unknown",
            ground_truth=sample.ground_truth,
        )
        for sample in labelled
    ]
    dev_seed, test_seed = stratified_dev_test_split(split_diags, dev_ratio=dev_ratio, seed=seed)
    dev_ids = {diag.sample_id for diag in dev_seed}
    test_ids = {diag.sample_id for diag in test_seed}
    return (
        [sample for sample in labelled if sample.sample_id in dev_ids],
        [sample for sample in labelled if sample.sample_id in test_ids],
    )


def _mode_report(
    test_diags: list[BugDiagnostic],
    dev_diags: list[BugDiagnostic],
    seed: int,
) -> tuple[list[BugDiagnostic], dict]:
    temperature = fit_temperature(dev_diags)
    dev_prior = compute_dev_class_prior(dev_diags)
    scaled_test_diags = temperature_scaled_diagnostics(test_diags, temperature)
    corrected_test_diags = apply_bayesian_prior_correction(scaled_test_diags, dev_prior)
    report = headline_metrics(corrected_test_diags, seed=seed)
    report["temperature_from_dev"] = round(temperature, 6)
    report["dev_prior"] = {k: round(v, 6) for k, v in dev_prior.items()}
    report["ece_pre_calibration"] = expected_calibration_error(test_diags, temperature=1.0, n_bins=10)
    report["ece_post_calibration"] = expected_calibration_error(test_diags, temperature=temperature, n_bins=10)
    return corrected_test_diags, report


def _write_split_bundle(
    results_dir: Path,
    dataset_name: str,
    mode: str,
    dev_diags: list[BugDiagnostic],
    test_diags: list[BugDiagnostic],
    metrics: dict,
) -> None:
    write_diagnostics(results_dir / f"diagnostics_{dataset_name}_dev_{mode}.jsonl", dev_diags)
    write_diagnostics(results_dir / f"diagnostics_{dataset_name}_test_{mode}.jsonl", test_diags)
    write_metrics(results_dir / f"metrics_{dataset_name}_{mode}.json", {"split": "test", **metrics})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", required=True, type=Path,
                    help="Directory containing cloned source repositories.")
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="Directory to write diagnostics and metrics JSON.")
    ap.add_argument("--mock", action="store_true",
                    help="Use the mock LLM (no API calls).")
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI model name (default: gpt-4o).")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of patterns to retrieve in RAG mode.")
    ap.add_argument("--bm25-floor", type=float, default=0.0,
                    help="Hard BM25 floor applied after ranking and framework boosting.")
    ap.add_argument("--dev-ratio", type=float, default=0.6,
                    help="Development split ratio for tau and temperature tuning.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Deterministic seed for dev/test splitting and bootstrap resampling.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on labelled samples per dataset for smoke runs.")
    args = ap.parse_args()

    work, results = args.work_dir, args.results_dir

    bugs4q_samples = build_bugs4q(work / "bugs4q_upstream")
    bqcp_samples = build_bugsqcp(work / "bqcp", quantum_only=True)
    print(f"Bugs4Q: {len(bugs4q_samples)} samples "
          f"({sum(1 for s in bugs4q_samples if s.ground_truth)} labelled)")
    print(f"Bugs-QCP (quantum-only): {len(bqcp_samples)} samples "
          f"({sum(1 for s in bqcp_samples if s.ground_truth)} labelled)")

    kb_roots = {
        "qiskit": work / "qiskit",
        "qiskit_aer": work / "qiskit_aer",
        "qiskit_ignis": work / "qiskit_ignis",
        "qiskit_ibm_runtime": work / "qiskit_ibm_runtime",
        "pennylane": work / "pennylane",
    }
    kb_patterns = build_validated_kb(kb_roots)
    print(f"KB: {len(kb_patterns)} validated patterns "
          f"by source {dict(Counter(p.source for p in kb_patterns))}")

    retriever = FrameworkAwareRetriever(kb_patterns, top_pool=20)
    llm = build_llm(use_mock=args.mock, model=args.model)

    datasets = {"bugs4q": bugs4q_samples, "bugsqcp": bqcp_samples}
    summary = {
        "model": args.model if not args.mock else "mock",
        "seed": args.seed,
        "split_policy": {"dev_ratio": args.dev_ratio, "test_ratio": round(1 - args.dev_ratio, 4)},
        "retrieval": {"top_k": args.top_k, "bm25_floor": args.bm25_floor},
        "prior_correction": {"epsilon": PRIOR_EPSILON},
        "kb_size": len(kb_patterns),
        "kb_sources": dict(Counter(p.source for p in kb_patterns)),
        "datasets": {},
    }

    results.mkdir(parents=True, exist_ok=True)

    for dataset_name, samples in datasets.items():
        dev_samples, test_samples = _split_samples(samples, args.dev_ratio, args.seed, args.limit)
        print(f"\n[{dataset_name}] dev={len(dev_samples)} test={len(test_samples)}")

        po_dev, _ = evaluate(
            dataset_name=f"{dataset_name}:dev",
            samples=dev_samples,
            mode="prompt_only",
            llm=llm,
            retriever=None,
            top_k=args.top_k,
            bm25_floor=args.bm25_floor,
        )
        rag_dev, _ = evaluate(
            dataset_name=f"{dataset_name}:dev",
            samples=dev_samples,
            mode="rag",
            llm=llm,
            retriever=retriever,
            top_k=args.top_k,
            bm25_floor=args.bm25_floor,
        )

        tau_info = tune_tau(po_dev, rag_dev)
        tau = float(tau_info["tau"])
        hybrid_dev = build_hybrid_diagnostics(po_dev, rag_dev, tau=tau)

        po_test, _ = evaluate(
            dataset_name=f"{dataset_name}:test",
            samples=test_samples,
            mode="prompt_only",
            llm=llm,
            retriever=None,
            top_k=args.top_k,
            bm25_floor=args.bm25_floor,
        )
        rag_test, _ = evaluate(
            dataset_name=f"{dataset_name}:test",
            samples=test_samples,
            mode="rag",
            llm=llm,
            retriever=retriever,
            top_k=args.top_k,
            bm25_floor=args.bm25_floor,
        )
        hybrid_test = build_hybrid_diagnostics(po_test, rag_test, tau=tau)

        po_test_corrected, metrics_prompt_only = _mode_report(po_test, po_dev, args.seed)
        rag_test_corrected, metrics_rag = _mode_report(rag_test, rag_dev, args.seed)
        hybrid_test_corrected, metrics_hybrid = _mode_report(hybrid_test, hybrid_dev, args.seed)
        metrics_hybrid["abstention_and_grounding"] = abstention_grounding_report(hybrid_test_corrected, rag_test_corrected)

        _write_split_bundle(results, dataset_name, "prompt_only", po_dev, po_test_corrected, metrics_prompt_only)
        _write_split_bundle(results, dataset_name, "rag", rag_dev, rag_test_corrected, metrics_rag)
        _write_split_bundle(results, dataset_name, "hybrid", hybrid_dev, hybrid_test_corrected, metrics_hybrid)

        summary["datasets"][dataset_name] = {
            "counts": {
                "total_labelled": len(dev_samples) + len(test_samples),
                "dev": len(dev_samples),
                "test": len(test_samples),
            },
            "tuning": {
                "tau": tau_info["tau_display"],
                "tau_candidates": tau_info["candidates"],
                "temperature_from_dev": {
                    "prompt_only": metrics_prompt_only["temperature_from_dev"],
                    "rag": metrics_rag["temperature_from_dev"],
                    "hybrid": metrics_hybrid["temperature_from_dev"],
                },
                "dev_prior": {
                    "prompt_only": metrics_prompt_only["dev_prior"],
                    "rag": metrics_rag["dev_prior"],
                    "hybrid": metrics_hybrid["dev_prior"],
                },
            },
            "test": {
                "prompt_only": metrics_prompt_only,
                "rag": metrics_rag,
                "hybrid": metrics_hybrid,
                "prompt_only_vs_rag_mcnemar": mcnemar_prompt_only_vs_rag(po_test_corrected, rag_test_corrected),
            },
        }

    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {results / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
