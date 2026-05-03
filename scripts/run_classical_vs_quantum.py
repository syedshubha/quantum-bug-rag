#!/usr/bin/env python3
"""Run the classical-vs-quantum binary classification experiment.

Refactored from
``quantum-software-bug-detection-rag-project-v6_classical.ipynb``.

Expected layout under ``--work-dir``:
  bugs4q/      — clone of github.com/Z-928/Bugs4Q
  bqcp/        — clone of github.com/MattePalte/Bugs-Quantum-Computing-Platforms
  qiskit/      — clone of github.com/Qiskit/qiskit
  qiskit_aer/  — clone of github.com/Qiskit/qiskit-aer
  pennylane/   — clone of github.com/PennyLaneAI/pennylane
  cpython/     — sparse-clone of github.com/python/cpython (Misc/NEWS.d)
  numpy/       — clone of github.com/numpy/numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical.analysis import (
    headline_metrics, mean_score, per_class_recall, quantum_rate,
)
from src.classical.dataset import build_bqcp, build_bugs4q
from src.classical.evaluator import run_dataset
from src.classical.kb import build_symmetric_kb, kb_summary
from src.classical.llm import build_llm
from src.classical.retriever import BalancedRetriever, BM25Retriever
from src.classical.schemas import CLASSES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--mock", action="store_true",
                    help="Use the mock LLM (no API calls).")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    work, results = args.work_dir, args.results_dir
    results.mkdir(parents=True, exist_ok=True)

    # 1. Datasets
    bqcp_samples, bqcp_stats = build_bqcp(work / "bqcp")
    bugs4q_samples = build_bugs4q(work / "bugs4q")
    print(f"BQCP   : {len(bqcp_samples)} samples (skipped: {bqcp_stats})")
    print(f"  class distribution: {dict(Counter(s.ground_truth for s in bqcp_samples))}")
    print(f"Bugs4Q : {len(bugs4q_samples)} samples (all quantum, external holdout)")

    # 2. Symmetric KB
    kb_quantum, kb_classical = build_symmetric_kb({
        "qiskit":     work / "qiskit",
        "qiskit_aer": work / "qiskit_aer",
        "pennylane":  work / "pennylane",
        "cpython":    work / "cpython",
        "numpy":      work / "numpy",
    }, seed=args.seed)
    print(f"Quantum   KB: {kb_summary(kb_quantum)}")
    print(f"Classical KB: {kb_summary(kb_classical)}")

    # 3. Retrievers and LLM
    retriever_biased = BM25Retriever(kb_quantum)
    retriever_balanced = BalancedRetriever(kb_quantum, kb_classical)
    llm = build_llm(use_mock=args.mock, model=args.model)

    # 4. Evaluate
    print("\nEvaluating BQCP ...")
    bqcp_diags = run_dataset("bqcp", bqcp_samples, llm,
                             retriever_biased, retriever_balanced)
    print("\nEvaluating Bugs4Q (holdout) ...")
    b4q_diags = run_dataset("bugs4q", bugs4q_samples, llm,
                            retriever_biased, retriever_balanced)

    # 5. Persist diagnostics
    for name, diags_by_mode in (("bqcp", bqcp_diags), ("bugs4q", b4q_diags)):
        for mode, diags in diags_by_mode.items():
            path = results / f"diagnostics_{name}_{mode}.jsonl"
            with path.open("w") as fh:
                for d in diags:
                    fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    # 6. Aggregate metrics
    summary = {
        "model": args.model if not args.mock else "mock",
        "seed": args.seed,
        "kb": {
            "quantum": kb_summary(kb_quantum),
            "classical": kb_summary(kb_classical),
        },
        "bqcp": {
            mode: {
                **headline_metrics(diags),
                "per_class_recall": {c: round(per_class_recall(diags, c), 4) for c in CLASSES},
            }
            for mode, diags in bqcp_diags.items()
        },
        "bugs4q_purity": {
            mode: {
                "predicted_quantum_rate": round(quantum_rate(diags), 4),
                "mean_score_quantum": round(mean_score(diags), 4),
                "n": len(diags),
            }
            for mode, diags in b4q_diags.items()
        },
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
