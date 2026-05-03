#!/usr/bin/env python3

"""Run the v6 forced-choice taxonomy evaluation.

Refactored from ``quantum_bug_detecttion_taxonomy.ipynb``.

Expected layout under ``--work-dir``:
  bugs4q_upstream/    — clone of github.com/Z-928/Bugs4Q
  bqcp/               — clone of github.com/MattePalte/Bugs-Quantum-Computing-Platforms
  qiskit/             — clone of github.com/Qiskit/qiskit
  qiskit_aer/         — clone of github.com/Qiskit/qiskit-aer
  qiskit_ignis/       — clone of github.com/Qiskit/qiskit-ignis
  qiskit_ibm_runtime/ — clone of github.com/Qiskit/qiskit-ibm-runtime
  pennylane/          — clone of github.com/PennyLaneAI/pennylane
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.taxonomy_v6.dataset import build_bugs4q, build_bugsqcp
from src.taxonomy_v6.evaluator import evaluate, paired_comparison
from src.taxonomy_v6.kb import build_validated_kb
from src.taxonomy_v6.llm import build_llm
from src.taxonomy_v6.retriever import FrameworkAwareRetriever


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
    args = ap.parse_args()

    work, results = args.work_dir, args.results_dir

    # 1. Datasets
    bugs4q_samples = build_bugs4q(work / "bugs4q_upstream")
    bqcp_samples = build_bugsqcp(work / "bqcp", quantum_only=True)
    print(f"Bugs4Q: {len(bugs4q_samples)} samples "
          f"({sum(1 for s in bugs4q_samples if s.ground_truth)} labelled)")
    print(f"Bugs-QCP (quantum-only): {len(bqcp_samples)} samples "
          f"({sum(1 for s in bqcp_samples if s.ground_truth)} labelled)")

    # 2. Validated KB
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

    # 3. Retriever and LLM
    retriever = FrameworkAwareRetriever(kb_patterns, top_pool=20)
    llm = build_llm(use_mock=args.mock, model=args.model)

    # 4. Evaluate over both datasets, both modes
    datasets = {"bugs4q": bugs4q_samples, "bugsqcp": bqcp_samples}
    modes = ["prompt_only", "rag"]
    all_results = {}
    for ds_name, samples in datasets.items():
        for mode in modes:
            diags, metrics = evaluate(
                ds_name, samples, mode, llm, retriever,
                results_dir=results, top_k=args.top_k,
            )
            all_results[(ds_name, mode)] = (diags, metrics)

    # 5. Paired comparison per dataset
    paired = {}
    for ds in datasets:
        paired[ds] = paired_comparison(
            all_results[(ds, "prompt_only")][0],
            all_results[(ds, "rag")][0],
        )
    print("\nPaired comparison:")
    print(json.dumps(paired, indent=2))

    # 6. Summary
    summary = {
        "model": args.model if not args.mock else "mock",
        "kb_size": len(kb_patterns),
        "kb_sources": dict(Counter(p.source for p in kb_patterns)),
        "results": [
            {"dataset": ds, "mode": mode, **m}
            for (ds, mode), (_, m) in all_results.items()
        ],
        "paired": paired,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {results / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
