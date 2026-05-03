# Outputs

This directory stores run artefacts produced by the repository scripts.

Output naming now differs by track.

## 1. Legacy Scaffold Outputs

Typical files:

- `diagnostics_<mode>_<run_id>_<timestamp>.jsonl`
- `metrics_<mode>_<run_id>_<timestamp>.json`

Example diagnostic object:

```json
{
  "sample_id": "bugs4q_0042",
  "mode": "rag",
  "bug_likelihood": 0.91,
  "taxonomy_class": "incorrect_qubit_mapping",
  "suspected_location": "qc.cx(0, 1)",
  "justification": "...",
  "ground_truth": "incorrect_qubit_mapping",
  "correct": true,
  "retrieved_patterns": ["BP001", "BP007"]
}
```

## 2. `taxonomy_v6` Outputs

Files written by `scripts/run_taxonomy_v6.py`:

- `diagnostics_bugs4q_prompt_only.jsonl`
- `diagnostics_bugs4q_rag.jsonl`
- `diagnostics_bugsqcp_prompt_only.jsonl`
- `diagnostics_bugsqcp_rag.jsonl`
- `metrics_bugs4q_prompt_only.json`
- `metrics_bugs4q_rag.json`
- `metrics_bugsqcp_prompt_only.json`
- `metrics_bugsqcp_rag.json`
- `summary.json`

Diagnostic shape:

```json
{
  "sample_id": "bugs4q_0007",
  "mode": "rag",
  "bug_likelihood": 0.84,
  "taxonomy_class": "measurement_error",
  "class_scores": {
    "incorrect_operator": 0.12,
    "incorrect_qubit_mapping": 0.19,
    "missing_barrier": 0.07,
    "wrong_initial_state": 0.10,
    "measurement_error": 0.84
  },
  "suspected_location": "qc.measure(...)",
  "justification": "...",
  "ground_truth": "measurement_error",
  "correct": true,
  "retrieved_patterns": ["qiskit_releasenotes_fixes_x", "lintq_ql-double-measurement"]
}
```

## 3. `classical` Outputs

Files written by `scripts/run_classical_vs_quantum.py`:

- `diagnostics_bqcp_prompt_only.jsonl`
- `diagnostics_bqcp_biased_rag.jsonl`
- `diagnostics_bqcp_balanced_rag.jsonl`
- `diagnostics_bugs4q_prompt_only.jsonl`
- `diagnostics_bugs4q_biased_rag.jsonl`
- `diagnostics_bugs4q_balanced_rag.jsonl`
- `summary.json`

Diagnostic shape:

```json
{
  "sample_id": "bqcp_123",
  "mode": "balanced_rag",
  "predicted": "quantum",
  "score_quantum": 0.77,
  "ground_truth": "quantum",
  "correct": true,
  "retrieved_ids": ["qiskit_rn_fixes_x", "numpy_1_24_notes_y"],
  "reasoning": "..."
}
```

## Notes

- Smoke/mock runs are useful for wiring checks, not benchmark claims.
- Notebook-only plotting and archive ZIP generation are not currently reproduced as separate output artefacts by the CLIs.
