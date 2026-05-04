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

- `diagnostics_<dataset>_dev_prompt_only.jsonl`
- `diagnostics_<dataset>_dev_rag.jsonl`
- `diagnostics_<dataset>_dev_hybrid.jsonl`
- `diagnostics_<dataset>_test_prompt_only.jsonl`
- `diagnostics_<dataset>_test_rag.jsonl`
- `diagnostics_<dataset>_test_hybrid.jsonl`
- `metrics_bugs4q_prompt_only.json`
- `metrics_bugs4q_rag.json`
- `metrics_bugs4q_hybrid.json`
- `metrics_bugsqcp_prompt_only.json`
- `metrics_bugsqcp_rag.json`
- `metrics_bugsqcp_hybrid.json`
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
  "evidence_ids": ["lintq_ql-double-measurement"],
  "suspected_location": "qc.measure(...)",
  "justification": "...",
  "ground_truth": "measurement_error",
  "correct": true,
  "retrieved_patterns": ["qiskit_releasenotes_fixes_x", "lintq_ql-double-measurement"],
  "top1_bm25_score": 7.412,
  "routed_mode": "rag",
  "final_mode": "rag",
  "abstained_to_prompt_only": false,
  "prompt_only_fallback_used": false,
  "fallback_reason": "",
  "attribution_failure": false,
  "grounded": true,
  "parse_retry_count": 0
}
```

`metrics_<dataset>_hybrid.json` contains an `abstention_and_grounding` block with:

- attribution failure rate;
- accuracy conditional on grounded vs ungrounded evidence;
- abstention rate;
- accuracy on the RAG-routed subset;
- accuracy on the prompt-only-routed subset;
- combined hybrid accuracy;
- pure-RAG accuracy on the same Test samples.

## 3. `study_i` Outputs

Files written by `scripts/run_study_i_codebert.py`:

- `summary.json`
- `per_fold.csv`
- `epoch_logs.json`
- `fig1_confusion_matrix.png`
- `fig2_fold_distribution.png`
- `fig3_roc_curve.png`
- `fig4_learning_curves.png`
- `fig5_summary_panel.png`

`summary.json` contains:

- dataset size and class distribution;
- repeated-CV setup (`n_folds`, seeds, number of fold-runs);
- the Study I hyperparameters;
- mean and standard deviation of accuracy, macro-F1, weighted F1, and ROC-AUC;
- pooled accuracy / macro-F1 / weighted F1 / ROC-AUC;
- a pooled classification report;
- a pooled confusion matrix;
- raw per-fold predictions and epoch logs.

`per_fold.csv` contains one row per fold-run with:

- `repeat`
- `fold`
- `cv_seed`
- `train_seed`
- `n_train`
- `n_validation`
- `n_test`
- `accuracy`
- `f1_macro`
- `f1_weighted`
- `roc_auc`

## 4. `classical` Outputs

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
