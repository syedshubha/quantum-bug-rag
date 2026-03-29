# Outputs

This directory stores run artefacts produced by the pipeline scripts.  All
files are excluded from version control (see `.gitignore`).

## Output Files

Each pipeline run writes two files using a consistent naming pattern:

| File | Description |
|------|-------------|
| `diagnostics_<mode>_<run_id>_<timestamp>.jsonl` | One JSON object per sample (see schema below). |
| `metrics_<mode>_<run_id>_<timestamp>.json` | Aggregate evaluation metrics for the run. |

## Diagnostic Schema (per-line in JSONL)

```json
{
  "sample_id":          "bugs4q_0042",
  "mode":               "rag",
  "bug_likelihood":     0.91,
  "taxonomy_class":     "incorrect_qubit_mapping",
  "suspected_location": "qc.cx(0, 1)",
  "justification":      "...",
  "ground_truth":       "incorrect_qubit_mapping",
  "correct":            true,
  "retrieved_patterns": ["BP001", "BP007"]
}
```

## Metrics Schema

```json
{
  "run_id":           "a3f2b1c4d5e6",
  "mode":             "rag",
  "num_samples":      100,
  "accuracy":         0.7400,
  "f1_macro":         0.7123,
  "precision_macro":  0.7301,
  "recall_macro":     0.6952,
  "per_class_f1":     { "incorrect_operator": 0.80, "..." : 0.0 },
  "notes":            ""
}
```

## Notes

- Outputs from smoke-test runs (source = `synthetic_smoke_test`) must not be
  included in reported benchmark results.
- All timestamps are in UTC (`YYYYMMDDTHHMMSS` format).
