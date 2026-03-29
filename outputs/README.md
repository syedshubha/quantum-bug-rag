# Outputs

This directory stores all pipeline outputs, logs, and evaluation results.

## Structure

```
outputs/
├── prompt_only/
│   ├── results.json    ← DiagnosticResult objects for each program
│   └── run.log
├── rag/
│   ├── results.json
│   └── run.log
├── static/
│   ├── results.json
│   └── run.log
└── subset_eval/
    ├── prompt_only_results.json
    ├── rag_results.json
    ├── static_results.json
    └── comparison.json   ← EvaluationSummary for all three modes
```

## Result schema

Each entry in `results.json` follows the `DiagnosticResult` schema
(see `src/schemas.py`):

```json
{
  "program_id": "bugs4q_0001",
  "bug_likelihood": 0.82,
  "taxonomy_class": "missing_measurement",
  "suspected_location": "circuit.py:12",
  "justification": "No .measure() call found ...",
  "retrieved_patterns": ["BP001"],
  "mode": "rag"
}
```

## Notes

- Result files are **not** committed to the repository by default.
- Add `outputs/*/results.json` and `outputs/*/run.log` to `.gitignore`
  if you do not want to track them.
- The `comparison.json` file from `run_subset_eval.py` is useful for
  quick comparisons between pipeline modes.
