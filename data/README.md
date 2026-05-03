# Data Directory

This directory documents the prepared-data workflow used by the legacy scaffold.

## What Lives Here

| Path | Purpose |
|------|---------|
| `data/bugs4q/` | Prepared Bugs4Q JSONL files for the legacy top-level pipeline |

The newer `taxonomy_v6` and `classical` CLIs do not read from `data/bugs4q/`. They expect raw upstream repository clones under a user-provided `--work-dir`.

## Dataset Roles Across The Repository

| Dataset | Legacy scaffold | `taxonomy_v6` | `classical` |
|---------|-----------------|---------------|-------------|
| `Bugs4Q` | Prepared evaluation dataset | Upstream clone, evaluated with mapped labels | Upstream clone, treated as all-quantum holdout |
| `Bugs-QCP` | KB enrichment source only | Upstream clone, quantum-only evaluation subset | Upstream clone, primary labelled binary dataset |

## Preparing Bugs4Q For The Legacy Scaffold

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q/
```

This creates prepared JSONL artefacts such as:

- `samples.real.jsonl`
- `samples.synthetic.jsonl`
- `active_dataset.json`

Inspect the current prepared dataset:

```bash
python scripts/inspect_dataset.py --data-dir data/bugs4q/
```

## Smoke-Test Mode

```bash
python scripts/prepare_bugs4q.py --smoke-test --output-dir data/bugs4q/
```

Synthetic smoke data is only for validating pipeline wiring. Do not report benchmark metrics from it.

## Bugs-QCP For The Legacy Scaffold

To enrich the legacy JSON knowledge base:

```bash
python scripts/prepare_bugsqcp_kb.py \
  --input-dir /path/to/bugsqcp \
  --output-dir knowledge_base/
```

## Raw Clone Layout For The Newer Tracks

The newer notebook-refactored CLIs expect upstream clones instead of prepared JSONL files.

`scripts/run_taxonomy_v6.py` expects:

```text
work-dir/
├── bugs4q_upstream/
├── bqcp/
├── qiskit/
├── qiskit_aer/
├── qiskit_ignis/
├── qiskit_ibm_runtime/
└── pennylane/
```

`scripts/run_classical_vs_quantum.py` expects:

```text
work-dir/
├── bugs4q/
├── bqcp/
├── qiskit/
├── qiskit_aer/
├── pennylane/
├── cpython/
└── numpy/
```
