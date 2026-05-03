# Methodology and Architecture Reference

> Last updated: May 3, 2026

## 1. System Overview

The repository has three distinct experiment paths:

| Track | Objective | Primary code |
|------|-----------|--------------|
| `legacy scaffold` | Evaluate prompt-only, RAG, and static baseline over prepared Bugs4Q data and a JSON KB | top-level `src/` modules |
| `taxonomy_v6` | Forced-choice 5-class quantum bug classification using validated release-note knowledge | `src/taxonomy_v6/` |
| `classical` | Distinguish quantum bugs from classical bugs in quantum-software repositories | `src/classical/` |

The two notebook-refactored tracks were introduced to preserve the experimental logic of:

- `quantum_bug_detecttion_taxonomy.ipynb`
- `quantum-software-bug-detection-rag-project-v6_classical.ipynb`

while making them scriptable and reusable.

## 2. Track Architecture

### 2.1 `taxonomy_v6`

Files:

- `src/taxonomy_v6/dataset.py`
- `src/taxonomy_v6/kb.py`
- `src/taxonomy_v6/retriever.py`
- `src/taxonomy_v6/prompts.py`
- `src/taxonomy_v6/llm.py`
- `src/taxonomy_v6/evaluator.py`
- `scripts/run_taxonomy_v6.py`

Execution flow:

1. Load Bugs4Q and quantum-only Bugs-QCP samples.
2. Build a validated quantum KB from release notes and LintQ summaries.
3. Detect the framework of each query snippet.
4. Retrieve diversified BM25 references with framework boosting.
5. Run either prompt-only or RAG classification.
6. Write diagnostics, metrics, and a cross-mode summary.

### 2.2 `classical`

Files:

- `src/classical/dataset.py`
- `src/classical/kb.py`
- `src/classical/retriever.py`
- `src/classical/prompts.py`
- `src/classical/llm.py`
- `src/classical/evaluator.py`
- `src/classical/analysis.py`
- `scripts/run_classical_vs_quantum.py`

Execution flow:

1. Load Bugs-QCP as the labelled binary dataset.
2. Load Bugs4Q as an external all-quantum holdout.
3. Extract quantum and classical release-note KB halves.
4. Downsample to a symmetric KB so retrieval pool composition is balanced.
5. Evaluate prompt-only, biased-RAG, and balanced-RAG modes.
6. Write diagnostics and summary metrics.

### 2.3 Legacy Scaffold

The original top-level `src/` modules remain for the project’s earlier prepared-data workflow. They use `data/bugs4q/` and `knowledge_base/` rather than raw upstream repository clones.

## 3. Data Handling

### 3.1 `taxonomy_v6` Data Handling

`build_bugs4q()`:

- scans the upstream Bugs4Q repository for buggy Python files;
- parses the upstream README table;
- maps raw type strings into the five forced taxonomy classes.

`build_bugsqcp()`:

- indexes Bugs-QCP minimal bugfix folders;
- reconstructs focused buggy snippets from unified diffs;
- optionally restricts to quantum-labelled entries.

### 3.2 `classical` Data Handling

`build_bqcp()`:

- reads `annotation_bugs.csv`;
- resolves each row to a `minimal_bugfixes` folder;
- concatenates source/build/script files from `before/`.

`build_bugs4q()`:

- finds buggy files in Bugs4Q;
- assigns every sample `ground_truth="quantum"`;
- uses the set only for purity analysis, not balanced binary training labels.

## 4. Knowledge Bases

### 4.1 `taxonomy_v6` Validated KB

Sources:

- Qiskit release-note YAML files;
- Qiskit Aer release-note YAML files;
- Qiskit Ignis release-note YAML files;
- IBM Runtime RST release notes;
- PennyLane changelog markdown;
- embedded LintQ rule summaries.

Processing:

- remove low-quality release-note items;
- classify remaining text into the project taxonomy with a keyword table;
- attach framework and section tags for retrieval-time use.

### 4.2 `classical` Symmetric KB

Quantum side:

- Qiskit release-note YAML files;
- Qiskit Aer release-note YAML files;
- PennyLane changelog entries.

Classical side:

- CPython `Misc/NEWS.d`;
- NumPy release notes.

The two halves are size-matched by proportional downsampling. This is part of the experiment design, not just an implementation detail.

## 5. Retrieval

### 5.1 `taxonomy_v6`

Retriever properties:

- BM25 over KB text;
- framework detection for `qiskit`, `pennylane`, `cirq`, `qsharp`, or `other`;
- 1.5x score boost for matching-framework KB entries;
- diversified selection across taxonomy classes.

### 5.2 `classical`

Retriever properties:

- `BM25Retriever` for a single pool;
- `BalancedRetriever` for separate quantum/classical pools.

This supports the comparison between:

- `biased_rag`: retrieval from a quantum-only KB;
- `balanced_rag`: retrieval from equal-sized quantum and classical KB halves.

## 6. Prompting And Model Interaction

Both notebook-refactored tracks keep lightweight per-track prompt builders and LLM clients instead of reusing the older project-wide abstractions.

Reason:

- the notebook experiments had different response schemas;
- the classical track predicts a continuous `score_quantum`;
- the taxonomy track predicts per-class scores and forced-choice labels.

Each track supports:

- `--mock` for offline deterministic smoke runs;
- OpenAI-backed runs via `OPENAI_API_KEY`.

## 7. Metrics And Outputs

### 7.1 `taxonomy_v6`

Per-mode outputs:

- `diagnostics_<dataset>_<mode>.jsonl`
- `metrics_<dataset>_<mode>.json`

Aggregate output:

- `summary.json`

Metrics:

- accuracy;
- top-2 accuracy;
- macro precision / recall / F1;
- per-class F1;
- label and prediction distributions;
- paired prompt-only vs RAG comparison.

### 7.2 `classical`

Per-mode outputs:

- `diagnostics_bqcp_<mode>.jsonl`
- `diagnostics_bugs4q_<mode>.jsonl`

Aggregate output:

- `summary.json`

Metrics:

- accuracy;
- macro F1;
- bootstrap CI on accuracy;
- per-class recall;
- Bugs4Q predicted-quantum rate;
- Bugs4Q mean `score_quantum`.

`src/classical/analysis.py` also exposes calibration helpers used in the notebook for Brier and reliability analysis.

## 8. Notebook Refactor Coverage

Core parity with the notebooks is now present for both tracks:

- schemas/dataclasses;
- dataset adapters;
- KB extraction;
- retrievers;
- prompts;
- LLM clients;
- evaluation loops;
- JSON artefact writing.

Still notebook-only:

- matplotlib figure generation;
- ad hoc printed disagreement/error inspections;
- result ZIP packaging.

These gaps are presentation-oriented rather than core execution gaps.

## 9. Current Constraints

- The automated test suite still primarily covers the legacy scaffold.
- The notebook-refactored tracks depend on external cloned repositories and have higher setup cost.
- Runtime validation in a fresh shell requires the dependencies in `requirements.txt`, especially `rank_bm25`.

## 10. Recommended Reading Order

- Top-level overview: `README.md`
- Concise experimental summary: `docs/methodology.md`
- Data and KB details: `data/README.md`, `knowledge_base/README.md`
- Output file conventions: `outputs/README.md`
