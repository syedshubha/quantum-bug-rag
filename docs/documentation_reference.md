# Documentation Reference

This file presents the three primary documentation files side by side so that
they can be reviewed and copied together.  Each section is the verbatim content
of the corresponding file; use the section anchors below to jump directly to
the one you need.

| Section | Source file |
|---------|------------|
| [1 — Project README](#1--project-readme) | `README.md` |
| [2 — Data Documentation](#2--data-documentation) | `data/README.md` |
| [3 — Knowledge-Base Documentation](#3--knowledge-base-documentation) | `knowledge_base/README.md` |

---

## 1 — Project README

> **Source:** [`README.md`](../README.md)

---

# quantum-bug-rag

A modular, research-grade Python repository for retrieval-augmented LLM-based bug detection and classification in Qiskit programs.  
Developed for **CSC 7135** — Quantum Software Testing and Analysis.

> **Status:** Repository scaffold. Core infrastructure is implemented and tested; full benchmark runs on Bugs4Q are in progress.

---

### Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Dataset Roles](#dataset-roles)
5. [Dataset Preparation](#dataset-preparation)
6. [Running Experiments](#running-experiments)
   - [Prompt-Only Baseline](#prompt-only-baseline)
   - [RAG Pipeline](#rag-pipeline)
   - [Static Baseline](#static-baseline)
   - [Subset Evaluation](#subset-evaluation)
7. [Outputs and Logs](#outputs-and-logs)
8. [Knowledge Base](#knowledge-base)
9. [Running Tests](#running-tests)
10. [Extension Points](#extension-points)
11. [Leakage Control and Evaluation Splits](#leakage-control-and-evaluation-splits)
12. [License](#license)

---

### Project Overview

We investigate whether retrieval-augmented generation (RAG) improves LLM-based detection and classification of bugs in Qiskit quantum programs.  
We compare three experimental modes:

| Mode | Description |
|------|-------------|
| `static` | Lightweight rule-based static baseline (pattern-matching heuristics; **not** a full LintQ implementation) |
| `prompt_only` | LLM inference with the buggy code snippet only |
| `rag` | LLM inference augmented with retrieved bug-pattern and taxonomy context from the knowledge base |

Each mode produces structured diagnostics:

```json
{
  "bug_likelihood": 0.87,
  "taxonomy_class": "incorrect_operator",
  "suspected_location": "circuit.cx(0, 1)",
  "justification": "..."
}
```

---

### Repository Structure

```
quantum-bug-rag/
├── README.md
├── LICENSE
├── requirements.txt
├── config.example.yaml          # Copy to config.yaml; never commit secrets
├── src/                         # Core library
│   ├── __init__.py
│   ├── schemas.py               # Pydantic models for inputs and outputs
│   ├── utils.py                 # Shared utilities (config loading, logging)
│   ├── dataset_loader.py        # Bugs4Q loader and normalisation
│   ├── benchmark_runner.py      # Orchestrates a full evaluation run
│   ├── retriever.py             # Local TF-IDF / cosine retriever
│   ├── prompt_builder.py        # Constructs LLM prompts for each mode
│   ├── llm_client.py            # LLM abstraction (mock / OpenAI / Gemini)
│   ├── baselines.py             # Static rule-based baseline
│   ├── evaluate.py              # Metrics computation
│   └── knowledge_ingest.py      # Knowledge-base loading and indexing
├── scripts/                     # CLI entry points
│   ├── prepare_bugs4q.py        # Fetch and normalise Bugs4Q
│   ├── prepare_bugsqcp_kb.py    # Ingest Bugs-QCP into knowledge base
│   ├── run_prompt_only.py
│   ├── run_rag.py
│   ├── run_static_baseline.py
│   └── run_subset_eval.py
├── knowledge_base/
│   ├── bug_patterns.json        # Starter bug-pattern entries
│   ├── taxonomy.json            # Starter taxonomy
│   └── README.md
├── data/
│   ├── README.md                # How to obtain and place datasets
│   └── bugs4q/                  # Populated by prepare_bugs4q.py
├── outputs/
│   └── README.md                # Output format documentation
├── docs/
│   └── methodology.md           # Research methodology notes
└── tests/
    ├── test_schemas.py
    ├── test_retriever.py
    ├── test_llm_client.py
    ├── test_baselines.py
    └── test_pipeline_integration.py
```

---

### Installation

We recommend a dedicated virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy and edit the configuration file:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml to choose your LLM backend (default: mock)
```

For OpenAI or Gemini backends, set the appropriate environment variable:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

---

### Dataset Roles

> **Important:** We use two distinct external datasets with clearly separated roles.

| Dataset | Role |
|---------|------|
| **Bugs4Q** ([GitHub](https://github.com/Z-928/Bugs4Q)) | **Primary benchmark and evaluation dataset.** Bugs4Q is an executable benchmark of real Qiskit bugs with ground-truth labels. All reported evaluation metrics are computed on this dataset. |
| **Bugs-QCP** (Zenodo 5834281) | **Secondary corpus for knowledge-base enrichment.** We use Bugs-QCP only for taxonomy grounding, bug-pattern retrieval context, and knowledge-base construction. It is **not** used as an evaluation dataset. |

> ⚠️ **Synthetic smoke-test data** (generated by `prepare_bugs4q.py --smoke-test`) is **for infrastructure validation only**. It must **never** be used when reporting benchmark results.

---

### Dataset Preparation

Neither Bugs4Q nor Bugs-QCP is bundled in this repository. Follow the steps below to prepare each dataset locally.

#### Bugs4Q

```bash
# Clone the Bugs4Q repository and normalise it into data/bugs4q/
python scripts/prepare_bugs4q.py --output-dir data/bugs4q/

# Smoke-test mode (generates synthetic samples; for pipeline testing only)
python scripts/prepare_bugs4q.py --smoke-test --output-dir data/bugs4q/
```

See `data/README.md` for full instructions.

#### Bugs-QCP Knowledge-Base Enrichment

```bash
# After downloading the Bugs-QCP archive from Zenodo 5834281:
python scripts/prepare_bugsqcp_kb.py \
    --input-dir /path/to/bugsqcp/ \
    --output-dir knowledge_base/
```

See `knowledge_base/README.md` for the expected JSON schema.

---

### Running Experiments

All scripts accept `--config` to point at a custom YAML configuration file (default: `config.yaml`).

#### Prompt-Only Baseline

```bash
python scripts/run_prompt_only.py \
    --data-dir data/bugs4q/ \
    --output-dir outputs/ \
    --config config.yaml
```

#### RAG Pipeline

```bash
python scripts/run_rag.py \
    --data-dir data/bugs4q/ \
    --kb-dir knowledge_base/ \
    --output-dir outputs/ \
    --config config.yaml
```

#### Static Baseline

```bash
python scripts/run_static_baseline.py \
    --data-dir data/bugs4q/ \
    --output-dir outputs/
```

> Note: The static baseline is a lightweight placeholder using rule-based heuristics. It is **not** a full re-implementation of LintQ or any other published quantum linter.

#### Subset Evaluation

For quick iteration on a small subset of Bugs4Q samples:

```bash
python scripts/run_subset_eval.py \
    --data-dir data/bugs4q/ \
    --subset-size 20 \
    --config config.yaml
```

---

### Outputs and Logs

Each run writes a timestamped JSONL file to `outputs/`. Each line is a structured diagnostic:

```json
{
  "sample_id": "bugs4q_042",
  "mode": "rag",
  "bug_likelihood": 0.91,
  "taxonomy_class": "incorrect_qubit_mapping",
  "suspected_location": "qc.cx(control, target)",
  "justification": "...",
  "ground_truth": "incorrect_qubit_mapping",
  "correct": true
}
```

A summary metrics file (`metrics_<run_id>.json`) is written alongside the JSONL file.

See `outputs/README.md` for the complete output schema.

---

### Knowledge Base

The `knowledge_base/` directory ships with starter content:

- `bug_patterns.json` — hand-curated and Bugs-QCP-derived bug-pattern entries specific to Qiskit and quantum programs.
- `taxonomy.json` — a starter taxonomy for quantum bug classification.

Run `prepare_bugsqcp_kb.py` to enrich the knowledge base with additional entries derived from the Bugs-QCP corpus.

---

### Running Tests

```bash
pytest tests/ -v
```

To run with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests use the mock LLM backend and in-memory synthetic data; no external APIs or datasets are required.

---

### Leakage Control and Evaluation Splits

When running full experiments with Bugs4Q, we apply strict train/eval split discipline:

- The knowledge base is constructed from the **Bugs-QCP corpus only** (or the Bugs4Q training split, if applicable), never from the evaluation split.
- Retrieved bug-pattern context in RAG mode must not expose evaluation-set ground-truth labels to the model.
- All reported metrics are computed on a held-out evaluation split.

See `docs/methodology.md` for details.

---

### Extension Points

| Component | How to extend |
|-----------|--------------|
| LLM backend | Add a new class in `src/llm_client.py` implementing `BaseLLMClient` |
| Retriever | Replace or augment `src/retriever.py` with dense retrieval (e.g., Sentence-Transformers + FAISS) |
| Dataset | Add a loader in `src/dataset_loader.py` following the `BugSample` schema |
| Taxonomy | Edit `knowledge_base/taxonomy.json` and re-index with `knowledge_ingest.py` |
| Metrics | Add new metric functions to `src/evaluate.py` |

---

### License

MIT — see [LICENSE](../LICENSE).

---

## 2 — Data Documentation

> **Source:** [`data/README.md`](../data/README.md)

---

# Data Directory

This directory stores locally prepared datasets.  **No raw external dataset
files are committed to this repository.**

## Subdirectories

| Path | Contents |
|------|----------|
| `bugs4q/` | Normalised Bugs4Q samples (populated by `scripts/prepare_bugs4q.py`). |

## Dataset Roles

| Dataset | Role |
|---------|------|
| **Bugs4Q** | Primary benchmark and evaluation dataset. All reported metrics are computed on Bugs4Q. |
| **Bugs-QCP** | Secondary corpus for knowledge-base enrichment and taxonomy grounding only. Not used as an evaluation dataset. |

## Obtaining Bugs4Q

Bugs4Q is an executable benchmark of real Qiskit bugs maintained at
<https://github.com/Z-928/Bugs4Q>.  To prepare it:

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q/
```

This clones the upstream repository and converts its contents into the
`BugSample` JSON schema under `data/bugs4q/`.

### Smoke-Test Mode

For pipeline infrastructure validation only (no real data required):

```bash
python scripts/prepare_bugs4q.py --smoke-test --output-dir data/bugs4q/
```

> ⚠️ Synthetic smoke-test samples must **never** be used when reporting
> benchmark results.  They exist solely to validate that the pipeline runs
> end-to-end without errors.

## Obtaining Bugs-QCP

Bugs-QCP is available from Zenodo at <https://zenodo.org/records/5834281>.
Download the archive, extract it, then run:

```bash
python scripts/prepare_bugsqcp_kb.py \
    --input-dir /path/to/bugsqcp/ \
    --output-dir knowledge_base/
```

This does **not** copy raw Bugs-QCP files into this repository; it only
enriches the knowledge base with normalised pattern entries.

## Split Discipline

When running full Bugs4Q evaluations, maintain a strict training/evaluation
split.  The knowledge base must not contain any samples from the evaluation
split.  See `docs/methodology.md` for details.

---

## 3 — Knowledge-Base Documentation

> **Source:** [`knowledge_base/README.md`](../knowledge_base/README.md)

---

# Knowledge Base

This directory contains the local knowledge base used by the RAG pipeline for
bug-pattern retrieval and taxonomy grounding.

## Files

| File | Description |
|------|-------------|
| `bug_patterns.json` | Starter bug-pattern entries derived from manual curation and Bugs-QCP-derived patterns. |
| `taxonomy.json` | Starter taxonomy for quantum bug classification. |

## Schema

### `bug_patterns.json`

A JSON array of objects matching the `BugPattern` Pydantic model (`src/schemas.py`):

```json
{
  "pattern_id": "BP001",
  "name": "CNOT Self-Loop",
  "taxonomy_class": "incorrect_qubit_mapping",
  "description": "...",
  "example_code": "...",
  "fix_hint": "...",
  "source": "manual | bugsqcp | ...",
  "tags": ["cx", "qubit_mapping"]
}
```

### `taxonomy.json`

A JSON array of objects matching the `TaxonomyEntry` Pydantic model:

```json
{
  "class_id": "incorrect_operator",
  "name": "Incorrect Operator",
  "description": "...",
  "parent_class": null,
  "examples": ["..."]
}
```

## Enriching the Knowledge Base

To add Bugs-QCP-derived entries, download the Bugs-QCP archive from
[Zenodo 5834281](https://zenodo.org/records/5834281) and run:

```bash
python scripts/prepare_bugsqcp_kb.py \
    --input-dir /path/to/bugsqcp/ \
    --output-dir knowledge_base/
```

The script normalises each entry to the `BugPattern` schema and merges it into
`bug_patterns.json` without overwriting existing manually-curated entries.

## Leakage Note

The knowledge base must be constructed from the **Bugs-QCP corpus** and the
**Bugs4Q training split only**.  Evaluation-split Bugs4Q samples must never
appear in the knowledge base or be used to populate retrieved context during
evaluation.  See `docs/methodology.md` for details.
