# Documentation Reference

This file now serves as the canonical documentation index. The previous copy-and-paste mirror format caused drift once the notebook-refactored tracks were added.

## Core Documents

| Document | Purpose |
|---------|---------|
| [`README.md`](../README.md) | Top-level repository overview, setup, and run commands |
| [`methodology.md`](./methodology.md) | Concise methodological summary of all three tracks |
| [`methodology_and_architecture.md`](./methodology_and_architecture.md) | Detailed architecture and experiment-flow reference |

## Supporting Readmes

| Document | Purpose |
|---------|---------|
| [`data/README.md`](../data/README.md) | Prepared-data workflow and dataset roles |
| [`knowledge_base/README.md`](../knowledge_base/README.md) | Legacy JSON knowledge-base format and notes on the newer validated KB flow |
| [`outputs/README.md`](../outputs/README.md) | Output file conventions for each track |

## Code Entry Points

| Path | Purpose |
|------|---------|
| [`scripts/run_taxonomy_v6.py`](../scripts/run_taxonomy_v6.py) | Refactored CLI for `quantum_bug_detecttion_taxonomy.ipynb` |
| [`scripts/run_classical_vs_quantum.py`](../scripts/run_classical_vs_quantum.py) | Refactored CLI for `quantum-software-bug-detection-rag-project-v6_classical.ipynb` |
| [`scripts/run_rag.py`](../scripts/run_rag.py) | Legacy scaffold RAG run |
| [`scripts/run_prompt_only.py`](../scripts/run_prompt_only.py) | Legacy scaffold prompt-only run |
| [`scripts/run_static_baseline.py`](../scripts/run_static_baseline.py) | Legacy scaffold static baseline |

## Notebook Provenance

| Notebook | Refactored code |
|----------|-----------------|
| [`quantum_bug_detecttion_taxonomy.ipynb`](../quantum_bug_detecttion_taxonomy.ipynb) | `src/taxonomy_v6/` and `scripts/run_taxonomy_v6.py` |
| [`quantum-software-bug-detection-rag-project-v6_classical.ipynb`](../quantum-software-bug-detection-rag-project-v6_classical.ipynb) | `src/classical/` and `scripts/run_classical_vs_quantum.py` |

## Refactor Coverage Notes

Covered in the reusable code:

- loaders;
- KB builders;
- retrievers;
- prompt builders;
- LLM clients;
- evaluation and metric export.

Still notebook-only:

- plotting;
- one-off console inspection helpers;
- result archive packaging.
