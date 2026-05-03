# Methodology

## Overview

This repository now contains three related experiment tracks for bug analysis in quantum software:

1. `legacy scaffold`: prompt-only, RAG, and static-baseline experiments over prepared `Bugs4Q` data and a JSON knowledge base.
2. `taxonomy_v6`: forced-choice five-class taxonomy classification for quantum bugs.
3. `classical`: binary classification of bugs as `quantum` or `classical`.

The two new tracks were refactored from the project notebooks into `src/taxonomy_v6`, `src/classical`, and matching CLI scripts.

## Datasets

### Bugs4Q

`Bugs4Q` is the main benchmark repository of buggy Qiskit programs.

- In the legacy scaffold, it is the primary evaluation dataset after local preparation into `data/bugs4q/`.
- In `taxonomy_v6`, it is one of two evaluated labelled datasets.
- In `classical`, it acts as an external all-quantum holdout used to measure purity of binary predictions.

### Bugs-QCP

`Bugs-QCP` has two different roles depending on the track:

- In the legacy scaffold, it is used for knowledge-base enrichment only.
- In `taxonomy_v6`, quantum-labelled Bugs-QCP samples are evaluated alongside Bugs4Q.
- In `classical`, Bugs-QCP is the primary labelled binary dataset containing both `classical` and `quantum` bugs.

### Release-Note Source Repositories

The new tracks also depend on upstream source repositories:

- `taxonomy_v6`: Qiskit, Qiskit Aer, Qiskit Ignis, Qiskit IBM Runtime, PennyLane.
- `classical`: Qiskit, Qiskit Aer, PennyLane, CPython, NumPy.

These repositories are not vendored into this project. The CLIs expect them to be cloned under a user-provided `--work-dir`.

## `taxonomy_v6` Track

### Task

The input is assumed to be buggy. The model must assign exactly one of:

- `incorrect_operator`
- `incorrect_qubit_mapping`
- `missing_barrier`
- `wrong_initial_state`
- `measurement_error`

### Dataset Construction

- `src/taxonomy_v6/dataset.py` loads Bugs4Q directly from the upstream repo, parsing its README type column and mapping raw labels to the forced taxonomy.
- The same module loads Bugs-QCP by locating minimal bugfix folders and reconstructing focused buggy snippets from `before/` vs `after/` diffs.

### Knowledge Base

`src/taxonomy_v6/kb.py` builds a validated KB from:

- Qiskit-family YAML release notes, using `fixes`, `deprecations`, and `upgrade` sections;
- IBM Runtime RST release notes;
- PennyLane changelog entries;
- hand-coded LintQ rule summaries.

Entries are filtered for quality and mapped into taxonomy classes with a keyword-based classifier.

### Retrieval

`src/taxonomy_v6/retriever.py` uses BM25 with two extra steps:

- framework detection from the query code;
- diversification so the retrieved pool spans distinct taxonomy classes when possible.

### Evaluation

`src/taxonomy_v6/evaluator.py` runs:

- `prompt_only`
- `rag`

Metrics include:

- accuracy;
- top-2 accuracy;
- macro F1 / precision / recall;
- per-class F1;
- paired prompt-only vs RAG comparison counts.

## `classical` Track

### Task

Classify a buggy snapshot as:

- `quantum`: the defect is in quantum-specific logic;
- `classical`: the defect is in surrounding non-quantum software logic.

### Dataset Construction

- `src/classical/dataset.py` builds a Bugs-QCP evaluation set by concatenating relevant files under each bug's `before/` directory.
- The same module treats every Bugs4Q sample as `ground_truth="quantum"` for external holdout purity analysis.

### Symmetric Knowledge Base

`src/classical/kb.py` extracts:

- quantum entries from Qiskit-family and PennyLane release notes;
- classical entries from CPython and NumPy release notes.

The larger side is proportionally downsampled so the quantum and classical KB halves have equal size. This is critical because the experiment explicitly studies the confound introduced by an asymmetric retrieval pool.

### Retrieval Conditions

`src/classical/retriever.py` implements:

- `BM25Retriever`: BM25 over a single pool;
- `BalancedRetriever`: one BM25 per domain, then top results from each domain are concatenated.

The evaluated modes are:

- `prompt_only`
- `biased_rag`
- `balanced_rag`

### Evaluation

`src/classical/evaluator.py` emits per-sample diagnostics.
`src/classical/analysis.py` computes:

- accuracy;
- macro F1;
- bootstrap confidence intervals;
- per-class recall;
- confusion matrices;
- Brier score;
- reliability-bin data;
- Bugs4Q quantum-rate and mean-score purity summaries.

## Legacy Scaffold

The top-level `src/` modules remain available for the original prompt-only / RAG / static-baseline pipeline over prepared local data in `data/` and `knowledge_base/`.

This is still useful for the earlier project workflow, but it is now only one of three supported experiment paths.

## Notebook Parity

The new module/script refactors preserve the core computational flow from both notebooks:

- loaders;
- KB builders;
- retrievers;
- prompts;
- LLM client logic;
- evaluation and metric export.

Notebook-only items that remain outside the reusable CLI paths:

- plotting cells;
- console-format disagreement/error displays;
- one-off result archiving cells.

## Leakage And Split Discipline

- Do not mix evaluation targets into retrieval-source corpora.
- In the legacy scaffold, keep `Bugs4Q` evaluation samples out of the KB.
- In `taxonomy_v6`, the validated KB is built from release notes and LintQ summaries, not from the evaluated snippets themselves.
- In `classical`, Bugs4Q is used only as an external holdout and not as a source for the classical-vs-quantum KB.
