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
- framework-aware score boosting;
- a hard BM25 floor after ranking;
- `top1_bm25_score` logging for abstention routing.

### Evaluation

`src/taxonomy_v6/evaluator.py` and `src/taxonomy_v6/analysis.py` run:

- `prompt_only`
- `rag`
- `hybrid` (RAG with Dev-tuned abstention to prompt-only)

Methodology constraints:

- each labelled dataset is split deterministically into 60% Dev and 40% Test;
- Dev is used only to tune the BM25 routing threshold `tau` and the temperature-scaling parameter `T`;
- Dev is also used to estimate the model prior `pi_hat(c)` as the empirical mean of raw class-score vectors;
- before final Test metrics are computed, scores are temperature-scaled and then prior-corrected with a smoothed floor `epsilon = 0.05`;
- all headline metrics are reported only on the strictly held-out Test split;
- prompt-only vs pure-RAG comparisons remain paired at the sample level.

Metrics include:

- accuracy and macro-F1;
- 95% bootstrap confidence intervals for accuracy and macro-F1;
- top-2 accuracy;
- macro precision / recall / F1;
- per-class F1;
- McNemar prompt-only vs pure-RAG on Test;
- ECE before and after Dev-fitted temperature scaling, with equal-frequency bin counts;
- abstention and grounding reporting for the hybrid system.

### Empirical Findings

- The Dev priors make the model bias visible: `incorrect_operator` dominates the average score mass, while rare classes such as `missing_barrier` can receive extremely small prior mass.
- Abstention routing based on `top1_bm25_score < tau` protects headline accuracy by avoiding weakly grounded RAG calls.
- Unsmoothed prior correction over-amplified rare classes and caused large Macro-F1 regressions.
- The current implementation therefore uses a smoothed prior floor of `epsilon = 0.05`, which stabilizes the correction numerically, though in the current `gpt-4o` held-out Test run it remains primarily a bias-diagnostic device rather than a net Macro-F1 win over the uncorrected baseline.

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
- In `taxonomy_v6`, tune `tau`, temperature `T`, and class-prior estimates `pi_hat(c)` on Dev only and reserve Test exclusively for final accuracy, F1, CI, McNemar, and ECE reporting.
- In `classical`, Bugs4Q is used only as an external holdout and not as a source for the classical-vs-quantum KB.
