# Methodology and Architecture Reference

> Last updated: May 3, 2026

## 1. System Overview

The repository has four distinct experiment paths:

| Track | Objective | Primary code |
|------|-----------|--------------|
| `legacy scaffold` | Evaluate prompt-only, RAG, and static baseline over prepared Bugs4Q data and a JSON KB | top-level `src/` modules |
| `study_i` | Fine-tune CodeBERT for binary `classical` vs `quantum` bug prediction | `src/study_i/` |
| `taxonomy_v6` | Forced-choice 5-class quantum bug classification using validated release-note knowledge | `src/taxonomy_v6/` |
| `classical` | Older LLM/RAG binary classifier for quantum-vs-classical bug distinction | `src/classical/` |

The paper-facing studies are `study_i` and `taxonomy_v6`. The notebook-refactored tracks were introduced to preserve the experimental logic of:

- `quantum-vs-classical-bug-prediction.ipynb`
- `quantum_bug_detecttion_taxonomy.ipynb`
- `quantum-software-bug-detection-rag-project-v6_classical.ipynb`

while making them scriptable and reusable.

## 2. Track Architecture

### 2.1 `study_i`

Files:

- `src/study_i/dataset.py`
- `src/study_i/training.py`
- `src/study_i/analysis.py`
- `src/study_i/plotting.py`
- `src/study_i/schemas.py`
- `scripts/run_study_i_codebert.py`

Execution flow:

1. Load an external labeled JSON dataset of `(name, description, code)` bug-report triples.
2. Filter to `bug_category ∈ {classical, quantum}`.
3. Concatenate the text triple into one CodeBERT input sequence.
4. Run 5-fold stratified cross-validation.
5. Repeat the 5-fold split across 5 random seeds for 25 fold-runs total.
6. Inside each fold:
   - minority-oversample the training set
   - compute inverse-frequency class weights
   - fit `microsoft/codebert-base` with weighted cross-entropy and label smoothing
   - hold out a 10% stratified validation split for early stopping on macro-F1
7. Aggregate per-fold metrics, pooled predictions, and publication-ready figures.

### 2.2 `taxonomy_v6`

Files:

- `src/taxonomy_v6/dataset.py`
- `src/taxonomy_v6/kb.py`
- `src/taxonomy_v6/retriever.py`
- `src/taxonomy_v6/prompts.py`
- `src/taxonomy_v6/llm.py`
- `src/taxonomy_v6/analysis.py`
- `src/taxonomy_v6/evaluator.py`
- `scripts/run_taxonomy_v6.py`

Execution flow:

1. Load Bugs4Q and quantum-only Bugs-QCP samples.
2. Build a validated quantum KB from release notes and LintQ summaries.
3. Split each labelled dataset into a deterministic 60% Dev / 40% Test partition.
4. On Dev, run prompt-only and pure-RAG baselines, then tune the abstention threshold `tau`, fit temperature-scaling parameter `T`, and estimate the model prior `pi_hat(c)` from average raw class-score vectors.
5. On Test, run prompt-only and pure-RAG baselines, derive the abstention-routed hybrid system with the frozen `tau`, apply temperature scaling, then apply smoothed Bayesian prior correction with `epsilon = 0.05` before final argmax classification.
6. Write split-specific diagnostics, per-mode Test metrics, and a summary with confidence intervals, McNemar, and ECE.

### 2.3 `classical`

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

### 2.4 Legacy Scaffold

The original top-level `src/` modules remain for the project’s earlier prepared-data workflow. They use `data/bugs4q/` and `knowledge_base/` rather than raw upstream repository clones.

## 3. Data Handling

### 3.1 `study_i` Data Handling

`load_labeled_bug_reports()`:

- reads a JSON list from a user-provided `--data-path`;
- keeps only `classical` / `quantum` labels;
- preserves `name`, `description`, `example_code` or `code`, and metadata;
- assigns a stable `sample_id`.

`to_training_arrays()`:

- concatenates `(name, description, code)` with newline separators;
- maps labels to `{classical: 0, quantum: 1}` arrays.

### 3.2 `taxonomy_v6` Data Handling

`build_bugs4q()`:

- scans the upstream Bugs4Q repository for buggy Python files;
- parses the upstream README table;
- maps raw type strings into the five forced taxonomy classes.

`build_bugsqcp()`:

- indexes Bugs-QCP minimal bugfix folders;
- reconstructs focused buggy snippets from unified diffs;
- optionally restricts to quantum-labelled entries.

### 3.3 `classical` Data Handling

`build_bqcp()`:

- reads `annotation_bugs.csv`;
- resolves each row to a `minimal_bugfixes` folder;
- concatenates source/build/script files from `before/`.

`build_bugs4q()`:

- finds buggy files in Bugs4Q;
- assigns every sample `ground_truth="quantum"`;
- uses the set only for purity analysis, not balanced binary training labels.

## 4. Knowledge Bases

### 4.1 `study_i`

Study I has no external retrieval knowledge base. It is a supervised text/code
fine-tuning track rather than a prompt-based RAG system.

### 4.2 `taxonomy_v6` Validated KB

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

### 4.3 `classical` Symmetric KB

Quantum side:

- Qiskit release-note YAML files;
- Qiskit Aer release-note YAML files;
- PennyLane changelog entries.

Classical side:

- CPython `Misc/NEWS.d`;
- NumPy release notes.

The two halves are size-matched by proportional downsampling. This is part of the experiment design, not just an implementation detail.

## 5. Retrieval

### 5.1 `study_i`

Study I has no retrieval stage. The model sees only the concatenated
`(name, description, code)` input sequence.

### 5.2 `taxonomy_v6`

Retriever properties:

- BM25 over KB text;
- framework detection for `qiskit`, `pennylane`, `cirq`, `qsharp`, or `other`;
- 1.5x score boost for matching-framework KB entries;
- raw top-K ranking with a hard BM25 score floor;
- `top1_bm25_score` recorded for abstention routing.

### 5.3 `classical`

Retriever properties:

- `BM25Retriever` for a single pool;
- `BalancedRetriever` for separate quantum/classical pools.

This supports the comparison between:

- `biased_rag`: retrieval from a quantum-only KB;
- `balanced_rag`: retrieval from equal-sized quantum and classical KB halves.

## 6. Model Interaction

The repository now contains both supervised-transformer and prompt-based tracks.

Study I:

- fine-tunes `microsoft/codebert-base` directly with Hugging Face Trainer;
- uses no external prompts or LLM API;
- predicts a binary label and calibrated class probabilities from the classifier head.

Prompt-based tracks (`taxonomy_v6` and `classical`):

- keep lightweight per-track prompt builders and LLM clients instead of reusing the older project-wide abstractions;
- have different response schemas and retrieval conditions.

Each track supports:

- Study I: a fully local transformer fine-tuning loop;
- `taxonomy_v6` / `classical`: `--mock` for offline deterministic smoke runs and OpenAI-backed runs via `OPENAI_API_KEY`.

For `taxonomy_v6`, the OpenAI request now uses Structured Outputs via
`response_format.type = "json_schema"` with strict schema enforcement.
For RAG calls, the `evidence_ids` field is constrained at request time to the
retrieved pattern IDs only. If strict parsing fails, the client retries once at
a slightly higher temperature and then falls back to prompt-only for that
sample.

Post-hoc prediction correction:

- estimate `pi_hat(c)` on Dev from the mean raw class-score vector;
- apply temperature scaling on Test using Dev-fitted `T`;
- divide each temperature-scaled class score by `max(pi_hat(c), 0.05)`;
- renormalize and take the final argmax.

## 7. Metrics And Outputs

### 7.1 `study_i`

Per-run outputs:

- `summary.json`
- `per_fold.csv`
- `epoch_logs.json`
- `fig1_confusion_matrix.png`
- `fig2_fold_distribution.png`
- `fig3_roc_curve.png`
- `fig4_learning_curves.png`
- `fig5_summary_panel.png`

Metrics:

- mean and standard deviation of accuracy across 25 fold-runs;
- mean and standard deviation of macro-F1 across 25 fold-runs;
- mean and standard deviation of weighted F1 across 25 fold-runs;
- mean and standard deviation of ROC-AUC across 25 fold-runs;
- pooled classification report across all fold predictions;
- pooled confusion matrix.

Observed behavior from the executed notebook:

- `233` labeled samples (`134` classical, `99` quantum)
- repeated-CV accuracy `0.767 ± 0.057`
- repeated-CV macro-F1 `0.763 ± 0.056`
- repeated-CV ROC-AUC `0.855 ± 0.044`
- pooled per-class F1 of `0.7875` for `classical` and `0.7410` for `quantum`

### 7.2 `taxonomy_v6`

Per-mode outputs:

- `diagnostics_<dataset>_dev_<mode>.jsonl`
- `diagnostics_<dataset>_test_<mode>.jsonl`
- `metrics_<dataset>_<mode>.json`

Aggregate output:

- `summary.json`

Metrics:

- accuracy and macro-F1 with paired bootstrap 95% CIs;
- top-2 accuracy;
- macro precision / recall / F1;
- per-class F1;
- McNemar prompt-only vs pure-RAG on Test only;
- pre/post temperature-scaled ECE on Test with equal-frequency 10-bin counts;
- abstention and grounding diagnostics for the hybrid system;
- label and prediction distributions.

Observed behavior in the current `gpt-4o` study:

- abstention routing improves robustness by refusing weak BM25 matches instead of forcing low-quality RAG evidence into every sample;
- Dev priors confirm strong majority-class bias toward `incorrect_operator`;
- unsmoothed prior correction was numerically unstable on rare classes such as `missing_barrier`;
- the smoothed correction with `epsilon = 0.05` stabilizes that failure mode, but does not currently beat the uncorrected Test Macro-F1 baseline.

### 7.3 `classical`

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
