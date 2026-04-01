# Methodology and Architecture Reference

> **Course**: CSC 7135 – Software Testing & Quality Assurance  
> **Project**: Quantum Software Bug Detection via Retrieval-Augmented Generation  
> **Last Updated**: April 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dataset Handling](#2-dataset-handling)
   - 2.1 [Bugs4Q Benchmark](#21-bugs4q-benchmark)
   - 2.2 [Label Extraction and Taxonomy Mapping](#22-label-extraction-and-taxonomy-mapping)
   - 2.3 [Leakage Control](#23-leakage-control)
3. [The Knowledge Base](#3-the-knowledge-base)
   - 3.1 [Bugs-QCP Ingestion](#31-bugs-qcp-ingestion)
   - 3.2 [Pattern Schema and Taxonomy Alignment](#32-pattern-schema-and-taxonomy-alignment)
   - 3.3 [Rationale for Retaining the Full Corpus](#33-rationale-for-retaining-the-full-corpus)
4. [Algorithms and Baselines](#4-algorithms-and-baselines)
   - 4.1 [Static Baseline (Rule-Based)](#41-static-baseline-rule-based)
   - 4.2 [Prompt-Only Mode](#42-prompt-only-mode)
   - 4.3 [RAG Mode (TF-IDF + LLM)](#43-rag-mode-tf-idf--llm)
5. [Evaluation Metrics](#5-evaluation-metrics)
   - 5.1 [Taxonomy and String-Matching Policy](#51-taxonomy-and-string-matching-policy)
   - 5.2 [Aggregate Metrics](#52-aggregate-metrics)
   - 5.3 [Per-Class F1](#53-per-class-f1)
6. [Pipeline Execution Flow](#6-pipeline-execution-flow)
7. [Configuration and Reproducibility](#7-configuration-and-reproducibility)
8. [Limitations and Future Work](#8-limitations-and-future-work)

---

## 1. System Overview

We built an end-to-end evaluation pipeline that compares two LLM-based approaches for classifying bugs in quantum programs written with IBM's Qiskit framework:

1. **Prompt-Only**: The LLM receives only a raw Qiskit code snippet and a structured system prompt defining our bug taxonomy; it must classify the bug without any external context.
2. **RAG (Retrieval-Augmented Generation)**: Before the LLM is invoked, we retrieve the most relevant bug-pattern entries from a locally-indexed knowledge base (KB) using TF-IDF cosine similarity and inject them into the prompt as additional context.

A third mode, **Static Baseline**, applies lightweight regex-based heuristics as a non-LLM control condition. All three modes share the same evaluation harness, output schema, and metric computation code, ensuring a fair comparison.

The pipeline is implemented in Python and structured as an installable package (`quantum-bug-rag`). Key components reside under `src/`:

| Module | Responsibility |
|--------|----------------|
| `dataset_loader.py` | Parse, validate, and load the Bugs4Q benchmark |
| `data_prep.py` | Clone, normalise, and prepare the raw Bugs4Q repository |
| `bugs4q_labels.py` | Extract ground-truth labels from the upstream README and map them to our taxonomy |
| `knowledge_ingest.py` | Load and index `bug_patterns.json` and `taxonomy.json` |
| `bugsqcp_ingest.py` | Ingest external Bugs-QCP CSV/JSON data into normalised `BugPattern` entries |
| `retriever.py` | TF-IDF vectoriser and cosine-similarity retrieval over the KB |
| `prompt_builder.py` | Construct chat-format prompts for each experimental mode |
| `llm_client.py` | Abstraction over OpenAI, Gemini, and GitHub Models backends |
| `baselines.py` | Regex-based static analyser baseline |
| `evaluate.py` | Compute accuracy, precision, recall, macro-F1, and per-class F1 |
| `benchmark_runner.py` | Orchestrate a complete evaluation run end-to-end |
| `benchmark_splits.py` | Deterministic train/dev/eval split generation with leakage control metadata |
| `schemas.py` | Pydantic v2 models for `BugSample`, `BugDiagnostic`, `BugPattern`, `TaxonomyEntry`, and `EvalSummary` |

---

## 2. Dataset Handling

### 2.1 Bugs4Q Benchmark

We use the [Bugs4Q](https://github.com/Z-928/Bugs4Q) dataset as our primary evaluation corpus. Bugs4Q is a curated collection of real-world buggy Qiskit programs sourced from GitHub issues, Stack Overflow posts, and the Qiskit repository itself. Each entry contains a buggy Python file and, in many cases, its corresponding fix.

Our preparation pipeline (`scripts/prepare_bugs4q.py` → `src/data_prep.py`) performs the following steps:

1. **Clone or update** the upstream Bugs4Q repository into `_tmp_bugs4q_clone/`.
2. **Walk the directory tree**, collecting every `.py` file that is not a known fix variant (we exclude filenames matching `fixed.py`, `fix.py`, `fixed_version.py`, `modify.py`, and `mod.py`).
3. **Normalise** each file into a `BugSample` Pydantic object carrying `sample_id`, `source`, `code`, and `ground_truth`.
4. **Extract labels** from the upstream `README.md` table (see Section 2.2).
5. **Write** the prepared dataset to `data/bugs4q/samples.real.jsonl` and register it in an `active_dataset.json` manifest.

After preparation, the dataset contains **105 total samples**, of which **45 carry ground-truth labels**. Only these 45 labelled samples participate in metric computation.

### 2.2 Label Extraction and Taxonomy Mapping

The upstream Bugs4Q repository does not store labels in a structured format. Instead, it provides a README table where each row links a buggy file path to a free-text "Type" description (e.g., `"parameter"`, `"output wrong"`, `"qr,qc"`).

We implemented a deterministic two-stage labelling process in `src/bugs4q_labels.py`:

1. **README parsing**: We extract `(buggy_path, upstream_type)` tuples by scanning the markdown table rows for hyperlink targets and their adjacent type cells.
2. **Taxonomy mapping**: Each upstream type string is mapped to one of our six canonical taxonomy classes via an explicit dictionary (`_UPSTREAM_TYPE_TO_TAXONOMY`). This dictionary contains 40+ manually curated entries covering every observed upstream label. Samples whose upstream type does not appear in the dictionary receive `ground_truth = None` and are excluded from evaluation.

Our six taxonomy classes are:

| Class ID | Description |
|----------|-------------|
| `incorrect_operator` | Wrong gate, rotation angle, or operator applied to a qubit |
| `incorrect_qubit_mapping` | Gate references the wrong qubit index or identical control/target |
| `missing_barrier` | Absent barrier instruction allowing spurious gate reordering |
| `wrong_initial_state` | Quantum register initialised to an incorrect state |
| `measurement_error` | Measurement operation incorrectly placed, targeting wrong qubit/cbit, or missing |
| `unknown` | Fallback when the bug cannot be confidently assigned |

The resulting label distribution across the 45 labelled samples is:

| Class | Count |
|-------|-------|
| `incorrect_operator` | 27 |
| `measurement_error` | 12 |
| `wrong_initial_state` | 5 |
| `incorrect_qubit_mapping` | 1 |

This distribution exhibits significant class imbalance, with `incorrect_operator` representing 60% of samples and `incorrect_qubit_mapping` represented by a single sample. We report both macro-averaged and per-class metrics to surface this imbalance transparently.

### 2.3 Leakage Control

We strictly enforce leakage control to ensure that no evaluation sample is present in or derivable from the retrieval context. Our policy is implemented at two levels:

1. **Corpus separation**: The evaluation dataset (Bugs4Q) and the knowledge base (Bugs-QCP) are drawn from entirely different upstream projects. No Bugs4Q code snippet appears in the Bugs-QCP corpus.
2. **Split-level safeguards**: Our split generation utility (`src/benchmark_splits.py`) produces deterministic train/dev/eval partitions with a `leakage_control` metadata block explicitly recording the policy: *"Do not use eval split samples as retrieval-source candidates."* The split manifest is persisted to `data/bugs4q/splits.json` for auditability.

---

## 3. The Knowledge Base

### 3.1 Bugs-QCP Ingestion

Our knowledge base is populated from the **Bugs-QCP** (Bugs in Quantum Computing Platforms) dataset, an independent corpus of annotated bug patterns drawn from real defects in quantum computing frameworks. Ingestion is performed by `scripts/prepare_bugsqcp_kb.py` → `src/bugsqcp_ingest.py`.

The ingestion pipeline:

1. **Discovers** all CSV and JSON files under `external_data/bugsqcp/`.
2. **Normalises** each record by probing for known column aliases (e.g., `bug_id` → `pattern_id`, `bug_description` → `description`, `bug_pattern` → `taxonomy_class`).
3. **Maps upstream type labels** to our taxonomy via `_TYPE_ALIAS_MAP`, a manually curated dictionary of 30+ alias entries covering the Bugs-QCP vocabulary (e.g., `"barrier related"` → `missing_barrier`, `"overlooked qubit order"` → `incorrect_qubit_mapping`).
4. **Generates stable `pattern_id`** values using SHA-256 hashing over the source filename and row content, ensuring idempotent re-ingestion.
5. **Writes** the final knowledge base to `knowledge_base/bug_patterns.json` and enriches `knowledge_base/taxonomy.json` with example entries drawn from the ingested patterns.

### 3.2 Pattern Schema and Taxonomy Alignment

Each knowledge-base entry conforms to the `BugPattern` Pydantic model:

```
BugPattern:
  pattern_id   : str       # Unique identifier (SHA-256 derived)
  name         : str       # Short human-readable name
  taxonomy_class : str     # One of our six taxonomy classes
  description  : str       # Detailed natural-language description
  example_code : str       # Illustrative code snippet (may be empty)
  fix_hint     : str       # Guidance on remediation
  source       : str       # Origin corpus identifier (e.g., "bugsqcp")
  tags         : list[str] # Keyword tags for retrieval enrichment
```

All 233 patterns are aligned to the same six-class taxonomy used for evaluation. The taxonomy itself is stored in `knowledge_base/taxonomy.json`, where each `TaxonomyEntry` carries a `class_id`, `name`, `description`, and illustrative `examples` array.

The KB class distribution is:

| Class | Pattern Count |
|-------|--------------|
| `unknown` | 128 |
| `incorrect_operator` | 75 |
| `incorrect_qubit_mapping` | 20 |
| `wrong_initial_state` | 4 |
| `missing_barrier` | 3 |
| `measurement_error` | 3 |

### 3.3 Rationale for Retaining the Full Corpus

We deliberately retain all 233 patterns—including the 128 `unknown`-class entries—rather than subsetting the KB to only taxonomy-aligned patterns. This design decision reflects our intent to simulate real-world retrieval noise:

- **In production**, a retrieval corpus will inevitably contain entries that are partially relevant, ambiguously labelled, or outside the target taxonomy. By keeping the full Bugs-QCP corpus, we test the LLM's ability to integrate useful signal while ignoring irrelevant context.
- **Subsetting would inflate accuracy artificially**: If we removed all `unknown` patterns, the retriever would be more likely to surface taxonomy-aligned patterns, giving the RAG pipeline an unrealistic advantage that would not generalise.
- **Transparency**: We report the KB composition openly (Section 3.2) so that readers can assess the retrieval difficulty.

---

## 4. Algorithms and Baselines

### 4.1 Static Baseline (Rule-Based)

The static baseline (`src/baselines.py`) applies five hand-crafted regex rules over the raw code text. Each rule maps a syntactic pattern to a taxonomy class:

| Rule | Pattern | Taxonomy Class |
|------|---------|----------------|
| R01 | `.measure(` call detected | `measurement_error` |
| R02 | `.cx(q, q)` — identical control and target | `incorrect_qubit_mapping` |
| R03 | `QuantumCircuit(1)` — single-qubit circuit | `wrong_initial_state` |
| R04 | `.h()` without subsequent `.barrier` | `missing_barrier` |
| R05 | `.rx/.ry/.rz` with hard-coded numeric angle | `incorrect_operator` |

When multiple rules match, the baseline selects the rule with the highest `likelihood` score. When no rule matches, it assigns `no_bug_detected`. This baseline is intentionally simplistic; we include it as a non-LLM control condition rather than as a faithful re-implementation of any published quantum static analyser.

### 4.2 Prompt-Only Mode

In prompt-only mode, the LLM receives a two-message chat sequence constructed by `build_prompt_only()` in `src/prompt_builder.py`:

**System message**: A structured instruction that:
- Establishes the LLM's role as a quantum software engineering expert.
- Defines the required JSON output schema (`bug_likelihood`, `taxonomy_class`, `suspected_location`, `justification`).
- Enumerates the six valid taxonomy classes with the directive that it must select exactly one.
- Specifies fallback behaviour: if uncertain, use `bug_likelihood = 0.5` and `taxonomy_class = "unknown"`.

**User message**: Contains only the raw Qiskit code snippet, wrapped in a Python code fence, with the instruction: *"Please analyse the following Qiskit code snippet and produce a bug diagnostic."*

There is no retrieved context, no few-shot examples, and no chain-of-thought scaffolding. The model must rely entirely on its parametric knowledge to classify the bug.

### 4.3 RAG Mode (TF-IDF + LLM)

The RAG pipeline augments the prompt with retrieved knowledge-base context. It proceeds in three stages:

#### Stage 1: Index Construction

At startup, `BugPatternRetriever.__init__()` builds a TF-IDF index over the 233 KB patterns. Each pattern is serialised into a retrieval document by `_pattern_to_text()`, which concatenates and duplicates high-signal fields:

```
document = [name, name, taxonomy_class, taxonomy_class,
            description, fix_hint, tags, tags, example_code]
```

The field duplication is a lightweight term-frequency boosting strategy: by repeating the `name` and `taxonomy_class`, we ensure these short but discriminative tokens receive higher TF-IDF weight relative to the longer `description` and `example_code` fields.

The vectoriser is configured as:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `analyzer` | `"word"` | Word-level tokenisation |
| `ngram_range` | `(1, 2)` | Unigrams and bigrams capture multi-word gate names (e.g., "cx gate") |
| `max_features` | `8192` | Feature budget sufficient for the 233-document corpus |
| `sublinear_tf` | `True` | Applies $1 + \log(\text{tf})$ scaling to dampen high-frequency terms |

#### Stage 2: Retrieval

Given a code snippet, `BugPatternRetriever.retrieve()`:

1. Transforms the snippet into the same TF-IDF feature space.
2. Computes cosine similarity against all 233 corpus vectors.
3. Returns the top-$k$ patterns (default $k = 5$) with similarity > 0.0.

The retrieval is purely lexical (sparse). We do not apply a minimum similarity threshold; any pattern with non-zero cosine overlap is eligible.

#### Stage 3: Prompt Augmentation

`build_rag_prompt()` constructs the user message by prepending the retrieved context before the code snippet:

```
## Retrieved Bug Patterns

### Pattern 1: <name> [<taxonomy_class>]
<description>
Fix hint: <fix_hint>

### Pattern 2: ...

## Relevant Taxonomy Classes
- **<class_id>**: <description>

## Code Snippet to Analyse

```python
<code>
```​

Using the retrieved context above, produce a structured bug diagnostic.
```

The system message is identical to the prompt-only mode. The LLM receives both the code and the retrieved patterns in a single user turn. Additionally, for each unique taxonomy class among the retrieved patterns, the corresponding `TaxonomyEntry` from `taxonomy.json` is appended, giving the LLM the canonical definition of each class that appeared in the retrieval results.

---

## 5. Evaluation Metrics

### 5.1 Taxonomy and String-Matching Policy

We evaluate predictions using **strict string equality** between the predicted `taxonomy_class` and the `ground_truth` label. There is no partial credit, no semantic similarity scoring, and no hierarchical class distance. A prediction is correct if and only if:

```python
diagnostic.taxonomy_class == diagnostic.ground_truth
```

This design choice prioritises simplicity and reproducibility. We acknowledge that it penalises near-miss predictions (e.g., predicting `incorrect_operator` for a sample labelled `incorrect_qubit_mapping` scores identically to predicting `unknown`), but it avoids the subjectivity inherent in soft-match schemes.

### 5.2 Aggregate Metrics

We compute four aggregate metrics using scikit-learn, applied only to the subset of diagnostics with non-null `ground_truth`:

| Metric | scikit-learn Function | Averaging | Description |
|--------|-----------------------|-----------|-------------|
| **Accuracy** | `accuracy_score(y_true, y_pred)` | — | Fraction of predictions that exactly match the ground truth |
| **Precision (macro)** | `precision_score(..., average="macro", zero_division=0)` | Macro | Unweighted mean of per-class precision; treats all classes equally regardless of support |
| **Recall (macro)** | `recall_score(..., average="macro", zero_division=0)` | Macro | Unweighted mean of per-class recall |
| **F1 (macro)** | `f1_score(..., average="macro", zero_division=0)` | Macro | Harmonic mean of macro-precision and macro-recall |

We use **macro averaging** to ensure that minority classes (e.g., `incorrect_qubit_mapping` with 1 sample) receive equal weight in the aggregate score. This is intentional: we want our metrics to penalise a pipeline that performs well only on the dominant class. The `zero_division=0` parameter ensures that classes with zero true positives contribute 0.0 rather than raising an exception, which is appropriate given the extreme class imbalance.

### 5.3 Per-Class F1

In addition to macro averages, we compute per-class F1 scores using:

```python
f1_score(y_true, y_pred, labels=sorted(set(y_true)), average=None, zero_division=0)
```

The `labels` parameter is set to the sorted set of ground-truth classes (not predicted classes), ensuring that we report F1 for every true class even if the pipeline never predicts it. Per-class F1 scores are stored in the `EvalSummary.per_class_f1` dictionary and included in all output artifacts.

---

## 6. Pipeline Execution Flow

A complete evaluation run proceeds as follows:

```
┌─────────────────────────────────────────────────────┐
│  1. Load Configuration (config.yaml)                │
│     - LLM backend, model, temperature, max_tokens   │
│     - Retrieval top_k, paths                        │
├─────────────────────────────────────────────────────┤
│  2. Load Dataset                                    │
│     - Read data/bugs4q/samples.real.jsonl            │
│     - Validate schema, check for duplicates          │
│     - Filter to labelled-only (if --labelled-only)   │
├─────────────────────────────────────────────────────┤
│  3. Initialise Mode Components                      │
│     ┌── static:     StaticBaseline()                │
│     ├── prompt_only: LLMClient(model)               │
│     └── rag:         LLMClient(model)               │
│                      KnowledgeBase(kb_dir)           │
│                      BugPatternRetriever(patterns)   │
├─────────────────────────────────────────────────────┤
│  4. Process Each Sample                             │
│     For sample in samples:                          │
│       a. [rag only] Retrieve top-k patterns          │
│       b. Build prompt (prompt_only or rag)           │
│       c. Call LLM → parse JSON response              │
│       d. Construct BugDiagnostic                     │
│       e. Compute correctness (strict string match)   │
├─────────────────────────────────────────────────────┤
│  5. Compute Metrics                                 │
│     - Accuracy, Precision, Recall, F1 (macro)        │
│     - Per-class F1                                   │
├─────────────────────────────────────────────────────┤
│  6. Write Outputs                                   │
│     - diagnostics_{mode}_{run_id}_{timestamp}.jsonl  │
│     - metrics_{mode}_{run_id}_{timestamp}.json       │
│     - Console summary                                │
└─────────────────────────────────────────────────────┘
```

Each run is uniquely identified by a UUID-based `run_id` and a UTC timestamp, ensuring that repeated executions never overwrite prior results. All output artifacts are written to the `outputs/` directory.

---

## 7. Configuration and Reproducibility

All tuneable parameters are centralised in `config.yaml`:

```yaml
llm:
  backend: openai          # openai | gemini | github_models | mock
  openai:
    model: gpt-4o
    max_tokens: 1024
    temperature: 0.0       # Deterministic decoding

retrieval:
  top_k: 5                 # Number of KB patterns to retrieve
  similarity_metric: cosine

paths:
  knowledge_base: knowledge_base/
  bugs4q_dir: data/bugs4q/
  outputs_dir: outputs/
```

**Reproducibility guarantees**:

- **Deterministic LLM output**: We set `temperature = 0.0` for all production runs, requesting the model's most-likely token at each step. While this does not guarantee bitwise-identical outputs across API versions, it minimises run-to-run variance.
- **Deterministic splits**: The split generator uses a seeded `random.Random(42)` instance, producing identical partitions across runs.
- **No API key leakage**: API keys are read exclusively from environment variables (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GITHUB_TOKEN`); the configuration file never stores secrets.
- **Pinned dependencies**: `requirements.txt` specifies exact package versions for scikit-learn, pydantic, openai, and all transitive dependencies.

---

## 8. Limitations and Future Work

We acknowledge the following limitations of our current pipeline:

1. **Small evaluation set**: With only 45 labelled samples and extreme class imbalance (27 `incorrect_operator`, 1 `incorrect_qubit_mapping`), per-class metrics are volatile and macro-averaged scores are dominated by minority-class performance.

2. **Sparse retrieval only**: Our TF-IDF retriever is purely lexical. Dense retrieval (e.g., Sentence-Transformers + FAISS) could better capture semantic similarity between code patterns and natural-language KB descriptions.

3. **No similarity threshold**: The retriever includes any pattern with cosine similarity > 0.0, which may introduce irrelevant context that confuses the LLM, particularly for minority classes.

4. **KB class imbalance**: 128 of 233 KB patterns map to `unknown`, creating a retrieval bias toward non-informative patterns. Class-stratified retrieval or KB curation could mitigate this.

5. **No few-shot prompting**: The current prompt provides zero labelled examples. Adding class-stratified few-shot exemplars is a well-established technique for improving minority-class recall without fine-tuning.

6. **Single-pass inference**: We do not employ self-consistency (majority-vote) decoding, which has been shown to improve classification accuracy by 5–15 percentage points on similar tasks.

7. **Evaluation on a single model**: Our primary results use GPT-4o. Cross-model evaluation (GPT-4o-mini, Gemini 1.5 Pro) produced qualitatively different results, suggesting that findings may not generalise across LLM families without further investigation.

Future directions include hybrid BM25 + dense retrieval, few-shot prompt engineering with train-split exemplars, self-consistency decoding, and expansion of the labelled evaluation set through community annotation.

---

*This document was generated as part of the CSC 7135 course project. For questions or contributions, see the repository README.*
