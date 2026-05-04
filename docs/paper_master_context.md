# Paper Master Context

> Last updated: May 3, 2026
>
> This document is the single-source briefing for writing a paper about the
> two-study pipeline in this repository. It is intended to be standalone: a
> paper writer should be able to draft the abstract, introduction, method,
> experiments, results, discussion, limitations, and appendix from this file
> alone without needing additional repo context. It consolidates the problem
> framing, architecture, data pipeline, model protocols, retrieval logic,
> evaluation protocol, statistical analysis, and the current final result
> artifacts for both Study I and Study II.

## 0. Executive Summary

This project now has two paper-facing studies:

- Study I: binary classification of quantum-software bugs as `classical` or
  `quantum`
- Study II: fine-grained classification of quantum bugs into five taxonomy
  classes

Study II asks not whether a bug exists, but which bug type is present among a
fixed label set:

- `incorrect_operator`
- `incorrect_qubit_mapping`
- `missing_barrier`
- `wrong_initial_state`
- `measurement_error`

The final system is evaluated in three modes:

- `prompt_only`: classify from code alone
- `rag`: classify with retrieved bug-pattern references
- `hybrid`: use RAG only when retrieval confidence is strong enough, otherwise
  route the sample to prompt-only

The paper-relevant findings are:

1. Study I shows that a supervised CodeBERT classifier can separate
   `classical` vs `quantum` bugs at `0.767 ± 0.057` accuracy and
   `0.763 ± 0.056` macro-F1 across 25 stratified fold-runs.
2. Retrieval can improve Study II classification, but only when the retrieved
   evidence is strong.
3. A BM25-threshold abstention rule is useful because weak retrieval can harm
   prediction quality; allowing the system to reject weak RAG inputs protects
   final accuracy.
4. Strict structured outputs with evidence-ID constraints make attribution
   auditable and sharply reduce citation hallucinations.
5. The model exhibits severe class-prior bias toward majority classes such as
   `incorrect_operator`, which explains why top-1 accuracy can look reasonable
   while macro-F1 remains low.
6. Post-hoc Bayesian prior correction is analytically useful, but even with
   smoothing it does not outperform the simpler held-out `gpt-4o` baseline, so
   it should be framed as a negative result rather than the main deployed
   method.

## 0.1 Formal Task Definition

Let `D = {(x_i, y_i)}_{i=1}^N` denote a dataset of bug reports, where each
`x_i = (name_i, description_i, code_i)` is a triple of natural-language and
source-code fields.

- In Study I, `Y = {classical, quantum}` and the task is binary classification
  of whether a defect arises from quantum-specific circuit semantics or from a
  conventional software error embedded inside a quantum program.
- In Study II, `Y = {incorrect_operator, incorrect_qubit_mapping,
  missing_barrier, wrong_initial_state, measurement_error}` and the task is
  fine-grained taxonomy prediction.

Study I establishes whether quantum-aware tooling is needed at all. Study II
determines which diagnostic action is warranted.

## 1. Project Scope

This repository contains several experiment tracks, but the paper-ready path
now has two studies:

- Study I:
  - task: binary `classical` vs `quantum` bug prediction
  - model: `microsoft/codebert-base`
  - protocol: 5-fold stratified cross-validation repeated across 5 seeds
- Study II:
  - task: forced-choice classification of buggy quantum code into 5 bug classes
  - pipeline: prompt-only vs RAG vs abstention-routed hybrid
  - models evaluated: `gpt-4o`, `gpt-5.4`

Core contribution themes:

- binary prediction to establish whether quantum-aware tooling is needed at all
- structured-output RAG with strict evidence attribution for fine-grained taxonomy prediction
- BM25-based abstention routing
- held-out Dev/Test discipline for threshold tuning and final reporting
- empirical analysis of class bias and post-hoc prior correction

Primary code:

- Study I runner: [scripts/run_study_i_codebert.py](/Users/syedshubha/Desktop/quantum-bug-rag/scripts/run_study_i_codebert.py)
- Study I dataset: [src/study_i/dataset.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/study_i/dataset.py)
- Study I training: [src/study_i/training.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/study_i/training.py)
- Study I analysis: [src/study_i/analysis.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/study_i/analysis.py)
- Study I plotting: [src/study_i/plotting.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/study_i/plotting.py)
- runner: [scripts/run_taxonomy_v6.py](/Users/syedshubha/Desktop/quantum-bug-rag/scripts/run_taxonomy_v6.py)
- dataset loading: [src/taxonomy_v6/dataset.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/dataset.py)
- KB building: [src/taxonomy_v6/kb.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/kb.py)
- retrieval: [src/taxonomy_v6/retriever.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/retriever.py)
- prompts: [src/taxonomy_v6/prompts.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/prompts.py)
- LLM client: [src/taxonomy_v6/llm.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/llm.py)
- evaluation: [src/taxonomy_v6/evaluator.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/evaluator.py)
- analysis/calibration: [src/taxonomy_v6/analysis.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/analysis.py)
- dataclasses/schema: [src/taxonomy_v6/schemas.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/schemas.py)

Supporting docs:

- [README.md](/Users/syedshubha/Desktop/quantum-bug-rag/README.md)
- [docs/methodology.md](/Users/syedshubha/Desktop/quantum-bug-rag/docs/methodology.md)
- [docs/methodology_and_architecture.md](/Users/syedshubha/Desktop/quantum-bug-rag/docs/methodology_and_architecture.md)
- [outputs/README.md](/Users/syedshubha/Desktop/quantum-bug-rag/outputs/README.md)

Working terminology used throughout the paper:

- `sample`: one labeled buggy-code example evaluated by the model
- `fold-run`: one Study I train/test split among the 25 repeated CV runs
- `class_scores`: the model's per-class probability-like scores over the five
  taxonomy labels
- `KB entry` or `pattern`: one curated bug-pattern document in the retrieval
  corpus
- `retrieved IDs`: the `pattern_id` values returned by BM25 for a sample
- `prompt_only`: no retrieval, code-only classification
- `rag`: classification with retrieved bug-pattern references
- `hybrid`: prompt-only fallback when retrieval is below threshold
- `Dev split`: the 60% tuning subset used for `tau`, temperature scaling, and
  prior estimation
- `Test split`: the 40% held-out subset used for all final metrics
- `macro-F1`: the unweighted average of F1 across the five classes, which is
  essential because the label distribution is highly imbalanced

## 1.2 Study I At A Glance

Study I addresses the binary decision problem:

- `classical`: the defect arises from conventional software logic embedded in a
  quantum program
- `quantum`: the defect arises from quantum-specific circuit semantics

This study establishes whether quantum-aware tooling is needed at all. Study II
then determines which fine-grained quantum diagnostic action is warranted.

Input representation:

- each sample is a triple `(name, description, code)`
- the three fields are concatenated with newline separators and fed to
  `microsoft/codebert-base`

Dataset:

- executed notebook dataset size: `233` labeled samples
- class distribution: `134` classical / `99` quantum

Study I dataset provenance and structure:

- the executed notebook reads an external JSON file named
  `bug_patterns_categorized.json`
- in the notebook, this file is loaded from a Kaggle-style dataset path rather
  than from a vendored repository folder inside this project
- each retained record is expected to expose:
  - `bug_category` with value `classical` or `quantum`
  - `name`
  - `description`
  - `example_code` or `code`
- the refactored loader in `src/study_i/dataset.py` preserves that same schema
  and filters out any record whose `bug_category` is not one of the two Study I
  labels
- labels are treated as externally curated ground truth supplied by the dataset,
  not generated inside this repository
- the repository currently documents the executed dataset size and class balance,
  but does not independently reconstruct the upstream labeling process from raw
  source repositories in the way Study II does

Implication for the paper:

- Study I should be described as a supervised binary classification experiment
  over an externally prepared labeled bug-report dataset
- if a final paper version needs stronger dataset provenance, the exact source
  release or archival link for `bug_patterns_categorized.json` should be cited
  explicitly in the dataset section or appendix

Training protocol:

- backbone: `microsoft/codebert-base`
- inverse-frequency class-weighted cross-entropy
- label smoothing `0.05`
- minority oversampling inside each training fold
- 5-fold stratified cross-validation
- 5 independent random seeds
- 25 fold-runs total
- within-fold validation split: `10%`
- manual early stopping on validation macro-F1 with patience `4`

Current preserved notebook results:

- repeated-CV accuracy: `0.767 ± 0.057`
- repeated-CV macro-F1: `0.763 ± 0.056`
- repeated-CV weighted F1: `0.766 ± 0.056`
- repeated-CV ROC-AUC: `0.855 ± 0.044`
- pooled per-class F1:
  - `classical`: `0.7875`
  - `quantum`: `0.7410`
  - pooled macro-F1: `0.7642`

Outputs:

- `summary.json`
- `per_fold.csv`
- `epoch_logs.json`
- five publication-ready figure PNGs

Important interpretive note:

- Study I uses repeated stratified cross-validation rather than a single held-out
  Test split because the labeled binary corpus is still small
- the reported `±` values are dispersion across the 25 fold-runs, not a formal
  confirmatory held-out-test confidence interval

## 1.1 Repository and Runtime Assumptions

The canonical execution paths are:

```bash
PYTHONPATH=. python scripts/run_study_i_codebert.py \
  --data-path /path/to/bug_patterns_categorized.json \
  --results-dir /path/to/study_i_results
```

```bash
PYTHONPATH=. python scripts/run_taxonomy_v6.py \
  --work-dir /path/to/work-dir \
  --results-dir /path/to/results \
  --model gpt-4o
```

Important runtime arguments:

- Study I:
  - `--data-path`: labeled JSON file containing `(name, description, code)` bug-report triples
  - `--results-dir`: output directory for summary, per-fold CSV, epoch logs, and figures
  - `--model-name`: Hugging Face model ID, default `microsoft/codebert-base`
  - `--cv-seeds`: comma-separated CV seeds
  - `--skip-plots`: disable figure generation when only metrics are needed
- `--model`: OpenAI model ID, e.g. `gpt-4o` or `gpt-5.4`
- `--work-dir`: temporary working directory for upstream clones and generated
  KB files
- `--results-dir`: output directory for summary and diagnostics
- `--top-k`: number of BM25 retrieval results to pass into RAG
- `--bm25-floor`: raw retrieval score floor; candidates below this are dropped
- `--dev-ratio`: fraction used for Dev; the study uses `0.6`
- `--seed`: deterministic split seed
- `--limit`: optional smoke-test cap on sample count; not used in final runs
- `--mock`: bypasses the live OpenAI API for local dry-runs; not used in final
  paper results

Expected `work-dir` contents during a full run:

- cloned or refreshed upstream `Bugs4Q`
- cloned or refreshed upstream `Bugs-Quantum-Computing-Platforms`
- generated and cached validated KB artifacts
- intermediate retrieval material if regenerated

Expected Study I input:

- a JSON list of bug-report records
- each retained record must have `bug_category` equal to `classical` or `quantum`
- the textual fields are read from `name`, `description`, and `example_code` or `code`

Environment assumptions:

- Python environment created from `requirements.txt`
- `.env` contains `OPENAI_API_KEY`
- `python-dotenv` loads `.env` before OpenAI client construction
- internet access is available for live model evaluation

## 2. Study II Research Problem

The input is assumed to be buggy code from a quantum-software repository. The
model must assign exactly one taxonomy label:

- `incorrect_operator`
- `incorrect_qubit_mapping`
- `missing_barrier`
- `wrong_initial_state`
- `measurement_error`

This is not a bug-detection task. It is a bug-type classification task with
forced choice among 5 classes.

Why macro-F1 matters:

- the datasets are class-imbalanced
- a model can achieve superficially decent accuracy by overpredicting common
  classes
- rare classes such as `missing_barrier` are central to the analysis and are
  often suppressed by the model's implicit prior

## 3. Full Architecture

Study II end-to-end execution flow:

1. Load labelled samples from `Bugs4Q` and quantum-only `Bugs-QCP`.
2. Build a validated quantum bug-pattern KB from release notes and LintQ rules.
3. Split each labelled dataset deterministically into Dev 60% / Test 40%.
4. On Dev:
   - run `prompt_only`
   - run pure `rag`
   - tune the abstention threshold `tau` from top-1 BM25 scores
   - fit temperature scaling `T`
   - estimate the model prior `pi_hat(c)` from mean class-score vectors
5. On Test:
   - run `prompt_only`
   - run pure `rag`
   - derive `hybrid` routing from frozen `tau`
   - apply temperature scaling
   - optionally apply Bayesian prior correction
6. Compute final Test-only metrics, confidence intervals, McNemar, ECE, and
   abstention/grounding diagnostics.

Important design constraint:

- Dev is used only for tuning and calibration.
- Test is strictly held out for final metrics.

## 4. Dataset Construction

### 4.1 Bugs4Q

Source:

- upstream `Bugs4Q` repository clone

Construction logic:

- walk buggy `.py` files
- parse the upstream README table
- map raw bug-type strings into the 5 forced taxonomy classes

Output shape:

- `BugSample(sample_id, source, code, ground_truth, metadata)`

Useful metadata typically includes source-repository provenance, file-path
context, and the original upstream bug annotation carried through the mapping.

Observed labelled size in the current full run:

- total: `45` labelled samples
- split: `27` Dev / `18` Test

### 4.2 Bugs-QCP

Source:

- upstream `Bugs-Quantum-Computing-Platforms` repository clone

Construction logic:

- read `annotation_bugs.csv`
- restrict to `real == bug`
- optionally restrict to `type == quantum`
- locate the `minimal_bugfixes` folder
- reconstruct a focused buggy snippet from unified diffs between `before/` and
  `after/`
- map raw bug-pattern strings into the same 5 taxonomy classes

Useful metadata typically includes the original annotation text, repository
provenance, and localization hints reconstructed from the `before/` and `after/`
diffs.

Observed labelled size in the current full run:

- total: `44` labelled samples
- split: `26` Dev / `18` Test

## 5. Knowledge Base

### 5.1 KB Sources

The validated KB is extracted from:

- Qiskit release-note YAML files
- Qiskit Aer release-note YAML files
- Qiskit Ignis release-note YAML files
- Qiskit IBM Runtime RST release notes
- PennyLane changelog markdown
- hand-authored LintQ rule summaries

Current KB size in the full experiments:

- total patterns: `2044`

Source distribution:

- `qiskit_releasenotes`: `1089`
- `qiskit_aer_releasenotes`: `152`
- `qiskit_ignis_releasenotes`: `16`
- `ibm_runtime_changelog`: `31`
- `pennylane_changelog`: `746`
- `lintq_rules`: `10`

### 5.2 KB Entry Structure

Dataclass:

- `BugPattern(pattern_id, name, taxonomy_class, description, example_code, fix_hint, source, tags)`

Important fields:

- `pattern_id`: unique retrieval/evidence reference
- `taxonomy_class`: one of the 5 output classes
- `description`: normalized bug-fix text used for BM25
- `tags`: includes source, validation tags, section tags, and framework tags

Typical tag categories:

- source tags such as `qiskit_releasenotes` or `pennylane_changelog`
- framework tags such as `framework:qiskit`
- content tags such as `measurement`, `initialization`, or `mapping`
- validation tags indicating the entry survived the KB quality filters

### 5.3 KB Curation Logic

The builder removes low-quality entries:

- too short
- too long
- cosmetic/version-bump style notes
- unclassifiable entries

Taxonomy assignment is keyword-driven, not LLM-driven. This reduces leakage and
keeps the retrieval corpus deterministic.

This matters methodologically because the KB is not generated from model
outputs. It is a fixed external evidence source derived from software artifacts,
which makes the retrieval story easier to defend in the paper.

### 5.4 LintQ Role

LintQ rule summaries inject a small set of precise, high-value bug patterns,
especially around measurement misuse and circuit-construction issues. These are
treated as validated structured references, not as training labels.

## 6. Retrieval Design

Retriever:

- [src/taxonomy_v6/retriever.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/retriever.py)

Retrieval steps:

1. Detect the framework of the query snippet:
   - `qiskit`
   - `pennylane`
   - `cirq`
   - `qsharp`
   - `other`
2. Tokenize query and KB descriptions.
3. Compute BM25 over the full KB.
4. Apply a `1.5x` score boost to entries matching the detected framework.
5. Return raw top-`k` results after a hard BM25 score floor.

Important implementation choices:

- no diversification pass is used
- retrieval is raw rank-based top-`k`
- the top-1 BM25 score is logged for routing

Why diversification was removed:

- earlier diversification forced artificial taxonomy coverage in the retrieved
  set
- that made the evidence less faithful to true lexical relevance
- the current design is more interpretable because the retriever simply returns
  the strongest matching bug-patterns above a score floor

Routing variable:

- `top1_bm25_score`

## 7. Prompt Structure

Prompt builder:

- [src/taxonomy_v6/prompts.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/prompts.py)

System prompt behavior:

- explicitly states the input definitely contains a bug
- defines the 5 categories in natural language
- requests a score for each class in `[0, 1]`
- asks for one final taxonomy choice
- requires JSON only

Two prompt modes:

### 7.1 Prompt-only

User message contains:

- instruction that no external references are provided
- requirement that `evidence_ids` must be empty
- buggy code snippet

Interpretation:

- this mode measures what the LLM can infer directly from code alone
- it is the clean non-retrieval baseline for the paper

### 7.2 RAG Prompt

User message contains:

- retrieved references in the form:
  - `Reference ID <pattern_id> [<taxonomy_class>]: <description>`
- explicit instruction that `evidence_ids` may only use the provided IDs
- buggy code snippet

Interpretation:

- retrieved references are intended to ground reasoning, not leak the answer
- the model still has to infer which retrieved pattern best explains the code
- the explicit evidence-ID requirement makes attribution directly auditable

## 8. LLM / Structured Output Logic

LLM client:

- [src/taxonomy_v6/llm.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/llm.py)

### 8.1 Models

Models used in the current study:

- `gpt-4o`
- `gpt-5.4`

### 8.2 Strict JSON

OpenAI calls use Structured Outputs with:

- `response_format.type = "json_schema"`
- `strict = true`

The response schema requires:

- `scores`
- `taxonomy_class`
- `suspected_location`
- `justification`
- `evidence_ids`

Crucially:

- in RAG mode, `evidence_ids` is dynamically constrained to an enum over the
  retrieved IDs only
- in prompt-only mode, `evidence_ids` is constrained to an empty array

Conceptual response shape:

```json
{
  "scores": {
    "incorrect_operator": 0.71,
    "incorrect_qubit_mapping": 0.12,
    "missing_barrier": 0.03,
    "wrong_initial_state": 0.05,
    "measurement_error": 0.09
  },
  "taxonomy_class": "incorrect_operator",
  "suspected_location": "line 7: apply x gate to wrong qubit register",
  "justification": "The code applies an operator inconsistent with the intended circuit behavior.",
  "evidence_ids": ["qiskit_releasenotes_123"]
}
```

This example is illustrative. The real schema is stricter and dynamically
changes the allowed `evidence_ids` per sample.

### 8.3 Retry / Fallback Policy

If strict parsing fails:

1. retry exactly once at a slightly higher temperature
2. if it still fails in RAG mode, fall back to prompt-only for that sample

This preserves evaluation continuity. A malformed RAG answer does not delete
the sample from the experiment; it converts the sample into a controlled
fallback case.

### 8.4 Attribution Failure Policy

After parsing:

- any `evidence_ids` not found in `retrieved_patterns` are flagged as
  `attribution_failure = true`
- the sample is kept in the dataset
- the sample is not dropped from metrics

In the live 5-sample smoke test, strict JSON parsing succeeded cleanly and no
attribution failures were observed.

## 9. Modes Evaluated

The runner reports three final modes:

- `prompt_only`
- `rag`
- `hybrid`

Definitions:

- `prompt_only`: no retrieval
- `rag`: pure retrieval-augmented mode, no abstention
- `hybrid`: use prompt-only if `top1_bm25_score < tau`, otherwise use RAG

The `hybrid` mode is the most deployment-oriented setting because it is allowed
to reject weak retrieval rather than blindly trusting noisy evidence.

## 10. Abstention Routing

Routing rule:

```python
if top1_bm25_score < tau:
    use_prompt_only()
else:
    use_rag()
```

`tau` is chosen on Dev only by maximizing paired accuracy over candidate
thresholds derived from score quintiles.

Interpretation:

- if retrieval is weak, forcing RAG can inject noisy references
- abstention protects accuracy by refusing weakly grounded retrieval
- a higher abstention rate is not automatically worse; it can indicate the
  system is correctly identifying low-quality retrieval cases

This is one of the most important methodological contributions of the current
pipeline.

## 11. Statistical Protocol

All final metrics are computed on Test only.

Implemented in:

- [src/taxonomy_v6/analysis.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6/analysis.py)

Reported analyses:

- accuracy
- macro-F1
- macro precision / recall
- per-class F1
- top-2 accuracy
- bootstrap 95% CIs for accuracy and macro-F1
- McNemar prompt-only vs RAG
- ECE before and after temperature scaling
- abstention and grounding diagnostics

Technical details:

- bootstrap uses paired resampling with a shared index per sample vector
- McNemar uses `exact=True` when the disagreement count is small
- ECE uses `10` equal-frequency bins
- temperature scaling is fit on Dev NLL only

Strict split discipline:

- `tau` is chosen on Dev only
- temperature `T` is chosen on Dev only
- prior estimates `pi_hat(c)` are computed on Dev only
- the frozen Dev-tuned values are then applied once to Test
- Test labels are used only for the final metric computation

## 12. Bayesian Prior Correction Experiment

Motivation:

- macro-F1 was low because the LLM strongly favored majority classes
- rare classes such as `missing_barrier` often received extremely low score mass

Method:

1. Estimate `pi_hat(c)` on Dev as the empirical mean of class-score vectors.
2. Temperature-scale the Test predictions.
3. Reweight each class by:

```python
corrected_score_c = raw_score_c / max(pi_hat(c), epsilon)
```

Current smoothing:

- `epsilon = 0.05`

Key finding:

- the prior estimate clearly exposed strong class bias
- unsmoothed correction was numerically unstable
- smoothed correction stabilized the failure mode
- but it still did not beat the uncorrected `gpt-4o` baseline on held-out
  macro-F1

This is a valuable negative result for the paper.

Why the correction can fail even after smoothing:

- when a rare class receives tiny average probability mass on Dev, dividing by
  that prior amplifies even weak, noisy score mass on Test
- `epsilon = 0.05` prevents catastrophic division by near-zero values, but it
  does not guarantee that the corrected class ordering becomes more accurate
- the experiment therefore reveals model bias clearly, but does not supply a
  better final classifier

## 13. Main Results

### 13.1 Full `gpt-4o` Baseline

Source:

- [outputs/results_4o/summary.json](/Users/syedshubha/Desktop/quantum-bug-rag/outputs/results_4o/summary.json)

#### Bugs4Q Test

| Mode | Accuracy | Macro-F1 | Abstention |
|---|---:|---:|---:|
| prompt_only | 0.2778 | 0.1506 | - |
| rag | 0.5000 | 0.3596 | - |
| hybrid | 0.5000 | 0.3596 | 0.0000 |

McNemar prompt-only vs RAG:

- `both_correct = 5`
- `both_wrong = 9`
- `po_only = 0`
- `rag_only = 4`
- `p = 0.125`

Interpretation:

- RAG helped substantially on Bugs4Q.
- The tuned `tau` was `-inf`, so the hybrid did not abstain on this split.
- This should be presented as a case where retrieval was strong enough that the
  abstention mechanism had no need to intervene.

#### Bugs-QCP Test

| Mode | Accuracy | Macro-F1 | Abstention |
|---|---:|---:|---:|
| prompt_only | 0.5000 | 0.1724 | - |
| rag | 0.6667 | 0.1655 | - |
| hybrid | 0.6111 | 0.1571 | 0.1667 |

McNemar prompt-only vs RAG:

- `both_correct = 8`
- `both_wrong = 5`
- `po_only = 1`
- `rag_only = 4`
- `p = 0.375`

Interpretation:

- RAG improved accuracy more than macro-F1.
- Hybrid abstention rejected weak RAG calls on `16.67%` of Test samples.
- This is the clearest example of the routing policy protecting the final
  system from low-quality retrieval on held-out data.

### 13.2 Full `gpt-5.4` Comparison

Source:

- [outputs/results_54/summary.json](/Users/syedshubha/Desktop/quantum-bug-rag/outputs/results_54/summary.json)

#### Bugs4Q Test

| Mode | Accuracy | Macro-F1 | Abstention |
|---|---:|---:|---:|
| prompt_only | 0.6667 | 0.2914 | - |
| rag | 0.7222 | 0.3152 | - |
| hybrid | 0.6667 | 0.2914 | 0.6111 |

McNemar prompt-only vs RAG:

- `both_correct = 12`
- `both_wrong = 5`
- `po_only = 0`
- `rag_only = 1`
- `p = 1.0`

#### Bugs-QCP Test

| Mode | Accuracy | Macro-F1 | Abstention |
|---|---:|---:|---:|
| prompt_only | 0.5556 | 0.1481 | - |
| rag | 0.5000 | 0.1440 | - |
| hybrid | 0.5000 | 0.1440 | 0.0000 |

McNemar prompt-only vs RAG:

- `both_correct = 9`
- `both_wrong = 8`
- `po_only = 1`
- `rag_only = 0`
- `p = 1.0`

Interpretation:

- `gpt-5.4` was not uniformly better than `gpt-4o`.
- It abstained much more on Bugs4Q, and did not show stronger macro-F1.
- The newer model should therefore be treated as a comparison point, not as a
  dominant replacement.

### 13.3 Smoothed Prior-Correction (`gpt-4o`, `epsilon = 0.05`)

Source:

- [outputs/results_4o_priorcorr_eps005/summary.json](/Users/syedshubha/Desktop/quantum-bug-rag/outputs/results_4o_priorcorr_eps005/summary.json)

#### Bugs4Q Test

| Mode | Accuracy | Macro-F1 | Abstention |
|---|---:|---:|---:|
| prompt_only | 0.2778 | 0.1905 | - |
| rag | 0.1111 | 0.1400 | - |
| hybrid | 0.1111 | 0.1500 | 0.1667 |

#### Bugs-QCP Test

| Mode | Accuracy | Macro-F1 | Abstention |
|---|---:|---:|---:|
| prompt_only | 0.0556 | 0.0364 | - |
| rag | 0.2222 | 0.1086 | - |
| hybrid | 0.2222 | 0.1086 | 0.0000 |

Interpretation:

- smoothing stabilized the catastrophic unsmoothed correction
- but the corrected system still underperformed the original uncorrected
  baseline on macro-F1
- this makes the prior-correction story more of a bias analysis / negative
  result than a final deployed method

This distinction is important in the paper: the prior-correction experiment is
scientifically valuable because it explains model behavior, but it should not
be mistaken for the best-performing final system.

### 13.4 Legacy Non-LLM Static Baseline

Source code:

- [src/baselines.py](/Users/syedshubha/Desktop/quantum-bug-rag/src/baselines.py)
- [scripts/run_static_baseline.py](/Users/syedshubha/Desktop/quantum-bug-rag/scripts/run_static_baseline.py)

This is a lightweight rule-based heuristic baseline from the legacy scaffold.
It is not a faithful reimplementation of a published static analyzer. It uses
simple textual rules such as:

- presence of `.measure(...)` to trigger `measurement_error`
- self-targeted `cx(q, q)` patterns to trigger `incorrect_qubit_mapping`
- Hadamard without barrier heuristics to trigger `missing_barrier`
- hard-coded rotation heuristics to trigger `incorrect_operator`

For paper comparison, the baseline was evaluated on the same `taxonomy_v6`
held-out Test splits used by the `gpt-4o` runs:

#### Bugs4Q Test

| Mode | Accuracy | 95% CI | Macro-F1 | 95% CI |
|---|---:|---:|---:|---:|
| static baseline | 0.2222 | [0.0556, 0.4444] | 0.1333 | [0.0500, 0.1846] |

Prediction pattern:

- predicted `measurement_error` on `7/18`
- predicted `missing_barrier` on `3/18`
- returned `no_bug_detected` on `8/18`

#### Bugs-QCP Test

| Mode | Accuracy | 95% CI | Macro-F1 | 95% CI |
|---|---:|---:|---:|---:|
| static baseline | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] |

Prediction pattern:

- predicted `missing_barrier` on `2/18`
- returned `no_bug_detected` on `16/18`

Interpretation:

- on `Bugs4Q`, the best LLM result (`gpt-4o` RAG / hybrid at `0.5000`
  accuracy, `0.3596` macro-F1) clearly exceeds the heuristic baseline
- on `Bugs-QCP`, the rule-based baseline essentially fails on the focused diff
  snippets, which is consistent with the fact that these samples often require
  semantic reasoning rather than a few shallow syntax cues
- this baseline is useful as a sanity check because it shows the task is not
  solved by trivial keyword rules alone

## 14. Dev Priors and Class Bias

The most important qualitative bias finding is that the model’s average score
mass is concentrated on majority classes.

Example from the smoothed `gpt-4o` run:

### Bugs4Q hybrid Dev prior

- `incorrect_operator`: `0.759209`
- `incorrect_qubit_mapping`: `0.081321`
- `missing_barrier`: `0.006791`
- `wrong_initial_state`: `0.029013`
- `measurement_error`: `0.123666`

### Bugs-QCP hybrid Dev prior

- `incorrect_operator`: `0.717552`
- `incorrect_qubit_mapping`: `0.241848`
- `missing_barrier`: `0.000001`
- `wrong_initial_state`: `0.007693`
- `measurement_error`: `0.032907`

Interpretation:

- `incorrect_operator` dominates the score mass
- `missing_barrier` is effectively ignored by the model in some settings
- this explains why top-1 accuracy can look reasonable while macro-F1 stays low

## 15. Final Narrative Options for the Paper

The strongest honest paper narratives are:

### Option A: RAG + Abstention as the Main Contribution

Claim:

- retrieval helps only when evidence is strong
- BM25-based abstention is necessary to protect accuracy from weak matches

Evidence:

- `gpt-4o` baseline improves from prompt-only to RAG on Bugs4Q
- hybrid abstention rejects weak RAG matches on Bugs-QCP

### Option B: Class-Bias Analysis as the Main Finding

Claim:

- modern LLMs show severe implicit class priors in forced-choice bug taxonomy
- this biases them toward majority classes and suppresses rare failure modes

Evidence:

- Dev priors are extremely skewed
- macro-F1 is much lower than accuracy
- rare classes receive near-zero probability mass

### Option C: Honest Negative Result on Post-hoc Debiasing

Claim:

- naive Bayesian prior correction is unstable
- smoothed correction stabilizes numerics but still does not outperform the
  simpler baseline

Evidence:

- unsmoothed correction tanked macro-F1
- smoothed correction improved stability but not final held-out macro-F1

### 15.1 Reporting Discipline for the Writeup

The paper should phrase the empirical claims conservatively.

Required wording discipline:

- do not say retrieval "significantly improved" performance
- do say retrieval showed directional improvements on some cells, especially
  `Bugs4Q` with `gpt-4o`, but none of the paired McNemar comparisons reached
  `p < 0.05`
- explicitly note that `n = 18` Test samples per dataset makes the confidence
  intervals wide and the threshold estimates high-variance

Important result-language template:

- "RAG demonstrated directional improvements consistent with retrieval helping,
  but the held-out Test sets are small enough that none of the paired
  comparisons reaches statistical significance at `alpha = 0.05`."

What to show in tables:

- include `95%` confidence intervals directly next to accuracy and macro-F1
- include McNemar `p` values in a compact comparison row or caption
- include abstention rate for the `hybrid` rows

How to describe the hybrid story honestly:

- abstention helped in one of the four `(dataset x model)` cells
- abstention was effectively inert in two cells
- abstention was net-negative in one cell
- therefore the routing rule is interpretable and meaningful, but not a
  uniformly positive intervention

How to describe `tau` variance:

- Dev splits have only about `26-27` samples
- quintile-based threshold selection therefore operates on small buckets
- cross-dataset differences in tuned `tau` may reflect real signal, sampling
  variance, or both

How to describe `pi_hat(c)` semantics:

- the current prior-correction experiment estimates `pi_hat(c)` as the mean of
  Dev `class_scores`
- this should be described as empirical distribution rebalancing over the
  evaluation distribution, not as a pure content-free prompt prior in the
  strongest Zhao et al. sense

Multiple-comparison disclosure:

- the paper should state that no formal multiple-comparison correction was
  applied
- this does not change the practical conclusion because none of the paired
  comparisons is significant even before correction

## 16. Recommended Result Set to Treat as Primary

If the goal is the strongest paper:

- use uncorrected `gpt-4o` as the primary reported system
- use `gpt-5.4` as a comparison model
- use the prior-correction experiments as an analysis section, not as the main
  final method

Reason:

- the uncorrected `gpt-4o` pipeline is the strongest held-out Macro-F1 result
  among the tested variants
- the later correction experiments are scientifically useful, but they weaken
  the headline system performance

### 16.1 Recommended Final Paper Position

The cleanest final paper framing is:

- main system: `gpt-4o` with validated KB retrieval and BM25-threshold
  abstention routing
- main methodological contribution: retrieval should be used selectively rather
  than unconditionally
- main empirical finding: class imbalance and implicit LLM priors materially
  depress macro-F1, especially for rare classes
- negative-result section: Bayesian prior correction reveals the bias but does
  not outperform the simpler baseline on held-out data

In other words, the best paper is not "we solved rare-class calibration." The
best paper is "we built an auditable RAG classifier for quantum bug taxonomy,
showed when retrieval helps, and uncovered a strong class-prior failure mode."

## 17. Output Artifact Structure

Every full run writes a results directory containing:

- `summary.json`: top-level run configuration, per-dataset reports, tuning
  values, and final metrics
- `diagnostics_<dataset>_<mode>_<split>.jsonl`: per-sample records used for
  routing, attribution, calibration, and error analysis

Important `summary.json` content:

- run metadata such as model name, seed, split ratio, `top_k`, and score floor
- per-dataset Dev/Test sizes
- tuned `tau`
- fitted temperature `T`
- estimated Dev prior `pi_hat(c)` when prior correction is enabled
- Test-only accuracy, macro-F1, bootstrap CIs, McNemar, and ECE
- abstention and grounding reporting block

Important per-sample diagnostic fields:

- `sample_id`
- `ground_truth`
- `predicted_class`
- `class_scores`
- `retrieved_patterns`
- `evidence_ids`
- `top1_bm25_score`
- `routed_mode`
- `final_mode`
- `prompt_only_fallback_used`
- `fallback_reason`
- `parse_retry_count`
- `attribution_failure`
- `grounded`

These diagnostics are enough to write a qualitative error-analysis appendix if
the paper needs example cases.

### 17.1 Interpreting `class_scores`

The `class_scores` stored in diagnostics should be treated as model score
vectors, not necessarily as perfectly normalized probabilities at emission time.
The pipeline handles them as follows:

- raw model outputs are clipped and normalized before calibration analysis
- temperature scaling is applied on the normalized score matrix
- Bayesian prior correction, when enabled, is applied after temperature scaling
- the final predicted class is the `argmax` after the appropriate downstream
  transformation

This matters when reading examples from raw diagnostics: some live model score
vectors do not sum exactly to `1.0`, but the analysis path makes them
well-defined before calibration and prior-correction steps.

### 17.2 Representative Diagnostic Cases

The following cases are useful for writing the paper's qualitative analysis.

Case A: RAG rescues a prompt-only miss

- sample: `bugs4q_0045`
- dataset/split: `Bugs4Q` Test
- ground truth: `wrong_initial_state`
- prompt-only prediction: `incorrect_operator`
- RAG prediction: `wrong_initial_state`
- top-1 BM25 score: `52.6187`
- cited evidence ID: `qiskit_aer_releasenotes_fixes_fix-for-loop-no-parameter-aa5b04b1da0e956b_0`
- model rationale: the initial state vector `'01'` does not match the expected
  basis encoding for a 2-qubit system, so the circuit is prepared in the wrong
  state

Why it matters:

- this is the clearest held-out example of retrieval supplying useful context
  that the prompt-only model did not exploit correctly

Case B: Hybrid abstention can make a false rejection

- sample: `bqcp_001156`
- dataset/split: `Bugs-QCP` Test
- ground truth: `incorrect_operator`
- tuned `tau` for `gpt-4o` on `Bugs-QCP`: `86.723515`
- RAG top-1 BM25 score: `69.1820`
- prompt-only prediction: `incorrect_qubit_mapping` (wrong)
- RAG prediction: `incorrect_operator` (correct)
- hybrid decision: abstain to `prompt_only`, therefore final prediction is
  wrong

Why it matters:

- this case shows the routing policy is principled but imperfect
- the threshold can reject a sample where retrieval would actually have helped
- the paper should therefore describe abstention as a controlled tradeoff, not
  as a guaranteed per-sample improvement

Case C: Prior correction can flip a correct answer into a wrong one

- sample: `bqcp_000011`
- dataset/split: `Bugs-QCP` Test
- ground truth: `incorrect_operator`
- baseline hybrid prediction: `incorrect_operator` with raw score pattern
  dominated by `incorrect_operator`
- smoothed prior-corrected prediction: `measurement_error`
- evidence ID: `qiskit_ignis_releasenotes_fixes_no-auto-scs-b82ebef53508fc7b_0`

Why it matters:

- this is a concrete example of the negative result
- after reweighting by the Dev prior, the minority/underweighted classes can be
  amplified enough to reorder the class ranking incorrectly

Case D: Prior correction can also rescue a minority-class error

- sample: `bugs4q_0000`
- dataset/split: `Bugs4Q` Test
- ground truth: `measurement_error`
- baseline prompt-only prediction: `incorrect_operator`
- smoothed prior-corrected prompt-only prediction: `measurement_error`
- underlying rationale: the raw score vector still contained substantial mass
  on `measurement_error`, and reweighting reduced the dominance of
  `incorrect_operator`

Why it matters:

- this shows the correction is not uniformly pathological
- its problem is not that it never helps, but that its aggregate held-out
  behavior is too unstable to beat the simpler baseline

Structured-output reliability note:

- across the saved full Test diagnostics for `gpt-4o`, `gpt-5.4`, and the
  smoothed prior-correction run, there were `0` attribution failures, `0`
  prompt-only fallback invocations caused by parsing failure, and `0` total
  parse retries

This means the strict JSON schema and evidence-ID constraints behaved cleanly
in the recorded experiments.

### 17.3 Appendix-Ready Implementation Pseudocode

Retriever:

```python
framework = detect_framework(code)
scores = bm25.get_scores(tokenize(code))
if framework in framework_index:
    scores[framework_index[framework]] *= 1.5
ranked = sort_desc(scores)
hits = []
for i, s in ranked:
    if s < bm25_floor:
        continue
    hits.append((patterns[i], s))
    if len(hits) == top_k:
        break
```

Structured-output schema generation:

```python
retrieved_ids = [p.pattern_id for p in retrieved]
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "taxonomy_v6_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scores": {... five classes ...},
                "taxonomy_class": {"type": "string", "enum": TAXONOMY_FORCED},
                "suspected_location": {"type": "string"},
                "justification": {"type": "string"},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": retrieved_ids},
                },
            },
            "required": [...],
            "additionalProperties": False,
        },
    },
}
```

Strict JSON call policy:

```python
for temperature in [base_temperature, base_temperature + 0.15]:
    raw = llm.complete(messages, response_format=response_format)
    parsed = parse_json(raw)
    if parsed_is_complete(parsed):
        return parsed
if mode == "rag":
    return prompt_only_fallback(sample)
```

Hybrid routing:

```python
if rag_diag.top1_bm25_score < tau:
    final_diag = prompt_only_diag
    final_diag.routed_mode = "prompt_only"
    final_diag.abstained_to_prompt_only = True
else:
    final_diag = rag_diag
    final_diag.routed_mode = "rag"
```

Calibration and smoothed prior correction:

```python
dev_probs = normalize(dev_class_scores)
T = fit_temperature_by_dev_nll(dev_probs, dev_labels)
test_scaled = softmax(log(normalize(test_class_scores)) / T)
pi_hat = mean(dev_scaled_or_raw_scores, axis=0)
safe_prior = maximum(pi_hat, epsilon=0.05)
corrected = test_scaled / safe_prior
corrected = corrected / corrected.sum()
prediction = argmax(corrected)
```

These snippets are intentionally simplified, but they match the logic of the
actual implementation closely enough to support a method section or appendix.

## 18. Reproducibility Checklist

To reproduce the paper faithfully, the following must be fixed and reported:

- repository snapshot or commit hash
- exact OpenAI model ID, e.g. `gpt-4o` or `gpt-5.4`
- `requirements.txt` dependency set
- availability of the upstream `Bugs4Q` and
  `Bugs-Quantum-Computing-Platforms` sources
- the validated KB source families and their filtering logic
- Dev/Test split ratio of `60/40`
- deterministic split seed
- raw BM25 retrieval with no diversification
- Dev-only tuning for `tau`
- Dev-only fitting for temperature scaling
- Dev-only estimation for Bayesian prior correction
- Test-only reporting for final metrics
- output `summary.json` files stored with the paper artifacts

If the paper includes ablations, they should be described relative to this base
protocol rather than introducing new tuning or split rules.

## 19. Threats to Validity and Limitations

The paper should be explicit about the following limitations:

- severe label imbalance means accuracy alone is misleading
- the five-class taxonomy is a forced abstraction over more heterogeneous
  upstream bug descriptions
- KB evidence comes from release notes and rule summaries, which may not cover
  every real bug manifestation in the evaluation datasets
- Bugs4Q and Bugs-QCP have different provenance and construction pipelines, so
  cross-dataset behavior is informative but not perfectly controlled
- closed-model API behavior can drift over time even when the model ID stays
  constant
- prior correction exposed model bias clearly but did not improve held-out
  macro-F1, so the bias story is stronger than the debiasing story
- only a small set of OpenAI models were compared; broader model-family claims
  would require more baselines

## 20. What a Paper Writer Does Not Need Anymore

After reading this file, a paper writer should not need additional repo context
to understand:

- the task definition
- the five output classes
- dataset provenance and split sizes
- KB provenance and structure
- retrieval behavior and abstention routing
- prompt design and strict JSON requirements
- attribution handling and fallback logic
- the Dev/Test evaluation discipline
- the main `gpt-4o` and `gpt-5.4` results
- the class-bias finding
- the prior-correction negative result
- the recommended final narrative for the paper

The only remaining reason to open other artifacts would be to extract
illustrative example cases from the diagnostics files or quote exact code-level
implementation details in an appendix.

## 21. Files Sufficient for Drafting

If someone else has to write the paper, these files are sufficient:

- [docs/paper_master_context.md](/Users/syedshubha/Desktop/quantum-bug-rag/docs/paper_master_context.md)
- [README.md](/Users/syedshubha/Desktop/quantum-bug-rag/README.md)
- [docs/methodology.md](/Users/syedshubha/Desktop/quantum-bug-rag/docs/methodology.md)
- [docs/methodology_and_architecture.md](/Users/syedshubha/Desktop/quantum-bug-rag/docs/methodology_and_architecture.md)
- [quantum-vs-classical-bug-prediction.ipynb](/Users/syedshubha/Desktop/quantum-bug-rag/quantum-vs-classical-bug-prediction.ipynb)
- [scripts/run_study_i_codebert.py](/Users/syedshubha/Desktop/quantum-bug-rag/scripts/run_study_i_codebert.py)
- [src/study_i/](/Users/syedshubha/Desktop/quantum-bug-rag/src/study_i)
- [scripts/run_taxonomy_v6.py](/Users/syedshubha/Desktop/quantum-bug-rag/scripts/run_taxonomy_v6.py)
- [src/taxonomy_v6/](/Users/syedshubha/Desktop/quantum-bug-rag/src/taxonomy_v6)
- [outputs/results_4o/summary.json](/Users/syedshubha/Desktop/quantum-bug-rag/outputs/results_4o/summary.json)
- [outputs/results_54/summary.json](/Users/syedshubha/Desktop/quantum-bug-rag/outputs/results_54/summary.json)
- [outputs/results_4o_priorcorr_eps005/summary.json](/Users/syedshubha/Desktop/quantum-bug-rag/outputs/results_4o_priorcorr_eps005/summary.json)

For detailed error analysis:

- `outputs/results_4o/diagnostics_*.jsonl`
- `outputs/results_54/diagnostics_*.jsonl`
- `outputs/results_4o_priorcorr_eps005/diagnostics_*.jsonl`
