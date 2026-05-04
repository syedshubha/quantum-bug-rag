# quantum-bug-rag

Research code for LLM-based bug analysis in quantum software. The repository now contains:

- the original modular scaffold for prompt-only, RAG, and static-baseline experiments over prepared `Bugs4Q` data;
- a modular Study I binary classifier refactored from `quantum-vs-classical-bug-prediction.ipynb`;
- a refactored five-class taxonomy track from `quantum_bug_detecttion_taxonomy.ipynb`;
- a refactored binary classical-vs-quantum track from `quantum-software-bug-detection-rag-project-v6_classical.ipynb`.

The notebooks are still kept as provenance and exploratory-analysis artifacts. The reusable implementations now live under `src/study_i`, `src/taxonomy_v6`, `src/classical`, and matching `scripts/` entry points.

## Project Tracks

| Track | Purpose | Main code | Entry point |
|------|---------|-----------|-------------|
| `legacy scaffold` | Prompt-only, RAG, and static baseline over prepared local datasets and `knowledge_base/` | `src/` top-level modules | `scripts/run_prompt_only.py`, `scripts/run_rag.py`, `scripts/run_static_baseline.py` |
| `study_i` | Repeated stratified CV fine-tuning of `microsoft/codebert-base` for binary `classical` vs `quantum` bug prediction | `src/study_i/` | `scripts/run_study_i_codebert.py` |
| `taxonomy_v6` | Forced-choice 5-class quantum bug classification with validated KB and framework-aware retrieval | `src/taxonomy_v6/` | `scripts/run_taxonomy_v6.py` |
| `classical` | Older LLM/RAG binary classification of bugs as `quantum` vs `classical`, including biased vs balanced retrieval | `src/classical/` | `scripts/run_classical_vs_quantum.py` |

## Repository Structure

```text
quantum-bug-rag/
├── README.md
├── quantum-vs-classical-bug-prediction.ipynb
├── quantum_bug_detecttion_taxonomy.ipynb
├── quantum-software-bug-detection-rag-project-v6_classical.ipynb
├── src/
│   ├── ...                         # legacy scaffold modules
│   ├── classical/
│   │   ├── analysis.py
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   ├── kb.py
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   ├── retriever.py
│   │   └── schemas.py
│   ├── study_i/
│   │   ├── analysis.py
│   │   ├── dataset.py
│   │   ├── plotting.py
│   │   ├── schemas.py
│   │   └── training.py
│   └── taxonomy_v6/
│       ├── dataset.py
│       ├── evaluator.py
│       ├── kb.py
│       ├── llm.py
│       ├── prompts.py
│       ├── retriever.py
│       └── schemas.py
├── scripts/
│   ├── run_classical_vs_quantum.py
│   ├── run_study_i_codebert.py
│   ├── run_taxonomy_v6.py
│   ├── run_prompt_only.py
│   ├── run_rag.py
│   ├── run_static_baseline.py
│   └── ...
├── docs/
├── data/
├── knowledge_base/
├── outputs/
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For OpenAI-backed runs:

```bash
export OPENAI_API_KEY="sk-..."
```

The notebook-refactored scripts use `rank_bm25`, `pyyaml`, `numpy`, `scikit-learn`, and, for Study I, the Hugging Face / PyTorch stack declared in `requirements.txt`.

## Datasets And Roles

| Dataset / Repo | Role |
|----------------|------|
| `Bugs4Q` | Benchmark corpus of buggy quantum programs. Used by the legacy scaffold, Study II, and the older binary LLM track. |
| `Bugs-QCP` | Auxiliary corpus. Used as a labelled source in the `classical` track and as a quantum sample source in `taxonomy_v6`. |
| external labeled JSON of bug reports | Study I dataset of `(name, description, code)` triples with `bug_category ∈ {classical, quantum}`. |
| `Qiskit`, `Qiskit Aer`, `Qiskit Ignis`, `Qiskit IBM Runtime`, `PennyLane` | Source repositories for validated KB extraction in `taxonomy_v6`. |
| `CPython`, `NumPy` | Classical release-note sources for the symmetric KB in the `classical` track. |

The legacy scaffold expects prepared local data under `data/` and a JSON knowledge base under `knowledge_base/`.

The notebook-refactored scripts expect cloned upstream repositories under a `--work-dir` that you provide, except for Study I, which expects a labeled JSON file path.

## Running The Legacy Scaffold

Prepare the legacy local dataset layout:

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q/
python scripts/prepare_bugsqcp_kb.py --input-dir /path/to/bugsqcp --output-dir knowledge_base/
```

Run the original modes:

```bash
python scripts/run_prompt_only.py --data-dir data/bugs4q --output-dir outputs --config config.yaml
python scripts/run_rag.py --data-dir data/bugs4q --kb-dir knowledge_base --output-dir outputs --config config.yaml
python scripts/run_static_baseline.py --data-dir data/bugs4q --output-dir outputs
```

## Running Study I

Study I fine-tunes `microsoft/codebert-base` for binary `classical` vs `quantum`
bug prediction from triples of:

- bug name
- natural-language description
- source-code snippet

Run:

```bash
PYTHONPATH=. python scripts/run_study_i_codebert.py \
  --data-path /path/to/bug_patterns_categorized.json \
  --results-dir outputs/study_i_codebert
```

This reproduces the notebook protocol:

- inverse-frequency class-weighted cross-entropy
- minority oversampling inside each training fold
- 5-fold stratified cross-validation
- 5 independent CV seeds
- 25 fold-runs total
- manual early stopping on validation macro-F1

Current notebook result snapshot:

- dataset: `233` labeled samples (`134` classical, `99` quantum)
- mean accuracy across `25` fold-runs: `0.767 ± 0.057`
- mean macro-F1 across `25` fold-runs: `0.763 ± 0.056`
- mean ROC-AUC across `25` fold-runs: `0.855 ± 0.044`
- pooled per-class F1:
  - `classical`: `0.7875`
  - `quantum`: `0.7410`
  - pooled macro-F1: `0.7642`

The script writes:

- `summary.json`
- `per_fold.csv`
- `epoch_logs.json`
- `fig1_confusion_matrix.png`
- `fig2_fold_distribution.png`
- `fig3_roc_curve.png`
- `fig4_learning_curves.png`
- `fig5_summary_panel.png`

## Running `taxonomy_v6`

Expected `--work-dir` layout:

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

Run:

```bash
python scripts/run_taxonomy_v6.py \
  --work-dir /path/to/work-dir \
  --results-dir outputs/taxonomy_v6 \
  --model gpt-4o
```

Use `--mock` for an offline smoke run.

This track evaluates three reported views over labelled samples from both `Bugs4Q` and `Bugs-QCP`:

- `prompt_only`
- `rag` (pure RAG, no abstention)
- `hybrid` (Dev-tuned abstention to prompt-only when `top1_bm25_score < tau`)

Methodology:

- each dataset is split deterministically into 60% Dev and 40% Test;
- Dev is used only to tune `tau` and the temperature-scaling parameter `T`;
- final accuracy, macro-F1, bootstrap CIs, McNemar, and ECE are reported only on Test;
- Test-time class predictions are temperature-scaled and then corrected with a smoothed Bayesian prior estimate from Dev, using `epsilon = 0.05` to avoid division by near-zero class priors.

It writes:

- `diagnostics_<dataset>_dev_<mode>.jsonl`
- `diagnostics_<dataset>_test_<mode>.jsonl`
- `metrics_<dataset>_<mode>.json`
- `summary.json`

## Current Findings

The paper-facing studies support two separate conclusions.

Study I:

- CodeBERT reaches `0.767 ± 0.057` mean accuracy and `0.763 ± 0.056` mean macro-F1 across `25` repeated stratified fold-runs on the binary `classical` vs `quantum` task.
- Pooled per-class F1 is `0.7875` for `classical` and `0.7410` for `quantum`, showing the binary task is learnable but not trivial.

Study II:

- Abstention routing helps preserve accuracy by sending weak-retrieval cases to `prompt_only` when the top-1 BM25 score falls below the frozen Dev-tuned threshold `tau`.
- Dev-set prior estimates show that the LLM is strongly biased toward majority classes such as `incorrect_operator`, while rare classes like `missing_barrier` can receive near-zero mass.
- A naive prior correction was numerically unstable, so the current implementation uses a smoothed Bayesian correction with `epsilon = 0.05`.
- That smoothed correction stabilizes the catastrophic over-correction failure mode, but in the current `gpt-4o` run it still does not outperform the uncorrected baseline on held-out Test macro-F1.

## Running `classical`

Expected `--work-dir` layout:

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

Run:

```bash
python scripts/run_classical_vs_quantum.py \
  --work-dir /path/to/work-dir \
  --results-dir outputs/classical_vs_quantum \
  --model gpt-4o
```

Use `--mock` for an offline smoke run.

This track evaluates:

- `prompt_only`
- `biased_rag`
- `balanced_rag`

It writes:

- `diagnostics_bqcp_<mode>.jsonl`
- `diagnostics_bugs4q_<mode>.jsonl`
- `summary.json`

`Bugs4Q` is treated as an external all-quantum holdout for purity checks in this binary setup.

## Notebook Refactor Coverage

The new module/script paths cover the core experiment logic from all three main notebooks:

- dataset adapters;
- KB construction;
- retrievers;
- prompt builders;
- LLM clients;
- transformer fine-tuning loops;
- evaluation loops and JSON artefact writing.

What remains notebook-only today:

- ad hoc plotting cells;
- console-only disagreement/error-print helpers;
- zip/archive packaging cells.

Those notebook-only pieces do not block the reusable experiment runs, but they are not exposed as standalone CLI commands yet.

## Outputs

See:

- [`outputs/README.md`](outputs/README.md)
- [`docs/methodology.md`](docs/methodology.md)
- [`docs/methodology_and_architecture.md`](docs/methodology_and_architecture.md)
- [`docs/documentation_reference.md`](docs/documentation_reference.md)

## Tests

```bash
pytest tests/ -v
```

The current automated tests mainly cover the legacy scaffold. The notebook-refactored tracks are still lighter on automated coverage, though Study I now has dedicated tests around dataset loading and result aggregation.

## License

MIT. See [`LICENSE`](LICENSE).
