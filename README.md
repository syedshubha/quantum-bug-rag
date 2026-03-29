# quantum-bug-rag

**Retrieval-Augmented LLM Bug Detection and Classification for Qiskit Programs**  
CSC 7135 Research Project · Bugs4Q Benchmark

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Preparing the Bugs4Q Dataset](#preparing-the-bugs4q-dataset)
5. [Running the Pipelines](#running-the-pipelines)
   - [Prompt-Only Baseline](#prompt-only-baseline)
   - [RAG Pipeline](#rag-pipeline)
   - [Static-Analysis Baseline](#static-analysis-baseline)
   - [Subset Evaluation (all three modes)](#subset-evaluation-all-three-modes)
6. [Configuration](#configuration)
7. [Outputs and Logs](#outputs-and-logs)
8. [Knowledge Base](#knowledge-base)
9. [Running Tests](#running-tests)
10. [Extending the Pipeline](#extending-the-pipeline)
11. [License](#license)

---

## Project Overview

I compare three approaches to detecting and classifying bugs in Qiskit
(quantum-computing) programs using the [Bugs4Q](https://github.com/Z-928/Bugs4Q)
benchmark:

| Mode | Description |
|------|-------------|
| `static` | Rule-based static analysis (fast, no API cost) |
| `prompt_only` | LLM with source code only; no retrieved context |
| `rag` | LLM augmented with top-k retrieved bug patterns from a local knowledge base |

Each mode produces **structured diagnostics** with:

- `bug_likelihood` — estimated probability the program contains a bug  
- `taxonomy_class` — predicted bug category (see `knowledge_base/taxonomy.json`)  
- `suspected_location` — file/line hint  
- `justification` — natural-language reasoning  

I evaluate all three modes using:

- **Detection**: Precision, Recall, F1 (binary: buggy/clean)  
- **Classification**: Macro-F1 over the Bugs4Q taxonomy classes  

---

## Directory Structure

```
quantum-bug-rag/
├── src/                        # Main Python package
│   ├── __init__.py
│   ├── schemas.py              # Pydantic data models
│   ├── utils.py                # Logging, config, JSON helpers
│   ├── dataset_loader.py       # Load Bugs4Q records
│   ├── retriever.py            # Keyword-based knowledge-base retriever
│   ├── prompt_builder.py       # Prompt construction for LLM calls
│   ├── llm_client.py           # LLM abstraction (mock / OpenAI / Gemini)
│   ├── baselines.py            # Static-analysis rules
│   ├── benchmark_runner.py     # Orchestrate a pipeline over a dataset
│   └── evaluate.py             # Detection and classification metrics
│
├── scripts/                    # Entry-point scripts
│   ├── prepare_bugs4q.py       # Prepare / convert Bugs4Q data
│   ├── run_prompt_only.py      # Prompt-only baseline
│   ├── run_rag.py              # RAG pipeline
│   ├── run_static_baseline.py  # Static-analysis baseline
│   └── run_subset_eval.py      # Compare all three modes on a subset
│
├── knowledge_base/
│   ├── bug_patterns.json       # Curated Qiskit bug patterns for retrieval
│   └── taxonomy.json           # Bug taxonomy definitions
│
├── data/
│   ├── README.md               # How to prepare Bugs4Q
│   └── bugs4q/                 # Prepared dataset (not committed)
│       └── bugs4q.json
│
├── outputs/                    # Pipeline results and logs (not committed)
│   └── README.md
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_schemas.py
│   ├── test_retriever.py
│   ├── test_llm_client.py
│   ├── test_baselines.py
│   └── test_pipeline_integration.py
│
├── docs/
│   └── methodology.md          # Research methodology note
│
├── config.example.yaml         # Configuration template
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

I recommend using a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

To use a real LLM backend, install the corresponding extra:

```bash
pip install openai               # for OpenAI (GPT-4o etc.)
pip install google-generativeai  # for Gemini
```

---

## Preparing the Bugs4Q Dataset

### Quick start – synthetic data (no download required)

```bash
python scripts/prepare_bugs4q.py
```

This generates 20 synthetic Qiskit program records in `data/bugs4q/bugs4q.json`
so you can run the full pipeline immediately.

### Using the real Bugs4Q benchmark

1. Clone the Bugs4Q repository:

   ```bash
   git clone https://github.com/Z-928/Bugs4Q /tmp/bugs4q_repo
   ```

2. Convert and place the data:

   ```bash
   python scripts/prepare_bugs4q.py --source /tmp/bugs4q_repo
   ```

3. Verify the data loaded correctly:

   ```bash
   python -c "from src.dataset_loader import load_bugs4q; print(len(load_bugs4q()), 'records')"
   ```

See `data/README.md` for the full JSON schema and additional options.

---

## Running the Pipelines

All scripts support `--help` for full option details.

### Prompt-Only Baseline

```bash
# Using the mock LLM (no API key required)
python scripts/run_prompt_only.py --backend mock --max-items 10

# Using OpenAI (requires OPENAI_API_KEY environment variable)
export OPENAI_API_KEY=sk-...
python scripts/run_prompt_only.py --backend openai --max-items 10

# Using Gemini (requires GEMINI_API_KEY environment variable)
export GEMINI_API_KEY=AIza...
python scripts/run_prompt_only.py --backend gemini --max-items 10
```

Results are saved to `outputs/prompt_only/results.json`.

### RAG Pipeline

```bash
python scripts/run_rag.py --backend mock --top-k 3 --max-items 10
```

Results are saved to `outputs/rag/results.json`.

### Static-Analysis Baseline

```bash
python scripts/run_static_baseline.py --max-items 10
```

Results are saved to `outputs/static/results.json`.

### Subset Evaluation (all three modes)

I run all three pipelines on a small subset and print a comparison table:

```bash
python scripts/run_subset_eval.py --n 10 --backend mock
```

This produces:

- `outputs/subset_eval/prompt_only_results.json`
- `outputs/subset_eval/rag_results.json`
- `outputs/subset_eval/static_results.json`
- `outputs/subset_eval/comparison.json` — side-by-side metrics

---

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit as needed:

```bash
cp config.example.yaml config.yaml
```

Pass it to any script with `--config config.yaml`.

Key settings:

```yaml
llm:
  backend: mock          # mock | openai | gemini
retrieval:
  top_k: 3
dataset:
  max_items: null        # null = all records
```

**Never commit `config.yaml` or API keys to version control.**  
Use environment variables instead:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
```

---

## Outputs and Logs

Each script writes results to a subdirectory of `outputs/`:

```
outputs/
├── prompt_only/results.json
├── rag/results.json
├── static/results.json
└── subset_eval/comparison.json
```

Each `results.json` is a JSON array of `DiagnosticResult` objects.

Log files (`run.log`) are written alongside `results.json` and mirror stdout.

See `outputs/README.md` for the full schema.

---

## Knowledge Base

`knowledge_base/bug_patterns.json` contains eight manually curated Qiskit
bug patterns (missing measurement, swapped qubits, deprecated gates, etc.).

`knowledge_base/taxonomy.json` defines the nine bug classes used for
classification.

To extend the knowledge base, add entries to `bug_patterns.json` following
the `BugPatternEntry` schema in `src/schemas.py`.

---

## Running Tests

```bash
pytest tests/ -v
```

To include coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Extending the Pipeline

| Goal | Where to look |
|------|---------------|
| Add a new LLM backend | `src/llm_client.py` → subclass `LLMClient` |
| Add static analysis rules | `src/baselines.py` → append to `STATIC_RULES` |
| Add knowledge-base patterns | `knowledge_base/bug_patterns.json` |
| Switch to dense retrieval | `src/retriever.py` → replace `KnowledgeBaseRetriever` |
| Change prompt format | `src/prompt_builder.py` |
| Add new evaluation metrics | `src/evaluate.py` |

---

## License

MIT – see [LICENSE](LICENSE).
