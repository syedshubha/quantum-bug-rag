# quantum-bug-rag

A retrieval-augmented generation (RAG) pipeline for quantum-computing bug detection and
classification, evaluated on the Bugs4Q benchmark.

---

## Overview

`quantum-bug-rag` combines a vector-based knowledge base of quantum bug patterns with a
large-language-model (LLM) reasoning layer to detect and classify bugs in Qiskit programs.
The system is designed to be evaluated rigorously on the [Bugs4Q](https://github.com/Z-928/Bugs4Q)
benchmark.  A secondary corpus, Bugs-QCP (Zenodo record 5834281), is used to enrich the
knowledge base with a broader taxonomy of quantum bug patterns; it is **not** used as an
evaluation set.

---

## Dataset Roles

| Dataset | Role | Used for evaluation? |
|---------|------|----------------------|
| **Bugs4Q** | Primary benchmark | ✅ Yes — all reported metrics are computed on Bugs4Q |
| **Bugs-QCP** (Zenodo 5834281) | Taxonomy corpus | ❌ No — used only for knowledge-base enrichment and bug-pattern retrieval |

### Bugs4Q — Executable Benchmark

Bugs4Q is the primary dataset for training, validation, and evaluation.  It provides
executable Qiskit programs paired with bug labels and is the sole basis for all quantitative
results reported in this project.  Download instructions and preparation steps are in
[`data/README.md`](data/README.md) and [`scripts/prepare_bugs4q.py`](scripts/prepare_bugs4q.py).

### Bugs-QCP — Broader Bug-Pattern and Taxonomy Corpus

Bugs-QCP (Zenodo DOI: [10.5281/zenodo.5834281](https://doi.org/10.5281/zenodo.5834281))
provides a wider catalogue of quantum computing bug patterns and a cross-platform taxonomy.
It is ingested into the knowledge base to improve retrieval quality and to ground the system
in a richer bug vocabulary.  It is **never** used to generate benchmark numbers.  Preparation
instructions are in [`scripts/prepare_bugs_qcp.py`](scripts/prepare_bugs_qcp.py), and
knowledge-base ingestion is documented in [`knowledge_base/README.md`](knowledge_base/README.md).

---

## Synthetic Data

A small synthetic dataset is included solely for **smoke testing** the pipeline (unit tests,
CI checks, and end-to-end pipeline validation).  **No benchmark results are reported on
synthetic data.**  Synthetic examples must not be mixed into Bugs4Q evaluation splits.

---

## Baseline

The current static baseline is a **lightweight, rule-based placeholder** that applies
hand-crafted heuristics to flag common Qiskit anti-patterns.  It is intended only to
establish a lower bound for comparison.  A faithful re-implementation of
[LintQ](https://github.com/nicoPy/LintQ)-style static analysis is planned as a more
representative baseline; until that is available, results attributed to the "static baseline"
refer to the rule-based heuristics described in this repository.

---

## Repository Layout

```
quantum-bug-rag/
├── data/
│   ├── README.md              # Dataset acquisition and preparation guide
│   ├── bugs4q/                # ← not committed; populate with prepare_bugs4q.py
│   └── bugs_qcp/              # ← not committed; populate with prepare_bugs_qcp.py
├── knowledge_base/
│   └── README.md              # Knowledge-base schema and ingestion guide
├── scripts/
│   ├── prepare_bugs4q.py      # Download and pre-process Bugs4Q
│   └── prepare_bugs_qcp.py    # Download and pre-process Bugs-QCP (Zenodo 5834281)
├── src/                       # Pipeline source code (retriever, generator, baseline)
├── tests/                     # Unit and integration tests (synthetic fixtures only)
├── .gitignore
├── LICENSE
└── README.md
```

> **Raw dataset files are never committed to this repository.**  Only preparation scripts
> and documentation are tracked.  See [`data/README.md`](data/README.md) for download
> instructions.

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/syedshubha/quantum-bug-rag.git
cd quantum-bug-rag
pip install -r requirements.txt   # (requirements file to be added with pipeline code)
```

### 2. Prepare Bugs4Q (primary benchmark)

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q
```

### 3. Prepare Bugs-QCP (knowledge-base enrichment)

```bash
python scripts/prepare_bugs_qcp.py --output-dir data/bugs_qcp
```

### 4. Ingest bug patterns into the knowledge base

See [`knowledge_base/README.md`](knowledge_base/README.md) for detailed ingestion steps.

### 5. Run smoke tests

```bash
pytest tests/          # Uses synthetic fixtures only; does not require downloaded datasets
```

---

## Citation

If you use this work, please cite the relevant datasets:

**Bugs4Q**
```
@inproceedings{bugs4q,
  title     = {Bugs4Q: A Benchmark of Real Bugs for Quantum Programs},
  author    = {Zhao, Guolong and others},
  booktitle = {ASE},
  year      = {2021}
}
```

**Bugs-QCP (Zenodo 5834281)**
```
@dataset{bugs_qcp_zenodo,
  title     = {Bugs-QCP: A Dataset of Quantum Computing Program Bugs},
  doi       = {10.5281/zenodo.5834281},
  publisher = {Zenodo}
}
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
