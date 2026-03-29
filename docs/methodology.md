# Methodology Note

**Project**: CSC 7135 – Retrieval-Augmented LLM Bug Detection for Qiskit Programs  
**Author**: (your name)  
**Date**: 2024

---

## 1. Motivation

Quantum software is increasingly important yet error-prone.  Classical static
analysis tools do not understand quantum semantics, and LLMs trained on
general-purpose code may lack sufficient Qiskit-specific knowledge.  I
hypothesise that augmenting LLM prompts with retrieved bug-pattern context
(RAG) improves both detection accuracy and taxonomy classification over
prompt-only and static-analysis baselines.

---

## 2. Research Questions

| ID  | Question |
|-----|----------|
| RQ1 | Does RAG improve bug-detection F1 over the prompt-only baseline? |
| RQ2 | Does RAG improve taxonomy classification Macro-F1? |
| RQ3 | How does the LLM-based approach compare to simple static analysis? |

---

## 3. Pipeline Overview

```
Bugs4Q record
    │
    ├─ [static] ──► StaticAnalyser ──────────────────────────────► DiagnosticResult
    │
    ├─ [prompt only] ──► PromptBuilder ──► LLMClient ───────────► DiagnosticResult
    │
    └─ [RAG] ──► KnowledgeBaseRetriever
                      │
                      └──► PromptBuilder (augmented) ──► LLMClient ──► DiagnosticResult
```

Each `DiagnosticResult` contains:

- `bug_likelihood` ∈ [0, 1]  
- `taxonomy_class` from the Bugs4Q taxonomy  
- `suspected_location` (file:line, if available)  
- `justification` (free-text reasoning)

---

## 4. Dataset

I use **Bugs4Q** (Zhao et al., ASE 2021), a curated benchmark of real bugs
from open-source Qiskit programs.  The benchmark provides:

- Buggy and fixed versions of each program.  
- A bug taxonomy covering eight categories (see `knowledge_base/taxonomy.json`).

For preliminary experiments I use a small subset (≤ 20 programs) to validate
the pipeline before scaling up.

---

## 5. Knowledge Base

The `knowledge_base/` directory contains:

- **`bug_patterns.json`** – hand-curated bug-pattern entries, each with a
  title, description, example code snippet, and keyword list.
- **`taxonomy.json`** – the eight top-level bug classes and their definitions.

Retrieval in the first version is keyword-based (overlap scoring).  Future
work will replace this with dense vector retrieval (FAISS + sentence-transformers).

---

## 6. Baselines

| Mode          | Description |
|---------------|-------------|
| `static`      | Rule-based regex/AST checks against a hand-crafted rule set. |
| `prompt_only` | LLM prompt with source code only; no additional context. |
| `rag`         | LLM prompt augmented with top-k retrieved bug patterns. |

---

## 7. Evaluation Metrics

| Metric                     | Task               |
|----------------------------|--------------------|
| Detection Precision/Recall/F1 | Binary bug detection |
| Classification Macro-F1    | Taxonomy class prediction |

A program is predicted as *buggy* when `bug_likelihood ≥ 0.5`.

---

## 8. Reproducibility

- All random seeds are fixed where applicable.  
- LLM temperature is set to 0.  
- The mock LLM mode enables fully deterministic pipeline runs.  
- Configuration is stored in `config.yaml` (see `config.example.yaml`).

---

## 9. Limitations and Future Work

- The first knowledge base is small and manually curated; future work should
  expand it automatically from Bugs4Q and similar datasets.
- Dense retrieval should replace keyword overlap for better semantic matching.
- Few-shot examples should be drawn dynamically from the knowledge base.
- Evaluation should cover the full Bugs4Q benchmark, not just a subset.
