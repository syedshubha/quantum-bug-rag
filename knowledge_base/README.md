# Knowledge Base

This directory documents the schema, ingestion pipeline, and query interface for the
`quantum-bug-rag` knowledge base (KB).  The KB stores quantum bug patterns in a vector
store to support retrieval-augmented generation (RAG).

---

## Overview

The knowledge base is populated from two sources:

| Source | Role |
|--------|------|
| **Bugs4Q** training split | Bug examples paired with labels — used to ground retrieval in verified benchmark data |
| **Bugs-QCP** (Zenodo 5834281) | Broader taxonomy of bug patterns — used to enrich retrieval vocabulary |

**Evaluation-only constraint:** The Bugs4Q *test* split must **never** be ingested into the
knowledge base.  Doing so would contaminate the evaluation and invalidate reported metrics.

---

## Schema

Each document stored in the KB has the following fields:

```json
{
  "id":           "<source>-<identifier>",
  "bug_type":     "<category string>",
  "description":  "<natural-language description of the bug pattern>",
  "platform":     "<qiskit | cirq | any | ...>",
  "code_snippet": "<illustrative code excerpt or null>",
  "tags":         ["<tag>", "..."],
  "source":       "<bugs4q-train | bugs_qcp>",
  "source_doi":   "<DOI or URL>",
  "embedding":    [0.123, ...]   // populated at ingestion time; not stored as plain JSON
}
```

The `embedding` field is stored natively by the vector store backend (e.g., ChromaDB,
FAISS, Weaviate).

---

## Ingestion Pipeline

### 1. Prerequisites

Ensure both datasets have been prepared:

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q
python scripts/prepare_bugs_qcp.py --output-dir data/bugs_qcp
```

Only the **training and validation splits** of Bugs4Q may be ingested.

### 2. Ingest Bugs4Q training patterns

```bash
python src/kb/ingest.py \
    --source bugs4q \
    --input  data/bugs4q/train/bugs4q.jsonl \
    --kb-dir knowledge_base/store
```

### 3. Ingest Bugs-QCP patterns

```bash
python src/kb/ingest.py \
    --source bugs_qcp \
    --input  data/bugs_qcp/patterns.jsonl \
    --kb-dir knowledge_base/store
```

### 4. Verify ingestion

```bash
python src/kb/stats.py --kb-dir knowledge_base/store
```

This prints the number of documents per source, a sample of stored entries, and the
embedding-space statistics.

---

## Querying the Knowledge Base

```python
from src.kb import KnowledgeBase

kb = KnowledgeBase(kb_dir="knowledge_base/store")

results = kb.retrieve(
    query="qubit measurement performed before circuit execution",
    top_k=5,
    filter={"platform": "qiskit"},   # optional metadata filter
)

for doc in results:
    print(doc["id"], doc["bug_type"], doc["description"])
```

---

## Extending the Knowledge Base

To add new bug patterns without re-ingesting existing data:

1. Prepare a JSON-Lines file following the schema above (omit the `embedding` field).
2. Run the ingestion script with `--mode append`:

```bash
python src/kb/ingest.py \
    --source custom \
    --input  /path/to/new_patterns.jsonl \
    --kb-dir knowledge_base/store \
    --mode   append
```

---

## Notes on Synthetic Data

Synthetic fixtures in `tests/fixtures/` are **not** ingested into the knowledge base.
They exist solely for pipeline smoke tests and CI validation.

---

## Versioning

Each ingestion run appends a record to `knowledge_base/store/ingest_log.jsonl` with
the following fields:

```json
{
  "timestamp":   "<ISO-8601>",
  "source":      "<bugs4q-train | bugs_qcp | custom>",
  "input_file":  "<path>",
  "doc_count":   123,
  "kb_version":  "1.0.0"
}
```

Increment `kb_version` whenever the schema or embedding model changes to ensure
reproducibility.
