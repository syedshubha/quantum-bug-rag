# Knowledge Base

This directory documents the JSON knowledge base used by the legacy scaffold.

## Scope

`knowledge_base/` is the active KB format for the original top-level RAG pipeline:

- `scripts/run_rag.py`
- `src/knowledge_ingest.py`
- `src/retriever.py`

The newer `taxonomy_v6` and `classical` tracks do not read `knowledge_base/bug_patterns.json` directly. They build their own KBs dynamically from upstream release notes and rule summaries.

## Files

| File | Purpose |
|------|---------|
| `bug_patterns.json` | Legacy JSON array of bug-pattern entries |
| `taxonomy.json` | Legacy taxonomy definition |
| `taxonomy_mapping_hints.json` | Auxiliary mapping hints used by the project |

## Legacy JSON Schema

Example `bug_patterns.json` entry:

```json
{
  "pattern_id": "BP001",
  "name": "CNOT Self-Loop",
  "taxonomy_class": "incorrect_qubit_mapping",
  "description": "...",
  "example_code": "...",
  "fix_hint": "...",
  "source": "manual | bugsqcp | ...",
  "tags": ["cx", "qubit_mapping"]
}
```

Example `taxonomy.json` entry:

```json
{
  "class_id": "incorrect_operator",
  "name": "Incorrect Operator",
  "description": "...",
  "parent_class": null,
  "examples": ["..."]
}
```

## Legacy KB Enrichment

```bash
python scripts/prepare_bugsqcp_kb.py \
  --input-dir /path/to/bugsqcp \
  --output-dir knowledge_base/
```

This normalizes Bugs-QCP data into the legacy JSON schema.

## KBs In The Newer Tracks

`taxonomy_v6` builds a validated quantum KB from:

- Qiskit-family release notes;
- IBM Runtime release notes;
- PennyLane changelogs;
- embedded LintQ summaries.

`classical` builds a symmetric KB with:

- quantum entries from Qiskit-family and PennyLane release notes;
- classical entries from CPython and NumPy release notes.

Those KBs are generated in memory at runtime and are not persisted in this directory by default.
