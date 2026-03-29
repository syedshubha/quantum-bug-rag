# Knowledge Base

This directory contains the local knowledge base used by the RAG pipeline for
bug-pattern retrieval and taxonomy grounding.

## Files

| File | Description |
|------|-------------|
| `bug_patterns.json` | Starter bug-pattern entries derived from manual curation and Bugs-QCP-derived patterns. |
| `taxonomy.json` | Starter taxonomy for quantum bug classification. |

## Schema

### `bug_patterns.json`

A JSON array of objects matching the `BugPattern` Pydantic model (`src/schemas.py`):

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

### `taxonomy.json`

A JSON array of objects matching the `TaxonomyEntry` Pydantic model:

```json
{
  "class_id": "incorrect_operator",
  "name": "Incorrect Operator",
  "description": "...",
  "parent_class": null,
  "examples": ["..."]
}
```

## Enriching the Knowledge Base

To add Bugs-QCP-derived entries, download the Bugs-QCP archive from
[Zenodo 5834281](https://zenodo.org/records/5834281) and run:

```bash
python scripts/prepare_bugsqcp_kb.py \
    --input-dir /path/to/bugsqcp/ \
    --output-dir knowledge_base/
```

The script normalises each entry to the `BugPattern` schema and merges it into
`bug_patterns.json` without overwriting existing manually-curated entries.

## Leakage Note

The knowledge base must be constructed from the **Bugs-QCP corpus** and the
**Bugs4Q training split only**.  Evaluation-split Bugs4Q samples must never
appear in the knowledge base or be used to populate retrieved context during
evaluation.  See `docs/methodology.md` for details.
