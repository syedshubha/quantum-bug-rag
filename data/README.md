# Data Directory

This directory holds locally prepared datasets.  **No raw dataset files are committed to
the repository.**  Use the preparation scripts in [`../scripts/`](../scripts/) to populate
each subdirectory.

---

## Directory Structure

```
data/
├── bugs4q/       # Bugs4Q benchmark (primary evaluation dataset)
└── bugs_qcp/     # Bugs-QCP taxonomy corpus (knowledge-base enrichment only)
```

Both subdirectories are listed in `.gitignore` and must be populated locally.

---

## Bugs4Q — Primary Benchmark

**Role:** Executable benchmark for all reported evaluation metrics.

**Source:** <https://github.com/Z-928/Bugs4Q>

### Preparation

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q
```

The script will:
1. Clone or download the Bugs4Q repository into a temporary location.
2. Validate the expected directory layout and file checksums.
3. Split the data into `train`, `validation`, and `test` splits under `data/bugs4q/`.
4. Write a `manifest.json` with provenance information (source URL, commit SHA, date).

**Important:** The `test` split must remain sealed during model development.  Do not
inspect test-split labels when tuning hyperparameters or building the knowledge base.

---

## Bugs-QCP — Taxonomy Corpus (Zenodo 5834281)

**Role:** Secondary corpus for bug-pattern retrieval and knowledge-base enrichment.
**Not used for evaluation.**

**Source:** <https://doi.org/10.5281/zenodo.5834281>

### Preparation

```bash
python scripts/prepare_bugs_qcp.py --output-dir data/bugs_qcp
```

The script will:
1. Download the Zenodo archive for record 5834281.
2. Extract and validate the archive against the published checksum.
3. Normalise bug entries into a JSON-Lines format under `data/bugs_qcp/patterns.jsonl`.
4. Write a `manifest.json` with DOI, download URL, and processing date.

### Output Schema (`patterns.jsonl`)

Each line is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique pattern identifier (prefixed `qcp-`) |
| `bug_type` | `str` | High-level bug category (e.g., `measurement_error`) |
| `description` | `str` | Natural-language description of the bug pattern |
| `platform` | `str` | Quantum framework (e.g., `qiskit`, `cirq`, `any`) |
| `code_snippet` | `str` \| `null` | Illustrative code excerpt, if available |
| `tags` | `list[str]` | Free-form taxonomy tags |
| `source_doi` | `str` | `10.5281/zenodo.5834281` |

---

## Synthetic Fixtures

A small set of hand-crafted synthetic examples lives in `tests/fixtures/`.  These are used
**exclusively** for smoke testing and CI validation.  They do not represent real bugs and
must never be included in evaluation runs.
