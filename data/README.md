# Data Directory

This directory stores locally prepared datasets.  **No raw external dataset
files are committed to this repository.**

## Subdirectories

| Path | Contents |
|------|----------|
| `bugs4q/` | Normalised Bugs4Q samples (populated by `scripts/prepare_bugs4q.py`). |

## Dataset Roles

| Dataset | Role |
|---------|------|
| **Bugs4Q** | Primary benchmark and evaluation dataset. All reported metrics are computed on Bugs4Q. |
| **Bugs-QCP** | Secondary corpus for knowledge-base enrichment and taxonomy grounding only. Not used as an evaluation dataset. |

## Obtaining Bugs4Q

Bugs4Q is an executable benchmark of real Qiskit bugs maintained at
<https://github.com/Z-928/Bugs4Q>.  To prepare it:

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q/
```

This clones the upstream repository and converts its contents into the
`BugSample` JSON schema under `data/bugs4q/`.

### Smoke-Test Mode

For pipeline infrastructure validation only (no real data required):

```bash
python scripts/prepare_bugs4q.py --smoke-test --output-dir data/bugs4q/
```

> ⚠️ Synthetic smoke-test samples must **never** be used when reporting
> benchmark results.  They exist solely to validate that the pipeline runs
> end-to-end without errors.

## Obtaining Bugs-QCP

Bugs-QCP is available from Zenodo at <https://zenodo.org/records/5834281>.
Download the archive, extract it, then run:

```bash
python scripts/prepare_bugsqcp_kb.py \
    --input-dir /path/to/bugsqcp/ \
    --output-dir knowledge_base/
```

This does **not** copy raw Bugs-QCP files into this repository; it only
enriches the knowledge base with normalised pattern entries.

## Split Discipline

When running full Bugs4Q evaluations, maintain a strict training/evaluation
split.  The knowledge base must not contain any samples from the evaluation
split.  See `docs/methodology.md` for details.
