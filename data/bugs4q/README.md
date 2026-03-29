# Bugs4Q Data Directory

Place normalised Bugs4Q samples here by running:

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q/
```

The script produces either individual JSON files (`bugs4q_NNNN.json`) or a
single `samples.jsonl` catalogue, depending on the source format.  The dataset
loader in `src/dataset_loader.py` supports both layouts.

See `data/README.md` for full instructions and dataset-role documentation.
