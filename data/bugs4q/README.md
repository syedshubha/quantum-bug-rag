# Bugs4Q Data Directory

Place normalised Bugs4Q samples here by running:

```bash
python scripts/prepare_bugs4q.py --output-dir data/bugs4q/
```

The preparation script writes separate prepared datasets:

- `samples.real.jsonl` for real Bugs4Q benchmark samples
- `samples.synthetic.jsonl` for smoke-test samples only
- `active_dataset.json` to declare which prepared dataset is currently active

We keep smoke-test and real data separate so the active dataset is always
explicit. A real preparation run replaces the active dataset manifest and
removes legacy unversioned dataset artifacts such as `samples.jsonl`.

See `data/README.md` for full instructions and dataset-role documentation.
