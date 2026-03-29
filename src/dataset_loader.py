"""
dataset_loader.py – Loaders and normalisers for the Bugs4Q benchmark dataset.

We treat Bugs4Q as the primary evaluation corpus.  Raw dataset files are NOT
bundled in this repository; use scripts/prepare_bugs4q.py to fetch and place
them under data/bugs4q/.

For infrastructure smoke-testing only, we provide a synthetic sample generator
that must never be used when reporting benchmark results.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from .schemas import BugSample
from .utils import get_logger

logger = get_logger(__name__)

# ── Bugs4Q loader ─────────────────────────────────────────────────────────────

def load_bugs4q(data_dir: str | Path) -> list[BugSample]:
    """
    Load all normalised Bugs4Q samples from *data_dir*.

    Each sample is expected to be a JSON file matching the BugSample schema,
    or the directory may contain a single ``samples.jsonl`` file produced by
    ``prepare_bugs4q.py``.

    Returns a list of BugSample objects.
    """
    data_dir = Path(data_dir)
    samples: list[BugSample] = []

    # Prefer a pre-built JSONL catalogue if present.
    jsonl_path = data_dir / "samples.jsonl"
    if jsonl_path.exists():
        logger.info("Loading Bugs4Q samples from %s", jsonl_path)
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(BugSample(**json.loads(line)))
        logger.info("Loaded %d samples.", len(samples))
        return samples

    # Fall back to individual JSON files.
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        logger.warning(
            "No Bugs4Q samples found in '%s'. "
            "Run scripts/prepare_bugs4q.py first.",
            data_dir,
        )
        return samples

    for jf in json_files:
        try:
            with jf.open(encoding="utf-8") as fh:
                data = json.load(fh)
            samples.append(BugSample(**data))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed file '%s': %s", jf, exc)

    logger.info("Loaded %d samples from %s.", len(samples), data_dir)
    return samples


def iter_bugs4q(data_dir: str | Path) -> Iterator[BugSample]:
    """Yield BugSample objects one at a time (memory-efficient)."""
    yield from load_bugs4q(data_dir)


# ── Synthetic smoke-test data ─────────────────────────────────────────────────

_SMOKE_TAXONOMY_CLASSES = [
    "incorrect_operator",
    "incorrect_qubit_mapping",
    "missing_barrier",
    "wrong_initial_state",
    "measurement_error",
]

_SMOKE_CODE_TEMPLATES = [
    # Template with a plausible Qiskit pattern.
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\n",
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(1)\nqc.x(0)\nqc.measure_all()\n",
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(3)\nqc.ccx(0, 1, 2)\n",
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.swap(0, 1)\n",
    "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.ry(1.57, 0)\nqc.cx(0, 1)\n",
]


def generate_smoke_samples(n: int = 10, seed: int = 42) -> list[BugSample]:
    """
    Generate *n* synthetic BugSample objects for smoke-testing only.

    ⚠️  These samples are NOT real bugs and must NEVER be used for reporting
    benchmark results.  They exist solely to validate pipeline infrastructure.
    """
    rng = random.Random(seed)
    samples: list[BugSample] = []
    for i in range(n):
        label = rng.choice(_SMOKE_TAXONOMY_CLASSES)
        code = rng.choice(_SMOKE_CODE_TEMPLATES)
        samples.append(
            BugSample(
                sample_id=f"smoke_{i:04d}",
                source="synthetic_smoke_test",
                code=code,
                ground_truth=label,
                metadata={"synthetic": True},
            )
        )
    return samples
