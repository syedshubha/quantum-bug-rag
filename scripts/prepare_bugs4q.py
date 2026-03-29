#!/usr/bin/env python3
"""
prepare_bugs4q.py
=================
Download and pre-process the Bugs4Q benchmark dataset.

Bugs4Q is the **primary evaluation benchmark** for quantum-bug-rag.  All reported
metrics must be computed on the test split produced by this script.

Source
------
  https://github.com/Z-928/Bugs4Q

Usage
-----
  python scripts/prepare_bugs4q.py --output-dir data/bugs4q [--seed 42]

Output layout
-------------
  <output-dir>/
    train/          # training split (JSONL)
    validation/     # validation split (JSONL)
    test/           # test split (JSONL) — keep sealed during development
    manifest.json   # provenance: source URL, commit SHA, preparation date
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BUGS4Q_REPO_URL = "https://github.com/Z-928/Bugs4Q.git"

# Approximate train / validation / test proportions.
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _clone_bugs4q(target_dir: Path) -> str:
    """Shallow-clone Bugs4Q into *target_dir* and return the HEAD commit SHA."""
    logger.info("Cloning Bugs4Q from %s …", BUGS4Q_REPO_URL)
    subprocess.run(
        ["git", "clone", "--depth", "1", BUGS4Q_REPO_URL, str(target_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["git", "-C", str(target_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".tox", ".venv", "venv"})


def _collect_entries(repo_dir: Path) -> list[dict]:
    """
    Walk the cloned Bugs4Q repository and collect bug entries.

    The Bugs4Q layout uses per-bug subdirectories.  Each directory is
    expected to contain at least a buggy Python file.  Adapt this function
    if the upstream layout changes.
    """
    entries: list[dict] = []
    for bug_dir in sorted(repo_dir.iterdir()):
        if not bug_dir.is_dir():
            continue
        if bug_dir.name.startswith(".") or bug_dir.name in _SKIP_DIRS:
            continue
        py_files = sorted(bug_dir.glob("*.py"))
        if not py_files:
            continue
        entry = {
            "id": bug_dir.name,
            "source": str(bug_dir.relative_to(repo_dir)),
            "files": [str(f.relative_to(repo_dir)) for f in py_files],
        }
        # If the directory contains a metadata file, merge it.
        meta_path = bug_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                entry.update(json.load(fh))
        entries.append(entry)
    return entries


def _split_entries(
    entries: list[dict],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[dict]]:
    """Randomly split *entries* according to *ratios*."""
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["validation"])
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _write_split(split_dir: Path, entries: list[dict]) -> None:
    """Write *entries* as a JSONL file inside *split_dir*."""
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / "bugs4q.jsonl"
    with open(out_path, "w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("  %s: %d entries → %s", split_dir.name, len(entries), out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bugs4q_clone_") as tmp:
        repo_dir = Path(tmp) / "Bugs4Q"
        commit_sha = _clone_bugs4q(repo_dir)

        entries = _collect_entries(repo_dir)
        if not entries:
            logger.error(
                "No bug entries found in the cloned repository.  "
                "The upstream layout may have changed — please inspect %s and update "
                "_collect_entries().",
                repo_dir,
            )
            sys.exit(1)

        logger.info("Collected %d bug entries.", len(entries))
        splits = _split_entries(entries, SPLIT_RATIOS, seed)

        for split_name, split_entries in splits.items():
            _write_split(output_dir / split_name, split_entries)

    manifest = {
        "dataset": "Bugs4Q",
        "source_url": BUGS4Q_REPO_URL,
        "commit_sha": commit_sha,
        "prepared_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": seed,
        "split_ratios": SPLIT_RATIOS,
        "split_sizes": {name: len(s) for name, s in splits.items()},
        "note": (
            "The test split must remain sealed during model development. "
            "Do not use test-split labels for tuning or knowledge-base construction."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest written to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bugs4q"),
        help="Directory to write processed splits (default: data/bugs4q)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split shuffling (default: 42)",
    )
    args = parser.parse_args()
    prepare(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
