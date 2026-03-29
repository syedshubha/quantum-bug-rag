#!/usr/bin/env python3
"""
prepare_bugs4q.py – Fetch and normalise the Bugs4Q benchmark dataset.

Bugs4Q is the primary evaluation dataset for this project.  Raw dataset files
are NOT bundled in this repository; this script clones the upstream repository
and converts its contents into the BugSample JSON schema used by our pipeline.

Usage
-----
    # Full preparation (requires internet access):
    python scripts/prepare_bugs4q.py --output-dir data/bugs4q/

    # Smoke-test mode (generates synthetic samples; for pipeline testing only):
    python scripts/prepare_bugs4q.py --smoke-test --output-dir data/bugs4q/

⚠️  Smoke-test data must NEVER be used for reporting benchmark results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import generate_smoke_samples
from src.utils import get_logger

logger = get_logger("prepare_bugs4q")

BUGS4Q_REPO_URL = "https://github.com/Z-928/Bugs4Q.git"


def _clone_or_update(target_dir: Path) -> None:
    if (target_dir / ".git").exists():
        logger.info("Updating existing Bugs4Q clone at %s …", target_dir)
        subprocess.run(["git", "-C", str(target_dir), "pull", "--ff-only"], check=True)
    else:
        logger.info("Cloning Bugs4Q into %s …", target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", BUGS4Q_REPO_URL, str(target_dir)], check=True)


def _normalise_bugs4q(source_dir: Path, output_dir: Path) -> int:
    """
    Walk *source_dir* for Python files, emit one BugSample JSON per file.

    This is a best-effort normalisation; adapt it to the actual Bugs4Q layout.
    Returns the count of samples written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Bugs4Q typically has subdirectories per bug; each contains buggy/ and fixed/ Python files.
    for py_file in sorted(source_dir.rglob("*.py")):
        # Skip __init__ and conftest files.
        if py_file.name.startswith("_") or py_file.stem == "conftest":
            continue
        code = py_file.read_text(encoding="utf-8", errors="replace")
        if not code.strip():
            continue

        # Derive a rough taxonomy class from the path structure when available.
        ground_truth: str | None = None
        parts = py_file.parts
        if "buggy" in parts:
            idx = parts.index("buggy")
            ground_truth = parts[idx - 1] if idx > 0 else None

        sample = {
            "sample_id": f"bugs4q_{py_file.stem}_{count:04d}",
            "source": "bugs4q",
            "code": code,
            "ground_truth": ground_truth,
            "metadata": {"path": str(py_file.relative_to(source_dir))},
        }
        out_path = output_dir / f"bugs4q_{count:04d}.json"
        out_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
        count += 1

    return count


def _write_smoke_test(output_dir: Path, n: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = generate_smoke_samples(n=n)
    jsonl_path = output_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")
    logger.warning(
        "⚠  Wrote %d SYNTHETIC smoke-test samples to %s. "
        "These must NOT be used for reporting benchmark results.",
        n,
        jsonl_path,
    )
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Bugs4Q dataset.")
    parser.add_argument("--output-dir", default="data/bugs4q/", help="Output directory.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Write synthetic smoke-test samples instead of real data.",
    )
    parser.add_argument(
        "--smoke-n", type=int, default=20, help="Number of synthetic samples (smoke-test only)."
    )
    parser.add_argument(
        "--bugs4q-dir",
        default=None,
        help="Path to an existing Bugs4Q clone (skips git clone/pull).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.smoke_test:
        count = _write_smoke_test(output_dir, args.smoke_n)
    else:
        bugs4q_src = Path(args.bugs4q_dir) if args.bugs4q_dir else Path("_tmp_bugs4q_clone")
        _clone_or_update(bugs4q_src)
        count = _normalise_bugs4q(bugs4q_src, output_dir)
        logger.info("Normalised %d Bugs4Q samples into '%s'.", count, output_dir)

    print(f"Done. {count} samples written to {output_dir}.")


if __name__ == "__main__":
    main()
