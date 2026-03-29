#!/usr/bin/env python3
"""
run_static_baseline.py – Rule-based static-analysis baseline.

I run the static analyser from ``src/baselines.py`` over the Bugs4Q programs
and save structured diagnostics to ``outputs/static/``.

Usage
-----
    python scripts/run_static_baseline.py [--max-items 10] [--out outputs/static]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import analyse_static
from src.benchmark_runner import run_benchmark
from src.dataset_loader import load_bugs4q
from src.schemas import DiagnosticResult
from src.utils import repo_root, save_json, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUT = repo_root() / "outputs" / "static"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=str(Path(args.out) / "run.log"))
    logger.info("=== Static-analysis baseline ===")

    records = load_bugs4q(data_dir=args.data_dir, max_items=args.max_items)

    def analyser(record: dict) -> DiagnosticResult:
        return analyse_static(
            program_id=record["id"],
            source_code=record["source_code"],
        )

    results = run_benchmark(records, analyser)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    save_json([r.model_dump() for r in results], out_file)
    logger.info("Saved %d results to %s", len(results), out_file)


if __name__ == "__main__":
    main()
