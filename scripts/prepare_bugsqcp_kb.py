#!/usr/bin/env python3
"""CLI wrapper for importing Bugs-QCP data into the local knowledge base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.bugsqcp_ingest import ingest_bugsqcp_into_kb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Bugs-QCP JSON/CSV records and merge them into knowledge_base.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/bugsqcp"),
        help="Directory containing Bugs-QCP JSON/CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("knowledge_base"),
        help="Knowledge base directory containing bug_patterns.json and taxonomy.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview import results without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = ingest_bugsqcp_into_kb(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Bugs-QCP ingestion completed")
    print(f"- discovered_files: {report.discovered_files}")
    print(f"- discovered_records: {report.discovered_records}")
    print(f"- imported_records: {report.imported_records}")
    print(f"- skipped_records: {report.skipped_records}")
    print(f"- duplicate_in_input: {report.duplicate_in_input}")
    print(f"- duplicate_with_existing: {report.duplicate_with_existing}")
    print(f"- manual_preserved_on_collision: {report.manual_preserved_on_collision}")
    print(f"- taxonomy_examples_added: {report.taxonomy_examples_added}")
    print(f"- final_pattern_count: {report.final_pattern_count}")
    print(f"- source_counts: {report.source_counts}")
    if args.dry_run:
        print("(dry-run mode: no files were modified)")


if __name__ == "__main__":
    main()
