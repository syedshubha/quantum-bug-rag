#!/usr/bin/env python3
"""Inspect the local knowledge base for coverage and quality issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.knowledge_base_inspect import inspect_knowledge_base, summarize_inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect knowledge base quality and coverage.")
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("knowledge_base"),
        help="Directory containing bug_patterns.json.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspection = inspect_knowledge_base(args.kb_dir)
    summary = summarize_inspection(inspection)

    if args.as_json:
        print(json.dumps(summary, indent=2))
        return

    print("Knowledge base inspection")
    print(f"- total_entries: {summary['total_entries']}")
    print(f"- valid_entries: {summary['valid_entries']}")
    print(f"- invalid_entries: {summary['invalid_entries']}")
    print(f"- duplicate_ids: {summary['duplicate_ids']}")
    print(f"- invalid_entry_indices: {summary['invalid_entry_indices']}")
    print(f"- by_source: {summary['by_source']}")
    print(f"- by_taxonomy_class: {summary['by_taxonomy_class']}")


if __name__ == "__main__":
    main()
