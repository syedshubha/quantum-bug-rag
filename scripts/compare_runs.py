#!/usr/bin/env python3
"""
compare_runs.py – Generate comparative analysis from prompt_only and RAG runs.

Reads the latest diagnostics JSONL for each mode from the outputs directory,
computes metrics, and produces:
  - comparative JSON
  - comparative CSV
  - markdown summary suitable for slides
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import compute_metrics
from src.schemas import BugDiagnostic, EvalSummary
from src.utils import get_logger, read_jsonl, save_json

logger = get_logger("compare_runs")


def find_latest_diagnostics(output_dir: Path, mode: str) -> Path | None:
    """Return the most recently modified diagnostics file for *mode*."""
    pattern = f"diagnostics_{mode}_*.jsonl"
    candidates = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_diagnostics(path: Path) -> list[BugDiagnostic]:
    records = read_jsonl(path)
    return [BugDiagnostic(**r) for r in records]


def summary_to_row(s: EvalSummary) -> dict:
    return {
        "mode": s.mode,
        "num_samples": s.num_samples,
        "accuracy": s.accuracy,
        "precision_macro": s.precision_macro,
        "recall_macro": s.recall_macro,
        "f1_macro": s.f1_macro,
        **{f"f1_{k}": v for k, v in (s.per_class_f1 or {}).items()},
    }


def error_analysis(
    po_diags: list[BugDiagnostic],
    rag_diags: list[BugDiagnostic],
    po_summary: EvalSummary,
    rag_summary: EvalSummary,
) -> str | None:
    """If metrics are identical, generate a short error-analysis explanation."""
    if (
        po_summary.accuracy == rag_summary.accuracy
        and po_summary.f1_macro == rag_summary.f1_macro
        and po_summary.precision_macro == rag_summary.precision_macro
        and po_summary.recall_macro == rag_summary.recall_macro
    ):
        # Check if predictions are literally the same
        po_map = {d.sample_id: d.taxonomy_class for d in po_diags if d.ground_truth is not None}
        rag_map = {d.sample_id: d.taxonomy_class for d in rag_diags if d.ground_truth is not None}
        identical_preds = all(po_map.get(k) == rag_map.get(k) for k in po_map)

        lines = [
            "## Error Analysis: Identical Metrics",
            "",
            f"Prompt-only and RAG produced **identical aggregate metrics** "
            f"(accuracy={po_summary.accuracy}, F1={po_summary.f1_macro}).",
            "",
        ]

        if identical_preds:
            lines.append(
                "**All individual predictions are identical.** Possible explanations:"
            )
        else:
            lines.append(
                "Individual predictions differ but cancel out in aggregate. Possible explanations:"
            )

        lines += [
            "",
            "1. **Retrieved context did not change the prompt meaningfully**: "
            "The TF-IDF retriever may have returned patterns with low relevance, "
            "leaving the effective prompt largely unchanged.",
            "",
            "2. **Model ignored retrieval context**: "
            "gpt-4o-mini may anchor strongly on the code snippet and system prompt, "
            "giving minimal weight to prepended context blocks.",
            "",
            "3. **Output normalisation collapsed differences**: "
            "Both modes map free-text LLM output to the same taxonomy class set. "
            "If the model consistently picks the same class regardless of context, "
            "normalisation masks any subtle shifts in reasoning.",
        ]
        return "\n".join(lines)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prompt_only vs RAG benchmark runs.")
    parser.add_argument("--output-dir", default="outputs/", help="Directory with diagnostics.")
    parser.add_argument("--po-file", default=None, help="Explicit prompt_only diagnostics JSONL.")
    parser.add_argument("--rag-file", default=None, help="Explicit RAG diagnostics JSONL.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    po_path = Path(args.po_file) if args.po_file else find_latest_diagnostics(output_dir, "prompt_only")
    rag_path = Path(args.rag_file) if args.rag_file else find_latest_diagnostics(output_dir, "rag")

    if not po_path or not po_path.exists():
        logger.error("No prompt_only diagnostics found in %s", output_dir)
        sys.exit(1)
    if not rag_path or not rag_path.exists():
        logger.error("No RAG diagnostics found in %s", output_dir)
        sys.exit(1)

    logger.info("prompt_only: %s", po_path.name)
    logger.info("rag:         %s", rag_path.name)

    po_diags = load_diagnostics(po_path)
    rag_diags = load_diagnostics(rag_path)

    po_summary = compute_metrics(po_diags, mode="prompt_only")
    rag_summary = compute_metrics(rag_diags, mode="rag")

    # ── Comparative JSON ──────────────────────────────────────────────────
    comparative = {
        "prompt_only": po_summary.model_dump(),
        "rag": rag_summary.model_dump(),
        "delta": {
            "accuracy": round(rag_summary.accuracy - po_summary.accuracy, 4),
            "f1_macro": round(rag_summary.f1_macro - po_summary.f1_macro, 4),
            "precision_macro": round(rag_summary.precision_macro - po_summary.precision_macro, 4),
            "recall_macro": round(rag_summary.recall_macro - po_summary.recall_macro, 4),
        },
    }

    json_path = output_dir / "comparative_results.json"
    save_json(comparative, json_path)
    logger.info("Wrote %s", json_path)

    # ── Comparative CSV ───────────────────────────────────────────────────
    csv_path = output_dir / "comparative_results.csv"
    rows = [summary_to_row(po_summary), summary_to_row(rag_summary)]
    fieldnames = list(rows[0].keys())
    # Merge any extra keys from RAG row
    for k in rows[1]:
        if k not in fieldnames:
            fieldnames.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", csv_path)

    # ── Markdown summary ─────────────────────────────────────────────────
    md_lines = [
        "# Benchmark Results: Prompt-Only vs RAG",
        "",
        f"**Dataset**: Bugs4Q (45 labelled samples)  ",
        f"**Backend**: GitHub Models / gpt-4o-mini  ",
        f"**Date**: auto-generated",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Prompt-Only | RAG | Delta |",
        "|--------|------------|-----|-------|",
        f"| Accuracy | {po_summary.accuracy:.4f} | {rag_summary.accuracy:.4f} | {rag_summary.accuracy - po_summary.accuracy:+.4f} |",
        f"| Precision (macro) | {po_summary.precision_macro:.4f} | {rag_summary.precision_macro:.4f} | {rag_summary.precision_macro - po_summary.precision_macro:+.4f} |",
        f"| Recall (macro) | {po_summary.recall_macro:.4f} | {rag_summary.recall_macro:.4f} | {rag_summary.recall_macro - po_summary.recall_macro:+.4f} |",
        f"| F1 (macro) | {po_summary.f1_macro:.4f} | {rag_summary.f1_macro:.4f} | {rag_summary.f1_macro - po_summary.f1_macro:+.4f} |",
        "",
    ]

    # Per-class F1
    all_classes = sorted(
        set(list((po_summary.per_class_f1 or {}).keys()) + list((rag_summary.per_class_f1 or {}).keys()))
    )
    if all_classes:
        md_lines += [
            "## Per-Class F1",
            "",
            "| Class | Prompt-Only | RAG | Delta |",
            "|-------|------------|-----|-------|",
        ]
        for cls in all_classes:
            po_v = (po_summary.per_class_f1 or {}).get(cls, 0.0)
            rag_v = (rag_summary.per_class_f1 or {}).get(cls, 0.0)
            md_lines.append(f"| {cls} | {po_v:.4f} | {rag_v:.4f} | {rag_v - po_v:+.4f} |")
        md_lines.append("")

    # Error analysis if needed
    ea = error_analysis(po_diags, rag_diags, po_summary, rag_summary)
    if ea:
        md_lines += ["", ea, ""]

    md_path = output_dir / "benchmark_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    # Print summary
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
