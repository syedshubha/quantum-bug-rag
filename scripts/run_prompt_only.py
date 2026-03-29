#!/usr/bin/env python3
"""
run_prompt_only.py – Prompt-only LLM baseline.

I load Bugs4Q, build a simple prompt for each program (no retrieval), call
the LLM, and save structured diagnostics to ``outputs/prompt_only/``.

Usage
-----
    python scripts/run_prompt_only.py [--config config.yaml] [--backend mock] \
        [--max-items 10] [--out outputs/prompt_only]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark_runner import run_benchmark
from src.dataset_loader import load_bugs4q
from src.llm_client import build_llm_client
from src.prompt_builder import build_prompt_only
from src.schemas import DiagnosticResult
from src.utils import load_config, repo_root, save_json, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUT = repo_root() / "outputs" / "prompt_only"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML file")
    parser.add_argument("--backend", default="mock", choices=["mock", "openai", "gemini"])
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=str(Path(args.out) / "run.log"))

    # I load config if provided; CLI flags override config values.
    cfg: dict = {}
    if args.config:
        cfg = load_config(args.config)

    backend = args.backend or cfg.get("llm", {}).get("backend", "mock")
    llm_kwargs = cfg.get("llm", {}).get("kwargs", {})

    logger.info("=== Prompt-only baseline  backend=%s ===", backend)

    llm = build_llm_client(backend, **llm_kwargs)
    records = load_bugs4q(data_dir=args.data_dir, max_items=args.max_items)

    def analyser(record: dict) -> DiagnosticResult:
        prompt = build_prompt_only(
            program_id=record["id"],
            source_code=record["source_code"],
            description=record.get("description"),
        )
        return llm.analyse(record["id"], prompt, mode="prompt_only")

    results = run_benchmark(records, analyser)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    save_json([r.model_dump() for r in results], out_file)
    logger.info("Saved %d results to %s", len(results), out_file)


if __name__ == "__main__":
    main()
