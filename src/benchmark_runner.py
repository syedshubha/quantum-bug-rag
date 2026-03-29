"""
benchmark_runner.py – Orchestrates running a pipeline mode over Bugs4Q.

I accept a dataset (list of program records) and a callable that maps each
record to a ``DiagnosticResult``.  I handle logging, error recovery, and
optional subset truncation so that the individual pipeline scripts stay thin.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .schemas import DiagnosticResult

logger = logging.getLogger(__name__)

# I type the analyser callable: program record → DiagnosticResult.
AnalyserFn = Callable[[Dict[str, Any]], DiagnosticResult]


def run_benchmark(
    records: List[Dict[str, Any]],
    analyser: AnalyserFn,
    max_items: Optional[int] = None,
    delay_seconds: float = 0.0,
) -> List[DiagnosticResult]:
    """
    I iterate over *records*, call *analyser* for each one, and return the
    collected results.

    Parameters
    ----------
    records:
        Program records produced by ``dataset_loader.load_bugs4q``.
    analyser:
        A function that takes a single record dict and returns a
        ``DiagnosticResult``.
    max_items:
        If set, I process only the first *max_items* records.  Useful for
        quick subset experiments.
    delay_seconds:
        Optional sleep between calls – handy for respecting rate limits
        when using a real LLM API.
    """
    if max_items is not None:
        records = records[:max_items]

    total = len(records)
    logger.info("Starting benchmark run: %d programs", total)

    results: List[DiagnosticResult] = []

    for i, record in enumerate(records, start=1):
        program_id = record.get("id", f"prog_{i}")
        logger.debug("[%d/%d] Analysing %s", i, total, program_id)

        try:
            result = analyser(record)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error analysing %s: %s", program_id, exc, exc_info=True)
            result = DiagnosticResult(
                program_id=program_id,
                bug_likelihood=0.5,
                taxonomy_class="unknown",
                justification=f"Pipeline error: {exc}",
                mode="error",
            )

        results.append(result)

        if delay_seconds > 0 and i < total:
            time.sleep(delay_seconds)

    logger.info("Benchmark run complete: %d results", len(results))
    return results
