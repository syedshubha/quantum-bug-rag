"""
taxonomy_v6 — Five-class quantum bug taxonomy classification.

This sub-package refactors the ``quantum_bug_detecttion_taxonomy.ipynb`` notebook
into reusable modules. The pipeline is forced-choice (no "no-bug" outcome): the
input is assumed buggy, and the classifier must assign one of five classes.

Modules:
  dataset      Bugs4Q + Bugs-QCP loaders that emit BugSample dataclasses.
  kb           Validated KB extractors (Qiskit/Aer/Ignis YAML release notes,
               IBM Runtime RST, PennyLane changelog, LintQ rules).
  retriever    Framework-aware BM25 retriever with a hard score floor.
  prompts      Forced-choice classifier prompt and prompt builders.
  llm          OpenAI / mock client used by the v6 evaluation runs.
  analysis     Dev/test split helpers, confidence intervals, McNemar, ECE.
  evaluator    Structured-output evaluation helpers and diagnostics writers.
"""

from .schemas import (
    BugSample,
    BugDiagnostic,
    BugPattern,
    TAXONOMY_FORCED,
    TAXONOMY_ALL,
)

__all__ = [
    "BugSample",
    "BugDiagnostic",
    "BugPattern",
    "TAXONOMY_FORCED",
    "TAXONOMY_ALL",
]
