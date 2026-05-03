"""
classical — Binary quantum-vs-classical bug classification.

This sub-package refactors the
``quantum-software-bug-detection-rag-project-v6_classical.ipynb`` notebook
into reusable modules. The pipeline classifies whether a buggy snapshot
from a quantum-software repository contains a *quantum* bug (defect in
quantum-specific logic) or a *classical* bug (defect in surrounding
classical infrastructure).

Three retrieval configurations are supported:

  - ``prompt_only``     : LLM classifies from code alone.
  - ``biased_rag``      : BM25 over a quantum-only KB. Control condition
                          to demonstrate the asymmetric-KB confound.
  - ``balanced_rag``    : BM25 within each domain separately, then
                          concatenate top-K from each. The recommended
                          configuration.

Modules:
  schemas      Dataclasses (BugSample, KBEntry, Diagnostic).
  dataset      BQCP and Bugs4Q loaders for the binary task.
  kb           Quantum + classical KB extractors and symmetric balancing.
  retriever    BM25Retriever and BalancedRetriever.
  prompts      Binary classifier prompt and prompt builders.
  llm          OpenAI / mock LLM client.
  evaluator    run_dataset, _predict, Diagnostic emission.
  analysis     Bootstrap CIs, Brier score, reliability bins, per-class recall.
"""

from .schemas import BugSample, KBEntry, Diagnostic, CLASSES

__all__ = ["BugSample", "KBEntry", "Diagnostic", "CLASSES"]
