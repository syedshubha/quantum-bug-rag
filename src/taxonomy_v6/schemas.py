"""Lightweight dataclasses for the v6 taxonomy track.

The notebook used plain dataclasses rather than the project's Pydantic
schemas in ``src/schemas.py``. We keep the same shape here so that the
refactored modules drop in unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Forced-choice taxonomy used by the classifier prompt and metrics.
TAXONOMY_FORCED: list[str] = [
    "incorrect_operator",
    "incorrect_qubit_mapping",
    "missing_barrier",
    "wrong_initial_state",
    "measurement_error",
]

# 'unknown' is reserved for KB-tagging fallbacks; it also appears as a
# parser-failure fallback in metric computation.
TAXONOMY_ALL: list[str] = TAXONOMY_FORCED + ["unknown"]


@dataclass
class BugSample:
    sample_id: str
    source: str
    code: str
    ground_truth: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BugDiagnostic:
    sample_id: str
    mode: str
    bug_likelihood: float
    taxonomy_class: str
    class_scores: dict = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    suspected_location: str = ""
    justification: str = ""
    ground_truth: Optional[str] = None
    correct: Optional[bool] = None
    retrieved_patterns: list = field(default_factory=list)
    top1_bm25_score: Optional[float] = None
    routed_mode: str = ""
    final_mode: str = ""
    abstained_to_prompt_only: bool = False
    prompt_only_fallback_used: bool = False
    fallback_reason: str = ""
    attribution_failure: bool = False
    grounded: Optional[bool] = None
    parse_retry_count: int = 0


@dataclass
class BugPattern:
    pattern_id: str
    name: str
    taxonomy_class: str
    description: str
    example_code: str = ""
    fix_hint: str = ""
    source: str = ""
    tags: list = field(default_factory=list)
