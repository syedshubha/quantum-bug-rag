"""
schemas.py – Pydantic data models for structured inputs and outputs.

I use these models throughout the pipeline to enforce type safety and
to produce consistent JSON-serialisable diagnostic records.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Bug taxonomy
# ---------------------------------------------------------------------------


class BugTaxonomyClass(str, Enum):
    """
    I enumerate the top-level bug categories drawn from the Bugs4Q taxonomy.
    These labels are used both as retrieval keys and as classification targets.
    """

    INCORRECT_GATE = "incorrect_gate"
    WRONG_QUBIT_ORDER = "wrong_qubit_order"
    MISSING_MEASUREMENT = "missing_measurement"
    CIRCUIT_STRUCTURE = "circuit_structure"
    INITIALISATION = "initialisation"
    CLASSICAL_CONTROL = "classical_control"
    RUNTIME_ERROR = "runtime_error"
    OTHER = "other"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Per-program diagnostic
# ---------------------------------------------------------------------------


class DiagnosticResult(BaseModel):
    """
    I represent the structured output that the pipeline produces for each
    Qiskit program under analysis.
    """

    program_id: str = Field(..., description="Unique identifier for the program / bug report")
    bug_likelihood: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated probability that the program contains a bug"
    )
    taxonomy_class: BugTaxonomyClass = Field(
        ..., description="Predicted top-level bug category"
    )
    suspected_location: Optional[str] = Field(
        None, description="File path and/or line range where the bug is suspected"
    )
    justification: str = Field(
        ..., description="Free-text explanation produced by the LLM or static analyser"
    )
    retrieved_patterns: List[str] = Field(
        default_factory=list,
        description="IDs of the knowledge-base patterns that were retrieved for this analysis",
    )
    mode: str = Field(
        "unknown",
        description="Pipeline mode used: 'prompt_only', 'rag', or 'static'",
    )

    model_config = ConfigDict(use_enum_values=True)


# ---------------------------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------------------------


class EvaluationSummary(BaseModel):
    """
    I summarise evaluation metrics across an entire benchmark run.
    """

    mode: str
    n_samples: int
    detection_precision: float
    detection_recall: float
    detection_f1: float
    classification_macro_f1: float
    per_class_f1: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knowledge-base entry
# ---------------------------------------------------------------------------


class BugPatternEntry(BaseModel):
    """
    I represent one entry in the local knowledge base used for retrieval.
    """

    id: str
    taxonomy_class: str
    title: str
    description: str
    code_snippet: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
