"""
schemas.py – Pydantic models for pipeline inputs and outputs.

We use Pydantic v2 throughout; update field validators if migrating to v1.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Input schema ─────────────────────────────────────────────────────────────

class BugSample(BaseModel):
    """A single quantum-program sample from the evaluation dataset."""

    sample_id: str = Field(..., description="Unique identifier for the sample.")
    source: str = Field(..., description="Dataset origin, e.g. 'bugs4q'.")
    code: str = Field(..., description="Raw Qiskit (Python) source code.")
    ground_truth: Optional[str] = Field(
        None,
        description=(
            "Ground-truth taxonomy label, if known. "
            "None for unlabelled samples."
        ),
    )
    metadata: dict = Field(default_factory=dict, description="Arbitrary extra fields.")

    @field_validator("code")
    @classmethod
    def code_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'code' must not be an empty string.")
        return v


# ── Output / diagnostic schema ────────────────────────────────────────────────

class BugDiagnostic(BaseModel):
    """Structured diagnostic produced for a single BugSample."""

    sample_id: str = Field(..., description="Matches BugSample.sample_id.")
    mode: str = Field(
        ...,
        description="Pipeline mode: 'static', 'prompt_only', or 'rag'.",
    )
    bug_likelihood: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the sample contains a bug.",
    )
    taxonomy_class: str = Field(
        ...,
        description="Predicted bug taxonomy class (e.g. 'incorrect_operator').",
    )
    suspected_location: str = Field(
        "",
        description="Best-effort identification of the offending code fragment.",
    )
    justification: str = Field(
        "",
        description="Natural-language explanation of the diagnosis.",
    )
    ground_truth: Optional[str] = Field(
        None,
        description="Ground-truth label copied from the input sample (if available).",
    )
    correct: Optional[bool] = Field(
        None,
        description="Whether taxonomy_class matches ground_truth. None if ground_truth is absent.",
    )
    retrieved_patterns: list[str] = Field(
        default_factory=list,
        description="IDs of knowledge-base patterns retrieved (RAG mode only).",
    )

    def compute_correctness(self) -> None:
        """Set self.correct based on ground_truth vs. taxonomy_class."""
        if self.ground_truth is not None:
            self.correct = self.taxonomy_class == self.ground_truth


# ── Knowledge-base entry schemas ─────────────────────────────────────────────

class BugPattern(BaseModel):
    """A single bug-pattern entry in the knowledge base."""

    pattern_id: str = Field(..., description="Unique pattern identifier.")
    name: str = Field(..., description="Short human-readable name.")
    taxonomy_class: str = Field(..., description="Associated taxonomy class.")
    description: str = Field(..., description="Detailed description of the pattern.")
    example_code: str = Field("", description="Illustrative (potentially synthetic) code snippet.")
    fix_hint: str = Field("", description="Guidance on how to fix this class of bug.")
    source: str = Field("", description="Origin corpus or reference (e.g. 'bugsqcp', 'manual').")
    tags: list[str] = Field(default_factory=list)


class TaxonomyEntry(BaseModel):
    """A single entry in the bug taxonomy."""

    class_id: str = Field(..., description="Unique taxonomy class identifier.")
    name: str = Field(..., description="Short display name.")
    description: str = Field(..., description="Definition of this taxonomy class.")
    parent_class: Optional[str] = Field(
        None, description="Parent class_id for hierarchical taxonomies."
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Short illustrative examples.",
    )


# ── Evaluation summary schema ─────────────────────────────────────────────────

class EvalSummary(BaseModel):
    """Aggregate evaluation metrics for a pipeline run."""

    run_id: str
    mode: str
    num_samples: int
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    per_class_f1: dict[str, float] = Field(default_factory=dict)
    notes: str = ""
