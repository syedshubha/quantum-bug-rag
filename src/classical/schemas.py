"""Dataclasses for the classical-vs-quantum binary track."""

from __future__ import annotations

from dataclasses import dataclass, field

CLASSES: list[str] = ["classical", "quantum"]


@dataclass
class BugSample:
    sample_id: str
    source: str            # "bqcp" or "bugs4q"
    code: str              # buggy code (truncated at prompt construction)
    ground_truth: str      # "classical" or "quantum"
    metadata: dict = field(default_factory=dict)


@dataclass
class KBEntry:
    entry_id: str
    domain: str            # "quantum" or "classical"
    framework: str         # qiskit / pennylane / cpython / numpy
    description: str


@dataclass
class Diagnostic:
    sample_id: str
    mode: str              # "prompt_only" | "biased_rag" | "balanced_rag"
    predicted: str
    score_quantum: float   # LLM probability that the bug is quantum
    ground_truth: str
    correct: bool
    retrieved_ids: list = field(default_factory=list)
    reasoning: str = ""
