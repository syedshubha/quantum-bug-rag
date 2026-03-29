"""
baselines.py – Lightweight rule-based static baseline.

This baseline uses simple heuristic rules over the source text to flag
potential quantum-program bugs.  It is intentionally lightweight and should
be interpreted as a placeholder proof-of-concept rather than a faithful
re-implementation of any published quantum static analyser such as LintQ.

The baseline assigns one of the same taxonomy classes used by the LLM modes
so that its output can be evaluated with the same metrics.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from .schemas import BugDiagnostic, BugSample
from .utils import get_logger

logger = get_logger(__name__)

# ── Rule definitions ──────────────────────────────────────────────────────────


class _Rule(NamedTuple):
    """A single heuristic rule."""

    rule_id: str
    pattern: re.Pattern[str]
    taxonomy_class: str
    suspected_fragment: str
    description: str
    likelihood: float


_RULES: list[_Rule] = [
    _Rule(
        rule_id="R01",
        pattern=re.compile(r"\.measure\s*\(", re.IGNORECASE),
        taxonomy_class="measurement_error",
        suspected_fragment="measure()",
        description="Explicit measurement call detected; check qubit/cbit mapping.",
        likelihood=0.55,
    ),
    _Rule(
        rule_id="R02",
        pattern=re.compile(r"\.cx\s*\(\s*(\w+)\s*,\s*\1\s*\)", re.IGNORECASE),
        taxonomy_class="incorrect_qubit_mapping",
        suspected_fragment="cx(q, q)",
        description="CNOT gate applied with identical control and target qubits.",
        likelihood=0.90,
    ),
    _Rule(
        rule_id="R03",
        pattern=re.compile(r"QuantumCircuit\s*\(\s*1\s*\)", re.IGNORECASE),
        taxonomy_class="wrong_initial_state",
        suspected_fragment="QuantumCircuit(1)",
        description="Single-qubit circuit; verify that entanglement is not expected.",
        likelihood=0.40,
    ),
    _Rule(
        rule_id="R04",
        pattern=re.compile(r"\.h\s*\(.*\)(?!.*\.barrier)", re.DOTALL),
        taxonomy_class="missing_barrier",
        suspected_fragment="h() without barrier",
        description="Hadamard gate used without a subsequent barrier; optimisation may reorder gates.",
        likelihood=0.50,
    ),
    _Rule(
        rule_id="R05",
        pattern=re.compile(r"\.(rx|ry|rz)\s*\(\s*[\d.]+\s*,", re.IGNORECASE),
        taxonomy_class="incorrect_operator",
        suspected_fragment="rotation gate with hard-coded angle",
        description="Rotation gate with a hard-coded numeric angle; verify the angle is correct.",
        likelihood=0.45,
    ),
]


# ── Analyser ─────────────────────────────────────────────────────────────────


class StaticBaseline:
    """
    Rule-based static analyser for Qiskit code snippets.

    ⚠️  This is a lightweight placeholder baseline.  It uses only textual
    pattern matching and does not perform any semantic or control-flow analysis.
    Do not compare its recall/precision directly to published static analysers.
    """

    def analyse(self, sample: BugSample) -> BugDiagnostic:
        """Analyse a single BugSample and return a BugDiagnostic."""
        code = sample.code
        matched_rules: list[_Rule] = []

        for rule in _RULES:
            if rule.pattern.search(code):
                matched_rules.append(rule)
                logger.debug("Rule %s matched for sample '%s'.", rule.rule_id, sample.sample_id)

        if not matched_rules:
            return BugDiagnostic(
                sample_id=sample.sample_id,
                mode="static",
                bug_likelihood=0.1,
                taxonomy_class="no_bug_detected",
                suspected_location="",
                justification="No heuristic rules matched. The snippet may be clean.",
                ground_truth=sample.ground_truth,
            )

        # Use the highest-likelihood match as the primary diagnosis.
        primary = max(matched_rules, key=lambda r: r.likelihood)
        justification = "; ".join(
            f"[{r.rule_id}] {r.description}" for r in matched_rules
        )

        diagnostic = BugDiagnostic(
            sample_id=sample.sample_id,
            mode="static",
            bug_likelihood=primary.likelihood,
            taxonomy_class=primary.taxonomy_class,
            suspected_location=primary.suspected_fragment,
            justification=justification,
            ground_truth=sample.ground_truth,
        )
        diagnostic.compute_correctness()
        return diagnostic

    def analyse_batch(self, samples: list[BugSample]) -> list[BugDiagnostic]:
        """Analyse a list of BugSamples and return one BugDiagnostic per sample."""
        return [self.analyse(s) for s in samples]
