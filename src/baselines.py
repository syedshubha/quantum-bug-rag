"""
baselines.py – Static-analysis and heuristic baselines.

I implement a simple rule-based static analyser that inspects Qiskit source
code for common anti-patterns without calling an LLM.  This serves as the
``static`` baseline in my three-way comparison.

The rules are intentionally conservative and readable; extending them is as
simple as adding a new entry to ``STATIC_RULES``.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .schemas import BugTaxonomyClass, DiagnosticResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------


@dataclass
class StaticRule:
    """I represent one static-analysis rule."""

    rule_id: str
    taxonomy_class: BugTaxonomyClass
    description: str
    # A list of (regex_pattern, description) pairs; any match triggers the rule.
    patterns: List[Tuple[str, str]] = field(default_factory=list)
    weight: float = 0.3  # contribution to overall bug_likelihood


# I define a curated set of rules for common Qiskit bugs.
STATIC_RULES: List[StaticRule] = [
    StaticRule(
        rule_id="R001",
        taxonomy_class=BugTaxonomyClass.MISSING_MEASUREMENT,
        description="Circuit has no measurement operations",
        patterns=[],  # I handle this rule via AST, not regex.
        weight=0.5,
    ),
    StaticRule(
        rule_id="R002",
        taxonomy_class=BugTaxonomyClass.WRONG_QUBIT_ORDER,
        description="CX/CNOT gate present (flag for manual review of qubit argument order)",
        patterns=[
            (r"\.cx\s*\(\s*\d+\s*,\s*\d+\s*\)", "cx gate with literal qubit indices detected"),
        ],
        weight=0.2,
    ),
    StaticRule(
        rule_id="R003",
        taxonomy_class=BugTaxonomyClass.INITIALISATION,
        description="QuantumCircuit initialised with zero qubits",
        patterns=[
            (r"QuantumCircuit\s*\(\s*0\s*\)", "QuantumCircuit(0)"),
        ],
        weight=0.6,
    ),
    StaticRule(
        rule_id="R004",
        taxonomy_class=BugTaxonomyClass.CIRCUIT_STRUCTURE,
        description="Barrier applied to a circuit that may have no gates",
        patterns=[
            (r"\.barrier\s*\(\s*\)", "empty barrier"),
        ],
        weight=0.1,
    ),
    StaticRule(
        rule_id="R005",
        taxonomy_class=BugTaxonomyClass.INCORRECT_GATE,
        description="Use of deprecated Qiskit gate names",
        patterns=[
            (r"\.u1\s*\(", "deprecated u1 gate"),
            (r"\.u2\s*\(", "deprecated u2 gate"),
            (r"\.u3\s*\(", "deprecated u3 gate"),
        ],
        weight=0.4,
    ),
    StaticRule(
        rule_id="R006",
        taxonomy_class=BugTaxonomyClass.CLASSICAL_CONTROL,
        description="Classical register size mismatch (simple heuristic)",
        patterns=[
            (r"ClassicalRegister\s*\(\s*(\d+)\s*\)", "ClassicalRegister size"),
        ],
        weight=0.2,
    ),
]


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------


def _has_measurement(source_code: str) -> bool:
    """I check (via simple text scan) whether the code contains measurement ops."""
    return bool(re.search(r"\.(measure|measure_all)\s*\(", source_code))


def _check_rule_patterns(rule: StaticRule, source_code: str) -> Optional[str]:
    """
    I check if any of the rule's regex patterns match the source code.
    Returns the matched description or None.
    """
    for pattern, description in rule.patterns:
        if re.search(pattern, source_code):
            return description
    return None


def analyse_static(
    program_id: str,
    source_code: str,
) -> DiagnosticResult:
    """
    I run all static rules against *source_code* and return a
    ``DiagnosticResult`` with aggregated findings.

    Parameters
    ----------
    program_id:
        Unique identifier for the program.
    source_code:
        Raw Python/Qiskit source code.
    """
    triggered: List[Tuple[StaticRule, str]] = []

    # Check R001 (missing measurement) separately.
    if not _has_measurement(source_code) and "QuantumCircuit" in source_code:
        triggered.append((STATIC_RULES[0], "no .measure() or .measure_all() call found"))

    # Check the remaining pattern-based rules.
    for rule in STATIC_RULES[1:]:
        match_desc = _check_rule_patterns(rule, source_code)
        if match_desc:
            triggered.append((rule, match_desc))

    if not triggered:
        return DiagnosticResult(
            program_id=program_id,
            bug_likelihood=0.05,
            taxonomy_class=BugTaxonomyClass.UNKNOWN,
            justification="Static analysis found no issues.",
            mode="static",
        )

    # I aggregate the likelihood from all triggered rules.
    combined_likelihood = min(1.0, sum(r.weight for r, _ in triggered))
    # I report the taxonomy of the highest-weight triggered rule.
    dominant_rule, dominant_match = max(triggered, key=lambda x: x[0].weight)

    justification_parts = ["Static analysis triggered the following rules:"]
    for rule, match_desc in triggered:
        justification_parts.append(f"  [{rule.rule_id}] {rule.description} – {match_desc}")

    return DiagnosticResult(
        program_id=program_id,
        bug_likelihood=combined_likelihood,
        taxonomy_class=dominant_rule.taxonomy_class,
        suspected_location=None,
        justification="\n".join(justification_parts),
        mode="static",
    )
