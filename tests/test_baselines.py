"""
test_baselines.py – Unit tests for src/baselines.py.

We verify that the StaticBaseline correctly applies rules and returns
BugDiagnostic objects with the expected fields.
"""

from __future__ import annotations

import pytest

from src.baselines import StaticBaseline
from src.schemas import BugSample


def _sample(code: str, gnd: str | None = None, sid: str = "s1") -> BugSample:
    return BugSample(sample_id=sid, source="test", code=code, ground_truth=gnd)


class TestStaticBaseline:
    def setup_method(self) -> None:
        self.baseline = StaticBaseline()

    # ── Rule R02: CNOT self-loop ──────────────────────────────────────────────

    def test_cx_self_loop_detected(self) -> None:
        code = "qc.cx(0, 0)"
        diag = self.baseline.analyse(_sample(code))
        assert diag.taxonomy_class == "incorrect_qubit_mapping"
        assert diag.bug_likelihood >= 0.8

    def test_cx_self_loop_mode_is_static(self) -> None:
        diag = self.baseline.analyse(_sample("qc.cx(1, 1)"))
        assert diag.mode == "static"

    # ── Rule R01: measurement call ────────────────────────────────────────────

    def test_measure_call_detected(self) -> None:
        code = "qc.measure(0, 0)"
        diag = self.baseline.analyse(_sample(code))
        assert diag.taxonomy_class in {
            "measurement_error",
            "incorrect_qubit_mapping",
            "missing_barrier",
        }

    # ── No match ──────────────────────────────────────────────────────────────

    def test_no_match_returns_no_bug_detected(self) -> None:
        code = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\n"
        diag = self.baseline.analyse(_sample(code))
        # h() without barrier rule R04 will match, so check it's not no_bug_detected only for truly empty
        # Use a snippet that intentionally matches nothing.
        clean_code = "pass"
        diag2 = self.baseline.analyse(_sample(clean_code))
        assert diag2.taxonomy_class == "no_bug_detected"
        assert diag2.bug_likelihood < 0.3

    # ── Correctness computation ───────────────────────────────────────────────

    def test_correctness_computed_when_ground_truth_given(self) -> None:
        code = "qc.cx(0, 0)"
        diag = self.baseline.analyse(_sample(code, gnd="incorrect_qubit_mapping"))
        assert diag.correct is True

    def test_correctness_none_when_no_ground_truth(self) -> None:
        diag = self.baseline.analyse(_sample("qc.cx(0, 0)"))
        # No ground_truth → correct should be None
        assert diag.correct is None

    # ── Batch analysis ────────────────────────────────────────────────────────

    def test_analyse_batch_length(self) -> None:
        samples = [_sample("qc.cx(0, 0)", sid=f"s{i}") for i in range(5)]
        diags = self.baseline.analyse_batch(samples)
        assert len(diags) == 5

    def test_analyse_batch_all_static_mode(self) -> None:
        samples = [_sample("qc.h(0)", sid=f"s{i}") for i in range(3)]
        diags = self.baseline.analyse_batch(samples)
        assert all(d.mode == "static" for d in diags)

    # ── Justification ─────────────────────────────────────────────────────────

    def test_justification_non_empty_on_match(self) -> None:
        diag = self.baseline.analyse(_sample("qc.cx(0, 0)"))
        assert len(diag.justification) > 0

    def test_sample_id_preserved(self) -> None:
        diag = self.baseline.analyse(_sample("qc.cx(0, 0)", sid="my_id"))
        assert diag.sample_id == "my_id"
