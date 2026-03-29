"""
test_baselines.py – Unit tests for the static-analysis baseline.
"""

import pytest

from src.baselines import analyse_static
from src.schemas import BugTaxonomyClass


CLEAN_CODE = """\
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
"""

MISSING_MEASURE_CODE = """\
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
"""

DEPRECATED_GATE_CODE = """\
from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
qc.u1(3.14, 0)
qc.measure_all()
"""

ZERO_QUBIT_CODE = """\
from qiskit import QuantumCircuit
qc = QuantumCircuit(0)
"""


class TestAnalyseStatic:
    def test_clean_code_low_likelihood(self):
        result = analyse_static("prog_clean", CLEAN_CODE)
        assert result.bug_likelihood < 0.5
        assert result.mode == "static"

    def test_missing_measurement_detected(self):
        result = analyse_static("prog_no_measure", MISSING_MEASURE_CODE)
        assert result.bug_likelihood >= 0.5
        assert result.taxonomy_class == BugTaxonomyClass.MISSING_MEASUREMENT.value

    def test_deprecated_gate_detected(self):
        result = analyse_static("prog_deprecated", DEPRECATED_GATE_CODE)
        assert result.bug_likelihood > 0.0
        assert result.taxonomy_class == BugTaxonomyClass.INCORRECT_GATE.value

    def test_zero_qubit_circuit(self):
        result = analyse_static("prog_zero", ZERO_QUBIT_CODE)
        # R003 triggers for QuantumCircuit(0)
        assert result.bug_likelihood >= 0.5

    def test_result_has_justification(self):
        result = analyse_static("prog_any", MISSING_MEASURE_CODE)
        assert len(result.justification) > 0

    def test_program_id_preserved(self):
        result = analyse_static("my_program_id", CLEAN_CODE)
        assert result.program_id == "my_program_id"

    def test_empty_source_code(self):
        result = analyse_static("prog_empty", "")
        # No QuantumCircuit detected → R001 should not fire.
        assert result.bug_likelihood < 0.5
