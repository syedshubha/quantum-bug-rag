"""
test_pipeline_integration.py – End-to-end integration test using mock data.

I test that the full pipeline (data loading → prompt building → LLM call →
evaluation) runs without errors on a small synthetic dataset.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.baselines import analyse_static
from src.benchmark_runner import run_benchmark
from src.dataset_loader import load_bugs4q
from src.evaluate import evaluate_results
from src.llm_client import MockLLMClient
from src.prompt_builder import build_prompt_only, build_rag_prompt
from src.retriever import KnowledgeBaseRetriever
from src.schemas import DiagnosticResult

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "bug_patterns.json"

SYNTHETIC_RECORDS = [
    {
        "id": "s001",
        "source_code": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2, 2)\nqc.h(0)\nqc.cx(0, 1)\n",
        "has_bug": True,
        "bug_class": "missing_measurement",
        "location": None,
        "description": "Missing measurement",
    },
    {
        "id": "s002",
        "source_code": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2, 2)\nqc.h(0)\nqc.cx(0, 1)\nqc.measure([0, 1], [0, 1])\n",
        "has_bug": False,
        "bug_class": None,
        "location": None,
        "description": "Clean circuit",
    },
]


class TestPromptOnlyPipeline:
    def test_end_to_end(self):
        llm = MockLLMClient()

        def analyser(record: dict) -> DiagnosticResult:
            prompt = build_prompt_only(record["id"], record["source_code"])
            return llm.analyse(record["id"], prompt, mode="prompt_only")

        results = run_benchmark(SYNTHETIC_RECORDS, analyser)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, DiagnosticResult)
            assert 0.0 <= r.bug_likelihood <= 1.0


class TestRAGPipeline:
    def test_end_to_end(self):
        llm = MockLLMClient()
        retriever = KnowledgeBaseRetriever(patterns_path=KB_PATH, top_k=2)

        def analyser(record: dict) -> DiagnosticResult:
            patterns = retriever.retrieve(record["source_code"])
            prompt = build_rag_prompt(record["id"], record["source_code"], patterns)
            result = llm.analyse(record["id"], prompt, mode="rag")
            result.retrieved_patterns = [p.id for p in patterns]
            return result

        results = run_benchmark(SYNTHETIC_RECORDS, analyser)
        assert len(results) == 2
        # I verify that patterns were actually retrieved.
        assert any(len(r.retrieved_patterns) > 0 for r in results)


class TestStaticPipeline:
    def test_end_to_end(self):
        def analyser(record: dict) -> DiagnosticResult:
            return analyse_static(record["id"], record["source_code"])

        results = run_benchmark(SYNTHETIC_RECORDS, analyser)
        assert len(results) == 2

    def test_buggy_gets_higher_score(self):
        buggy = SYNTHETIC_RECORDS[0]
        clean = SYNTHETIC_RECORDS[1]
        buggy_result = analyse_static(buggy["id"], buggy["source_code"])
        clean_result = analyse_static(clean["id"], clean["source_code"])
        assert buggy_result.bug_likelihood > clean_result.bug_likelihood


class TestEvaluationIntegration:
    def test_evaluate_runs(self):
        llm = MockLLMClient()

        def analyser(record: dict) -> DiagnosticResult:
            prompt = build_prompt_only(record["id"], record["source_code"])
            return llm.analyse(record["id"], prompt, mode="prompt_only")

        results = run_benchmark(SYNTHETIC_RECORDS, analyser)
        summary = evaluate_results(results, SYNTHETIC_RECORDS, mode="prompt_only")
        assert summary.n_samples == len(SYNTHETIC_RECORDS)
        assert 0.0 <= summary.detection_f1 <= 1.0


class TestDatasetLoader:
    def test_load_from_tmp_dir(self, tmp_path):
        """I verify that load_bugs4q can read a hand-crafted JSON file."""
        data = [
            {"id": "t001", "source_code": "qc = QuantumCircuit(1)", "has_bug": True, "bug_class": "initialisation"},
            {"id": "t002", "source_code": "qc = QuantumCircuit(2)", "has_bug": False},
        ]
        (tmp_path / "bugs4q.json").write_text(json.dumps(data))
        records = load_bugs4q(data_dir=tmp_path)
        assert len(records) == 2
        assert records[0]["id"] == "t001"
        assert records[0]["has_bug"] is True

    def test_split_buggy(self, tmp_path):
        data = [
            {"id": "t001", "source_code": "x", "has_bug": True},
            {"id": "t002", "source_code": "y", "has_bug": False},
        ]
        (tmp_path / "bugs4q.json").write_text(json.dumps(data))
        buggy = load_bugs4q(data_dir=tmp_path, split="buggy")
        assert all(r["has_bug"] for r in buggy)

    def test_max_items(self, tmp_path):
        data = [{"id": f"t{i:03d}", "source_code": "x", "has_bug": True} for i in range(10)]
        (tmp_path / "bugs4q.json").write_text(json.dumps(data))
        records = load_bugs4q(data_dir=tmp_path, max_items=3)
        assert len(records) == 3
