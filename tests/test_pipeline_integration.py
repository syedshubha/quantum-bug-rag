"""
test_pipeline_integration.py – End-to-end integration tests for the pipeline.

We run each experimental mode on a small set of synthetic smoke-test samples
using the MockLLMClient and in-memory knowledge base.  These tests validate
that all components integrate correctly without requiring real datasets or
external APIs.

⚠️  These tests use synthetic data for infrastructure validation only.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.benchmark_runner import BenchmarkRunner
from src.dataset_loader import generate_smoke_samples
from src.evaluate import compute_metrics
from src.knowledge_ingest import KnowledgeBase
from src.schemas import BugPattern, TaxonomyEntry


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def smoke_samples():
    return generate_smoke_samples(n=10, seed=0)


@pytest.fixture()
def tmp_kb_dir(tmp_path: Path) -> Path:
    """Create a minimal knowledge-base directory in a temp location."""
    patterns = [
        {
            "pattern_id": "BP001",
            "name": "CNOT Self-Loop",
            "taxonomy_class": "incorrect_qubit_mapping",
            "description": "CNOT applied with same qubit as control and target.",
            "example_code": "qc.cx(0, 0)",
            "fix_hint": "Use distinct qubit indices.",
            "source": "test",
            "tags": ["cx"],
        },
        {
            "pattern_id": "BP002",
            "name": "Missing Measurement",
            "taxonomy_class": "measurement_error",
            "description": "Classical register read without measurement.",
            "example_code": "qc.measure(0, 0)",
            "fix_hint": "Add measure().",
            "source": "test",
            "tags": ["measure"],
        },
    ]
    taxonomy = [
        {
            "class_id": "incorrect_qubit_mapping",
            "name": "Incorrect Qubit Mapping",
            "description": "Wrong qubit index used.",
            "parent_class": None,
            "examples": [],
        },
        {
            "class_id": "measurement_error",
            "name": "Measurement Error",
            "description": "Measurement incorrectly placed.",
            "parent_class": None,
            "examples": [],
        },
    ]
    (tmp_path / "bug_patterns.json").write_text(json.dumps(patterns))
    (tmp_path / "taxonomy.json").write_text(json.dumps(taxonomy))
    return tmp_path


@pytest.fixture()
def mock_config(tmp_kb_dir: Path) -> dict:
    return {
        "llm": {"backend": "mock"},
        "paths": {"knowledge_base": str(tmp_kb_dir)},
        "retrieval": {"top_k": 3},
    }


# ── Static mode ───────────────────────────────────────────────────────────────

class TestStaticMode:
    def test_run_returns_correct_counts(self, smoke_samples, tmp_path) -> None:
        runner = BenchmarkRunner(mode="static", config={}, output_dir=tmp_path)
        diagnostics, summary = runner.run(smoke_samples)
        assert len(diagnostics) == len(smoke_samples)
        assert summary.num_samples == len(smoke_samples)

    def test_diagnostics_have_static_mode(self, smoke_samples, tmp_path) -> None:
        runner = BenchmarkRunner(mode="static", config={}, output_dir=tmp_path)
        diagnostics, _ = runner.run(smoke_samples)
        assert all(d.mode == "static" for d in diagnostics)

    def test_output_files_written(self, smoke_samples, tmp_path) -> None:
        runner = BenchmarkRunner(mode="static", config={}, output_dir=tmp_path)
        runner.run(smoke_samples)
        jsonl_files = list(tmp_path.glob("diagnostics_static_*.jsonl"))
        metrics_files = list(tmp_path.glob("metrics_static_*.json"))
        assert len(jsonl_files) == 1
        assert len(metrics_files) == 1


# ── Prompt-only mode ──────────────────────────────────────────────────────────

class TestPromptOnlyMode:
    def test_run_returns_correct_counts(self, smoke_samples, mock_config, tmp_path) -> None:
        runner = BenchmarkRunner(mode="prompt_only", config=mock_config, output_dir=tmp_path)
        diagnostics, summary = runner.run(smoke_samples)
        assert len(diagnostics) == len(smoke_samples)

    def test_diagnostics_have_prompt_only_mode(self, smoke_samples, mock_config, tmp_path) -> None:
        runner = BenchmarkRunner(mode="prompt_only", config=mock_config, output_dir=tmp_path)
        diagnostics, _ = runner.run(smoke_samples)
        assert all(d.mode == "prompt_only" for d in diagnostics)

    def test_bug_likelihood_in_range(self, smoke_samples, mock_config, tmp_path) -> None:
        runner = BenchmarkRunner(mode="prompt_only", config=mock_config, output_dir=tmp_path)
        diagnostics, _ = runner.run(smoke_samples)
        for d in diagnostics:
            assert 0.0 <= d.bug_likelihood <= 1.0


# ── RAG mode ──────────────────────────────────────────────────────────────────

class TestRAGMode:
    def test_run_returns_correct_counts(self, smoke_samples, mock_config, tmp_path) -> None:
        runner = BenchmarkRunner(mode="rag", config=mock_config, output_dir=tmp_path)
        diagnostics, summary = runner.run(smoke_samples)
        assert len(diagnostics) == len(smoke_samples)

    def test_diagnostics_have_rag_mode(self, smoke_samples, mock_config, tmp_path) -> None:
        runner = BenchmarkRunner(mode="rag", config=mock_config, output_dir=tmp_path)
        diagnostics, _ = runner.run(smoke_samples)
        assert all(d.mode == "rag" for d in diagnostics)

    def test_retrieved_patterns_field_present(self, smoke_samples, mock_config, tmp_path) -> None:
        runner = BenchmarkRunner(mode="rag", config=mock_config, output_dir=tmp_path)
        diagnostics, _ = runner.run(smoke_samples)
        # All diagnostics should have the retrieved_patterns field (may be empty list).
        for d in diagnostics:
            assert isinstance(d.retrieved_patterns, list)


# ── Invalid mode ──────────────────────────────────────────────────────────────

class TestInvalidMode:
    def test_invalid_mode_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            BenchmarkRunner(mode="nonexistent", config={}, output_dir=tmp_path)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_labelled_samples(self, smoke_samples, tmp_path) -> None:
        runner = BenchmarkRunner(mode="static", config={}, output_dir=tmp_path)
        diagnostics, summary = runner.run(smoke_samples)
        # Smoke samples have ground truth; metrics should be computed.
        assert summary.num_samples == len(smoke_samples)
        assert 0.0 <= summary.accuracy <= 1.0
        assert 0.0 <= summary.f1_macro <= 1.0

    def test_no_labelled_samples_returns_zeros(self) -> None:
        from src.schemas import BugDiagnostic

        diagnostics = [
            BugDiagnostic(
                sample_id=f"s{i}",
                mode="rag",
                bug_likelihood=0.5,
                taxonomy_class="unknown",
                ground_truth=None,  # no label
            )
            for i in range(5)
        ]
        summary = compute_metrics(diagnostics)
        assert summary.num_samples == 0
        assert summary.accuracy == 0.0


# ── KnowledgeBase integration ─────────────────────────────────────────────────

class TestKnowledgeBaseIntegration:
    def test_loads_patterns(self, tmp_kb_dir) -> None:
        kb = KnowledgeBase(tmp_kb_dir)
        assert len(kb.patterns) == 2

    def test_loads_taxonomy(self, tmp_kb_dir) -> None:
        kb = KnowledgeBase(tmp_kb_dir)
        assert len(kb.taxonomy) == 2

    def test_get_pattern(self, tmp_kb_dir) -> None:
        kb = KnowledgeBase(tmp_kb_dir)
        p = kb.get_pattern("BP001")
        assert p is not None
        assert p.name == "CNOT Self-Loop"

    def test_search_patterns(self, tmp_kb_dir) -> None:
        kb = KnowledgeBase(tmp_kb_dir)
        results = kb.search_patterns("CNOT qubit")
        assert len(results) >= 1
        assert results[0].pattern_id == "BP001"
