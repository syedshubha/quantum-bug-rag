"""
benchmark_runner.py – Orchestrates a complete evaluation run.

A BenchmarkRunner ties together the dataset loader, LLM client, retriever,
prompt builder, and evaluator into a single coherent pipeline for a given mode.
"""

from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from .baselines import StaticBaseline
from .dataset_loader import load_bugs4q
from .evaluate import compute_metrics, print_summary
from .knowledge_ingest import KnowledgeBase
from .llm_client import BaseLLMClient, build_llm_client
from .prompt_builder import build_prompt_only, build_rag_prompt
from .retriever import BugPatternRetriever
from .schemas import BugDiagnostic, BugSample, EvalSummary
from .utils import append_jsonl, get_logger, new_run_id, save_json

logger = get_logger(__name__)


class BenchmarkRunner:
    """
    High-level runner for a single experimental mode on a dataset.

    Parameters
    ----------
    mode:
        One of ``"static"``, ``"prompt_only"``, or ``"rag"``.
    config:
        Loaded configuration dict (see config.example.yaml).
    output_dir:
        Directory where JSONL diagnostics and summary files are written.
    """

    def __init__(
        self,
        mode: str,
        config: dict,
        output_dir: str | Path = "outputs",
    ) -> None:
        if mode not in {"static", "prompt_only", "rag"}:
            raise ValueError(f"Unknown mode '{mode}'. Choose: static, prompt_only, rag.")
        self.mode = mode
        self.config = config
        self.output_dir = Path(output_dir)
        self.run_id = new_run_id()

        # Initialise components based on mode.
        self._llm: Optional[BaseLLMClient] = None
        self._retriever: Optional[BugPatternRetriever] = None
        self._kb: Optional[KnowledgeBase] = None
        self._static: Optional[StaticBaseline] = None

        # Self-consistency decoding settings.
        sc_cfg = config.get("self_consistency", {})
        self._sc_enabled: bool = bool(sc_cfg.get("enabled", False))
        self._sc_n: int = int(sc_cfg.get("n", 3))
        self._sc_temperature: float = float(sc_cfg.get("temperature", 0.4))

        # Post-hoc calibration settings.
        cal_cfg = config.get("calibration", {})
        self._cal_enabled: bool = bool(cal_cfg.get("enabled", False))
        self._cal_rules: list[dict[str, Any]] = cal_cfg.get("rules", [])

        if mode == "static":
            self._static = StaticBaseline()
        elif mode in {"prompt_only", "rag"}:
            self._llm = build_llm_client(config)
            if mode == "rag":
                kb_dir = config.get("paths", {}).get("knowledge_base", "knowledge_base")
                self._kb = KnowledgeBase(kb_dir)
                top_k = config.get("retrieval", {}).get("top_k", 5)
                min_score = config.get("retrieval", {}).get("min_score", 1.0)
                exclude_classes = config.get("retrieval", {}).get(
                    "exclude_classes", ["unknown"]
                )
                dense_cfg = config.get("retrieval", {}).get("dense", {})
                dense_model = (
                    dense_cfg.get("model")
                    if dense_cfg.get("enabled", False)
                    else None
                )
                dense_weight = float(dense_cfg.get("weight", 0.5))
                self._retriever = BugPatternRetriever(
                    self._kb.all_patterns(),
                    min_score=min_score,
                    exclude_classes=exclude_classes,
                    dense_model=dense_model,
                    dense_weight=dense_weight,
                )
                self._top_k = top_k

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, samples: list[BugSample]) -> tuple[list[BugDiagnostic], EvalSummary]:
        """
        Run the pipeline on *samples* and return (diagnostics, summary).

        Results are also written to ``output_dir``.
        """
        logger.info(
            "Starting run '%s' in mode '%s' on %d samples.",
            self.run_id,
            self.mode,
            len(samples),
        )
        diagnostics: list[BugDiagnostic] = []
        for idx, sample in enumerate(samples, 1):
            logger.info("Processing sample %d/%d: %s", idx, len(samples), sample.sample_id)
            try:
                diag = self._process_one(sample)
            except Exception as exc:
                logger.error("Sample %s failed: %s", sample.sample_id, exc)
                diag = BugDiagnostic(
                    sample_id=sample.sample_id,
                    mode=self.mode,
                    bug_likelihood=0.5,
                    taxonomy_class="unknown",
                    suspected_location="",
                    justification=f"Error: {exc}",
                    ground_truth=sample.ground_truth,
                )
                diag.compute_correctness()
            diagnostics.append(diag)

        summary = compute_metrics(diagnostics, run_id=self.run_id, mode=self.mode)
        self._save_results(diagnostics, summary)
        print_summary(summary)
        return diagnostics, summary

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_one(self, sample: BugSample) -> BugDiagnostic:
        if self.mode == "static":
            assert self._static is not None
            return self._static.analyse(sample)

        assert self._llm is not None
        retrieved_ids: list[str] = []

        if self.mode == "rag":
            assert self._retriever is not None
            assert self._kb is not None
            patterns = self._retriever.retrieve(sample.code, top_k=self._top_k)
            retrieved_ids = [p.pattern_id for p in patterns]
            taxonomy_entries = [
                self._kb.get_taxonomy_entry(p.taxonomy_class)
                for p in patterns
                if self._kb.get_taxonomy_entry(p.taxonomy_class) is not None
            ]
            messages = build_rag_prompt(sample, patterns, taxonomy_entries)  # type: ignore[arg-type]
        else:
            messages = build_prompt_only(sample)

        # ── Self-consistency or single-pass inference ─────────────────────
        if self._sc_enabled:
            parsed = self._infer_with_self_consistency(messages)
        else:
            parsed = self._llm.complete_and_parse(messages)

        taxonomy_class = str(parsed.get("taxonomy_class", "unknown"))
        bug_likelihood = float(parsed.get("bug_likelihood", 0.5))

        # ── Post-hoc calibration ──────────────────────────────────────────
        if self._cal_enabled:
            taxonomy_class, bug_likelihood = self._apply_calibration(
                taxonomy_class, bug_likelihood,
            )

        diag = BugDiagnostic(
            sample_id=sample.sample_id,
            mode=self.mode,
            bug_likelihood=bug_likelihood,
            taxonomy_class=taxonomy_class,
            suspected_location=str(parsed.get("suspected_location", "")),
            justification=str(parsed.get("justification", "")),
            ground_truth=sample.ground_truth,
            retrieved_patterns=retrieved_ids,
        )
        diag.compute_correctness()
        return diag

    def _infer_with_self_consistency(
        self, messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Query the LLM *n* times and return the response whose taxonomy_class
        wins the majority vote.  Ties are broken by selecting the response
        whose bug_likelihood is highest.
        """
        assert self._llm is not None
        responses = self._llm.complete_and_parse_n(
            messages, n=self._sc_n, temperature=self._sc_temperature,
        )
        # Extract taxonomy votes.
        votes: list[str] = [
            str(r.get("taxonomy_class", "unknown")) for r in responses
        ]
        vote_counts = Counter(votes)
        winner, _ = vote_counts.most_common(1)[0]

        # Among responses that voted for the winner, pick the one with the
        # highest bug_likelihood to use as the canonical response.
        best: dict[str, Any] = {"bug_likelihood": -1.0}
        for r, v in zip(responses, votes):
            if v == winner:
                if float(r.get("bug_likelihood", 0.0)) > float(best.get("bug_likelihood", -1.0)):
                    best = r

        logger.info(
            "Self-consistency votes: %s → winner '%s'",
            dict(vote_counts), winner,
        )
        return best

    def _apply_calibration(
        self, taxonomy_class: str, bug_likelihood: float,
    ) -> tuple[str, float]:
        """
        Apply post-hoc calibration rules.

        Each rule in ``self._cal_rules`` has:
          - predicted_class: the class to intercept
          - max_likelihood: if bug_likelihood < this, override
          - override_class: replacement class
        """
        for rule in self._cal_rules:
            if taxonomy_class == rule.get("predicted_class"):
                threshold = float(rule.get("max_likelihood", 1.0))
                if bug_likelihood < threshold:
                    override = str(rule.get("override_class", "unknown"))
                    logger.info(
                        "Calibration override: %s (%.2f < %.2f) → %s",
                        taxonomy_class, bug_likelihood, threshold, override,
                    )
                    return override, bug_likelihood
        return taxonomy_class, bug_likelihood

    def _save_results(
        self, diagnostics: list[BugDiagnostic], summary: EvalSummary
    ) -> None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        jsonl_path = self.output_dir / f"diagnostics_{self.mode}_{self.run_id}_{ts}.jsonl"
        for diag in diagnostics:
            append_jsonl(diag.model_dump(), jsonl_path)

        summary_path = self.output_dir / f"metrics_{self.mode}_{self.run_id}_{ts}.json"
        save_json(summary.model_dump(), summary_path)
        logger.info("Results written to %s", self.output_dir)
