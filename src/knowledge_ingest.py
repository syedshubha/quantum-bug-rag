"""
knowledge_ingest.py – Load and index knowledge-base artifacts.

We support two knowledge-base files:
  - knowledge_base/bug_patterns.json  (list of BugPattern entries)
  - knowledge_base/taxonomy.json      (list of TaxonomyEntry entries)

An index is built in memory at startup for fast lookup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .schemas import BugPattern, TaxonomyEntry
from .utils import get_logger, load_json

logger = get_logger(__name__)


class KnowledgeBase:
    """
    In-memory knowledge base loaded from JSON files.

    Provides O(1) lookup by ID and simple text-based search helpers.
    """

    def __init__(
        self,
        kb_dir: str | Path = "knowledge_base",
    ) -> None:
        self.kb_dir = Path(kb_dir)
        self.patterns: dict[str, BugPattern] = {}
        self.taxonomy: dict[str, TaxonomyEntry] = {}
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        patterns_path = self.kb_dir / "bug_patterns.json"
        taxonomy_path = self.kb_dir / "taxonomy.json"

        if patterns_path.exists():
            raw = load_json(patterns_path)
            for entry in raw:
                bp = BugPattern(**entry)
                self.patterns[bp.pattern_id] = bp
            logger.info("Loaded %d bug patterns.", len(self.patterns))
        else:
            logger.warning("Bug-patterns file not found: %s", patterns_path)

        if taxonomy_path.exists():
            raw = load_json(taxonomy_path)
            for entry in raw:
                te = TaxonomyEntry(**entry)
                self.taxonomy[te.class_id] = te
            logger.info("Loaded %d taxonomy entries.", len(self.taxonomy))
        else:
            logger.warning("Taxonomy file not found: %s", taxonomy_path)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_pattern(self, pattern_id: str) -> Optional[BugPattern]:
        return self.patterns.get(pattern_id)

    def get_taxonomy_entry(self, class_id: str) -> Optional[TaxonomyEntry]:
        return self.taxonomy.get(class_id)

    def all_patterns(self) -> list[BugPattern]:
        return list(self.patterns.values())

    def all_taxonomy_classes(self) -> list[str]:
        return list(self.taxonomy.keys())

    # ── Simple keyword search ─────────────────────────────────────────────────

    def search_patterns(self, query: str, top_k: int = 5) -> list[BugPattern]:
        """
        Return the *top_k* patterns whose description contains any query token.

        This is a lightweight fallback; replace with the TF-IDF retriever for
        better recall.
        """
        tokens = set(query.lower().split())
        scored: list[tuple[int, BugPattern]] = []
        for bp in self.patterns.values():
            text = (bp.name + " " + bp.description + " " + bp.fix_hint).lower()
            score = sum(1 for t in tokens if t in text)
            if score > 0:
                scored.append((score, bp))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [bp for _, bp in scored[:top_k]]

    # ── Persistence helpers ───────────────────────────────────────────────────

    def add_pattern(self, pattern: BugPattern) -> None:
        """Add or replace a pattern entry (used by ingestion scripts)."""
        self.patterns[pattern.pattern_id] = pattern

    def save_patterns(self, path: Optional[str | Path] = None) -> None:
        """Persist the current pattern set back to JSON."""
        from .utils import save_json  # local import to avoid circulars

        target = Path(path) if path else self.kb_dir / "bug_patterns.json"
        data = [p.model_dump() for p in self.patterns.values()]
        save_json(data, target)
        logger.info("Saved %d patterns to %s.", len(data), target)
