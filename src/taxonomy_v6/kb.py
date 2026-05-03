"""Validated knowledge-base extractors for the v6 taxonomy track.

The v6 KB pulls bug-fix-flavoured entries from quantum framework release
notes, mapping each entry to a taxonomy class via a hand-curated keyword
table (see ``classify_text_to_taxonomy``). Entries that do not classify, or
that look like cosmetic/version-bump notes, are dropped at extraction time.

Sources:
  - Qiskit / Qiskit-Aer / Qiskit-Ignis ``releasenotes/notes/*.yaml``
  - Qiskit IBM Runtime ``release-notes/*.rst``
  - PennyLane ``doc/releases/changelog-*.md``
  - Hand-coded LintQ rules (Paltenghi & Pradel, FSE 2024 ruleset summary)
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .schemas import BugPattern

# ── Taxonomy keyword table ──────────────────────────────────────────────────

TAX_KEYWORDS: dict[str, list[str]] = {
    "measurement_error": [
        "measure", "measurement", "classical bit", "cbit", "counts",
        "qasm_simulator", "shots", "final_measurement",
        "remove_final_measurements", "clbit", "bit_string",
    ],
    "incorrect_qubit_mapping": [
        "qubit index", "qubit order", "register", "msb", "lsb",
        "cnot", "control qubit", "target qubit", "wires", "qargs", "cargs",
        "layout", "qubit mapping", "physical qubit", "virtual qubit",
        "qubit count", "swap mapper",
    ],
    "missing_barrier": [
        "barrier", "reorder", "transpiler reorder", "optimization across",
        "removed by optimization",
    ],
    "wrong_initial_state": [
        "initialize", "statevector", "state preparation", "amplitude",
        "normaliz", "initial state", "random_unitary", "rand_circuit",
        "prep_state", "save_statevector",
    ],
    "incorrect_operator": [
        "gate", "operator", "rotation", "matrix", "transpile",
        "decomposition", "iden ", "identity gate", "unitary", "parameter",
        "deprecated", "controlled gate", "conditional gate",
        "angle", "radians", "degrees", "u1", "u2", "u3", "swap gate",
        "pauli", "hadamard", "phase", "compile", "removed", "renamed",
    ],
}


def classify_text_to_taxonomy(text: str) -> str:
    """Map free-text release-note descriptions to a taxonomy class.

    Returns the class with the highest keyword-hit count, or ``"unknown"``
    if nothing matched.
    """
    text_l = text.lower()
    scores = {cls: sum(1 for kw in kws if kw in text_l) for cls, kws in TAX_KEYWORDS.items()}
    best_cls, best_score = max(scores.items(), key=lambda x: x[1])
    return best_cls if best_score > 0 else "unknown"


def _is_low_quality_entry(text: str) -> bool:
    if len(text) < 80 or len(text) > 1500:
        return True
    text_l = text.lower()
    if re.search(r"renamed to", text_l) and len(text) < 200:
        return True
    if any(p in text_l for p in ["bump version", "updated dependency", "removed deprecated alias for"]):
        return True
    return False


SOURCE_TO_FRAMEWORK: dict[str, str] = {
    "qiskit_releasenotes": "qiskit",
    "qiskit_aer_releasenotes": "qiskit",
    "qiskit_ignis_releasenotes": "qiskit",
    "ibm_runtime_changelog": "qiskit",
    "pennylane_changelog": "pennylane",
    "lintq_rules": "qiskit",
}

# v6: extract from MULTIPLE YAML sections (was: only "fixes")
YAML_SECTIONS: list[str] = ["fixes", "deprecations", "upgrade"]


# ── YAML release-note extractor (Qiskit family) ─────────────────────────────

def extract_yaml_releasenote_entries(
    repo_root: Path,
    source_label: str,
    notes_subdir: str = "releasenotes/notes",
) -> list[BugPattern]:
    """Extract bug-fix / deprecation / upgrade entries from Qiskit-style
    YAML release notes."""
    repo_root = Path(repo_root)
    patterns: list[BugPattern] = []
    notes_dir = repo_root / notes_subdir
    if not notes_dir.exists():
        return patterns
    framework = SOURCE_TO_FRAMEWORK.get(source_label, "other")
    for yf in notes_dir.rglob("*.yaml"):
        try:
            d = yaml.safe_load(yf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for section_name in YAML_SECTIONS:
            entries = d.get(section_name)
            if not entries or not isinstance(entries, list):
                continue
            for i, item in enumerate(entries):
                if not isinstance(item, str):
                    continue
                text = item.strip()
                cleaned = re.sub(r":[a-z_]+:`([^`]+)`", r"\1", text)
                cleaned = re.sub(r"`([^`]+)`_", r"\1", cleaned)
                cleaned = re.sub(r"<https?://[^>]+>", "", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if _is_low_quality_entry(cleaned):
                    continue
                cls = classify_text_to_taxonomy(cleaned)
                if cls == "unknown":
                    continue
                patterns.append(BugPattern(
                    pattern_id=f"{source_label}_{section_name}_{yf.stem}_{i}",
                    name=f"{source_label} {section_name}: {yf.stem[:40]}",
                    taxonomy_class=cls,
                    description=cleaned[:800],
                    source=source_label,
                    tags=[source_label, "validated", section_name, framework],
                ))
    return patterns


# ── PennyLane changelog extractor ───────────────────────────────────────────

def extract_pennylane_changelog_entries(pl_root: Path) -> list[BugPattern]:
    patterns: list[BugPattern] = []
    changelog_dir = Path(pl_root) / "doc" / "releases"
    if not changelog_dir.exists():
        return patterns
    for md in changelog_dir.glob("changelog-*.md"):
        text = md.read_text(encoding="utf-8", errors="replace")
        sections = re.split(r"<h3>([^<]+)</h3>", text)
        for i in range(1, len(sections), 2):
            heading = sections[i].lower()
            body = sections[i + 1] if i + 1 < len(sections) else ""
            if "bug" in heading or "fix" in heading:
                section_tag = "fixes"
            elif "deprecat" in heading or "breaking" in heading or "removed" in heading:
                section_tag = "deprecations"
            else:
                continue
            for j, m in enumerate(re.finditer(r"\n\* +(.+?)(?=\n\* |\n<h3>|\Z)", body, re.DOTALL)):
                item = m.group(1).strip()
                cleaned = re.sub(r"```[^`]*```", "", item, flags=re.DOTALL)
                cleaned = re.sub(r"\[\(#\d+\)\]\([^)]+\)", "", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if _is_low_quality_entry(cleaned):
                    continue
                cls = classify_text_to_taxonomy(cleaned)
                if cls == "unknown":
                    continue
                patterns.append(BugPattern(
                    pattern_id=f"pennylane_{section_tag}_{md.stem}_{j}",
                    name=f"PennyLane {section_tag} in {md.stem}",
                    taxonomy_class=cls, description=cleaned[:800],
                    source="pennylane_changelog",
                    tags=["pennylane", "validated", section_tag, "pennylane"],
                ))
    return patterns


# ── IBM Runtime RST extractor ───────────────────────────────────────────────

def extract_ibm_runtime_entries(ibmr_root: Path) -> list[BugPattern]:
    patterns: list[BugPattern] = []
    notes_dir = Path(ibmr_root) / "release-notes"
    if not notes_dir.exists():
        return patterns
    section_re = re.compile(
        r"(Bug Fixes|Deprecation Notes|Upgrade Notes)\s*\n[-=]+\s*\n(.*?)"
        r"(?=\n[A-Z][a-zA-Z\s]+\s*\n[-=]+|\Z)", re.DOTALL,
    )
    for rst in notes_dir.glob("*.rst"):
        text = rst.read_text(encoding="utf-8", errors="replace")
        for sm in section_re.finditer(text):
            section_name, body = sm.group(1), sm.group(2)
            section_tag = "fixes" if "Bug" in section_name else "deprecations"
            for j, m in enumerate(re.finditer(r"\n-  +(.+?)(?=\n-  |\Z)", body, re.DOTALL)):
                item = m.group(1).strip()
                cleaned = re.sub(r"`([^`]+)`__", r"\1", item)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if _is_low_quality_entry(cleaned):
                    continue
                cls = classify_text_to_taxonomy(cleaned)
                if cls == "unknown":
                    continue
                patterns.append(BugPattern(
                    pattern_id=f"ibmr_{section_tag}_{rst.stem}_{j}",
                    name=f"IBM Runtime {section_tag} {rst.stem}",
                    taxonomy_class=cls, description=cleaned[:800],
                    source="ibm_runtime_changelog",
                    tags=["ibm_runtime", "validated", section_tag, "qiskit"],
                ))
    return patterns


# ── LintQ rule summaries ────────────────────────────────────────────────────

LINTQ_RULES: list[tuple[str, str, str]] = [
    ("ql-unmeasurable-qubits", "measurement_error",
     "Qubits in a register never measured but circuit uses qasm_simulator. Unmeasured qubits cannot contribute to classical output."),
    ("ql-constant-classic-bit", "measurement_error",
     "Classical register bit read but never written by any measure always reads zero."),
    ("ql-operation-after-measurement", "measurement_error",
     "Quantum gate applied to a qubit AFTER measurement is meaningless on hardware."),
    ("ql-conditional-without-measurement", "measurement_error",
     "c_if referencing a cbit never written by measure: condition always reads zero."),
    ("ql-double-measurement", "measurement_error",
     "Same qubit measured twice with no intervening operation; second measure is redundant."),
    ("ql-measure-all-abuse", "measurement_error",
     "measure_all allocates new cbits, leading to mismatched cbit indexing afterward."),
    ("ql-oversized-circuit", "incorrect_operator",
     "QuantumCircuit initialised with more qubits than operations actually use."),
    ("ql-deprecated-identity", "incorrect_operator",
     "Code calls deprecated identity gate API like iden() that has been removed."),
    ("ql-ghost-composition", "incorrect_operator",
     "Subcircuit built but never composed into main circuit; usually incomplete construction."),
    ("ql-op-after-optimization", "incorrect_operator",
     "Gates added to circuit AFTER transpile bypass optimization and may break hardware constraints."),
]


def build_lintq_patterns() -> list[BugPattern]:
    return [BugPattern(
        pattern_id=f"lintq_{rid}", name=rid, taxonomy_class=tax,
        description=desc, source="lintq_rules",
        tags=["lintq", "fse2024", "validated", "fixes", "qiskit"],
    ) for rid, tax, desc in LINTQ_RULES]


# ── Top-level builder ───────────────────────────────────────────────────────

def build_validated_kb(roots: dict[str, Path]) -> list[BugPattern]:
    """Build the v6 KB from multiple cloned source repositories.

    ``roots`` maps source labels to repository paths::

        {
          "qiskit":              Path("/work/qiskit"),
          "qiskit_aer":          Path("/work/qiskit_aer"),
          "qiskit_ignis":        Path("/work/qiskit_ignis"),
          "qiskit_ibm_runtime":  Path("/work/qiskit_ibm_runtime"),
          "pennylane":           Path("/work/pennylane"),
        }

    Missing roots are skipped silently.
    """
    kb: list[BugPattern] = []
    if "qiskit" in roots:
        kb += extract_yaml_releasenote_entries(roots["qiskit"], "qiskit_releasenotes")
    if "qiskit_aer" in roots:
        kb += extract_yaml_releasenote_entries(roots["qiskit_aer"], "qiskit_aer_releasenotes")
    if "qiskit_ignis" in roots:
        kb += extract_yaml_releasenote_entries(roots["qiskit_ignis"], "qiskit_ignis_releasenotes")
    if "qiskit_ibm_runtime" in roots:
        kb += extract_ibm_runtime_entries(roots["qiskit_ibm_runtime"])
    if "pennylane" in roots:
        kb += extract_pennylane_changelog_entries(roots["pennylane"])
    kb += build_lintq_patterns()
    return kb
