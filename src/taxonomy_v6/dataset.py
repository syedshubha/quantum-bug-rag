"""Bugs4Q and Bugs-QCP loaders for the v6 forced-choice taxonomy track.

These mirror the notebook's adapters. ``build_bugs4q`` parses the upstream
README's type column and walks the buggy ``.py`` files. ``build_bugsqcp``
extracts focused diff snippets between ``before/`` and ``after/`` directories
of each ``minimal_bugfixes`` sub-folder for quantum-only bugs.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .schemas import BugSample

# ── Bugs4Q ──────────────────────────────────────────────────────────────────

BUGS4Q_TYPE_MAP: dict[str, str] = {
    "parameter": "incorrect_operator", "qr,qc": "incorrect_qubit_mapping",
    "empty circuit": "incorrect_operator", "qasm": "incorrect_operator",
    "output wrong": "measurement_error", "wrong circuit design": "incorrect_operator",
    "wrong command": "incorrect_operator",
    "being not familiar with the usage of measuring all bit using existing registers.": "measurement_error",
    "qiskit distinguishes operations in `gate`s": "incorrect_operator",
    "quantumcircuit.parameters` only tracks unbound parameters.": "incorrect_operator",
    "not fully understanding qasm and statevector/eval computation.": "wrong_initial_state",
    "the circuit library requires `decompose` for \"lin_comb\".": "incorrect_operator",
    "ignoring the impact of measurement": "measurement_error",
    "order during measurement": "measurement_error",
    "oversized resource consumption": "measurement_error",
    "unfamiliar with api": "incorrect_operator", "figure problem": "incorrect_operator",
    "name conflict": "incorrect_operator",
    "label convention is reversed(|011>&|110>)": "wrong_initial_state",
    "wrong operation with gate": "incorrect_operator",
    "qft operation*": "incorrect_operator", "qfe output wrong": "measurement_error",
    "ccx": "incorrect_operator", "random gates": "incorrect_operator",
    "not a dag": "incorrect_operator", "wait()": "incorrect_operator",
    "grover algrithm": "incorrect_operator", "only for simulator": "measurement_error",
    "compiler() removerd": "incorrect_operator", "obtain amplitude": "measurement_error",
    "output": "measurement_error", "threads": "incorrect_operator",
    "statevector": "wrong_initial_state", "initialization": "wrong_initial_state",
    "`transpile` required": "incorrect_operator", "transpile` required": "incorrect_operator",
    "outdated grammar": "incorrect_operator", "start state is reversed": "wrong_initial_state",
    "no output": "measurement_error", "random number error": "wrong_initial_state",
    "wrong circuit operation": "incorrect_operator", "call wrong function": "incorrect_operator",
}

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_FIXED_FN = {"fixed.py", "fix.py", "fixed_version.py", "modify.py", "mod.py"}


def _map_bugs4q_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return BUGS4Q_TYPE_MAP.get(re.sub(r"\s+", " ", raw.lower().strip()))


def _parse_bugs4q_readme(readme_path: Path) -> dict[str, Optional[str]]:
    """Extract (buggy-path, type-string) pairs from the upstream Bugs4Q README."""
    cases: dict[str, Optional[str]] = {}
    if not readme_path.exists():
        return cases
    headers, buggy_idx, type_idx = None, None, None
    for line in readme_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        lower = [c.lower() for c in cells]
        if "buggy" in lower and "type" in lower:
            headers, buggy_idx, type_idx = cells, lower.index("buggy"), lower.index("type")
            continue
        if headers is None or all(set(c) <= {"-", ":"} for c in cells):
            continue
        if len(cells) > len(headers):
            trail = max(len(headers) - type_idx - 1, 0)
            end = len(cells) - trail
            cells = cells[:type_idx] + [" | ".join(cells[type_idx:end])] + cells[end:]
        if len(cells) <= max(buggy_idx, type_idx):
            continue
        bcell, tcell = cells[buggy_idx], cells[type_idx].strip("`").strip()
        for tgt in _LINK_RE.findall(bcell):
            cleaned = tgt.replace("\\", "/").lstrip("./").rstrip("/")
            if cleaned:
                cases[cleaned] = tcell if tcell and tcell != "---" else None
                break
    return cases


def _is_buggy_file(py: Path) -> bool:
    n = py.name.lower()
    if n in _FIXED_FN:
        return False
    if n in {"buggy.py", "bug_version.py"} or n.startswith("bug"):
        return True
    if "buggy" in {p.lower() for p in py.parts}:
        return True
    return False


def build_bugs4q(repo_root: Path) -> list[BugSample]:
    """Build BugSample list from a cloned upstream Bugs4Q repository."""
    repo_root = Path(repo_root)
    cases = _parse_bugs4q_readme(repo_root / "README.md")
    sorted_paths = sorted(cases.keys(), key=len, reverse=True)

    def lookup_type(rel: Path) -> Optional[str]:
        rel_s = str(rel).replace("\\", "/").lstrip("./")
        for cp in sorted_paths:
            if rel_s == cp or rel_s.startswith(cp + "/"):
                return cases[cp]
        return None

    samples: list[BugSample] = []
    for py in sorted(repo_root.rglob("*.py")):
        if not _is_buggy_file(py):
            continue
        try:
            code = py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not code.strip():
            continue
        rel = py.relative_to(repo_root)
        raw_type = lookup_type(rel)
        label = _map_bugs4q_type(raw_type)
        samples.append(BugSample(
            sample_id=f"bugs4q_{len(samples):04d}",
            source="bugs4q", code=code, ground_truth=label,
            metadata={"path": str(rel), "bug_pattern_raw": raw_type},
        ))
    return samples


# ── Bugs-QCP ────────────────────────────────────────────────────────────────

BQCP_PATTERN_MAP: dict[str, str] = {
    "qubit mapping": "incorrect_qubit_mapping", "qubit index": "incorrect_qubit_mapping",
    "index": "incorrect_qubit_mapping", "operator": "incorrect_operator",
    "gate": "incorrect_operator", "wrong gate": "incorrect_operator",
    "barrier": "missing_barrier", "initial": "wrong_initial_state",
    "statevector": "wrong_initial_state", "measurement": "measurement_error",
    "measure": "measurement_error", "barrier related": "missing_barrier",
    "overlooked qubit order": "incorrect_qubit_mapping",
    "msb-lsb convention": "incorrect_qubit_mapping",
    "msb-lsb convention mismatches": "incorrect_qubit_mapping",
    "wrong identifier": "incorrect_qubit_mapping",
    "incorrect numerical computation": "incorrect_operator",
    "incorrect ir - wrong information": "incorrect_operator",
    "incorrect ir - missing information": "incorrect_operator",
    "incorrect circuit": "incorrect_operator",
    "api misuse - internal": "incorrect_operator",
    "api misuse - external": "incorrect_operator", "api misuse": "incorrect_operator",
    "typo": "incorrect_operator", "type problem": "incorrect_operator",
    "incorrect final measurement": "measurement_error",
    "incorrect randomness handling": "wrong_initial_state",
}


def _map_bqcp(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    for piece in (p.strip().lower() for p in raw.split(",")):
        m = BQCP_PATTERN_MAP.get(piece)
        if m and m != "unknown":
            return m
    return None


def _build_bqcp_index(fixes_root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for mp in fixes_root.rglob("metadata.json"):
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        proj, bid = meta.get("project_name"), meta.get("id")
        if proj and bid is not None:
            index[(proj, str(bid))] = mp.parent
    return index


def extract_focused_snippet(folder: Path, context_lines: int = 10) -> Optional[str]:
    """Return the most-modified file's diff hunks as a focused buggy snippet.

    Runs unified ``diff -u`` between ``before/`` and ``after/``, picks the file
    with the most deletions, and reconstructs its pre-fix content with the
    surrounding context lines. Returns ``None`` if no diff is available.
    """
    before_dir = folder / "before"
    after_dir = folder / "after"
    if not before_dir.exists() or not after_dir.exists():
        return None
    try:
        d = subprocess.run(
            ["diff", "-u", f"-U{context_lines}", "-r", str(before_dir), str(after_dir)],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return None
    if not d.stdout.strip():
        return None

    per_file: dict[str, list[str]] = defaultdict(list)
    current_file, current = None, []
    for line in d.stdout.splitlines():
        if line.startswith("--- "):
            if current and current_file:
                per_file[current_file].append("\n".join(current))
                current = []
            full = line[4:].split("\t")[0]
            current_file = full.replace(str(before_dir) + "/", "").replace(str(before_dir), "")
        elif line.startswith("+++"):
            continue
        elif line.startswith("@@"):
            if current and current_file:
                per_file[current_file].append("\n".join(current))
                current = []
            current.append(f"# {line}")
        elif line.startswith("+"):
            continue
        elif line.startswith("-"):
            current.append(line[1:])
        elif line.startswith(" "):
            current.append(line[1:])
    if current and current_file:
        per_file[current_file].append("\n".join(current))
    if not per_file:
        return None

    deletions_per_file: dict[str, int] = defaultdict(int)
    cur_f = None
    for line in d.stdout.splitlines():
        if line.startswith("--- "):
            full = line[4:].split("\t")[0]
            cur_f = full.replace(str(before_dir) + "/", "").replace(str(before_dir), "")
        elif line.startswith("-") and not line.startswith("---") and cur_f:
            deletions_per_file[cur_f] += 1

    best_file = max(per_file.keys(), key=lambda f: deletions_per_file.get(f, 0) or len(per_file[f]))
    hunks = [h for h in per_file[best_file] if h.strip()]
    if not hunks:
        return None
    return f"# === {best_file} ===\n" + "\n\n".join(hunks)


def build_bugsqcp(repo_root: Path, *, quantum_only: bool = True) -> list[BugSample]:
    """Build BugSample list from a cloned Bugs-QCP repository."""
    repo_root = Path(repo_root)
    csv_path = repo_root / "artifacts" / "annotation_bugs.csv"
    fixes_root = repo_root / "artifacts" / "minimal_bugfixes"
    index = _build_bqcp_index(fixes_root)

    samples: list[BugSample] = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["real"] != "bug":
                continue
            if quantum_only and row["type"].strip().lower() != "quantum":
                continue
            suffix = row["repo"].split("/")[-1]
            folder = None
            for cand in (suffix, suffix.lower(), suffix.capitalize()):
                if (cand, row["id"]) in index:
                    folder = index[(cand, row["id"])]
                    break
            if folder is None:
                continue
            snippet = extract_focused_snippet(folder, context_lines=10)
            if not snippet:
                continue
            samples.append(BugSample(
                sample_id=f"bqcp_{row['id'].replace(',', '_'):>06}",
                source="bugsqcp", code=snippet,
                ground_truth=_map_bqcp(row["bug_pattern"]),
                metadata={
                    "repo": row["repo"],
                    "bug_pattern_raw": row["bug_pattern"],
                    "type": row["type"],
                },
            ))
    return samples
