"""
bugs4q_labels.py – Explicit Bugs4Q label extraction and taxonomy mapping.

We extract case metadata from the upstream Bugs4Q README table and map upstream
"Type" fields into our local taxonomy classes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .schemas import BugSample
from .utils import get_logger

logger = get_logger(__name__)

_LINK_TARGET_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_PATH_TOKEN_RE = re.compile(r"(?:(?:\./)?)([A-Za-z0-9_\-#%/]+)")

_UPSTREAM_TYPE_TO_TAXONOMY: dict[str, str] = {
    "parameter": "incorrect_operator",
    "qr,qc": "incorrect_qubit_mapping",
    "empty circuit": "incorrect_operator",
    "qasm": "incorrect_operator",
    "output wrong": "measurement_error",
    "wrong circuit design": "incorrect_operator",
    "wrong command": "incorrect_operator",
    "being not familiar with the usage of measuring all bit using existing registers.": "measurement_error",
    "qiskit distinguishes operations in `gate`s": "incorrect_operator",
    "quantumcircuit.parameters` only tracks unbound parameters.": "incorrect_operator",
    "not fully understanding qasm and statevector/eval computation.": "wrong_initial_state",
    "the circuit library requires `decompose` for \"lin_comb\".": "incorrect_operator",
    "ignoring the impact of measurement": "measurement_error",
    "order during measurement": "measurement_error",
    "oversized resource consumption": "measurement_error",
    "unfamiliar with api": "incorrect_operator",
    "figure problem": "incorrect_operator",
    "name conflict": "incorrect_operator",
    "label convention is reversed(|011>&|110>)": "wrong_initial_state",
    "wrong operation with gate": "incorrect_operator",
    "qft operation*": "incorrect_operator",
    "qfe output wrong": "measurement_error",
    "ccx": "incorrect_operator",
    "random gates": "incorrect_operator",
    "not a dag": "incorrect_operator",
    "wait()": "incorrect_operator",
    "grover algrithm": "incorrect_operator",
    "only for simulator": "measurement_error",
    "compiler() removerd": "incorrect_operator",
    "obtain amplitude": "measurement_error",
    "output": "measurement_error",
    "threads": "incorrect_operator",
    "statevector": "wrong_initial_state",
    "initialization": "wrong_initial_state",
    "`transpile` required": "incorrect_operator",
    "transpile` required": "incorrect_operator",
    "outdated grammar": "incorrect_operator",
    "start state is reversed": "wrong_initial_state",
    "no output": "measurement_error",
    "random number error": "wrong_initial_state",
    "wrong circuit operation": "incorrect_operator",
    "call wrong function": "incorrect_operator",
}


@dataclass(frozen=True)
class Bugs4QCase:
    buggy_path: str
    upstream_type: str | None


@dataclass(frozen=True)
class LabelStats:
    total_samples: int
    labelled_samples: int
    unlabelled_samples: int
    label_distribution: dict[str, int]
    unmapped_types: dict[str, int]


def write_type_mapping_manifest(output_dir: Path) -> Path:
    """Write the explicit upstream-type to taxonomy mapping for inspection."""
    path = output_dir / "labels.type_mapping.json"
    path.write_text(
        json.dumps(_UPSTREAM_TYPE_TO_TAXONOMY, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def extract_cases_from_readme(readme_path: Path) -> list[Bugs4QCase]:
    lines = readme_path.read_text(encoding="utf-8", errors="replace").splitlines()
    cases: list[Bugs4QCase] = []

    header_cols: list[str] | None = None
    buggy_idx: int | None = None
    type_idx: int | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        lower_cells = [cell.lower() for cell in cells]

        if "buggy" in lower_cells and "type" in lower_cells:
            header_cols = cells
            buggy_idx = lower_cells.index("buggy")
            type_idx = lower_cells.index("type")
            continue

        if header_cols is None:
            continue

        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue

        if buggy_idx is None or type_idx is None:
            continue
        cells = _coerce_row_width(cells, expected_len=len(header_cols), type_idx=type_idx)
        if len(cells) <= max(buggy_idx, type_idx):
            continue

        buggy_cell = cells[buggy_idx]
        type_cell = cells[type_idx]

        buggy_path = _extract_buggy_path(buggy_cell)
        if buggy_path is None:
            continue

        upstream_type = _clean_type(type_cell)
        cases.append(Bugs4QCase(buggy_path=buggy_path, upstream_type=upstream_type))

    dedup: dict[str, Bugs4QCase] = {}
    for case in cases:
        dedup[case.buggy_path] = case
    return sorted(dedup.values(), key=lambda c: c.buggy_path)


def apply_labels_to_samples(
    samples: list[BugSample],
    cases: list[Bugs4QCase],
) -> tuple[list[BugSample], LabelStats, list[dict]]:
    """Apply upstream-case labels to samples and return updated records and stats."""
    case_map = {case.buggy_path: case for case in cases}
    path_index = sorted(case_map.keys(), key=len, reverse=True)

    labelled_samples: list[BugSample] = []
    label_counter: Counter[str] = Counter()
    unmapped_type_counter: Counter[str] = Counter()
    mapping_rows: list[dict] = []

    for sample in samples:
        sample_data = sample.model_dump()
        metadata = dict(sample_data.get("metadata", {}))
        rel_path = str(metadata.get("path", ""))

        matched_path = _match_case_path(rel_path, path_index)
        case = case_map.get(matched_path) if matched_path else None

        upstream_type = case.upstream_type if case else None
        taxonomy_label = _map_upstream_type(upstream_type)
        label_status = "labelled" if taxonomy_label is not None else "unlabelled"

        if taxonomy_label is not None:
            sample_data["ground_truth"] = taxonomy_label
            label_counter[taxonomy_label] += 1
        else:
            sample_data["ground_truth"] = None
            if upstream_type:
                unmapped_type_counter[upstream_type] += 1

        metadata.update(
            {
                "upstream_case_path": matched_path,
                "upstream_type": upstream_type,
                "label_status": label_status,
            }
        )
        sample_data["metadata"] = metadata
        labelled_samples.append(BugSample(**sample_data))

        mapping_rows.append(
            {
                "sample_id": sample.sample_id,
                "path": rel_path,
                "upstream_case_path": matched_path,
                "upstream_type": upstream_type,
                "taxonomy_label": taxonomy_label,
                "label_status": label_status,
            }
        )

    labelled_count = sum(1 for sample in labelled_samples if sample.ground_truth is not None)
    total_count = len(labelled_samples)
    stats = LabelStats(
        total_samples=total_count,
        labelled_samples=labelled_count,
        unlabelled_samples=total_count - labelled_count,
        label_distribution=dict(sorted(label_counter.items())),
        unmapped_types=dict(sorted(unmapped_type_counter.items())),
    )
    return labelled_samples, stats, mapping_rows


def _extract_buggy_path(cell: str) -> str | None:
    for target in _LINK_TARGET_RE.findall(cell):
        cleaned = _normalise_rel_path(target)
        if cleaned:
            return cleaned

    for token in _PATH_TOKEN_RE.findall(cell):
        cleaned = _normalise_rel_path(token)
        if cleaned and "bug" in cleaned.lower():
            return cleaned
    return None


def _normalise_rel_path(path_value: str) -> str | None:
    value = path_value.strip()
    if not value:
        return None
    value = value.lstrip("./")
    value = value.rstrip("/")
    return value or None


def _clean_type(raw_type: str) -> str | None:
    value = raw_type.strip().strip("`")
    value = re.sub(r"\s+", " ", value)
    if not value or value == "---":
        return None
    return value


def _match_case_path(sample_path: str, case_paths: list[str]) -> str | None:
    normalized_sample_path = _normalise_rel_path(sample_path)
    if normalized_sample_path is None:
        return None

    for case_path in case_paths:
        if normalized_sample_path == case_path:
            return case_path
        if normalized_sample_path.startswith(case_path + "/"):
            return case_path
    return None


def _coerce_row_width(cells: list[str], expected_len: int, type_idx: int) -> list[str]:
    """Normalize row width when type values contain inline pipe characters."""
    if len(cells) <= expected_len:
        return cells

    trailing_count = max(expected_len - type_idx - 1, 0)
    type_end = len(cells) - trailing_count
    if type_end <= type_idx:
        return cells

    merged_type = " | ".join(cells[type_idx:type_end]).strip()
    return cells[:type_idx] + [merged_type] + cells[type_end:]


def _map_upstream_type(upstream_type: str | None) -> str | None:
    if upstream_type is None:
        return None
    normalized = upstream_type.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return _UPSTREAM_TYPE_TO_TAXONOMY.get(normalized)
