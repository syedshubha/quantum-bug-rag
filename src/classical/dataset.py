"""Dataset construction for the classical-vs-quantum binary track.

* BQCP — primary evaluation set, both classes. Ground truth is the ``type``
  column of ``annotation_bugs.csv`` (Paltenghi & Pradel, 2022).
* Bugs4Q — external all-quantum holdout for the purity check.

For BQCP we concatenate the contents of ``before/`` to form the buggy
snippet, keeping source/build/test/script files because real classical
bugs include build- and packaging-system fixes.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .schemas import BugSample


# Source-code, build, and script extensions.  Keeping build files
# (CMakeLists, .csproj, .ps1) is intentional: BQCP's classical-labelled
# bugs include real build/test fixes; excluding them would bias the
# dataset toward only-Python bugs.
KEEP_EXT: set[str] = {
    ".py", ".cc", ".cpp", ".cxx", ".hpp", ".h", ".cs", ".qs",
    ".pyx", ".fs", ".fsproj", ".csproj", ".txt", ".ps1",
    ".targets", ".cmake", ".ipynb",
}
KEEP_NAME: set[str] = {"CMakeLists.txt", "build.ps1", "pack.ps1", "test.ps1"}


def _read_text_safe(p: Path, max_bytes: int = 200_000) -> str:
    try:
        data = p.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _concat_before(folder: Path, max_total: int = 50_000) -> str:
    """Concatenate every file under ``before/`` that we recognise as code."""
    before = folder / "before"
    if not before.exists():
        return ""
    parts: list[str] = []
    total = 0
    for fp in sorted(before.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in KEEP_EXT and fp.name not in KEEP_NAME:
            continue
        text = _read_text_safe(fp)
        if not text:
            continue
        rel = fp.relative_to(before)
        chunk = f"### file: {rel}\n{text}\n"
        if total + len(chunk) > max_total:
            chunk = chunk[: max_total - total] + "\n# [truncated]\n"
        parts.append(chunk)
        total += len(chunk)
        if total >= max_total:
            break
    return "\n".join(parts)


def build_bqcp(repo_root: Path) -> tuple[list[BugSample], dict]:
    """Build BugSample list from a cloned Bugs-QCP repository.

    Returns ``(samples, stats)`` where ``stats`` reports how many CSV rows
    were skipped due to missing folders or empty snippets — useful for
    smoke-checking that the layout is correct.
    """
    repo_root = Path(repo_root)
    fixes = repo_root / "artifacts" / "minimal_bugfixes"
    csv_p = repo_root / "artifacts" / "annotation_bugs.csv"

    idx: dict[tuple[str, str], Path] = {}
    for mp in fixes.rglob("metadata.json"):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        proj, bid = m.get("project_name"), m.get("id")
        if proj and bid is not None:
            idx[(proj, str(bid))] = mp.parent

    samples: list[BugSample] = []
    missing, empty = 0, 0
    with csv_p.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["real"] != "bug":
                continue
            t = row["type"].strip().lower()
            if t not in ("classical", "quantum"):
                continue

            suffix = row["repo"].split("/")[-1]
            folder = None
            for cand in (suffix, suffix.lower(), suffix.capitalize()):
                if (cand, row["id"]) in idx:
                    folder = idx[(cand, row["id"])]
                    break
            if folder is None:
                missing += 1
                continue

            code_str = _concat_before(folder)
            if not code_str.strip():
                empty += 1
                continue

            samples.append(BugSample(
                sample_id=f"bqcp_{row['id']}",
                source="bqcp",
                code=code_str,
                ground_truth=t,
                metadata={
                    "repo": row["repo"],
                    "component": row["component"],
                    "bug_pattern": row["bug_pattern"],
                    "symptom": row["symptom"],
                },
            ))

    stats = {
        "matched_csv_rows": len(samples) + missing + empty,
        "missing_folder": missing,
        "empty_snippet": empty,
    }
    return samples, stats


def build_bugs4q(repo_root: Path) -> list[BugSample]:
    """Treat every Bugs4Q sample as ``ground_truth='quantum'``."""
    repo_root = Path(repo_root)
    samples: list[BugSample] = []
    for root, _dirs, files in os.walk(repo_root):
        fl = {f.lower(): f for f in files}
        b = fl.get("buggy.py") or fl.get("bug_version.py")
        if not b:
            continue
        bp = Path(root) / b
        text = _read_text_safe(bp)
        if not text.strip():
            continue
        rel = bp.relative_to(repo_root)
        sid = "bugs4q_" + str(rel.parent).replace("/", "_").replace(" ", "_")
        samples.append(BugSample(
            sample_id=sid, source="bugs4q",
            code=text, ground_truth="quantum",
            metadata={"buggy_path": str(rel)},
        ))
    return samples
