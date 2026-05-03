    skip_prefixes = (
        "speed up ", "document ", "add new ", "added new ", "added: ",
        "documentation ", "patch by ", "port the ",
    )
    if any(text.lower().startswith(p) for p in skip_prefixes):
        return True
    return False


# ── Quantum side ────────────────────────────────────────────────────────────

YAML_SECTIONS_TO_KEEP = ["fixes", "deprecations", "upgrade"]


def extract_yaml_releasenotes(
    repo_root: Path,
    framework: str,
    label_prefix: str,
    notes_subdir: str = "releasenotes/notes",
) -> list[KBEntry]:
    out: list[KBEntry] = []
    nd = Path(repo_root) / notes_subdir
    if not nd.exists():
        return out
    for yf in nd.rglob("*.yaml"):
        try:
            doc = yaml.safe_load(yf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        for section in YAML_SECTIONS_TO_KEEP:
            entries = doc.get(section)
            if not isinstance(entries, list):
                continue
            for i, item in enumerate(entries):
                if not isinstance(item, str):
                    continue
                t = _strip_rst(item)
                if _is_low_quality(t):
                    continue
                out.append(KBEntry(
                    entry_id=f"{label_prefix}_{section}_{yf.stem}_{i}",
                    domain="quantum", framework=framework,
                    description=t[:800],
                ))
    return out


def extract_pennylane_changelog(pl_root: Path) -> list[KBEntry]:
    out: list[KBEntry] = []
    cd = Path(pl_root) / "doc" / "releases"
    if not cd.exists():
        return out
    for md in cd.glob("changelog-*.md"):
        text = md.read_text(encoding="utf-8", errors="replace")
        sections = re.split(r"<h3>([^<]+)</h3>", text)
        for i in range(1, len(sections), 2):
            heading = sections[i].lower()
            body = sections[i + 1] if i + 1 < len(sections) else ""
            if not any(k in heading for k in ("bug", "fix", "deprecat", "breaking", "removed")):
                continue
            for j, m in enumerate(re.finditer(r"\n\* +(.+?)(?=\n\* |\n<h3>|\Z)",
                                              body, re.DOTALL)):
                t = m.group(1).strip()
                t = re.sub(r"```[^`]*```", "", t, flags=re.DOTALL)
                t = re.sub(r"\[\(#\d+\)\]\([^)]+\)", "", t)
                t = re.sub(r"\s+", " ", t).strip()
                if _is_low_quality(t):
                    continue
                out.append(KBEntry(
                    entry_id=f"pennylane_{md.stem}_{j}",
                    domain="quantum", framework="pennylane",
                    description=t[:800],
                ))
    return out


# ── Classical side ──────────────────────────────────────────────────────────

CPYTHON_BUGFIX_SECTIONS: set[str] = {
    "Security", "Core and Builtins", "Library", "C API", "Windows", "macOS",
}


def extract_cpython_news(cpy_root: Path, max_per_file: int = 25) -> list[KBEntry]:
    """CPython NEWS.d entries from bug-fix-flavoured sections.

    NEWS.d uses RST with a small front-matter block; entries are separated
    by lines containing only ``..``.
    """
    out: list[KBEntry] = []
    nd = Path(cpy_root) / "Misc" / "NEWS.d"
    if not nd.exists():
        return out
    files = sorted(nd.glob("*.rst"))
    files = [f for f in files if not f.name.startswith("next")]
    for rst in files:
        text = rst.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r"\n\.\.\s*\n", text)
        n_kept = 0
        for ei, entry in enumerate(entries):
            if n_kept >= max_per_file:
                break
            if not entry.strip():
                continue
            section_match = re.search(r"^\.\. section:\s*(.+)$", entry, re.MULTILINE)
            if not section_match:
                continue
            section = section_match.group(1).strip()
            if section not in CPYTHON_BUGFIX_SECTIONS:
                continue
            lines = entry.splitlines()
            body_start = 0
            for i, line in enumerate(lines):
                if line.startswith("..") or line.startswith("   "):
                    body_start = i + 1
                elif line.strip() == "":
                    body_start = i + 1
                else:
                    break
            body = "\n".join(lines[body_start:]).strip()
            body = _strip_rst(body)
            if _is_low_quality(body):
                continue
            out.append(KBEntry(
                entry_id=f"cpython_{rst.stem}_{ei}",
                domain="classical", framework="cpython",
                description=body[:800],
            ))
            n_kept += 1
    return out


def extract_numpy_releasenotes(np_root: Path, max_per_file: int = 60) -> list[KBEntry]:
    """NumPy release-note entries from Bug-fixes / Changes / Compatibility."""
    out: list[KBEntry] = []
    candidates = [
        Path(np_root) / "doc" / "source" / "release",
        Path(np_root) / "doc" / "release",
    ]
    nd = next((p for p in candidates if p.exists()), None)
    if nd is None:
        return out

    keep_sections = ("Bug fixes", "Changes", "Compatibility notes")
    for rst in nd.glob("*-notes.rst"):
        text = rst.read_text(encoding="utf-8", errors="replace")
        sections = re.split(r"\n([A-Z][^\n]+)\n([=\-]{3,})\n", text)
        n_kept = 0
        for k in range(1, len(sections) - 2, 3):
            header, body = sections[k].strip(), sections[k + 2]
            if not any(header.startswith(s) for s in keep_sections):
                continue
            for ei, para in enumerate(re.split(r"\n\s*\n", body)):
                if n_kept >= max_per_file:
                    break
                p = _strip_rst(para)
                if not p or set(p) <= set("=-~^") or p.startswith(".. ") or p.startswith("("):
                    continue
                if _is_low_quality(p):
                    continue
                out.append(KBEntry(
                    entry_id=f"numpy_{rst.stem}_{ei}",
                    domain="classical", framework="numpy",
                    description=p[:800],
                ))
                n_kept += 1
    return out


# ── Symmetric balancing ─────────────────────────────────────────────────────

def proportional_downsample(
    pool: list[KBEntry],
    target_n: int,
    seed: int = 42,
) -> list[KBEntry]:
    """Downsample ``pool`` to ``target_n`` while preserving framework shares."""
    if len(pool) <= target_n:
        return list(pool)
    rng = random.Random(seed)
    by_fw: dict[str, list[KBEntry]] = {}
    for e in pool:
        by_fw.setdefault(e.framework, []).append(e)
    out: list[KBEntry] = []
    for items in by_fw.values():
        share = max(1, int(round(target_n * len(items) / len(pool))))
        share = min(share, len(items))
        out.extend(rng.sample(items, share))
    if len(out) > target_n:
        out = rng.sample(out, target_n)
    return out


def build_symmetric_kb(roots: dict[str, Path], seed: int = 42) -> tuple[list[KBEntry], list[KBEntry]]:
    """Construct the (kb_quantum, kb_classical) pair, balanced to equal size.

    ``roots`` maps framework labels to repository paths.  Recognised keys:
    ``qiskit``, ``qiskit_aer``, ``pennylane``, ``cpython``, ``numpy``.
    Missing roots are skipped silently.
    """
    kb_quantum: list[KBEntry] = []
    if "qiskit" in roots:
        kb_quantum += extract_yaml_releasenotes(roots["qiskit"], "qiskit", "qiskit_rn")
    if "qiskit_aer" in roots:
        kb_quantum += extract_yaml_releasenotes(roots["qiskit_aer"], "qiskit", "qiskit_aer_rn")
    if "pennylane" in roots:
        kb_quantum += extract_pennylane_changelog(roots["pennylane"])

    kb_classical: list[KBEntry] = []
    if "cpython" in roots:
        kb_classical += extract_cpython_news(roots["cpython"])
    if "numpy" in roots:
        kb_classical += extract_numpy_releasenotes(roots["numpy"])

    target = min(len(kb_quantum), len(kb_classical))
    if target > 0:
        kb_quantum = proportional_downsample(kb_quantum, target, seed=seed)
        kb_classical = proportional_downsample(kb_classical, target, seed=seed)
    return kb_quantum, kb_classical


def kb_summary(kb: list[KBEntry]) -> dict:
    return {
        "n": len(kb),
        "by_framework": dict(Counter(e.framework for e in kb)),
        "by_domain": dict(Counter(e.domain for e in kb)),
    }
