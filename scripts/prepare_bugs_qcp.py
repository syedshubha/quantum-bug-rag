#!/usr/bin/env python3
"""
prepare_bugs_qcp.py
===================
Download and pre-process the Bugs-QCP dataset from Zenodo (record 5834281).

Bugs-QCP serves as a **secondary taxonomy corpus** for knowledge-base enrichment and
bug-pattern retrieval.  It is **not** used for evaluation; all benchmark metrics are
computed exclusively on Bugs4Q.

Source
------
  https://doi.org/10.5281/zenodo.5834281

Usage
-----
  python scripts/prepare_bugs_qcp.py --output-dir data/bugs_qcp

Output layout
-------------
  <output-dir>/
    patterns.jsonl   # normalised bug-pattern entries (one JSON object per line)
    manifest.json    # provenance: DOI, download URL, preparation date, record checksum

Output schema (patterns.jsonl)
-------------------------------
  {
    "id":           "qcp-<n>",             # unique identifier
    "bug_type":     "<category>",          # high-level bug category
    "description":  "<text>",             # natural-language pattern description
    "platform":     "<framework|any>",    # target quantum framework
    "code_snippet": "<code>|null",        # illustrative excerpt, if available
    "tags":         ["<tag>", ...],       # taxonomy tags
    "source_doi":   "10.5281/zenodo.5834281"
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ZENODO_RECORD_ID = "5834281"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
SOURCE_DOI = "10.5281/zenodo.5834281"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_zenodo_metadata() -> dict:
    """Retrieve the Zenodo record metadata via the REST API."""
    logger.info("Fetching Zenodo record metadata for record %s …", ZENODO_RECORD_ID)
    # ZENODO_API_URL is a hard-coded HTTPS constant, so urlopen is safe here.
    req = urllib.request.Request(
        ZENODO_API_URL,
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (HTTPS constant)
        return json.loads(resp.read().decode())


def _download_file(url: str, dest: Path, expected_checksum: str | None = None) -> None:
    """Download *url* to *dest*, optionally verifying the MD5 checksum."""
    logger.info("Downloading %s …", url)
    # The URL is obtained from the Zenodo API response (HTTPS); urlretrieve is
    # acceptable here since we verify integrity via the Zenodo-provided checksum.
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (URL from HTTPS Zenodo API)
    if expected_checksum:
        # MD5 is used here solely to match the checksum format published by Zenodo.
        # It is not used for cryptographic security.
        actual = hashlib.md5(dest.read_bytes()).hexdigest()  # noqa: S324 (Zenodo checksum format)
        if actual != expected_checksum:
            logger.error(
                "Checksum mismatch for %s: expected %s, got %s",
                dest.name,
                expected_checksum,
                actual,
            )
            sys.exit(1)
        logger.info("Checksum verified ✓")


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP archive."""
    logger.info("Extracting %s …", archive_path.name)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)


def _normalise_entry(raw: dict, index: int) -> dict:
    """
    Normalise a raw Bugs-QCP entry into the canonical pattern schema.

    This function adapts upstream field names to the internal schema.  Adjust
    the field mappings if the upstream dataset structure changes.
    """
    return {
        "id": f"qcp-{index:05d}",
        "bug_type": raw.get("bug_type") or raw.get("category") or "unknown",
        "description": raw.get("description") or raw.get("text") or "",
        "platform": raw.get("platform") or raw.get("framework") or "any",
        "code_snippet": raw.get("code_snippet") or raw.get("code") or None,
        "tags": raw.get("tags") or [],
        "source_doi": SOURCE_DOI,
    }


def _parse_bugs_qcp(extracted_dir: Path) -> list[dict]:
    """
    Walk the extracted Bugs-QCP directory and collect normalised entries.

    The upstream archive may use CSV, JSON, or a nested directory layout.
    This implementation handles JSON and JSON-Lines files; extend as needed
    to support other formats present in the archive.
    """
    entries: list[dict] = []
    idx = 0

    for json_file in sorted(extracted_dir.rglob("*.json")):
        try:
            with open(json_file) as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON file: %s", json_file)
            continue

        if isinstance(data, list):
            for raw in data:
                if isinstance(raw, dict):
                    entries.append(_normalise_entry(raw, idx))
                    idx += 1
        elif isinstance(data, dict):
            entries.append(_normalise_entry(data, idx))
            idx += 1

    for jsonl_file in sorted(extracted_dir.rglob("*.jsonl")):
        with open(jsonl_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    entries.append(_normalise_entry(raw, idx))
                    idx += 1
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in %s", jsonl_file)

    return entries


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write *entries* as a JSON-Lines file."""
    with open(path, "w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Zenodo metadata to discover the download URL(s) and checksums.
    try:
        metadata = _fetch_zenodo_metadata()
    except Exception as exc:
        logger.error(
            "Failed to fetch Zenodo metadata: %s\n"
            "Check your internet connection or visit %s manually.",
            exc,
            f"https://zenodo.org/record/{ZENODO_RECORD_ID}",
        )
        sys.exit(1)

    files = metadata.get("files", [])
    if not files:
        logger.error(
            "No files found in Zenodo record %s.  "
            "Visit https://zenodo.org/record/%s to inspect the record manually.",
            ZENODO_RECORD_ID,
            ZENODO_RECORD_ID,
        )
        sys.exit(1)

    # 2. Download the first (typically only) archive file.
    file_meta = files[0]
    download_url: str = file_meta.get("links", {}).get("self", "")
    if not download_url:
        logger.error("Could not determine download URL from Zenodo metadata.")
        sys.exit(1)

    expected_md5: str | None = (
        file_meta.get("checksum", "").replace("md5:", "") or None
    )

    archive_path = output_dir / file_meta.get("key", "bugs_qcp.zip")
    _download_file(download_url, archive_path, expected_checksum=expected_md5)

    # 3. Extract the archive.
    extract_dir = output_dir / "raw"
    extract_dir.mkdir(exist_ok=True)
    if archive_path.suffix == ".zip":
        _extract_archive(archive_path, extract_dir)
    else:
        logger.warning(
            "Unexpected archive format: %s.  Attempting extraction anyway.",
            archive_path.suffix,
        )
        _extract_archive(archive_path, extract_dir)

    # 4. Parse and normalise entries.
    entries = _parse_bugs_qcp(extract_dir)
    if not entries:
        logger.warning(
            "No entries were parsed from the Bugs-QCP archive.  "
            "The upstream layout may have changed — inspect %s and update "
            "_parse_bugs_qcp().",
            extract_dir,
        )

    logger.info("Parsed %d bug-pattern entries from Bugs-QCP.", len(entries))

    # 5. Write normalised output.
    patterns_path = output_dir / "patterns.jsonl"
    _write_jsonl(patterns_path, entries)
    logger.info("Patterns written to %s", patterns_path)

    # 6. Write manifest.
    manifest = {
        "dataset": "Bugs-QCP",
        "source_doi": SOURCE_DOI,
        "zenodo_record_id": ZENODO_RECORD_ID,
        "download_url": download_url,
        "archive_sha256": _sha256(archive_path),
        "prepared_at": datetime.now(tz=timezone.utc).isoformat(),
        "entry_count": len(entries),
        "role": (
            "Secondary taxonomy corpus for knowledge-base enrichment. "
            "Not used for evaluation — all metrics are computed on Bugs4Q."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    # 7. Remove the raw archive to save disk space (the extracted files remain).
    archive_path.unlink()
    logger.info("Removed raw archive %s to save disk space.", archive_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bugs_qcp"),
        help="Directory to write processed output (default: data/bugs_qcp)",
    )
    args = parser.parse_args()
    prepare(args.output_dir)


if __name__ == "__main__":
    main()
