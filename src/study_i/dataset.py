"""Dataset loading for Study I CodeBERT experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schemas import LABEL_TO_ID, StudyISample


def load_labeled_bug_reports(path: Path) -> list[StudyISample]:
    """Load the external JSON dataset used in Study I.

    Expected input is a JSON list of objects. Each object should contain a
    binary label under ``bug_category`` and natural-language / code fields under
    some combination of:

    - ``name``
    - ``description``
    - ``example_code`` or ``code``
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected a JSON list in {path}")

    samples: list[StudyISample] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        label = str(item.get("bug_category", "")).strip().lower()
        if label not in LABEL_TO_ID:
            continue
        sample_id = str(
            item.get("sample_id")
            or item.get("id")
            or f"study_i_{idx:04d}"
        )
        code = str(item.get("example_code") or item.get("code") or "").strip()
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"sample_id", "id", "bug_category", "name", "description", "example_code", "code"}
        }
        samples.append(StudyISample(
            sample_id=sample_id,
            name=name,
            description=description,
            code=code,
            label=label,
            metadata=metadata,
        ))
    return samples


def to_training_arrays(samples: list[StudyISample]) -> tuple[np.ndarray, np.ndarray]:
    texts = np.array([sample.to_text() for sample in samples], dtype=object)
    labels = np.array([LABEL_TO_ID[sample.label] for sample in samples], dtype=int)
    return texts, labels


def dataset_summary(samples: list[StudyISample]) -> dict:
    texts = [sample.to_text() for sample in samples]
    n_classical = sum(1 for sample in samples if sample.label == "classical")
    n_quantum = sum(1 for sample in samples if sample.label == "quantum")
    avg_chars = float(np.mean([len(text) for text in texts])) if texts else 0.0
    return {
        "n_samples": len(samples),
        "class_distribution": {
            "classical": n_classical,
            "quantum": n_quantum,
        },
        "imbalance_ratio": round((n_classical / n_quantum), 4) if n_quantum else None,
        "avg_text_chars": round(avg_chars, 2),
    }
