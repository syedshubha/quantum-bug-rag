"""Study I: binary quantum-vs-classical bug prediction with CodeBERT.

This package refactors the ``quantum-vs-classical-bug-prediction.ipynb``
notebook into reusable modules and a scriptable CLI.
"""

from .dataset import dataset_summary, load_labeled_bug_reports, to_training_arrays
from .schemas import ID_TO_LABEL, LABELS, LABEL_TO_ID, EpochMetric, FoldRunResult, StudyISample
from .training import CodeBERTConfig, CodeBERTStudyRunner, oversample_to_balance

__all__ = [
    "CodeBERTConfig",
    "CodeBERTStudyRunner",
    "EpochMetric",
    "FoldRunResult",
    "ID_TO_LABEL",
    "LABELS",
    "LABEL_TO_ID",
    "StudyISample",
    "dataset_summary",
    "load_labeled_bug_reports",
    "oversample_to_balance",
    "to_training_arrays",
]
