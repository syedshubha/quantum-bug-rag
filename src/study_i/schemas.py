"""Dataclasses for Study I binary classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

LABELS: list[str] = ["classical", "quantum"]
LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class StudyISample:
    """One labeled bug-report example for Study I."""

    sample_id: str
    name: str
    description: str
    code: str
    label: str
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        return "\n".join(part for part in [
            self.name.strip(),
            self.description.strip(),
            self.code.strip(),
        ] if part)


@dataclass
class EpochMetric:
    """Validation metrics captured once per epoch for one fold."""

    epoch: float
    eval_loss: float
    eval_accuracy: float
    eval_f1_macro: float


@dataclass
class FoldRunResult:
    """Metrics and predictions for one fold-run."""

    repeat: int
    fold: int
    cv_seed: int
    train_seed: int
    n_train: int
    n_validation: int
    n_test: int
    accuracy: float
    f1_macro: float
    f1_weighted: float
    roc_auc: float
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    probs: list[list[float]] = field(default_factory=list)
    epoch_log: list[EpochMetric] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["epoch_log"] = [asdict(metric) for metric in self.epoch_log]
        return payload
