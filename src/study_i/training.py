"""Training loop for Study I CodeBERT experiments."""

from __future__ import annotations

import gc
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .schemas import EpochMetric, FoldRunResult, ID_TO_LABEL, LABELS, LABEL_TO_ID


@dataclass
class CodeBERTConfig:
    """Hyperparameters for Study I."""

    model_name: str = "microsoft/codebert-base"
    max_len: int = 256
    n_folds: int = 5
    cv_seeds: list[int] = field(default_factory=lambda: [42, 7, 2024, 99, 123])
    num_epochs: int = 12
    learning_rate: float = 2e-5
    batch_size: int = 8
    weight_decay: float = 0.05
    warmup_ratio: float = 0.15
    dropout: float = 0.2
    val_split: float = 0.10
    es_patience: int = 4
    label_smoothing: float = 0.05


def oversample_to_balance(
    texts: list[str],
    labels: list[int],
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    """Duplicate minority-class samples until all classes have equal size."""
    grouped: dict[int, list[str]] = defaultdict(list)
    for text, label in zip(texts, labels):
        grouped[label].append(text)
    max_n = max(len(bucket) for bucket in grouped.values())
    out_texts: list[str] = []
    out_labels: list[int] = []
    for label, bucket in grouped.items():
        reps = max_n // len(bucket)
        rem = max_n - reps * len(bucket)
        expanded = bucket * reps + (rng.sample(bucket, rem) if rem else [])
        out_texts.extend(expanded)
        out_labels.extend([label] * len(expanded))
    order = list(range(len(out_texts)))
    rng.shuffle(order)
    return [out_texts[i] for i in order], [out_labels[i] for i in order]


def _compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


class CodeBERTStudyRunner:
    """Repeated stratified cross-validation runner for Study I."""

    def __init__(self, config: CodeBERTConfig) -> None:
        self.config = config
        self._torch = None
        self._datasets = None
        self._transformers = None
        self._tokenizer = None
        self._data_collator = None
        self.device = "cpu"

    def _ensure_stack(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch
            from datasets import Dataset
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
                DataCollatorWithPadding,
                Trainer,
                TrainerCallback,
                TrainingArguments,
                set_seed,
            )
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "Study I requires torch, datasets, transformers, and accelerate. "
                "Install them from requirements.txt before running the CodeBERT CLI."
            ) from exc

        self._torch = torch
        self._datasets = {"Dataset": Dataset}
        self._transformers = {
            "AutoConfig": AutoConfig,
            "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
            "AutoTokenizer": AutoTokenizer,
            "DataCollatorWithPadding": DataCollatorWithPadding,
            "Trainer": Trainer,
            "TrainerCallback": TrainerCallback,
            "TrainingArguments": TrainingArguments,
            "set_seed": set_seed,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)

    def _tokenize_batch(self, batch: dict[str, list[str]]) -> dict[str, Any]:
        assert self._tokenizer is not None
        return self._tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.config.max_len,
        )

    def _make_trainer_types(self, epoch_log: list[EpochMetric]) -> tuple[type, list[Any]]:
        assert self._transformers is not None
        Trainer = self._transformers["Trainer"]
        TrainerCallback = self._transformers["TrainerCallback"]
        label_smoothing = self.config.label_smoothing
        torch = self._torch
        assert torch is not None

        class ManualEarlyStoppingCallback(TrainerCallback):
            def __init__(self, patience: int) -> None:
                self.patience = patience
                self.best_f1 = -1.0
                self.wait = 0

            def on_train_begin(self, args, state, control, **kwargs):
                self.best_f1 = -1.0
                self.wait = 0

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                current_f1 = metrics.get("eval_f1_macro", 0.0) if metrics else 0.0
                if current_f1 > self.best_f1 + 1e-4:
                    self.best_f1 = current_f1
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        control.should_training_stop = True

        class EpochLogCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics:
                    epoch_log.append(EpochMetric(
                        epoch=float(state.epoch or 0.0),
                        eval_loss=float(metrics.get("eval_loss", float("nan"))),
                        eval_accuracy=float(metrics.get("eval_accuracy", float("nan"))),
                        eval_f1_macro=float(metrics.get("eval_f1_macro", float("nan"))),
                    ))

        class WeightedTrainer(Trainer):
            def set_class_weights(self, weights):
                self._class_weights = weights.to(self.args.device)

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss = torch.nn.CrossEntropyLoss(
                    weight=getattr(self, "_class_weights", None),
                    label_smoothing=label_smoothing,
                )(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        callbacks = [
            ManualEarlyStoppingCallback(patience=self.config.es_patience),
            EpochLogCallback(),
        ]
        return WeightedTrainer, callbacks

    def run_fold(
        self,
        train_texts: list[str],
        train_labels: list[int],
        test_texts: list[str],
        test_labels: list[int],
        *,
        repeat_idx: int,
        fold_idx: int,
        cv_seed: int,
        train_seed: int,
        output_dir: Path,
    ) -> FoldRunResult:
        self._ensure_stack()
        assert self._datasets is not None
        assert self._transformers is not None
        assert self._tokenizer is not None
        assert self._data_collator is not None
        assert self._torch is not None

        torch = self._torch
        Dataset = self._datasets["Dataset"]
        AutoConfig = self._transformers["AutoConfig"]
        AutoModelForSequenceClassification = self._transformers["AutoModelForSequenceClassification"]
        TrainingArguments = self._transformers["TrainingArguments"]
        set_seed = self._transformers["set_seed"]

        set_seed(train_seed)
        rng = random.Random(train_seed)
        train_texts_bal, train_labels_bal = oversample_to_balance(train_texts, train_labels, rng)
        tr_t, val_t, tr_y, val_y = train_test_split(
            train_texts_bal,
            train_labels_bal,
            test_size=self.config.val_split,
            random_state=train_seed,
            stratify=train_labels_bal if len(set(train_labels_bal)) > 1 else None,
        )

        train_ds = Dataset.from_dict({"text": tr_t, "labels": tr_y}).map(self._tokenize_batch, batched=True)
        val_ds = Dataset.from_dict({"text": val_t, "labels": val_y}).map(self._tokenize_batch, batched=True)
        test_ds = Dataset.from_dict({"text": list(test_texts), "labels": list(test_labels)}).map(
            self._tokenize_batch,
            batched=True,
        )

        config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            hidden_dropout_prob=self.config.dropout,
            attention_probs_dropout_prob=self.config.dropout,
            ignore_mismatched_sizes=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        counts = Counter(tr_y)
        freq = np.array([counts.get(idx, 1) for idx in range(len(LABELS))], dtype=np.float32)
        class_weights = torch.tensor(freq.sum() / (len(LABELS) * freq), dtype=torch.float)

        epoch_log: list[EpochMetric] = []
        WeightedTrainer, callbacks = self._make_trainer_types(epoch_log)
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=64,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_steps=500,
            report_to="none",
            seed=train_seed,
            fp16=(self.device == "cuda"),
            disable_tqdm=True,
        )

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=lambda pred: _compute_metrics(pred.predictions, pred.label_ids),
            callbacks=callbacks,
        )
        trainer.set_class_weights(class_weights)
        trainer.train()

        prediction = trainer.predict(test_ds)
        logits = prediction.predictions
        y_true = prediction.label_ids
        y_pred = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs[:, 1])),
        }

        del trainer, model
        gc.collect()
        if self.device == "cuda":  # pragma: no cover - depends on local env
            torch.cuda.empty_cache()
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

        return FoldRunResult(
            repeat=repeat_idx + 1,
            fold=fold_idx + 1,
            cv_seed=cv_seed,
            train_seed=train_seed,
            n_train=len(tr_t),
            n_validation=len(val_t),
            n_test=len(test_texts),
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            f1_weighted=metrics["f1_weighted"],
            roc_auc=metrics["roc_auc"],
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            probs=probs.tolist(),
            epoch_log=epoch_log,
        )

    def run_repeated_cv(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        work_dir: Path,
    ) -> list[FoldRunResult]:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        results: list[FoldRunResult] = []
        for repeat_idx, cv_seed in enumerate(self.config.cv_seeds):
            skf = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=cv_seed,
            )
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
                result = self.run_fold(
                    texts[train_idx].tolist(),
                    labels[train_idx].tolist(),
                    texts[test_idx].tolist(),
                    labels[test_idx].tolist(),
                    repeat_idx=repeat_idx,
                    fold_idx=fold_idx,
                    cv_seed=cv_seed,
                    train_seed=cv_seed * 100 + fold_idx,
                    output_dir=work_dir / f"tmp_repeat_{repeat_idx}_fold_{fold_idx}",
                )
                results.append(result)
        return results
