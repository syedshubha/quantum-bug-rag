#!/usr/bin/env python3
"""Run Study I: CodeBERT binary classification of quantum vs classical bugs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.study_i.analysis import aggregate_results, write_epoch_logs_json, write_per_fold_csv, write_summary_json
from src.study_i.dataset import dataset_summary, load_labeled_bug_reports, to_training_arrays
from src.study_i.plotting import generate_all_figures
from src.study_i.training import CodeBERTConfig, CodeBERTStudyRunner


def _parse_cv_seeds(raw: str) -> list[int]:
    values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("cv seeds must not be empty")
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid cv seed list: {raw}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", required=True, type=Path, help="Path to the labeled Study I JSON dataset.")
    ap.add_argument("--results-dir", required=True, type=Path, help="Directory for Study I outputs.")
    ap.add_argument("--model-name", default="microsoft/codebert-base", help="Hugging Face model id.")
    ap.add_argument("--max-len", type=int, default=256, help="Tokenizer max sequence length.")
    ap.add_argument("--n-folds", type=int, default=5, help="Stratified folds per seed.")
    ap.add_argument("--cv-seeds", type=_parse_cv_seeds, default=[42, 7, 2024, 99, 123], help="Comma-separated CV seeds.")
    ap.add_argument("--epochs", type=int, default=12, help="Maximum epochs per fold.")
    ap.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    ap.add_argument("--batch-size", type=int, default=8, help="Per-device training batch size.")
    ap.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay.")
    ap.add_argument("--warmup-ratio", type=float, default=0.15, help="Warmup ratio.")
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    ap.add_argument("--val-split", type=float, default=0.10, help="Validation split inside each training fold.")
    ap.add_argument("--es-patience", type=int, default=4, help="Manual early-stopping patience on validation macro-F1.")
    ap.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for weighted cross-entropy.")
    ap.add_argument("--limit", type=int, default=None, help="Optional sample cap for smoke runs.")
    ap.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    args = ap.parse_args()

    samples = load_labeled_bug_reports(args.data_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("no labeled Study I samples found")

    summary = dataset_summary(samples)
    print("Study I dataset summary:")
    print(json.dumps(summary, indent=2))

    texts, labels = to_training_arrays(samples)
    config = CodeBERTConfig(
        model_name=args.model_name,
        max_len=args.max_len,
        n_folds=args.n_folds,
        cv_seeds=args.cv_seeds,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        val_split=args.val_split,
        es_patience=args.es_patience,
        label_smoothing=args.label_smoothing,
    )
    runner = CodeBERTStudyRunner(config)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results = runner.run_repeated_cv(texts, labels, args.results_dir / "_tmp")
    summary_payload = aggregate_results(samples, results, {
        "model_name": config.model_name,
        "max_len": config.max_len,
        "n_folds": config.n_folds,
        "cv_seeds": config.cv_seeds,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "dropout": config.dropout,
        "val_split": config.val_split,
        "es_patience": config.es_patience,
        "label_smoothing": config.label_smoothing,
    })

    write_summary_json(args.results_dir / "summary.json", summary_payload)
    write_per_fold_csv(args.results_dir / "per_fold.csv", results)
    write_epoch_logs_json(args.results_dir / "epoch_logs.json", results)

    generated = []
    if not args.skip_plots:
        generated = generate_all_figures(samples, results, args.results_dir)

    print("\nStudy I complete.")
    print(f"  mean accuracy : {summary_payload['mean_accuracy']:.3f} ± {summary_payload['std_accuracy']:.3f}")
    print(f"  mean macro-F1 : {summary_payload['mean_f1_macro']:.3f} ± {summary_payload['std_f1_macro']:.3f}")
    print(f"  mean ROC-AUC  : {summary_payload['mean_roc_auc']:.3f} ± {summary_payload['std_roc_auc']:.3f}")
    print(f"  results dir   : {args.results_dir}")
    if generated:
        print("  figures       :")
        for filename in generated:
            print(f"    - {filename}")


if __name__ == "__main__":
    main()
