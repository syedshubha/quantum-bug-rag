"""Plotting helpers for Study I."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve

from .schemas import LABELS, FoldRunResult, StudyISample

PALETTE = {
    "classical": "#e74c3c",
    "quantum": "#2980b9",
    "mean": "#2c3e50",
    "auc": "#27ae60",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(results: list[FoldRunResult], output_path: Path) -> None:
    y_true = np.concatenate([np.asarray(result.y_true, dtype=int) for result in results])
    y_pred = np.concatenate([np.asarray(result.y_pred, dtype=int) for result in results])
    probs = np.concatenate([np.asarray(result.probs, dtype=float) for result in results], axis=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = float(np.nan)
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, probs[:, 1]))
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Prediction count")
    ax.set_xticks(range(2), LABELS)
    ax.set_yticks(range(2), LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        "Confusion matrix — pooled across CV folds\n"
        f"Accuracy = {acc:.1%}   Macro-F1 = {f1:.1%}   AUC = {auc:.3f}"
    )
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i - 0.08, f"{cm[i, j]}", ha="center", va="center", fontsize=16, fontweight="bold", color=color)
            ax.text(j, i + 0.14, f"({cm_pct[i, j]:.1f}%)", ha="center", va="center", fontsize=10, color=color)
    _save(fig, output_path)


def save_fold_distribution(results: list[FoldRunResult], output_path: Path) -> None:
    fold_accs = np.array([result.accuracy for result in results], dtype=float)
    fold_f1s = np.array([result.f1_macro for result in results], dtype=float)
    fold_aucs = np.array([result.roc_auc for result in results], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    rng = np.random.default_rng(42)
    metrics = [
        (axes[0], fold_accs, "Accuracy", PALETTE["classical"]),
        (axes[1], fold_f1s, "Macro-F1", PALETTE["quantum"]),
        (axes[2], fold_aucs, "ROC-AUC", PALETTE["auc"]),
    ]
    for ax, values, label, color in metrics:
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        jitter = rng.uniform(-0.15, 0.15, len(values))
        ax.axhspan(mean - std, mean + std, alpha=0.12, color=color, label=f"±1 std ({std:.3f})")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Random (50%)")
        ax.axhline(mean, color=PALETTE["mean"], linestyle="--", linewidth=2, label=f"Mean = {mean:.3f}")
        ax.scatter(jitter, values, s=60, color=color, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.2, 1.02)
        ax.set_xticks([])
        ax.set_ylabel(label)
        ax.set_title(f"{label} across {len(values)} folds", fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "Per-fold performance — repeated stratified CV\n"
        "CodeBERT binary classifier: classical vs. quantum bug detection",
        fontsize=13,
        y=1.02,
    )
    _save(fig, output_path)


def save_roc_curve(results: list[FoldRunResult], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    base_fpr = np.linspace(0.0, 1.0, 200)
    tprs: list[np.ndarray] = []
    aucs: list[float] = []
    for result in results:
        y_true = np.asarray(result.y_true, dtype=int)
        probs = np.asarray(result.probs, dtype=float)
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        tprs.append(np.interp(base_fpr, fpr, tpr))
        try:
            from sklearn.metrics import roc_auc_score

            aucs.append(float(roc_auc_score(y_true, probs[:, 1])))
        except Exception:
            pass
        ax.plot(base_fpr, tprs[-1], color=PALETTE["quantum"], alpha=0.10, linewidth=0.8)
    arr = np.array(tprs)
    mean_tpr = arr.mean(axis=0)
    std_tpr = arr.std(axis=0, ddof=1)
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    ax.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=PALETTE["quantum"], alpha=0.20, label="±1 std band")
    ax.plot(base_fpr, mean_tpr, color=PALETTE["quantum"], linewidth=2.5, label=f"Mean ROC (AUC = {mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — classical vs. quantum bug classification")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="lower right")
    _save(fig, output_path)


def save_learning_curves(results: list[FoldRunResult], output_path: Path) -> None:
    valid_logs = [result.epoch_log for result in results if result.epoch_log]
    min_epochs = min((len(log) for log in valid_logs), default=0)
    if min_epochs <= 0:
        return
    trimmed = [log[:min_epochs] for log in valid_logs]
    epoch_x = np.array([entry.epoch for entry in trimmed[0]], dtype=float)
    val_losses = np.array([[entry.eval_loss for entry in log] for log in trimmed], dtype=float)
    val_accs = np.array([[entry.eval_accuracy for entry in log] for log in trimmed], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, values, ylabel, color in [
        (axes[0], val_losses, "Validation loss", PALETTE["classical"]),
        (axes[1], val_accs, "Validation accuracy", PALETTE["quantum"]),
    ]:
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        ax.fill_between(epoch_x, mean - std, mean + std, alpha=0.15, color=color)
        ax.plot(epoch_x, mean, color=color, linewidth=2.5, marker="o", markersize=5, label="Mean ± 1 std")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per epoch", fontweight="bold")
        ax.legend(fontsize=10)
    fig.suptitle("Learning curves — CodeBERT binary classifier", fontsize=13, y=1.02)
    _save(fig, output_path)


def save_summary_panel(samples: list[StudyISample], results: list[FoldRunResult], output_path: Path) -> None:
    fold_accs = np.array([result.accuracy for result in results], dtype=float)
    fold_f1s = np.array([result.f1_macro for result in results], dtype=float)
    fold_aucs = np.array([result.roc_auc for result in results], dtype=float)
    y_true = np.concatenate([np.asarray(result.y_true, dtype=int) for result in results])
    y_pred = np.concatenate([np.asarray(result.y_pred, dtype=int) for result in results])
    class_counts = [
        sum(1 for sample in samples if sample.label == "classical"),
        sum(1 for sample in samples if sample.label == "quantum"),
    ]
    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)

    fig = plt.figure(figsize=(14, 5))
    grid = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    metrics_summary = {
        "Accuracy": (float(fold_accs.mean()), float(fold_accs.std(ddof=1))),
        "Macro-F1": (float(fold_f1s.mean()), float(fold_f1s.std(ddof=1))),
        "ROC-AUC": (float(fold_aucs.mean()), float(fold_aucs.std(ddof=1))),
    }

    ax0 = fig.add_subplot(grid[0])
    names = list(metrics_summary.keys())
    means = [value[0] for value in metrics_summary.values()]
    stds = [value[1] for value in metrics_summary.values()]
    bars = ax0.bar(names, means, yerr=stds, capsize=6, color=[PALETTE["classical"], PALETTE["quantum"], PALETTE["auc"]], alpha=0.85, edgecolor="black", linewidth=0.8)
    ax0.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Random (50%)")
    ax0.set_ylim(0.0, 1.05)
    ax0.set_ylabel("Score")
    ax0.set_title("Mean performance\n(25 CV folds)", fontweight="bold")
    ax0.legend(fontsize=9)
    for bar, mean, std in zip(bars, means, stds):
        ax0.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, f"{mean:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax1 = fig.add_subplot(grid[1])
    wedges, _texts, autotexts = ax1.pie(
        class_counts,
        labels=LABELS,
        colors=[PALETTE["classical"], PALETTE["quantum"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 12},
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
    ax1.set_title(f"Dataset distribution\n(n={len(samples)} samples)", fontweight="bold")

    ax2 = fig.add_subplot(grid[2])
    per_class_f1 = [report["classical"]["f1-score"], report["quantum"]["f1-score"]]
    ax2.bar(LABELS, per_class_f1, color=[PALETTE["classical"], PALETTE["quantum"]], alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("F1-score")
    ax2.set_title("Per-class F1\n(pooled predictions)", fontweight="bold")
    for idx, value in enumerate(per_class_f1):
        ax2.text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=12, fontweight="bold")

    fig.suptitle(
        "CodeBERT Binary Classifier — Classical vs. Quantum Bug Detection\n"
        f"Dataset: {len(samples)} samples   Model: microsoft/codebert-base   CV: 5-fold × 5 seeds",
        fontsize=12,
        y=1.04,
        fontweight="bold",
    )
    _save(fig, output_path)


def generate_all_figures(samples: list[StudyISample], results: list[FoldRunResult], output_dir: Path) -> list[str]:
    output_dir = Path(output_dir)
    paths = [
        ("fig1_confusion_matrix.png", save_confusion_matrix),
        ("fig2_fold_distribution.png", save_fold_distribution),
        ("fig3_roc_curve.png", save_roc_curve),
        ("fig4_learning_curves.png", save_learning_curves),
        ("fig5_summary_panel.png", lambda rs, out: save_summary_panel(samples, rs, out)),
    ]
    generated: list[str] = []
    for filename, fn in paths:
        path = output_dir / filename
        fn(results, path)
        if path.exists():
            generated.append(filename)
    return generated
