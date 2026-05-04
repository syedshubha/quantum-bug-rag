#!/usr/bin/env python3
"""Generate tight SVG figures for the project report / paper."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
NOTEBOOK = ROOT / "quantum-vs-classical-bug-prediction.ipynb"

TAXONOMY = [
    "incorrect_operator",
    "incorrect_qubit_mapping",
    "missing_barrier",
    "wrong_initial_state",
    "measurement_error",
]

SHORT_CLASSES = {
    "incorrect_operator": "operator",
    "incorrect_qubit_mapping": "mapping",
    "missing_barrier": "barrier",
    "wrong_initial_state": "initialize",
    "measurement_error": "measure",
}

MODEL_COLORS = {
    "gpt-4o": "#2F5597",
    "gpt-5.4": "#C55A11",
}

MODE_STYLES = {
    "prompt_only": {"label": "PO", "alpha": 0.55},
    "rag": {"label": "RAG", "alpha": 0.82},
    "hybrid": {"label": "Hybrid", "alpha": 1.00},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.6,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0,
    "svg.fonttype": "none",
})


def _save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _load_notebook() -> dict:
    with NOTEBOOK.open() as fh:
        return json.load(fh)


def _find_output_text(nb: dict, needle: str) -> str:
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            if needle in text:
                return text
    raise RuntimeError(f"Could not find notebook output containing: {needle!r}")


def load_study1_cv_df() -> pd.DataFrame:
    nb = _load_notebook()
    text = _find_output_text(nb, "Per-fold results:")
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
    )
    rows: list[dict[str, float | int]] = []
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        repeat, fold, acc, f1m, f1w, auc = m.groups()
        rows.append({
            "repeat": int(repeat),
            "fold": int(fold),
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "f1_weighted": float(f1w),
            "roc_auc": float(auc),
        })
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 Study I fold rows, found {len(rows)}")
    return pd.DataFrame(rows)


def load_study1_confusion_matrix() -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    nb = _load_notebook()
    text = _find_output_text(nb, "Classification report (pooled across all folds):")
    report: dict[str, dict[str, float]] = {}
    pattern = re.compile(
        r"^\s*(classical|quantum)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$"
    )
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        label, precision, recall, f1, support = m.groups()
        report[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
    if set(report) != {"classical", "quantum"}:
        raise RuntimeError("Could not parse pooled Study I classification report")

    classical_support = report["classical"]["support"]
    quantum_support = report["quantum"]["support"]
    tp_classical = int(round(report["classical"]["recall"] * classical_support))
    tp_quantum = int(round(report["quantum"]["recall"] * quantum_support))
    fn_classical = classical_support - tp_classical
    fn_quantum = quantum_support - tp_quantum
    cm = np.array([
        [tp_classical, fn_classical],
        [fn_quantum, tp_quantum],
    ])
    return cm, report


def _load_summary(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_study1_metric_distribution(cv_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.6), constrained_layout=True)
    metrics = [
        ("accuracy", "Accuracy", "#D55E00"),
        ("f1_macro", "Macro-F1", "#0072B2"),
        ("f1_weighted", "Weighted-F1", "#009E73"),
        ("roc_auc", "ROC-AUC", "#6A3D9A"),
    ]
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(metrics) + 1)
    all_values = []
    all_colors = []
    for col, _, color in metrics:
        values = cv_df[col].to_numpy()
        all_values.append(values)
        all_colors.append(color)
    bp = ax.boxplot(
        all_values,
        vert=True,
        widths=0.42,
        patch_artist=True,
        positions=positions,
        medianprops={"color": "#333333", "linewidth": 1.8},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )
    for patch, color in zip(bp["boxes"], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for pos, (col, label, color) in zip(positions, metrics):
        values = cv_df[col].to_numpy()
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=26,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.85,
            zorder=3,
        )
        mean = values.mean()
        std = values.std(ddof=1)
        ax.plot([pos - 0.26, pos + 0.26], [mean, mean], color="#333333", linestyle="--", linewidth=1.1)
        ax.text(
            pos,
            values.min() - 0.028,
            label,
            va="top",
            ha="center",
            fontsize=10,
            color="#333333",
        )
        ax.text(
            pos,
            mean + 0.02,
            f"{mean:.3f} ± {std:.3f}",
            va="center",
            ha="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_xlim(0.5, len(metrics) + 0.5)
    ax.set_ylim(0.58, 0.96)
    ax.set_xticks([])
    ax.set_ylabel("")
    fig.suptitle("Metric distribution across 25 fold-runs", y=1.02)
    _save_svg(fig, FIG_DIR / "study1_metric_distribution.svg")


def plot_study1_confusion_matrix(cm: np.ndarray, report: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    labels = ["classical", "quantum"]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "#1a1a1a"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({row_pct[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        "Study I: pooled confusion matrix\n"
        f"classical F1={report['classical']['f1']:.4f}  quantum F1={report['quantum']['f1']:.4f}"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.02)
    cbar.set_label("Count")
    _save_svg(fig, FIG_DIR / "study1_confusion_matrix.svg")


def _study2_metric_frame(metric: str) -> pd.DataFrame:
    model_paths = {
        "gpt-4o": ROOT / "outputs" / "results_4o" / "summary.json",
        "gpt-5.4": ROOT / "outputs" / "results_54" / "summary.json",
    }
    rows = []
    for model, path in model_paths.items():
        summary = _load_summary(path)
        for dataset in ("bugs4q", "bugsqcp"):
            for mode in ("prompt_only", "rag", "hybrid"):
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "mode": mode,
                    "value": summary["datasets"][dataset]["test"][mode][metric],
                })
    return pd.DataFrame(rows)


def plot_study2_grouped_metric(metric: str, title: str, out_name: str, ylabel: str) -> None:
    df = _study2_metric_frame(metric)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2), sharey=True, constrained_layout=True)
    x = np.arange(6)
    global_max = df["value"].max()
    order = [
        ("gpt-4o", "prompt_only"),
        ("gpt-4o", "rag"),
        ("gpt-4o", "hybrid"),
        ("gpt-5.4", "prompt_only"),
        ("gpt-5.4", "rag"),
        ("gpt-5.4", "hybrid"),
    ]
    xticklabels = [
        "4o PO",
        "4o RAG",
        "4o Hyb",
        "5.4 PO",
        "5.4 RAG",
        "5.4 Hyb",
    ]
    for ax, dataset, panel in zip(axes, ("bugs4q", "bugsqcp"), ("Panel A: Bugs4Q", "Panel B: Bugs-QCP")):
        vals = []
        colors = []
        for model, mode in order:
            value = df.loc[
                (df["dataset"] == dataset)
                & (df["model"] == model)
                & (df["mode"] == mode),
                "value",
            ].iloc[0]
            vals.append(value)
            colors.append(MODEL_COLORS[model])
        bars = ax.bar(
            x,
            vals,
            color=colors,
            edgecolor="#444444",
            linewidth=0.6,
        )
        for bar, (_, mode) in zip(bars, order):
            bar.set_alpha(MODE_STYLES[mode]["alpha"])
        ax.set_xticks(x, xticklabels)
        ax.tick_params(axis="x", labelsize=9, pad=5)
        ax.set_ylim(0, global_max * 1.18 + 0.04)
        ax.set_title(panel)
        ax.set_ylabel(ylabel)
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.axvspan(-0.45, 2.45, color=MODEL_COLORS["gpt-4o"], alpha=0.04, zorder=0)
        ax.axvspan(2.55, 5.45, color=MODEL_COLORS["gpt-5.4"], alpha=0.04, zorder=0)
    _save_svg(fig, FIG_DIR / out_name)


def _load_true_dev_frequencies(dataset: str) -> dict[str, float]:
    diag_path = ROOT / "outputs" / "results_4o_priorcorr_eps005" / f"diagnostics_{dataset}_dev_prompt_only.jsonl"
    rows = _load_jsonl(diag_path)
    counts = Counter(row["ground_truth"] for row in rows if row.get("ground_truth"))
    total = sum(counts.values())
    return {cls: counts.get(cls, 0) / total for cls in TAXONOMY}


def plot_study2_class_prior_bias() -> None:
    summary = _load_summary(ROOT / "outputs" / "results_4o_priorcorr_eps005" / "summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8), sharey=True, constrained_layout=True)
    x = np.arange(len(TAXONOMY))
    width = 0.24
    for ax, dataset, panel in zip(axes, ("bugs4q", "bugsqcp"), ("Bugs4Q Dev", "Bugs-QCP Dev")):
        true_freq = _load_true_dev_frequencies(dataset)
        dev_prior = summary["datasets"][dataset]["tuning"]["dev_prior"]
        series = {
            "True label frequency": [true_freq[c] for c in TAXONOMY],
            "GPT-4o PO mean score": [dev_prior["prompt_only"][c] for c in TAXONOMY],
            "GPT-4o RAG mean score": [dev_prior["rag"][c] for c in TAXONOMY],
        }
        colors = ["#595959", "#4F81BD", "#C0504D"]
        for idx, (label, values) in enumerate(series.items()):
            bars = ax.bar(
                x + (idx - 1) * width,
                values,
                width=width,
                label=label,
                color=colors[idx],
                edgecolor="#444444",
                linewidth=0.5,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x, [SHORT_CLASSES[c] for c in TAXONOMY], rotation=0, ha="center")
        ax.set_title(panel)
        ax.set_ylim(0, 0.84)
        ax.set_ylabel("")
    axes[1].legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.52),
        fontsize=9,
        frameon=False,
        labelspacing=0.9,
        handletextpad=0.8,
        borderaxespad=0.2,
    )
    _save_svg(fig, FIG_DIR / "study2_class_prior_bias.svg")


def plot_study2_retrieval_score_vs_correctness() -> None:
    summary = _load_summary(ROOT / "outputs" / "results_4o" / "summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, dataset, title in zip(axes, ("bugs4q", "bugsqcp"), ("Bugs4Q", "Bugs-QCP")):
        rows = _load_jsonl(ROOT / "outputs" / "results_4o" / f"diagnostics_{dataset}_test_rag.jsonl")
        xs = np.array([row["top1_bm25_score"] for row in rows], dtype=float)
        ys = np.array([1.0 if row["correct"] else 0.0 for row in rows], dtype=float)
        jitter = rng.uniform(-0.06, 0.06, len(rows))
        colors = np.where(ys > 0.5, "#1B9E77", "#D95F02")
        ax.scatter(xs, ys + jitter, c=colors, s=48, alpha=0.88, edgecolors="white", linewidths=0.6)
        tau = summary["datasets"][dataset]["tuning"]["tau"]
        if isinstance(tau, (int, float)) and math.isfinite(tau):
            ax.axvline(tau, color="#444444", linestyle="--", linewidth=1.3)
            ax.text(tau, 1.12, f"tau={tau:.2f}", rotation=90, va="top", ha="right", fontsize=8)
        else:
            ax.text(0.98, 0.98, "tau=-inf", transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title(f"{title} — gpt-4o RAG Test")
        ax.set_xlabel("Top-1 boosted BM25 score")
        ax.set_yticks([0, 1], ["incorrect", "correct"])
        ax.set_ylim(-0.18, 1.18)
    _save_svg(fig, FIG_DIR / "study2_retrieval_score_vs_correctness.svg")


def plot_kb_source_distribution() -> None:
    summary = _load_summary(ROOT / "outputs" / "results_4o" / "summary.json")
    sources = summary["kb_sources"]
    order = [
        "qiskit_releasenotes",
        "pennylane_changelog",
        "qiskit_aer_releasenotes",
        "ibm_runtime_changelog",
        "qiskit_ignis_releasenotes",
        "lintq_rules",
    ]
    labels = [s.replace("_", "\n") for s in order]
    values = [sources[s] for s in order]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    x = np.arange(len(order))
    bars = ax.bar(x, values, color="#4C78A8", alpha=0.86, edgecolor="#444444", linewidth=0.5)
    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=8, pad=6)
    ax.set_ylabel("")
    ax.set_title("Knowledge base source distribution")
    for bar, value in zip(bars, values):
        y_pos = max(value - 42, value * 0.80)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            str(value),
            va="top",
            ha="center",
            fontsize=9,
            color="white" if value > 60 else "#1a1a1a",
            fontweight="bold",
        )
    _save_svg(fig, FIG_DIR / "kb_source_distribution.svg")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cv_df = load_study1_cv_df()
    cm, report = load_study1_confusion_matrix()
    plot_study1_metric_distribution(cv_df)
    plot_study1_confusion_matrix(cm, report)
    plot_study2_grouped_metric(
        metric="accuracy",
        title="",
        out_name="study2_accuracy_grouped.svg",
        ylabel="Accuracy",
    )
    plot_study2_grouped_metric(
        metric="macro_f1",
        title="",
        out_name="study2_macrof1_grouped.svg",
        ylabel="Macro-F1",
    )
    plot_study2_class_prior_bias()
    plot_study2_retrieval_score_vs_correctness()
    plot_kb_source_distribution()
    print(f"Wrote SVG figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
