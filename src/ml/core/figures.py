"""Every figure the assignment grades: ROC, confusion matrix, performance curve.

Forces the Agg backend so figures render identically in tests, in notebooks and on
a machine with no display.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _save(figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curves(curves: dict, path: Path, *, title: str = "ROC curve") -> None:
    """curves: {model_name: (fpr, tpr, auc)}"""
    figure, axes = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in sorted(curves.items()):
        axes.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={auc:.3f})")
    axes.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(title)
    axes.legend(fontsize="small", loc="lower right")
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_confusion_matrix(
    matrix, path: Path, *, title: str = "Confusion matrix", labels: tuple[str, str] = ("negative", "positive")
) -> None:
    figure, axes = plt.subplots(figsize=(5, 4.5))
    image = axes.imshow(matrix, cmap="Blues")
    largest = max(max(row) for row in matrix)
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            axes.text(
                j, i, f"{value:,}", ha="center", va="center",
                color="white" if value > largest / 2 else "black",
            )
    axes.set_xticks([0, 1], [f"predicted {labels[0]}", f"predicted {labels[1]}"])
    axes.set_yticks([0, 1], [f"actual {labels[0]}", f"actual {labels[1]}"])
    axes.set_title(title)
    figure.colorbar(image, ax=axes)
    _save(figure, path)


def plot_performance_curve(history: dict, path: Path, *, title: str = "Training loss") -> None:
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(history["iteration"], history["loss"], linewidth=1.5, marker="o" if len(history["iteration"]) < 20 else None)
    axes.set_xlabel("iteration")
    axes.set_ylabel("loss")
    axes.set_title(title)
    axes.grid(alpha=0.3)
    _save(figure, path)


def write_leaderboard(rows: list[dict], path: Path, *, sort_by: str, descending: bool = False) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty leaderboard")
    ordered = sorted(
        rows, key=lambda row: (row.get(sort_by) is None, row.get(sort_by)), reverse=descending
    )
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
