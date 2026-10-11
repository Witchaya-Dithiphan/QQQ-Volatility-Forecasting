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


# Eight validated categorical hues; a second line style doubles them, because the ROC
# and PR overlays carry 13 models and a repeated colour made two of them indistinguishable.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES_LINESTYLES = ("-", "--")


def series_styles(names, *, universe=None) -> dict:
    """{name: {"color", "linestyle"}}, assigned by position in the sorted universe of names."""
    order = sorted(universe if universe is not None else names)
    capacity = len(SERIES_COLORS) * len(SERIES_LINESTYLES)
    if len(order) > capacity:
        raise ValueError(f"Only {capacity} distinct line styles exist, got {len(order)} series")
    table = {
        name: {"color": SERIES_COLORS[i % len(SERIES_COLORS)],
               "linestyle": SERIES_LINESTYLES[i // len(SERIES_COLORS)]}
        for i, name in enumerate(order)
    }
    return {name: table[name] for name in names}


def plot_roc_curves(curves: dict, path: Path, *, title: str = "ROC curve") -> None:
    """curves: {model_name: (fpr, tpr, auc)}"""
    figure, axes = plt.subplots(figsize=(7, 6))
    styles = series_styles(list(curves))
    for name, (fpr, tpr, auc) in sorted(curves.items()):
        axes.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={auc:.3f})", **styles[name])
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
    # Undefined metrics go last in both directions; reversing a (is None, value) key
    # would have put them first and compared None with None.
    defined = [row for row in rows if row.get(sort_by) is not None]
    undefined = [row for row in rows if row.get(sort_by) is None]
    ordered = sorted(defined, key=lambda row: row[sort_by], reverse=descending) + undefined
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


def plot_pr_curves(curves: dict, path: Path, *, prevalence: float, title: str = "Precision-recall curve") -> None:
    """curves: {model_name: (recall, precision, average_precision)}"""
    figure, axes = plt.subplots(figsize=(7, 6))
    styles = series_styles(list(curves))
    for name, (recall, precision, average_precision) in sorted(curves.items()):
        axes.step(recall, precision, where="post", linewidth=1.5,
                  label=f"{name} (AP={average_precision:.3f})", **styles[name])
    axes.axhline(prevalence, color="k", linestyle="--", linewidth=0.8,
                 label=f"positive rate ({prevalence:.3f})")
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_ylim(0, 1.02)
    axes.set_title(title)
    axes.legend(fontsize="small", loc="upper right")
    axes.grid(alpha=0.3)
    _save(figure, path)


def _grid(count: int, columns: int, width: float, height: float):
    rows = -(-count // columns)
    figure, grid = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False,
                                # Panel titles otherwise sit on the tick labels of the row above.
                                layout="constrained")
    for axes in grid.ravel()[count:]:
        axes.axis("off")
    return figure, grid.ravel()[:count]


def plot_confusion_grid(matrices: dict, path: Path, *, title: str = "Confusion matrices", columns: int = 4) -> None:
    """matrices: {model_name: [[tn, fp], [fn, tp]]}"""
    names = sorted(matrices)
    figure, panels = _grid(len(names), columns, 3.2, 3.0)
    for axes, name in zip(panels, names):
        matrix = matrices[name]
        axes.imshow(matrix, cmap="Blues")
        largest = max(max(row) for row in matrix)
        for i in range(2):
            for j in range(2):
                axes.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center",
                          color="white" if matrix[i][j] > largest / 2 else "black")
        axes.set_xticks([0, 1], ["pred 0", "pred 1"])
        axes.set_yticks([0, 1], ["actual 0", "actual 1"])
        axes.set_title(name, fontsize="medium")
    figure.suptitle(title)
    _save(figure, path)


def plot_predicted_vs_actual(series: dict, path: Path, *, title: str = "Predicted vs actual") -> None:
    """series: {model_name: (actual, predicted)}"""
    names = sorted(series)
    figure, panels = _grid(len(names), len(names), 4.0, 4.0)
    for axes, name in zip(panels, names):
        actual, predicted = series[name]
        limit = float(max(max(actual), max(predicted)))
        axes.scatter(actual, predicted, s=8, alpha=0.5)
        axes.plot([0, limit], [0, limit], "k--", linewidth=0.8)
        axes.set_xlabel("actual")
        axes.set_ylabel("predicted")
        axes.set_title(name, fontsize="small")
        axes.grid(alpha=0.3)
    figure.suptitle(title)
    _save(figure, path)


def plot_residuals(series: dict, path: Path, *, title: str = "Residuals over time") -> None:
    """series: {model_name: (dates, actual - predicted)}"""
    figure, axes = plt.subplots(figsize=(10, 4.5))
    for name, (dates, residual) in sorted(series.items()):
        axes.plot(dates, residual, linewidth=1.0, label=name)
    axes.axhline(0, color="k", linewidth=0.8)
    axes.set_xlabel("date")
    axes.set_ylabel("actual - predicted")
    axes.set_title(title)
    axes.legend(fontsize="small")
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_performance_grid(histories: dict, path: Path, *, title: str = "Training loss", columns: int = 4) -> None:
    """histories: {model_name: {"iteration": [...], "loss": [...]}}"""
    names = sorted(histories)
    figure, panels = _grid(len(names), min(columns, len(names)), 3.6, 2.8)
    for axes, name in zip(panels, names):
        history = histories[name]
        axes.plot(history["iteration"], history["loss"], linewidth=1.2,
                  marker="o" if len(history["iteration"]) < 20 else None)
        axes.set_title(name, fontsize="small")
        axes.set_xlabel("iteration")
        axes.set_ylabel("loss")
        axes.grid(alpha=0.3)
    figure.suptitle(title)
    _save(figure, path)


def plot_interval_bars(rows: list[tuple], path: Path, *, title: str, xlabel: str) -> None:
    """rows: [(label, value, low, high)]; low/high are None when no interval could be computed."""
    figure, axes = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.5))
    for position, (label, value, low, high) in enumerate(rows):
        axes.barh(position, value, color="tab:blue" if low is not None else "lightgray")
        if low is not None:
            axes.plot([low, high], [position, position], color="k", linewidth=1.2)
    axes.axvline(0, color="k", linewidth=0.8)
    axes.set_yticks(range(len(rows)), [row[0] for row in rows])
    axes.invert_yaxis()
    axes.set_xlabel(xlabel)
    axes.set_title(title)
    axes.grid(alpha=0.3, axis="x")
    _save(figure, path)


def plot_heatmap(matrix, row_labels, column_labels, path: Path, *, title: str, colorbar_label: str) -> None:
    figure, axes = plt.subplots(figsize=(1.1 * len(column_labels) + 2.5, 0.4 * len(row_labels) + 2))
    image = axes.imshow(matrix, cmap="Blues", aspect="auto")
    axes.set_xticks(range(len(column_labels)), column_labels, rotation=45, ha="right")
    axes.set_yticks(range(len(row_labels)), row_labels)
    axes.set_title(title)
    figure.colorbar(image, ax=axes, label=colorbar_label)
    _save(figure, path)


def _cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(
    rows: list[dict], path: Path, *, columns: list[str], title: str, notes: tuple[str, ...] = ()
) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty table")
    lines = [f"### {title}", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(_cell(row.get(column)) for column in columns) + " |" for row in rows]
    lines += ["", *(f"> {note}" for note in notes)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
