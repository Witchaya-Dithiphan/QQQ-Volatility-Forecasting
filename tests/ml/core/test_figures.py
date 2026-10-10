import csv

import pytest

from src.ml.core.figures import (
    plot_confusion_matrix,
    plot_performance_curve,
    plot_roc_curves,
    write_leaderboard,
)


def test_roc_figure_is_written(tmp_path):
    path = tmp_path / "roc.png"
    plot_roc_curves({"model_a": ([0.0, 0.5, 1.0], [0.0, 0.8, 1.0], 0.85)}, path, title="ROC")
    assert path.is_file() and path.stat().st_size > 0


def test_roc_figure_handles_several_models(tmp_path):
    path = tmp_path / "roc.png"
    curves = {f"m{i}": ([0.0, 0.5, 1.0], [0.0, 0.7 + i / 50, 1.0], 0.8) for i in range(13)}
    plot_roc_curves(curves, path)
    assert path.stat().st_size > 0


def test_confusion_matrix_figure_is_written(tmp_path):
    path = tmp_path / "cm.png"
    plot_confusion_matrix([[300, 20], [15, 90]], path, title="Confusion")
    assert path.is_file() and path.stat().st_size > 0


def test_performance_curve_figure_is_written(tmp_path):
    path = tmp_path / "curve.png"
    plot_performance_curve({"iteration": [0, 1, 2], "loss": [1.0, 0.5, 0.25]}, path, title="Loss")
    assert path.is_file() and path.stat().st_size > 0


def test_figure_directory_is_created_when_missing(tmp_path):
    path = tmp_path / "nested" / "deeper" / "roc.png"
    plot_roc_curves({"a": ([0.0, 1.0], [0.0, 1.0], 0.5)}, path)
    assert path.is_file()


def test_leaderboard_rows_are_sorted_by_the_primary_metric(tmp_path):
    path = tmp_path / "leaderboard.csv"
    write_leaderboard(
        [
            {"model": "b", "variant": "with_spike", "rmse": 0.5},
            {"model": "a", "variant": "with_spike", "rmse": 0.2},
        ],
        path,
        sort_by="rmse",
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [row["model"] for row in rows] == ["a", "b"]


def test_leaderboard_can_sort_descending_for_auc(tmp_path):
    path = tmp_path / "leaderboard.csv"
    write_leaderboard(
        [{"model": "a", "auc": 0.7}, {"model": "b", "auc": 0.9}], path, sort_by="auc", descending=True
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [row["model"] for row in rows] == ["b", "a"]


def test_leaderboard_puts_undefined_metrics_last(tmp_path):
    path = tmp_path / "leaderboard.csv"
    write_leaderboard(
        [{"model": "a", "rmse": None}, {"model": "b", "rmse": 0.3}], path, sort_by="rmse"
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [row["model"] for row in rows] == ["b", "a"]


def test_empty_leaderboard_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty leaderboard"):
        write_leaderboard([], tmp_path / "x.csv", sort_by="rmse")
