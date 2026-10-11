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


import numpy as np

from src.ml.core.figures import (
    plot_confusion_grid,
    plot_heatmap,
    plot_interval_bars,
    plot_performance_grid,
    plot_pr_curves,
    plot_predicted_vs_actual,
    plot_residuals,
    write_markdown_table,
)


def test_leaderboard_puts_undefined_metrics_last_when_descending(tmp_path):
    path = tmp_path / "leaderboard.csv"
    write_leaderboard(
        [{"model": "a", "ap": None}, {"model": "b", "ap": 0.2}, {"model": "c", "ap": None},
         {"model": "d", "ap": 0.3}],
        path, sort_by="ap", descending=True,
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [row["model"] for row in rows] == ["d", "b", "a", "c"]


def test_pr_figure_is_written(tmp_path):
    path = tmp_path / "pr.png"
    plot_pr_curves({"m": ([0.0, 0.5, 1.0], [1.0, 0.4, 0.1], 0.3)}, path, prevalence=0.09)
    assert path.stat().st_size > 0


def test_confusion_grid_handles_a_partly_filled_last_row(tmp_path):
    path = tmp_path / "grid.png"
    plot_confusion_grid({f"m{i}": [[300, 20], [15, 9]] for i in range(13)}, path)
    assert path.stat().st_size > 0


def test_regression_figures_are_written(tmp_path):
    actual, predicted = np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.18, 0.25])
    dates = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    plot_predicted_vs_actual({"a": (actual, predicted)}, tmp_path / "pva.png")
    plot_residuals({"a": (dates, actual - predicted)}, tmp_path / "res.png")
    assert (tmp_path / "pva.png").stat().st_size > 0 and (tmp_path / "res.png").stat().st_size > 0


def test_performance_grid_is_written(tmp_path):
    path = tmp_path / "perf.png"
    plot_performance_grid({"a": {"iteration": [1, 2, 3], "loss": [1.0, 0.6, 0.5]}}, path)
    assert path.stat().st_size > 0


def test_interval_bars_accept_rows_without_an_interval(tmp_path):
    path = tmp_path / "delta.png"
    plot_interval_bars([("a", 0.02, -0.01, 0.05), ("b", -0.03, None, None)], path,
                       title="Delta", xlabel="non_spike - with_spike")
    assert path.stat().st_size > 0


def test_heatmap_is_written(tmp_path):
    path = tmp_path / "heat.png"
    plot_heatmap(np.array([[0.1, 0.0], [0.02, 0.3]]), ["m1", "m2"], ["f1", "f2"], path,
                 title="Importance", colorbar_label="drop in AP")
    assert path.stat().st_size > 0


def test_markdown_table_formats_missing_values_and_flags(tmp_path):
    path = tmp_path / "table.md"
    write_markdown_table(
        [{"model": "a", "ap": 0.25481, "tied": True}, {"model": "b", "ap": None, "tied": False}],
        path, columns=["model", "ap", "tied"], title="Leaderboard", notes=("selection score",),
    )
    text = path.read_text(encoding="utf-8")
    assert "### Leaderboard" in text
    assert "| a | 0.2548 | yes |" in text and "| b | n/a | no |" in text
    assert "> selection score" in text


def test_empty_markdown_table_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty table"):
        write_markdown_table([], tmp_path / "x.md", columns=["a"], title="t")


def test_thirteen_overlaid_models_each_get_a_distinct_line_style():
    from src.ml.core.figures import series_styles

    names = [f"model_{i:02d}" for i in range(13)]
    styles = series_styles(names)
    assert len({(style["color"], style["linestyle"]) for style in styles.values()}) == 13
    # The style follows the name, not its rank, so dropping a model never repaints the rest.
    assert series_styles(names[3:], universe=names)["model_05"] == styles["model_05"]


def test_more_models_than_distinct_styles_is_rejected():
    from src.ml.core.figures import series_styles

    with pytest.raises(ValueError, match="distinct line styles"):
        series_styles([f"m{i}" for i in range(17)])
