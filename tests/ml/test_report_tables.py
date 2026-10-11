import csv
import shutil

import pytest

from src.ml.core.contracts import VARIANTS
from src.ml.core.trainer import OUTPUT_ROOT
from src.ml.registry import model_names
from src.ml.report_tables import (
    headline_vs_classroom,
    model_inventory,
    spike_policy_on_holdout,
    write_tables,
)

RECEIPT = OUTPUT_ROOT / "finalize_receipt.json"
needs_receipt = pytest.mark.skipif(not RECEIPT.is_file(), reason="finalize has not been run")


def _rows(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def test_inventory_has_one_row_per_model_with_both_variants():
    rows = model_inventory(OUTPUT_ROOT)
    assert [row["model"] for row in rows] == model_names()
    knn = next(row for row in rows if row["model"] == "knn")
    assert knn["candidates"] == 8 and knn["reference"] == "KNeighborsClassifier"
    assert '"k": 15' in knn["params_with_spike"] and '"k": 9' in knn["params_non_spike"]


def test_validation_tables_are_written_without_a_receipt(tmp_path):
    for variant in VARIANTS:
        shutil.copytree(OUTPUT_ROOT / variant, tmp_path / variant,
                        ignore=shutil.ignore_patterns("test_metrics.json", "test_predictions.csv"))
    shutil.copytree(OUTPUT_ROOT / "report", tmp_path / "report", ignore=shutil.ignore_patterns("tables", "cards"))
    shutil.copy(OUTPUT_ROOT / "selection_lock.json", tmp_path / "selection_lock.json")
    written = write_tables(artifact_root=tmp_path, replicates=20)
    names = {path.name for path in written}
    assert {"model_inventory.md", "validation_headline_vs_classroom.md", "walk_forward_with_spike.md",
            "feature_importance_non_spike.md", "classification_metrics_with_spike.md"} <= names
    assert not any(name.startswith("test_") for name in names)
    assert len(_rows(tmp_path / "report" / "tables" / "classification_metrics_with_spike.md")) == 15


def test_headline_is_compared_with_the_classroom_model_of_its_task():
    rows = headline_vs_classroom("validation", OUTPUT_ROOT, replicates=20)
    assert [(row["task"], row["classroom_model"]) for row in rows] == [
        ("classification", "logistic"), ("classification", "logistic"),
        ("regression", "multiple_linear"), ("regression", "multiple_linear")]
    first = rows[0]
    assert first["delta"] == pytest.approx(first["headline_score"] - first["classroom_score"])


@needs_receipt
def test_holdout_comparison_reproduces_the_scores_of_the_final_leaderboard():
    rows = spike_policy_on_holdout(OUTPUT_ROOT, replicates=20)
    assert len(rows) == 17
    with (OUTPUT_ROOT / "final" / "with_spike" / "leaderboard_classification.csv").open(encoding="utf-8") as stream:
        final = {row["model"]: float(row["average_precision"]) for row in csv.DictReader(stream)}
    mlp = next(row for row in rows if row["model"] == "mlp")
    assert mlp["with_spike"] == pytest.approx(final["mlp"])
    assert mlp["delta"] == pytest.approx(mlp["non_spike"] - mlp["with_spike"])


@needs_receipt
def test_holdout_tables_are_written_once_finalize_left_a_receipt(tmp_path):
    written = write_tables(report_root=tmp_path, replicates=20)
    names = {path.name for path in written}
    assert {"test_segments_classification.md", "test_segments_regression.md", "test_headline_vs_classroom.md",
            "test_spike_policy.md", "test_classification_metrics_with_spike.md"} <= names
    segments = _rows(tmp_path / "tables" / "test_segments_classification.md")
    assert {row[2] for row in segments} == {"spike_affected", "non_spike"}
    # Counts are printed as whole numbers, and no column is empty for the whole table.
    assert {row[3] for row in segments} == {"54", "319"} and {row[4] for row in segments} == {"38", "35"}
    assert not any(cell == "n/a" for row in segments for cell in row)
    summary = _rows(tmp_path / "tables" / "test_headline_summary.md")
    assert [(row[0], row[1], row[2]) for row in summary] == [
        ("classification", "with_spike", "mlp"), ("classification", "non_spike", "random_forest"),
        ("regression", "with_spike", "elastic_net"), ("regression", "non_spike", "elastic_net")]
