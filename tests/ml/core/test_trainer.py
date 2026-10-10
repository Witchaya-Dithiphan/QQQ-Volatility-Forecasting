import inspect
import json

import pytest

from src.ml.core.trainer import expand_grid, train_model
from src.ml.regression.multiple_linear import MultipleLinear

REQUIRED_ARTIFACTS = (
    "config.json",
    "model.npz",
    "preprocessor.npz",
    "search_results.json",
    "validation_metrics.json",
    "validation_predictions.csv",
    "load_verification.json",
)


def test_expand_grid_produces_every_combination_in_a_stable_order():
    assert expand_grid({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_empty_grid_yields_one_default_candidate():
    assert expand_grid({}) == [{}]


def test_train_model_writes_every_required_artifact(tmp_path):
    result = train_model(
        MultipleLinear, grid={"fit_intercept": [True, False]}, variant="with_spike", output_root=tmp_path
    )
    directory = tmp_path / "with_spike" / "regression" / "multiple_linear"
    for filename in REQUIRED_ARTIFACTS:
        assert (directory / filename).is_file(), filename
    assert result["load_verification"]["passed"] is True
    assert len(json.loads((directory / "search_results.json").read_text(encoding="utf-8"))) == 2


def test_artifacts_are_valid_json_without_stringified_arrays(tmp_path):
    """Guards the bug where json.dumps(default=str) wrote '[0.1 0.2 ...]' into metrics."""
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path)
    directory = tmp_path / "with_spike" / "regression" / "multiple_linear"
    metrics = json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8"))
    assert isinstance(metrics["clipped_metrics"]["rmse"]["value"], float)
    assert "raw_prediction" not in metrics


def test_selected_candidate_is_the_best_on_validation(tmp_path):
    result = train_model(
        MultipleLinear, grid={"fit_intercept": [True, False]}, variant="with_spike", output_root=tmp_path
    )
    rmse = [c["metrics"]["clipped_metrics"]["rmse"]["value"] for c in result["search_results"]]
    assert result["validation_metrics"]["clipped_metrics"]["rmse"]["value"] == pytest.approx(min(rmse))


def test_both_variants_train_and_record_their_own_row_counts(tmp_path):
    for variant, rows in (("with_spike", 1733), ("non_spike", 1520)):
        train_model(MultipleLinear, variant=variant, output_root=tmp_path)
        config = json.loads(
            (tmp_path / variant / "regression" / "multiple_linear" / "config.json").read_text(encoding="utf-8")
        )
        assert config["train_rows"] == rows and config["variant"] == variant


def test_validation_predictions_cover_every_validation_row(tmp_path):
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path)
    path = tmp_path / "with_spike" / "regression" / "multiple_linear" / "validation_predictions.csv"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 372  # header + 371 validation rows


def test_rerunning_the_same_model_overwrites_instead_of_crashing(tmp_path):
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path)
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path)


def test_trainer_source_never_references_the_test_split():
    """A monkeypatch cannot catch `from .data import load_test`, so check the source."""
    import src.ml.core.trainer as trainer_module

    source = inspect.getsource(trainer_module)
    assert "load_test" not in source
    assert "TEST_PATH" not in source
