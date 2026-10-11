# tests/ml/core/test_artifacts.py
import numpy as np
import pytest

from src.ml.core.artifacts import load_model, read_run, read_runs, run_directory, sha256_file
from src.ml.core.contracts import VARIANTS
from src.ml.core.data import load
from src.ml.core.trainer import OUTPUT_ROOT, evaluate
from src.ml.registry import model_names


def test_run_directory_follows_the_artifact_layout(tmp_path):
    assert run_directory(tmp_path, "with_spike", "knn") == tmp_path / "with_spike" / "classification" / "knn"


def test_read_run_exposes_predictions_as_arrays():
    run = read_run(OUTPUT_ROOT, "with_spike", "knn")
    assert run.task == "classification" and run.config["selected_candidate"] == "knn-07"
    assert run.dates.shape == run.target.shape == run.score.shape == run.prediction.shape == (371,)
    assert str(run.dates[0]) == "2023-08-28"
    assert int(run.target.sum()) == 34


def test_history_is_none_when_a_model_has_no_training_curve():
    assert read_run(OUTPUT_ROOT, "with_spike", "knn").history is None
    assert len(read_run(OUTPUT_ROOT, "with_spike", "adaboost").history["loss"]) == 100


def test_read_runs_filters_by_task():
    assert [run.name for run in read_runs(OUTPUT_ROOT, "with_spike", task="regression")] == [
        "elastic_net", "multiple_linear", "polynomial", "simple_linear"]
    assert len(read_runs(OUTPUT_ROOT, "non_spike", task="classification")) == 13


def test_missing_runs_fail_loudly_unless_explicitly_tolerated(tmp_path):
    with pytest.raises(FileNotFoundError, match="no finished run"):
        read_runs(tmp_path, "with_spike")
    assert read_runs(tmp_path, "with_spike", require_all=False) == []


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", model_names())
def test_every_saved_model_reproduces_its_stored_validation_predictions(variant, name):
    run = read_run(OUTPUT_ROOT, variant, name)
    model, transform = load_model(run.directory)
    validation = load(variant).validation
    _, prediction, score = evaluate(
        model, transform(validation.X), validation.y_regression, validation.y_classification, run.task
    )
    if run.task == "regression":
        np.testing.assert_allclose(prediction, run.prediction, rtol=1e-10, atol=1e-12)
    else:
        np.testing.assert_array_equal(prediction, run.prediction)
    np.testing.assert_allclose(score, run.score, rtol=1e-10, atol=1e-12)


def test_sha256_is_the_hex_digest_of_the_file_bytes(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
