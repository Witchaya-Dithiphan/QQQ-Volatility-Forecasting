# tests/ml/core/test_report.py
import csv
import json
import shutil

import numpy as np
import pytest

from src.ml.core.artifacts import load_model, run_directory
from src.ml.core.bootstrap import block_indices
from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.core.data import load
from src.ml.core.report import (
    SELECTION_NOTE,
    baselines,
    build_report,
    classification_scores,
    half_vote,
    leaderboard,
    score_entries,
)
from src.ml.core.trainer import OUTPUT_ROOT


def test_regression_baselines_use_only_train_statistics_and_todays_feature():
    data = load("with_spike")
    rules = baselines("regression", data.train, data.validation, data.threshold)
    assert set(rules) == {"baseline_train_mean", "baseline_persistence"}
    assert np.all(rules["baseline_train_mean"][1] == data.train.y_regression.mean())
    column = FEATURE_COLUMNS.index("historical_volatility_5d")
    np.testing.assert_array_equal(rules["baseline_persistence"][1], data.validation.X[:, column])


def test_persistence_baseline_is_identical_for_both_variants():
    a, b = load("with_spike"), load("non_spike")
    first = baselines("classification", a.train, a.validation, a.threshold)["baseline_persistence"]
    second = baselines("classification", b.train, b.validation, b.threshold)["baseline_persistence"]
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_majority_baseline_reproduces_the_audited_numbers():
    data = load("with_spike")
    score, label = baselines("classification", data.train, data.validation, data.threshold)["baseline_majority"]
    scores = classification_scores(data.validation.y_classification, score, label)
    assert scores["accuracy"] == pytest.approx(337 / 371)
    assert scores["average_precision"] == pytest.approx(34 / 371)
    assert scores["roc_auc"] == pytest.approx(0.5)
    assert scores["recall"] == 0.0 and scores["f1"] == 0.0


def _entry(name, target, score):
    return {"model": name, "kind": "model", "target": target, "score": score,
            "prediction": (score > 0.5).astype(float)}


def test_leaderboard_separates_a_clearly_worse_model_but_not_an_identical_one():
    rng = np.random.RandomState(0)
    target = (rng.rand(400) < 0.3).astype(float)
    strong = target * 0.3 + rng.rand(400) * 0.7
    scored = score_entries(
        "classification",
        [_entry("strong", target, strong), _entry("copy", target, strong.copy()),
         _entry("inverted", target, 1 - strong)],
        block_indices(400, replicates=200),
    )
    rows = {row["model"]: row for row in leaderboard("classification", scored)}
    assert rows["strong"]["tied_with_leader"] is True
    assert rows["copy"]["tied_with_leader"] is True
    assert rows["inverted"]["tied_with_leader"] is False
    assert rows["inverted"]["gap_high"] < 0
    assert rows["strong"]["average_precision_low"] < rows["strong"]["average_precision"] < rows["strong"]["average_precision_high"]


def test_leaderboard_needs_at_least_one_model_row():
    with pytest.raises(ValueError, match="No model rows"):
        leaderboard("classification", [])


def test_build_report_writes_every_table_and_figure(tmp_path):
    result = build_report("with_spike", report_root=tmp_path / "report", figure_root=tmp_path / "fig", replicates=50)
    report, figures = tmp_path / "report" / "with_spike", tmp_path / "fig" / "with_spike"
    rows = list(csv.DictReader((report / "leaderboard_classification.csv").open(encoding="utf-8")))
    assert len(rows) == 15
    for column in ("loss", "tn", "fp", "fn", "tp", "accuracy", "precision", "sensitivity", "specificity",
                   "tnr", "f1", "roc_auc", "average_precision", "average_precision_low",
                   "average_precision_high", "tied_with_leader"):
        assert column in rows[0]
    assert len(list(csv.DictReader((report / "leaderboard_regression.csv").open(encoding="utf-8")))) == 6
    assert len(list(csv.DictReader((report / "leaderboard_clustering.csv").open(encoding="utf-8")))) == 2
    assert SELECTION_NOTE in (report / "leaderboard_classification.md").read_text(encoding="utf-8")
    for name in ("roc_overlay.png", "pr_overlay.png", "confusion_matrices.png", "performance_curves.png",
                 "regression_predicted_vs_actual.png", "regression_residuals.png"):
        assert (figures / name).stat().st_size > 0
    assert all(path.is_file() for path in result["written"])


def test_stale_artifact_is_rejected(tmp_path):
    source = run_directory(OUTPUT_ROOT, "with_spike", "multiple_linear")
    target = run_directory(tmp_path, "with_spike", "multiple_linear")
    shutil.copytree(source, target)
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    config["validation_sha256"] = "0" * 64
    (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="different Validation file"):
        build_report("with_spike", artifact_root=tmp_path, report_root=tmp_path / "r",
                     figure_root=tmp_path / "f", replicates=20, require_all=False)


def test_leaderboard_says_whether_a_model_beats_the_best_baseline():
    rng = np.random.RandomState(0)
    target = (rng.rand(400) < 0.3).astype(float)
    strong = target * 0.3 + rng.rand(400) * 0.7
    noise = rng.rand(400)
    scored = score_entries(
        "classification",
        [_entry("strong", target, strong), _entry("inverted", target, 1 - strong),
         {**_entry("baseline_noise", target, noise), "kind": "baseline"}],
        block_indices(400, replicates=50),
    )
    rows = {row["model"]: row for row in leaderboard("classification", scored)}
    assert rows["strong"]["beats_best_baseline"] is True
    assert rows["inverted"]["beats_best_baseline"] is False
    assert rows["baseline_noise"]["beats_best_baseline"] is None
    assert rows["strong"]["best_baseline"] == "baseline_noise"


def test_leaderboard_without_baselines_leaves_the_comparison_undefined():
    rng = np.random.RandomState(0)
    target = (rng.rand(200) < 0.3).astype(float)
    scored = score_entries("classification", [_entry("only", target, rng.rand(200))],
                           block_indices(200, replicates=20))
    assert leaderboard("classification", scored)[0]["beats_best_baseline"] is None


def test_markdown_leaderboard_names_the_loss_and_the_baseline_verdict(tmp_path):
    build_report("with_spike", report_root=tmp_path / "report", figure_root=tmp_path / "fig", replicates=20)
    header = (tmp_path / "report" / "with_spike" / "leaderboard_classification.md").read_text(
        encoding="utf-8").splitlines()[2]
    assert "loss_name" in header and "beats_best_baseline" in header


def _classification_rows(tmp_path, variant="with_spike"):
    build_report(variant, report_root=tmp_path / "report", figure_root=tmp_path / "fig", replicates=20)
    path = tmp_path / "report" / variant / "leaderboard_classification.csv"
    return {row["model"]: row for row in csv.DictReader(path.open(encoding="utf-8"))}


def test_half_vote_undoes_the_sigmoid_and_the_vote_normalisation():
    # A saved AdaBoost score is sigmoid(2F / sum(alpha)). With sum(alpha) = 2 the votes
    # F = 2, -2 and 0.5 are stored as sigmoid(2), sigmoid(-2) and sigmoid(0.5).
    stored = 1.0 / (1.0 + np.exp(-np.array([2.0, -2.0, 0.5])))
    np.testing.assert_allclose(half_vote(stored, weight_sum=2.0), [1.0, -1.0, 0.25])


def test_every_fitted_classifier_gets_a_loss_in_the_leaderboard(tmp_path):
    rows = _classification_rows(tmp_path)
    assert [name for name, row in rows.items() if row["kind"] == "model" and row["loss"] == ""] == []


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
def test_adaboost_loss_is_the_one_its_training_curve_tracks(tmp_path, variant):
    """exp(-yF/2) on the unnormalised vote F, rebuilt here from the saved stumps."""
    row = _classification_rows(tmp_path, variant)["adaboost"]
    validation = load(variant).validation
    model, transform = load_model(run_directory(OUTPUT_ROOT, variant, "adaboost"))
    X = transform(validation.X)
    vote = sum(alpha * (2 * stump.predict(X) - 1) for stump, alpha in zip(model.stumps_, model.estimator_weights_))
    signed = 2 * validation.y_classification - 1
    assert float(row["loss"]) == pytest.approx(np.mean(np.exp(-0.5 * signed * vote)), rel=1e-9)
