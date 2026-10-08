"""Truthful metrics, ties, clipping, loss domains and undefined reasons."""
import numpy as np
import pytest
from sklearn import metrics as reference
from src.modeling.metrics import regression_metrics, classification_metrics, loss, select_threshold, regression_outputs, select_candidate, clustering_metrics, train_cluster_mapping


def test_regression_raw_clipped_and_constant_target():
    result = regression_outputs([0., 2.], [-1., 2.])
    assert result["negative_count"] == 1
    np.testing.assert_array_equal(result["raw_prediction"], [-1., 2.])
    np.testing.assert_array_equal(result["final_prediction"], [0., 2.])
    assert result["raw_metrics"]["rmse"]["value"] == pytest.approx(np.sqrt(.5))
    assert result["clipped_metrics"]["rmse"]["value"] == 0
    assert regression_metrics([1., 1.], [1., 1.])["r2"]["reason"]


def test_binary_metrics_ranking_ties_and_curves():
    y = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([.2, .8, .8, .8, .5, .2])
    result = classification_metrics(y, scores, threshold=.5)
    predicted = scores >= .5
    for key, expected in {"accuracy": reference.accuracy_score(y, predicted), "precision": reference.precision_score(y, predicted), "recall": reference.recall_score(y, predicted), "f1": reference.f1_score(y, predicted), "roc_auc": reference.roc_auc_score(y, scores), "average_precision": reference.average_precision_score(y, scores)}.items():
        assert result[key]["value"] == pytest.approx(expected)
    assert result["confusion_matrix"] == [[2, 1], [0, 3]]
    assert result["specificity"]["value"] == pytest.approx(2 / 3)
    assert result["roc_curve"]["value"]["fpr"][0] == 0


def test_undefined_metrics_have_reasons_and_no_fake_probabilities():
    result = classification_metrics([0, 0], [0., 0.], threshold=.5)
    for key in ["precision", "recall", "f1", "roc_auc", "average_precision"]:
        assert result[key]["value"] is None and result[key]["reason"]
    assert loss("log_loss", [0, 1], scores=[-2, 2])["value"] is None
    assert select_threshold([1, 1], [.1, .8])["reason"]


def test_threshold_exhaustive_f1_recall_lower_ties_and_default():
    y, score = [0, 1, 1, 0], [.1, .4, .8, .9]
    result = select_threshold(y, score)
    assert result["threshold"] == .4
    assert result["default_threshold"] == .5
    assert result["selection_split"] == "validation"
    assert select_threshold(y, score, score_kind="decision")["default_threshold"] == 0
    with pytest.raises(ValueError): select_threshold(y, score, split="test")
    # Equal F1: threshold .2 recalls all positives; .8 recalls one.
    tied = select_threshold([0, 0, 1, 0, 1], [.1, .3, .2, .4, .8])
    assert tied["threshold"] == .2


def test_loss_values_and_extreme_probability_clipping():
    for name in ["binary_cross_entropy", "log_loss"]:
        assert loss(name, [0, 1], probabilities=[0, 1])["value"] == pytest.approx(0, abs=1e-14)
        assert np.isfinite(loss(name, [0, 1], probabilities=[1, 0])["value"])
    assert loss("binary_logistic_loss", [0, 1], scores=[-1000, 1000])["value"] == 0
    assert loss("hinge_loss", [0, 1], scores=[-.5, .5])["value"] == .5
    assert loss("exponential_loss", [0, 1], scores=[-1, 1])["value"] == pytest.approx(np.exp(-1))
    assert loss("perceptron_criterion", [0, 1], scores=[1, -2])["value"] == 1.5
    with pytest.raises(ValueError): loss("log_loss", [0, 1], probabilities=[-1, 2])


@pytest.mark.parametrize("y,p", [([], []), ([0, 2], [0, 1]), ([0, 1], [float("nan"), 0]), ([0], [0, 1])])
def test_invalid_binary_input(y, p):
    with pytest.raises(ValueError): classification_metrics(y, p)


def test_selection_ties_and_na():
    candidates = [{"id": "b", "metrics": {"clipped_metrics": {"rmse": {"value": 1}, "mae": {"value": .8}}}}, {"id": "a", "metrics": {"clipped_metrics": {"rmse": {"value": 1}, "mae": {"value": .5}}}}]
    assert select_candidate(candidates, task="regression")["id"] == "a"
    candidates = [{"id": "a", "metrics": {"average_precision": {"value": None}, "roc_auc": {"value": .9}}}, {"id": "b", "metrics": {"average_precision": {"value": .8}, "roc_auc": {"value": .7}}}]
    assert select_candidate(candidates, task="classification")["id"] == "b"


def test_clustering_metrics_and_train_only_mapping():
    X, labels = np.array([[0.], [1.], [10.], [11.]]), [0, 0, 1, 1]
    result = clustering_metrics(X, labels)
    assert result["silhouette"]["value"] == pytest.approx(reference.silhouette_score(X, labels))
    assert result["inertia"]["value"] == 1
    assert train_cluster_mapping(labels, [0, 1, 1, 1]) == {0: 1, 1: 1}
    with pytest.raises(ValueError): train_cluster_mapping(labels, [0, 1, 1, 1], split="validation")


def test_perceptron_loss_reports_actual_mistake_rate():
    result = loss("perceptron_criterion", [0, 1, 1], scores=[1, 0, 2])
    assert result["mistake_rate"]["value"] == pytest.approx(1 / 3)



def test_candidate_selection_consumes_regression_outputs_and_final_id_tie():
    best = {"id": "a", "metrics": regression_outputs([0., 2.], [-10., 2.])}
    worse = {"id": "b", "metrics": regression_outputs([0., 2.], [1., 2.])}
    assert select_candidate([worse, best], task="regression") is best
    tied = {"id": "z", "metrics": regression_outputs([0., 2.], [0., 2.])}
    for candidates in ([tied, best], [best, tied]):
        assert select_candidate(candidates, task="regression")["id"] == "a"


@pytest.mark.parametrize("task,metrics", [
    ("classification", {"average_precision": {"value": .8}, "roc_auc": {"value": .9}}),
    ("clustering", {"silhouette": {"value": .7}}),
])
def test_candidate_final_id_tie_for_other_tasks(task, metrics):
    a, z = {"id": "a", "metrics": metrics}, {"id": "z", "metrics": metrics}
    assert select_candidate([z, a], task=task) is a
    assert select_candidate([a, z], task=task) is a
    # Without candidate IDs, the explicitly supplied enumeration is the tie-break.
    first, second = {"metrics": metrics}, {"metrics": metrics}
    assert select_candidate([first, second], task=task) is first
