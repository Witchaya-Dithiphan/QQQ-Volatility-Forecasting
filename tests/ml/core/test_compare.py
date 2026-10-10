"""The parity gate decides whether a scratch model counts as correct."""
import numpy as np
import pytest

from src.ml.core.compare import ParityError, align_clusters, assert_parity


class _Fixed:
    """Stands in for a fitted model: returns whatever it was constructed with."""

    def __init__(self, values, scores=None, proba=None):
        self.values = np.asarray(values, dtype=np.float64)
        self._scores = scores
        self._proba = proba

    def predict(self, X):
        return self.values

    def decision_function(self, X):
        if self._scores is None:
            raise NotImplementedError
        return np.asarray(self._scores, dtype=np.float64)

    def predict_proba(self, X):
        if self._proba is None:
            raise NotImplementedError
        return np.asarray(self._proba, dtype=np.float64)


X3 = np.zeros((3, 1))
X2 = np.zeros((2, 1))


def test_identical_regression_predictions_pass():
    assert_parity(_Fixed([1.0, 2.0]), _Fixed([1.0, 2.0]), X2, task="regression")


def test_regression_difference_beyond_tolerance_fails():
    with pytest.raises(ParityError, match="max abs diff"):
        assert_parity(_Fixed([1.0, 2.0]), _Fixed([1.0, 2.5]), X2, task="regression")


def test_regression_difference_within_tolerance_passes():
    assert_parity(_Fixed([1.0, 2.0]), _Fixed([1.0, 2.0 + 1e-9]), X2, task="regression")


def test_failure_message_names_the_worst_index_and_both_values():
    """Diagnosing the mismatch is the whole point, so the message must carry the evidence."""
    with pytest.raises(ParityError) as error:
        assert_parity(_Fixed([1.0, 2.0, 3.0]), _Fixed([1.0, 2.0, 9.0]), X3, task="regression")
    message = str(error.value)
    assert "index 2" in message and "3.0" in message and "9.0" in message


def test_shape_mismatch_is_reported_as_such():
    with pytest.raises(ParityError, match="shape"):
        assert_parity(_Fixed([1.0, 2.0]), _Fixed([1.0, 2.0, 3.0]), X2, task="regression")


def test_classification_requires_every_label_to_match():
    with pytest.raises(ParityError, match="labels differ"):
        assert_parity(_Fixed([0, 1, 1]), _Fixed([0, 1, 0]), X3, task="classification")


def test_classification_also_compares_decision_scores():
    with pytest.raises(ParityError, match="decision_function"):
        assert_parity(
            _Fixed([0, 1], scores=[-1.0, 1.0]),
            _Fixed([0, 1], scores=[-1.0, 5.0]),
            X2,
            task="classification",
        )


def test_classification_unwraps_two_column_predict_proba():
    """sklearn returns (n, 2); scratch models return (n,). Same numbers, different shape."""
    assert_parity(
        _Fixed([0, 1], proba=[0.25, 0.75]),
        _Fixed([0, 1], proba=[[0.75, 0.25], [0.25, 0.75]]),
        X2,
        task="classification",
    )


def test_missing_optional_method_is_skipped_not_failed():
    assert_parity(_Fixed([0, 1]), _Fixed([0, 1], scores=[-1.0, 1.0]), X2, task="classification")


def test_cluster_labels_are_aligned_before_comparison():
    assert np.array_equal(align_clusters(np.array([0, 0, 1]), np.array([1, 1, 0])), np.array([0, 0, 1]))


def test_clustering_parity_ignores_cluster_renaming():
    assert_parity(_Fixed([0, 0, 1]), _Fixed([1, 1, 0]), X3, task="clustering")


def test_clustering_fails_when_the_partition_really_differs():
    with pytest.raises(ParityError, match="partition"):
        assert_parity(_Fixed([0, 0, 1]), _Fixed([0, 1, 1]), X3, task="clustering")


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="Unknown task"):
        assert_parity(_Fixed([1.0]), _Fixed([1.0]), np.zeros((1, 1)), task="nonsense")


def test_passing_result_reports_what_was_checked():
    result = assert_parity(_Fixed([1.0, 2.0]), _Fixed([1.0, 2.0]), X2, task="regression")
    assert result["passed"] is True and result["n_samples"] == 2 and result["task"] == "regression"
