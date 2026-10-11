"""CART split mechanics and strict comparison with sklearn."""

import numpy as np
import pytest

from src.ml.classification.decision_tree import DecisionTree, _best_split
from src.ml.core.compare import assert_parity
from src.ml.core.data import load
from src.ml.core.persist import load_npz, save_npz


def test_best_split_uses_weighted_gini_and_smallest_threshold_on_a_tie():
    X = np.array([[0, 0], [1, 0], [2, 1], [3, 1]], dtype=float)
    y = np.array([0, 1, 0, 1])
    feature, threshold, gain = _best_split(X, y, np.ones(4), 1)
    assert (feature, threshold) == (0, 0.5)
    assert gain == pytest.approx(1 / 6)

    feature, threshold, gain = _best_split(X, y, np.array([1, 1, 10, 1]), 1)
    assert (feature, threshold) == (0, 2.5)
    assert gain > 0


def test_best_split_rejects_pure_node_and_too_small_leaves():
    X = np.arange(4.0).reshape(-1, 1)
    assert _best_split(X, np.zeros(4, dtype=int), np.ones(4), 1) is None
    assert _best_split(X, np.array([0, 0, 1, 1]), np.ones(4), 3) is None
    assert _best_split(X, np.array([0, 0, 1, 1]), np.ones(4), 2)[:2] == (0, 1.5)


def test_pure_tree_has_one_root_and_readonly_input_is_untouched():
    X = np.array([[0.0], [1.0], [2.0]])
    X.setflags(write=False)
    model = DecisionTree(max_depth=3, min_samples_leaf=1).fit(X, [1, 1, 1])
    assert model.tree_.feature.tolist() == [-2]
    assert np.array_equal(model.predict(X), [1, 1, 1])
    assert np.array_equal(model.predict_proba(X), [1.0, 1.0, 1.0])
    assert not X.flags.writeable


def test_weighted_tree_matches_reference():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    weights = np.array([1, 1, 10, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    scratch = DecisionTree(max_depth=2, min_samples_leaf=1).fit(X, y, sample_weight=weights)
    reference = scratch.reference().fit(X, y, sample_weight=weights)
    assert np.array_equal(scratch.tree_.feature, reference.tree_.feature)
    assert np.allclose(scratch.tree_.threshold, reference.tree_.threshold, rtol=0, atol=1e-7)
    assert_parity(scratch, reference, X, y, task="classification")


@pytest.mark.parametrize("min_samples_leaf, expected", [(2, (0, 3.5)), (3, None)])
def test_zero_weight_rows_do_not_count_toward_minimum_leaf(min_samples_leaf, expected):
    X = np.arange(6.0).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    weights = np.array([1, 0, 0, 1, 1, 1], dtype=float)
    split = _best_split(X, y, weights, min_samples_leaf)
    assert (None if split is None else split[:2]) == expected
    scratch = DecisionTree(min_samples_leaf=min_samples_leaf).fit(X, y, sample_weight=weights)
    reference = scratch.reference().fit(X, y, sample_weight=weights)
    assert np.array_equal(scratch.tree_.feature, reference.tree_.feature)
    assert np.array_equal(scratch.tree_.threshold, reference.tree_.threshold)
    assert_parity(scratch, reference, X, y, task="classification")


@pytest.mark.parametrize("seed", [0, 3, 11, 42])
@pytest.mark.parametrize("constant_features", [False, True])
def test_sqrt_samples_features_per_node_with_bootstrap_weights(seed, constant_features):
    rng = np.random.RandomState(29)
    X = rng.normal(size=(180, 8))
    y = (X[:, 0] + 0.7 * X[:, 1] * X[:, 2] > 0).astype(int)
    if constant_features:
        X[:, 3:6] = 1
        X[:, 6] = X[:, 0] > 0  # Becomes constant in some descendants.
        X[:, 7] = X[:, 1] > 0
    X.setflags(write=False)
    weights = np.bincount(np.random.RandomState(seed).randint(0, len(X), len(X)), minlength=len(X))
    scratch = DecisionTree(max_depth=5, min_samples_leaf=5, random_state=seed, max_features="sqrt")
    scratch.fit(X, y, sample_weight=weights)
    reference = scratch.reference().fit(X, y, sample_weight=weights)
    assert np.array_equal(scratch.tree_.feature, reference.tree_.feature)
    assert np.array_equal(scratch.tree_.threshold, reference.tree_.threshold)
    assert np.array_equal(scratch.tree_.children_left, reference.tree_.children_left)
    assert np.array_equal(scratch.tree_.children_right, reference.tree_.children_right)
    assert_parity(scratch, reference, X, y, task="classification")
    assert not X.flags.writeable


def test_sqrt_search_continues_past_constant_features():
    X = np.zeros((4, 8))
    X[:, 7] = [0, 1, 2, 3]
    scratch = DecisionTree(min_samples_leaf=1, max_features="sqrt").fit(X, [0, 0, 1, 1])
    assert scratch.tree_.feature.tolist() == [7, -2, -2]
    assert scratch.tree_.threshold.tolist() == [1.5, -2, -2]
    reference = scratch.reference().fit(X, [0, 0, 1, 1])
    assert_parity(scratch, reference, X, [0, 0, 1, 1], task="classification")


def test_rejects_unsupported_max_features():
    with pytest.raises(ValueError, match="max_features"):
        DecisionTree(max_features="log2").fit([[0], [1]], [0, 1])


def test_rejects_invalid_labels_depth_and_weights():
    X = np.arange(4.0).reshape(-1, 1)
    with pytest.raises(ValueError):
        DecisionTree(min_samples_leaf=1).fit(X, [0, 0.5, 1, 1])
    with pytest.raises(ValueError):
        DecisionTree(max_depth=0, min_samples_leaf=1).fit(X, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        DecisionTree(min_samples_leaf=1).fit(X, [0, 0, 1, 1], sample_weight=np.zeros(4))


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_tree_structure_and_outputs_match_sklearn_on_synthetic_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(180, 5))
    y = (X[:, 0] + 0.7 * X[:, 1] * X[:, 2] > 0).astype(int)
    scratch = DecisionTree(max_depth=3, min_samples_leaf=5).fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert np.array_equal(scratch.tree_.feature, reference.tree_.feature)
    assert np.allclose(scratch.tree_.threshold, reference.tree_.threshold, rtol=0, atol=1e-7)
    assert_parity(scratch, reference, X, y, task="classification")


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
@pytest.mark.parametrize("max_depth", [2, 3, 5])
@pytest.mark.parametrize("min_samples_leaf", [5, 10])
@pytest.mark.parametrize("max_features", [None, "sqrt"])
def test_parity_on_real_train_and_validation(variant, max_depth, min_samples_leaf, max_features):
    data = load(variant)
    scratch = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features).fit(
        data.train.X, data.train.y_classification
    )
    reference = scratch.reference().fit(data.train.X, data.train.y_classification)
    assert np.array_equal(scratch.tree_.feature, reference.tree_.feature)
    assert np.allclose(scratch.tree_.threshold, reference.tree_.threshold, rtol=0, atol=1e-7)
    assert_parity(scratch, reference, data.train.X, data.train.y_classification, task="classification")
    assert_parity(scratch, reference, data.validation.X, data.validation.y_classification, task="classification")


@pytest.mark.parametrize("max_features", [None, "sqrt"])
def test_save_load_preserves_labels_and_probabilities(tmp_path, max_features):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    original = DecisionTree(max_depth=2, min_samples_leaf=1, max_features=max_features).fit(X, [0, 0, 1, 1])
    direct = DecisionTree().load_state(original.state_dict(), {"params": original.get_params()})
    assert direct.get_params() == original.get_params()
    assert np.array_equal(original.predict_proba(X), direct.predict_proba(X))
    path = tmp_path / "model.npz"
    save_npz(path, original.state_dict(), {"params": original.get_params()})
    arrays, metadata = load_npz(path)
    restored = DecisionTree().load_state(arrays, metadata)
    assert restored.get_params() == original.get_params()
    assert np.array_equal(original.predict(X), restored.predict(X))
    assert np.array_equal(original.predict_proba(X), restored.predict_proba(X))
