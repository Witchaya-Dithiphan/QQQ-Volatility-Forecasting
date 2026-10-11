"""Bootstrap forest mechanics and strict sklearn parity."""

import numpy as np
import pytest
from sklearn.ensemble._forest import _generate_sample_indices

from src.ml.classification.decision_tree import DecisionTree
from src.ml.classification.random_forest import RandomForest
from src.ml.core.compare import assert_parity
from src.ml.core.data import load
from src.ml.core.persist import load_npz, save_npz
from src.ml.core.preprocess import PCA, Standardizer
from src.ml.registry import build, model_names


def test_single_unbootstrapped_tree_uses_derived_seed():
    X = np.arange(12.0).reshape(6, 2)
    y = np.array([0, 0, 1, 1, 0, 1])
    seed = int(np.random.RandomState(17).randint(np.iinfo(np.int32).max))
    forest = RandomForest(n_estimators=1, max_depth=3, min_samples_leaf=1,
                          max_features=None, bootstrap=False, random_state=17).fit(X, y)
    tree = DecisionTree(max_depth=3, min_samples_leaf=1, max_features=None,
                        random_state=seed).fit(X, y)
    assert np.array_equal(forest.trees_[0].tree_.feature, tree.tree_.feature)
    assert np.array_equal(forest.predict_proba(X), tree.predict_proba(X))


def test_bootstrap_counts_match_sklearn_and_ignore_omitted_rows(monkeypatch):
    X = np.arange(24.0).reshape(12, 2)
    y = np.array([0, 1] * 6)
    fitted_weights = []
    original_fit = DecisionTree.fit

    def capture_fit(tree, X, y=None, sample_weight=None):
        fitted_weights.append(sample_weight.copy())
        return original_fit(tree, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(DecisionTree, "fit", capture_fit)
    forest = RandomForest(n_estimators=3, max_depth=3, min_samples_leaf=1,
                          random_state=9).fit(X, y)
    monkeypatch.setattr(DecisionTree, "fit", original_fit)
    seeds = np.random.RandomState(9).randint(np.iinfo(np.int32).max, size=3)
    assert len(fitted_weights) == len(seeds)
    for seed, tree, actual_counts in zip(seeds, forest.trees_, fitted_weights):
        indices = _generate_sample_indices(int(seed), len(X), len(X), None)
        counts = np.bincount(indices, minlength=len(X))
        assert np.array_equal(actual_counts, counts)
        expected = DecisionTree(max_depth=3, min_samples_leaf=1,
                                max_features="sqrt", random_state=int(seed)).fit(
                                    X, y, sample_weight=counts)
        assert np.array_equal(tree.tree_.feature, expected.tree_.feature)
        assert np.array_equal(tree.tree_.threshold, expected.tree_.threshold)


def test_forest_averages_probabilities_and_preserves_readonly_input():
    X = np.arange(20.0).reshape(10, 2)
    X.setflags(write=False)
    y = np.array([0, 1] * 5)
    forest = RandomForest(n_estimators=5, max_depth=2, min_samples_leaf=1).fit(X, y)
    expected = np.mean([tree.predict_proba(X) for tree in forest.trees_], axis=0)
    assert np.array_equal(forest.predict_proba(X), expected)
    assert np.array_equal(forest.predict(X), (expected > 0.5).astype(int))
    assert not X.flags.writeable


@pytest.mark.parametrize("kwargs", [
    {"n_estimators": 0}, {"max_depth": 0}, {"min_samples_leaf": 0},
    {"max_features": "log2"}, {"bootstrap": "yes"}, {"random_state": -1},
])
def test_invalid_hyperparameters_fail(kwargs):
    with pytest.raises(ValueError):
        RandomForest(**kwargs).fit([[0], [1]], [0, 1])


def test_invalid_training_data_fail():
    with pytest.raises(ValueError):
        RandomForest().fit([[0], [1]], [0, 2])
    with pytest.raises(ValueError):
        RandomForest().fit([[0], [np.nan]], [0, 1])
    # Seed 42 draws [1, 1] for n=2; invalid row 0 must still be rejected.
    with pytest.raises(ValueError):
        RandomForest(n_estimators=1).fit([[0], [1]], [2, 1])


@pytest.mark.parametrize("seed", [3, 11])
def test_synthetic_structure_and_strict_parity(seed):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(120, 8))
    X[:, 5:7] = 1.0
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
    scratch = RandomForest(n_estimators=8, max_depth=5, min_samples_leaf=2,
                           random_state=seed).fit(X, y)
    reference = scratch.reference().fit(X, y)
    for mine, theirs in zip(scratch.trees_, reference.estimators_):
        assert np.array_equal(mine.tree_.feature, theirs.tree_.feature)
        assert np.array_equal(mine.tree_.threshold, theirs.tree_.threshold)
    assert_parity(scratch, reference, X, y, task="classification")


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
@pytest.mark.parametrize("n_estimators,max_depth", [(50, 3), (100, 5)])
def test_real_train_and_validation_parity(variant, n_estimators, max_depth):
    data = load(variant)
    scratch = RandomForest(n_estimators=n_estimators, max_depth=max_depth).fit(
        data.train.X, data.train.y_classification)
    reference = scratch.reference().fit(data.train.X, data.train.y_classification)
    for mine, theirs in zip(scratch.trees_, reference.estimators_):
        assert np.array_equal(mine.tree_.feature, theirs.tree_.feature)
        assert np.array_equal(mine.tree_.threshold, theirs.tree_.threshold)
    assert_parity(scratch, reference, data.train.X, data.train.y_classification, task="classification")
    assert_parity(scratch, reference, data.validation.X, data.validation.y_classification, task="classification")


@pytest.mark.parametrize("variant,n_estimators,max_depth", [
    ("with_spike", 100, 3), ("non_spike", 50, 5),
])
@pytest.mark.parametrize("components", [2, 4, 6])
def test_pca_train_and_validation_parity(variant, n_estimators, max_depth, components):
    data = load(variant)
    standardizer = Standardizer().fit(data.train.X)
    train_scaled = standardizer.transform(data.train.X)
    pca = PCA(components).fit(train_scaled)
    train_X = pca.transform(train_scaled)
    validation_X = pca.transform(standardizer.transform(data.validation.X))
    scratch = RandomForest(n_estimators=n_estimators, max_depth=max_depth).fit(
        train_X, data.train.y_classification)
    reference = scratch.reference().fit(train_X, data.train.y_classification)
    assert_parity(scratch, reference, train_X, data.train.y_classification, task="classification")
    assert_parity(scratch, reference, validation_X, data.validation.y_classification, task="classification")


def test_save_load_preserves_probabilities(tmp_path):
    X = np.arange(24.0).reshape(12, 2)
    y = np.array([0, 1] * 6)
    original = RandomForest(n_estimators=3, max_depth=3, min_samples_leaf=1).fit(X, y)
    path = tmp_path / "forest.npz"
    save_npz(path, original.state_dict(), {"params": original.get_params()})
    arrays, metadata = load_npz(path)
    restored = RandomForest().load_state(arrays, metadata)
    assert restored.get_params() == original.get_params()
    assert np.array_equal(restored.predict_proba(X), original.predict_proba(X))
    assert np.array_equal(restored.predict(X), original.predict(X))


def test_registry_exposes_forest():
    assert "random_forest" in model_names()
    assert isinstance(build("random_forest"), RandomForest)
