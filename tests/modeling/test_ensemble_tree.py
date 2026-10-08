"""Scratch CART tree primitive shared by Random Forest and Gradient Boosting."""
import numpy as np
import pytest

from src.modeling.ensemble._tree import apply_tree, build_tree, predict_tree, tree_depth


def test_gini_tree_hand_fixture():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    t = np.array([0.0, 0.0, 1.0, 1.0])
    tree = build_tree(X, t, criterion="gini", max_depth=3, min_samples_leaf=1)
    assert tree["feature"][0] == 0 and tree["threshold"][0] == 2.5
    np.testing.assert_array_equal(predict_tree(tree, X), t)


def test_mse_tree_leaf_means_and_depth_limit():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    t = X[:, 1] * 2.0
    tree = build_tree(X, t, criterion="mse", max_depth=2, min_samples_leaf=1)
    assert tree_depth(tree) <= 2
    leaves = apply_tree(tree, X)
    for leaf in np.unique(leaves):
        assert predict_tree(tree, X[leaves == leaf])[0] == pytest.approx(t[leaves == leaf].mean())


def test_min_samples_leaf_enforced():
    X, t = np.random.default_rng(1).standard_normal((60, 2)), np.random.default_rng(2).integers(0, 2, 60).astype(float)
    tree = build_tree(X, t, criterion="gini", max_depth=6, min_samples_leaf=5)
    counts = np.bincount(apply_tree(tree, X))
    assert counts[counts > 0].min() >= 5


def test_feature_subset_sampler_is_deterministic_and_restricts_splits():
    X, t = np.random.default_rng(3).standard_normal((100, 6)), np.random.default_rng(4).integers(0, 2, 100).astype(float)
    kw = dict(criterion="gini", max_depth=3, min_samples_leaf=1, max_features=1)
    a = build_tree(X, t, rng=np.random.default_rng(7), **kw)
    b = build_tree(X, t, rng=np.random.default_rng(7), **kw)
    c = build_tree(X, t, rng=np.random.default_rng(8), **kw)
    assert a == b and a != c


def test_constant_features_make_a_leaf():
    tree = build_tree(np.ones((10, 2)), np.arange(10) % 2.0, criterion="gini", max_depth=3, min_samples_leaf=1)
    assert len(tree["feature"]) == 1 and tree["feature"][0] == -1
