"""Second-order regularised boosting worked by hand, then tree-by-tree xgboost parity."""

import json

import numpy as np
import pytest

from src.ml.classification.gradient_boosting import GradientBoosting
from src.ml.classification.xgboost import XGBoost
from src.ml.core.compare import assert_parity
from src.ml.core.data import load
from src.ml.core.persist import load_npz, save_npz
from src.ml.registry import build, model_names

HAND_X = np.array([[1.0], [2.0], [3.0], [4.0]])
BALANCED = np.array([0, 0, 1, 1])


def _one_round(y=BALANCED, **kwargs):
    params = {"n_estimators": 1, "max_depth": 1, "learning_rate": 1.0, "reg_lambda": 1.0, "gamma": 0.0}
    return XGBoost(**{**params, **kwargs}).fit(HAND_X, y)


def test_one_split_matches_hand_computation():
    # F0 = 0 -> p = 1/2, so g = p - y = [1/2, 1/2, -1/2, -1/2] and h = 1/4 each.
    # Root: G = 0, H = 1 -> score 0. Split at 2.5 with lambda = 1:
    #   left : G = 1,  H = 1/2 -> score 1 / 1.5 = 2/3, w* = -1 / 1.5 = -2/3
    #   right: G = -1, H = 1/2 -> score 2/3,           w* =  2/3
    #   gain = 2/3 + 2/3 - 0 = 4/3  (the library omits the textbook factor 1/2)
    model = _one_round()
    assert model.threshold_[0] == 2.5
    assert np.isclose(model.loss_change_[0], 4 / 3, rtol=1e-6)
    assert np.allclose(model.decision_function(HAND_X), [-2 / 3, -2 / 3, 2 / 3, 2 / 3], rtol=1e-6)
    assert np.array_equal(model.predict(HAND_X), BALANCED)


def test_initial_score_is_zero_not_prior_log_odds():
    # base_score = 0.5 means F0 = 0 even when the classes are unbalanced. With every
    # split pruned the root answers alone: g = [1/2, 1/2, 1/2, -1/2] -> G = 1, H = 1,
    # so w* = -G / (H + 1) = -1/2. Starting from the prior would give ln(1/3) instead.
    model = XGBoost(n_estimators=1, max_depth=1, learning_rate=1.0, gamma=10.0).fit(
        HAND_X, np.array([0, 0, 0, 1]))
    assert model.tree_sizes_.tolist() == [1]
    assert np.allclose(model.decision_function(HAND_X), -0.5, rtol=1e-6)


def test_lambda_shrinks_leaf_weights():
    plain = _one_round(reg_lambda=0.0)
    ridge = _one_round(reg_lambda=10.0)
    # w* = -G / (H + lambda): 1 / 0.5 = 2 without the penalty, 1 / 10.5 with it.
    assert np.allclose(np.abs(plain.decision_function(HAND_X)), 2.0, rtol=1e-6)
    assert np.allclose(np.abs(ridge.decision_function(HAND_X)), 1 / 10.5, rtol=1e-6)


def test_without_regularisation_equals_gradient_boosting_newton_step():
    # Balanced classes make both models start from F0 = 0, so they must agree exactly.
    newton = GradientBoosting(n_estimators=1, learning_rate=1.0, max_depth=1,
                              min_samples_leaf=1).fit(HAND_X, BALANCED)
    model = _one_round(reg_lambda=0.0)
    assert np.allclose(model.decision_function(HAND_X), newton.decision_function(HAND_X), rtol=1e-6)


def test_gamma_prunes_a_split_that_gains_too_little():
    # The only split gains 4/3: it survives gamma = 1 and is pruned at gamma = 2.
    assert _one_round(gamma=1.0).tree_sizes_.tolist() == [3]
    pruned = _one_round(gamma=2.0)
    assert pruned.tree_sizes_.tolist() == [1]
    assert np.array_equal(pruned.decision_function(HAND_X), np.zeros(4))


def test_gamma_prunes_bottom_up_only():
    # 14 rows: G = -1, H = 3.5 at the root. Splitting on x0 gains only
    # 0 + 1/1.5 - 1/3.5 = 0.381, but its children then split on x1 for 8 and 16/3.
    X = np.array([[0, 0]] * 4 + [[0, 1]] * 4 + [[1, 0]] * 4 + [[1, 1]] * 2, dtype=float)
    y = np.array([0] * 4 + [1] * 4 + [1] * 4 + [0] * 2)

    def fit(gamma):
        return XGBoost(n_estimators=1, max_depth=2, learning_rate=1.0, reg_lambda=0.0, gamma=gamma).fit(X, y)

    assert np.allclose(fit(0.0).loss_change_[:3], [8 / 21, 8.0, 16 / 3], rtol=1e-6)
    # The weak root survives gamma = 0.5 because both subtrees below it are kept.
    assert fit(0.5).tree_sizes_.tolist() == [7]
    # gamma = 6 removes the 16/3 split; the root still has a split child, so it stays.
    assert fit(6.0).tree_sizes_.tolist() == [5]
    # Once both children are leaves the root's own 0.381 is finally tested, and it goes.
    assert fit(9.0).tree_sizes_.tolist() == [1]


def test_learning_rate_shrinks_each_leaf_once():
    full = _one_round()
    tenth = _one_round(learning_rate=0.1)
    assert np.allclose(tenth.decision_function(HAND_X), 0.1 * full.decision_function(HAND_X), rtol=1e-6)


def test_history_is_training_log_loss():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=80) > 0).astype(int)
    model = XGBoost(n_estimators=5, max_depth=2).fit(X, y)
    score = model.decision_function(X)
    assert model.history_["iteration"] == [1, 2, 3, 4, 5]
    assert np.isclose(model.history_["loss"][-1], np.mean(np.logaddexp(0.0, score) - y * score), rtol=1e-12)
    assert all(a > b for a, b in zip(model.history_["loss"], model.history_["loss"][1:]))


def test_readonly_input_is_preserved():
    X = np.arange(40.0).reshape(20, 2)
    X.setflags(write=False)
    XGBoost(n_estimators=3).fit(X, np.array([0, 1] * 10))
    assert not X.flags.writeable


@pytest.mark.parametrize("kwargs", [
    {"n_estimators": 0}, {"max_depth": 0}, {"learning_rate": 0.0},
    {"reg_lambda": -1.0}, {"gamma": -0.1}, {"random_state": -1},
])
def test_invalid_hyperparameters_fail(kwargs):
    with pytest.raises(ValueError):
        XGBoost(**kwargs).fit(HAND_X, BALANCED)


def test_invalid_training_data_fail():
    with pytest.raises(ValueError):
        XGBoost().fit([[0], [1]], [0, 2])
    with pytest.raises(ValueError):
        XGBoost().fit([[0], [np.nan]], [0, 1])


def _reference_trees(reference):
    """Each reference tree as (feature, threshold, leaf value) lists in preorder."""
    trees = []
    for dump in reference.get_booster().get_dump(dump_format="json"):
        feature, threshold, value = [], [], []

        def walk(node):
            if "leaf" in node:
                feature.append(-1)
                threshold.append(0.0)
                value.append(node["leaf"])
                return
            feature.append(int(node["split"][1:]))
            threshold.append(node["split_condition"])
            value.append(0.0)
            children = {child["nodeid"]: child for child in node["children"]}
            walk(children[node["yes"]])
            walk(children[node["no"]])

        walk(json.loads(dump))
        trees.append((feature, threshold, value))
    return trees


def _scratch_trees(scratch):
    trees, offset = [], 0
    for size in scratch.tree_sizes_:
        nodes = slice(offset, offset + size)
        feature, threshold = scratch.feature_[nodes], scratch.threshold_[nodes]
        left, right, value = scratch.children_left_[nodes], scratch.children_right_[nodes], scratch.value_[nodes]
        out = ([], [], [])

        def walk(node):
            leaf = feature[node] < 0
            out[0].append(int(feature[node]))
            out[1].append(0.0 if leaf else float(threshold[node]))
            out[2].append(float(value[node]) if leaf else 0.0)
            if not leaf:
                walk(left[node])
                walk(right[node])

        walk(0)
        trees.append(out)
        offset += size
    return trees


def _assert_trees_match(scratch, reference, X):
    """Tree by tree, so a mismatch names the first round that went wrong."""
    theirs = _reference_trees(reference)
    mine = _scratch_trees(scratch)
    assert len(mine) == len(theirs)
    for stage, (a, b) in enumerate(zip(mine, theirs)):
        assert a[0] == b[0], f"tree {stage}: features"
        assert np.allclose(a[1], b[1], rtol=1e-6, atol=0), f"tree {stage}: thresholds"
        assert np.allclose(a[2], b[2], rtol=1e-5, atol=1e-9), f"tree {stage}: leaf values"
    for stage, margin in enumerate(scratch.staged_decision_function(X), start=1):
        expected = reference.predict(X, output_margin=True, iteration_range=(0, stage))
        assert np.allclose(margin, expected, rtol=1e-5, atol=1e-7), f"round {stage}: margin"


FIELDS = ("n_estimators", "max_depth", "learning_rate", "reg_lambda", "gamma")
GRID = [(50, 1, 0.1, 1, 0), (50, 2, 0.1, 1, 0), (100, 1, 0.03, 1, 0), (100, 2, 0.03, 1, 0),
        (100, 2, 0.1, 1, 0), (100, 2, 0.1, 10, 0), (100, 2, 0.1, 1, 0.1), (100, 2, 0.03, 10, 0.1)]


@pytest.mark.parametrize("seed", [0, 3, 11])
@pytest.mark.parametrize("max_depth,reg_lambda,gamma", [(1, 1.0, 0.0), (2, 10.0, 0.1), (3, 0.0, 0.5)])
def test_synthetic_tree_by_tree_and_strict_parity(seed, max_depth, reg_lambda, gamma):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * rng.normal(size=300) > 0).astype(int)
    train_X, train_y, held_X, held_y = X[:200], y[:200], X[200:], y[200:]
    scratch = XGBoost(n_estimators=40, max_depth=max_depth, learning_rate=0.1,
                      reg_lambda=reg_lambda, gamma=gamma).fit(train_X, train_y)
    reference = scratch.reference().fit(train_X, train_y)
    _assert_trees_match(scratch, reference, train_X)
    assert_parity(scratch, reference, train_X, train_y, task="classification")
    assert_parity(scratch, reference, held_X, held_y, task="classification")


def test_tied_and_constant_features_follow_reference_order():
    # Duplicate columns tie exactly on gain; xgboost keeps the lower feature index.
    rng = np.random.RandomState(8)
    base = rng.normal(size=(150, 3))
    X = np.column_stack([base[:, 0], base[:, 0], np.ones(150), base[:, 1], base[:, 1], base[:, 2]])
    y = (base[:, 0] - base[:, 1] + 0.4 * rng.normal(size=150) > 0).astype(int)
    scratch = XGBoost(n_estimators=30, max_depth=2).fit(X, y)
    reference = scratch.reference().fit(X, y)
    _assert_trees_match(scratch, reference, X)
    assert_parity(scratch, reference, X, y, task="classification")


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
@pytest.mark.parametrize("candidate", [GRID[0], GRID[5], GRID[6], GRID[7]])
def test_real_train_and_validation_parity(variant, candidate):
    data = load(variant)
    scratch = XGBoost(**dict(zip(FIELDS, candidate))).fit(data.train.X, data.train.y_classification)
    reference = scratch.reference().fit(data.train.X, data.train.y_classification)
    _assert_trees_match(scratch, reference, data.train.X)
    assert_parity(scratch, reference, data.train.X, data.train.y_classification, task="classification")
    assert_parity(scratch, reference, data.validation.X, data.validation.y_classification, task="classification")


def test_save_load_preserves_scores(tmp_path):
    rng = np.random.RandomState(2)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    original = XGBoost(n_estimators=6, max_depth=2, learning_rate=0.03, reg_lambda=10.0, gamma=0.1).fit(X, y)
    path = tmp_path / "xgboost.npz"
    save_npz(path, original.state_dict(), {"params": original.get_params()})
    arrays, metadata = load_npz(path)
    restored = XGBoost().load_state(arrays, metadata)
    assert restored.get_params() == original.get_params()
    assert np.array_equal(restored.decision_function(X), original.decision_function(X))
    assert np.array_equal(restored.predict_proba(X), original.predict_proba(X))
    assert np.array_equal(restored.predict(X), original.predict(X))


def test_registry_exposes_xgboost():
    assert "xgboost" in model_names()
    assert isinstance(build("xgboost"), XGBoost)
