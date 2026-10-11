"""Discrete SAMME mechanics worked by hand, then strict sklearn parity."""

import numpy as np
import pytest

from src.ml.classification.adaboost import AdaBoost
from src.ml.classification.decision_tree import DecisionTree
from src.ml.core.compare import assert_parity
from src.ml.core.data import load
from src.ml.core.persist import load_npz, save_npz
from src.ml.registry import build, model_names

# Four points no single stump can separate. Worked by hand:
#   round 1: w = 1/4 each; split at 2.5 leaves a 1-1 tie on the left, which resolves
#            to class 0, so the stump predicts 0 everywhere and misses only row 1.
#            err = 1/4, alpha = ln 3, w -> [1/6, 1/2, 1/6, 1/6]
#   round 2: same split, but the left side is now 3/4 positive by weight, so it
#            predicts [1, 1, 0, 0] and misses only row 0.
#            err = 1/6, alpha = ln 5, w -> [1/2, 3/10, 1/10, 1/10]
HAND_X = np.array([[1.0], [2.0], [3.0], [4.0]])
HAND_Y = np.array([0, 1, 0, 0])


def test_two_rounds_match_hand_computation():
    model = AdaBoost(n_estimators=2, learning_rate=1.0).fit(HAND_X, HAND_Y)
    assert [stump.tree_.threshold[0] for stump in model.stumps_] == [2.5, 2.5]
    assert np.allclose(model.estimator_errors_, [1 / 4, 1 / 6], rtol=1e-12)
    assert np.allclose(model.estimator_weights_, [np.log(3), np.log(5)], rtol=1e-12)
    assert np.allclose(model.sample_weight_, [0.5, 0.3, 0.1, 0.1], rtol=1e-12)
    assert np.array_equal(model.predict(HAND_X), [1, 1, 0, 0])
    # F = sum(alpha * h) with h in {-1, +1}, reported as 2F / sum(alpha) like sklearn.
    margin = np.array([np.log(5 / 3), np.log(5 / 3), -np.log(15), -np.log(15)])
    assert np.allclose(model.decision_function(HAND_X), 2 * margin / np.log(15), rtol=1e-12)


def test_history_is_the_product_of_round_normalisers():
    model = AdaBoost(n_estimators=2, learning_rate=1.0).fit(HAND_X, HAND_Y)
    # Z_m = 2 * sqrt(err * (1 - err)); the exponential loss after m rounds is prod(Z).
    first = 2 * np.sqrt(1 / 4 * 3 / 4)
    second = first * 2 * np.sqrt(1 / 6 * 5 / 6)
    assert model.history_["iteration"] == [1, 2]
    assert np.allclose(model.history_["loss"], [first, second], rtol=1e-12)


def test_learning_rate_scales_alpha_once():
    model = AdaBoost(n_estimators=1, learning_rate=0.5).fit(HAND_X, HAND_Y)
    assert np.allclose(model.estimator_weights_, [0.5 * np.log(3)], rtol=1e-12)
    # exp(0.5 * ln 3) = sqrt(3) on the one missed row.
    expected = np.array([1.0, np.sqrt(3), 1.0, 1.0])
    assert np.allclose(model.sample_weight_, expected / expected.sum(), rtol=1e-12)


@pytest.mark.parametrize("n_estimators", [1, 2, 7])
def test_weights_stay_normalised(n_estimators):
    rng = np.random.RandomState(5)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    model = AdaBoost(n_estimators=n_estimators).fit(X, y)
    assert np.isclose(model.sample_weight_.sum(), 1.0, rtol=1e-12)
    assert (model.sample_weight_ > 0).all()


def test_perfect_stump_stops_without_infinite_alpha():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    model = AdaBoost(n_estimators=10).fit(X, y)
    assert model.status_ == "completed_perfect"
    assert len(model.stumps_) == 1
    assert np.array_equal(model.estimator_errors_, [0.0])
    assert np.array_equal(model.estimator_weights_, [1.0])
    assert np.isfinite(model.decision_function(X)).all()
    assert np.array_equal(model.predict(X), y)


def test_first_stump_no_better_than_chance_fails():
    with pytest.raises(ValueError, match="worse than random"):
        AdaBoost().fit(np.ones((4, 1)), np.array([0, 1, 0, 1]))


def test_single_round_equals_one_stump():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(80, 4))
    y = (X[:, 2] + 0.3 * rng.normal(size=80) > 0).astype(int)
    model = AdaBoost(n_estimators=1, random_state=7).fit(X, y)
    seed = int(np.random.RandomState(7).randint(np.iinfo(np.int32).max))
    stump = DecisionTree(max_depth=1, min_samples_leaf=1, random_state=seed).fit(X, y)
    assert model.status_ == "completed"
    assert np.array_equal(model.predict(X), stump.predict(X))


def test_readonly_input_is_preserved():
    X = np.arange(20.0).reshape(10, 2)
    X.setflags(write=False)
    AdaBoost(n_estimators=3).fit(X, np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0]))
    assert not X.flags.writeable


@pytest.mark.parametrize("kwargs", [
    {"n_estimators": 0}, {"learning_rate": 0.0}, {"learning_rate": -1.0}, {"random_state": -1},
])
def test_invalid_hyperparameters_fail(kwargs):
    with pytest.raises(ValueError):
        AdaBoost(**kwargs).fit(HAND_X, HAND_Y)


def test_invalid_training_data_fail():
    with pytest.raises(ValueError):
        AdaBoost().fit([[0], [1]], [0, 2])
    with pytest.raises(ValueError):
        AdaBoost().fit([[0], [np.nan]], [0, 1])


def _assert_boosting_matches(scratch, reference):
    """Compare round by round: a label-only check hides a wrong alpha behind a right sign."""
    rounds = len(scratch.stumps_)
    assert rounds == len(reference.estimators_)
    for mine, theirs in zip(scratch.stumps_, reference.estimators_):
        assert np.array_equal(mine.tree_.feature, theirs.tree_.feature)
        assert np.array_equal(mine.tree_.threshold, theirs.tree_.threshold)
    assert np.allclose(scratch.estimator_errors_, reference.estimator_errors_[:rounds],
                       rtol=1e-9, atol=0)
    assert np.allclose(scratch.estimator_weights_, reference.estimator_weights_[:rounds],
                       rtol=1e-9, atol=0)


def test_reference_has_no_algorithm_argument():
    # sklearn 1.9 removed `algorithm`; SAMME is the only behaviour left.
    assert "algorithm" not in AdaBoost().reference().get_params()


@pytest.mark.parametrize("seed", [0, 3, 11])
@pytest.mark.parametrize("learning_rate", [0.5, 1.0])
def test_synthetic_round_by_round_and_strict_parity(seed, learning_rate):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * rng.normal(size=300) > 0).astype(int)
    train_X, train_y, held_X, held_y = X[:200], y[:200], X[200:], y[200:]
    scratch = AdaBoost(n_estimators=40, learning_rate=learning_rate, random_state=seed).fit(train_X, train_y)
    reference = scratch.reference().fit(train_X, train_y)
    _assert_boosting_matches(scratch, reference)
    assert_parity(scratch, reference, train_X, train_y, task="classification")
    assert_parity(scratch, reference, held_X, held_y, task="classification")


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
@pytest.mark.parametrize("n_estimators,learning_rate", [(25, 0.5), (100, 1.0)])
def test_real_train_and_validation_parity(variant, n_estimators, learning_rate):
    data = load(variant)
    scratch = AdaBoost(n_estimators=n_estimators, learning_rate=learning_rate).fit(
        data.train.X, data.train.y_classification)
    reference = scratch.reference().fit(data.train.X, data.train.y_classification)
    _assert_boosting_matches(scratch, reference)
    assert_parity(scratch, reference, data.train.X, data.train.y_classification, task="classification")
    assert_parity(scratch, reference, data.validation.X, data.validation.y_classification, task="classification")


def test_save_load_preserves_scores(tmp_path):
    rng = np.random.RandomState(2)
    X = rng.normal(size=(50, 3))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    original = AdaBoost(n_estimators=6, learning_rate=0.5).fit(X, y)
    path = tmp_path / "adaboost.npz"
    save_npz(path, original.state_dict(), {"params": original.get_params()})
    arrays, metadata = load_npz(path)
    restored = AdaBoost().load_state(arrays, metadata)
    assert restored.get_params() == original.get_params()
    assert np.array_equal(restored.decision_function(X), original.decision_function(X))
    assert np.array_equal(restored.predict_proba(X), original.predict_proba(X))
    assert np.array_equal(restored.predict(X), original.predict(X))


def test_registry_exposes_adaboost():
    assert "adaboost" in model_names()
    assert isinstance(build("adaboost"), AdaBoost)
