"""Newton-step boosting worked by hand, then stage-by-stage sklearn parity."""

import numpy as np
import pytest

from src.ml.classification.gradient_boosting import GradientBoosting, _log_odds
from src.ml.core.compare import assert_parity
from src.ml.core.data import load
from src.ml.core.persist import load_npz, save_npz
from src.ml.registry import build, model_names

HAND_X = np.array([[1.0], [2.0], [3.0], [4.0]])


def _one_round(y, **kwargs):
    return GradientBoosting(n_estimators=1, learning_rate=1.0, max_depth=1,
                            min_samples_leaf=1, **kwargs).fit(HAND_X, y)


def test_initial_score_is_prior_log_odds():
    model = _one_round(np.array([0, 0, 0, 1]))
    # p = 1/4 -> F0 = ln(p / (1 - p)) = -ln 3, not 0.
    assert np.isclose(model.init_, -np.log(3), rtol=1e-12)


def test_initial_score_matches_reference_to_the_last_bit():
    from scipy.special import logit

    # sklearn starts from scipy's logit, which switches formula near p = 0.5. Being one
    # ulp off there changed which tied feature a depth-3 tree picked 36 rounds later.
    fractions = np.unique(np.concatenate([np.arange(1, n) / n for n in range(2, 200)]))
    assert all(_log_odds(float(p)) == logit(p) for p in fractions)


def test_one_round_matches_hand_computation():
    # p = 1/4, residual = y - p = [-1/4, -1/4, -1/4, 3/4]. Squared error splits at 3.5.
    #   left : sum(residual) = -3/4, sum(p(1-p)) = 3 * 3/16  -> gamma = -4/3
    #   right: sum(residual) =  3/4, sum(p(1-p)) =     3/16  -> gamma =  4
    # The mean residual would give -1/4 and 3/4 instead, which is the classic mistake.
    model = _one_round(np.array([0, 0, 0, 1]))
    assert model.threshold_[0] == 3.5
    assert np.allclose(sorted(model.value_[model.feature_ < 0]), [-4 / 3, 4.0], rtol=1e-12)
    expected = -np.log(3) + np.array([-4 / 3, -4 / 3, -4 / 3, 4.0])
    assert np.allclose(model.decision_function(HAND_X), expected, rtol=1e-12)
    assert np.array_equal(model.predict(HAND_X), [0, 0, 0, 1])


def test_balanced_classes_start_at_zero():
    # p = 1/2 -> F0 = 0, residual = +-1/2, gamma = (2 * +-1/2) / (2 * 1/4) = +-2.
    model = _one_round(np.array([0, 0, 1, 1]))
    assert model.init_ == 0.0
    assert np.allclose(model.decision_function(HAND_X), [-2.0, -2.0, 2.0, 2.0], rtol=1e-12)


def test_learning_rate_shrinks_each_leaf_once():
    full = _one_round(np.array([0, 0, 0, 1]))
    half = GradientBoosting(n_estimators=1, learning_rate=0.5, max_depth=1,
                            min_samples_leaf=1).fit(HAND_X, np.array([0, 0, 0, 1]))
    assert np.allclose(half.decision_function(HAND_X) - half.init_,
                       0.5 * (full.decision_function(HAND_X) - full.init_), rtol=1e-12)


def test_tree_splits_on_squared_error_not_gini():
    # The labels are symmetric, so Gini can never prefer 1.5 over 5.5. Whichever end
    # round 1 isolates, its residual shrinks and squared error must take the other end.
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1, 0, 0, 0, 0, 1])
    model = GradientBoosting(n_estimators=2, learning_rate=1.0, max_depth=1,
                             min_samples_leaf=1).fit(X, y)
    assert sorted([model.threshold_[0], model.threshold_[3]]) == [1.5, 5.5]


def test_min_samples_leaf_blocks_small_children():
    model = GradientBoosting(n_estimators=1, max_depth=2, min_samples_leaf=2).fit(
        HAND_X, np.array([0, 0, 0, 1]))
    # Only 2.5 leaves two rows on each side, and neither child can split again.
    assert model.tree_sizes_.tolist() == [3]
    assert model.threshold_[0] == 2.5


def test_history_is_training_log_loss():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=80) > 0).astype(int)
    model = GradientBoosting(n_estimators=5).fit(X, y)
    score = model.decision_function(X)
    final = np.mean(np.log1p(np.exp(score)) - y * score)
    assert model.history_["iteration"] == [1, 2, 3, 4, 5]
    assert np.isclose(model.history_["loss"][-1], final, rtol=1e-12)
    assert all(a > b for a, b in zip(model.history_["loss"], model.history_["loss"][1:]))


def test_readonly_input_is_preserved():
    X = np.arange(40.0).reshape(20, 2)
    X.setflags(write=False)
    GradientBoosting(n_estimators=3).fit(X, np.array([0, 1] * 10))
    assert not X.flags.writeable


@pytest.mark.parametrize("kwargs", [
    {"n_estimators": 0}, {"learning_rate": 0.0}, {"max_depth": 0},
    {"min_samples_leaf": 0}, {"random_state": -1},
])
def test_invalid_hyperparameters_fail(kwargs):
    with pytest.raises(ValueError):
        GradientBoosting(**kwargs).fit(HAND_X, np.array([0, 0, 0, 1]))


def test_invalid_training_data_fail():
    with pytest.raises(ValueError):
        GradientBoosting().fit([[0], [1]], [0, 2])
    with pytest.raises(ValueError):
        GradientBoosting().fit([[0], [np.nan]], [0, 1])
    with pytest.raises(ValueError):
        GradientBoosting().fit([[0], [1]], [1, 1])


def _assert_stages_match(scratch, reference, X):
    """Stage by stage, so a mismatch names the first tree that went wrong."""
    offset = 0
    for stage, theirs in enumerate(reference.estimators_[:, 0]):
        size = int(scratch.tree_sizes_[stage])
        assert size == theirs.tree_.node_count, f"stage {stage}: node count"
        mine = slice(offset, offset + size)
        assert np.array_equal(scratch.feature_[mine], theirs.tree_.feature), f"stage {stage}: feature"
        assert np.array_equal(scratch.threshold_[mine], theirs.tree_.threshold), f"stage {stage}: threshold"
        offset += size
    for stage, (mine, theirs) in enumerate(zip(scratch.staged_decision_function(X),
                                               reference.staged_decision_function(X))):
        assert np.allclose(mine, theirs.ravel(), rtol=1e-9, atol=1e-12), f"stage {stage}: score"
    # sklearn's train_score_ is the binomial deviance, which is twice the log loss.
    assert np.allclose(2 * np.asarray(scratch.history_["loss"]), reference.train_score_, rtol=1e-9)


@pytest.mark.parametrize("seed", [0, 3, 11])
@pytest.mark.parametrize("max_depth,learning_rate", [(1, 0.1), (2, 0.03), (3, 0.1)])
def test_synthetic_stage_by_stage_and_strict_parity(seed, max_depth, learning_rate):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * rng.normal(size=300) > 0).astype(int)
    train_X, train_y, held_X, held_y = X[:200], y[:200], X[200:], y[200:]
    scratch = GradientBoosting(n_estimators=40, learning_rate=learning_rate, max_depth=max_depth,
                               random_state=seed).fit(train_X, train_y)
    reference = scratch.reference().fit(train_X, train_y)
    _assert_stages_match(scratch, reference, train_X)
    assert_parity(scratch, reference, train_X, train_y, task="classification")
    assert_parity(scratch, reference, held_X, held_y, task="classification")


def test_tied_and_constant_features_follow_reference_order():
    # Duplicate columns tie exactly on gain, so only sklearn's visiting order decides.
    rng = np.random.RandomState(8)
    base = rng.normal(size=(150, 3))
    X = np.column_stack([base[:, 0], base[:, 0], np.ones(150), base[:, 1], base[:, 1], base[:, 2]])
    y = (base[:, 0] - base[:, 1] + 0.4 * rng.normal(size=150) > 0).astype(int)
    scratch = GradientBoosting(n_estimators=30, max_depth=2).fit(X, y)
    reference = scratch.reference().fit(X, y)
    _assert_stages_match(scratch, reference, X)
    assert_parity(scratch, reference, X, y, task="classification")


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
@pytest.mark.parametrize("n_estimators,max_depth,learning_rate", [(50, 1, 0.03), (100, 2, 0.1)])
def test_real_train_and_validation_parity(variant, n_estimators, max_depth, learning_rate):
    data = load(variant)
    scratch = GradientBoosting(n_estimators=n_estimators, max_depth=max_depth,
                               learning_rate=learning_rate).fit(data.train.X, data.train.y_classification)
    reference = scratch.reference().fit(data.train.X, data.train.y_classification)
    _assert_stages_match(scratch, reference, data.train.X)
    assert_parity(scratch, reference, data.train.X, data.train.y_classification, task="classification")
    assert_parity(scratch, reference, data.validation.X, data.validation.y_classification, task="classification")


def test_save_load_preserves_scores(tmp_path):
    rng = np.random.RandomState(2)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    original = GradientBoosting(n_estimators=6, max_depth=2, learning_rate=0.03).fit(X, y)
    path = tmp_path / "gradient_boosting.npz"
    save_npz(path, original.state_dict(), {"params": original.get_params()})
    arrays, metadata = load_npz(path)
    restored = GradientBoosting().load_state(arrays, metadata)
    assert restored.get_params() == original.get_params()
    assert np.array_equal(restored.decision_function(X), original.decision_function(X))
    assert np.array_equal(restored.predict_proba(X), original.predict_proba(X))
    assert np.array_equal(restored.predict(X), original.predict(X))


def test_registry_exposes_gradient_boosting():
    assert "gradient_boosting" in model_names()
    assert isinstance(build("gradient_boosting"), GradientBoosting)
