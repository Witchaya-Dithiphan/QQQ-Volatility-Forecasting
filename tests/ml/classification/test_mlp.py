"""Backpropagation checked against finite differences, then weight-level sklearn parity."""

import warnings

import numpy as np
import pytest

from src.ml.classification.mlp import MLP
from src.ml.core.compare import assert_parity
from src.ml.core.data import load
from src.ml.core.persist import load_npz, save_npz
from src.ml.core.preprocess import Standardizer
from src.ml.registry import build, model_names


def _problem(seed=0, rows=120, columns=4):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(rows, columns))
    y = (X[:, 0] * X[:, 1] + 0.3 * rng.normal(size=rows) > 0).astype(np.float64)
    return X, y


@pytest.mark.parametrize("hidden_layers", [[5], [6, 3]])
@pytest.mark.parametrize("l2", [0.0, 0.5])
def test_backprop_matches_numerical_gradient_in_every_layer(hidden_layers, l2):
    X, y = _problem(rows=7)
    model = MLP(hidden_layers=hidden_layers, l2=l2)
    coefs, intercepts = model._initial_parameters(X.shape[1])
    _, coef_grads, intercept_grads = model._loss_and_gradients(X, y, coefs, intercepts)

    def loss():
        return model._loss_and_gradients(X, y, coefs, intercepts)[0]

    step = 1e-6
    for layer, (group, grads) in enumerate([(c, g) for c, g in zip(coefs, coef_grads)]
                                           + [(b, g) for b, g in zip(intercepts, intercept_grads)]):
        numerical = np.empty_like(group)
        for index in np.ndindex(group.shape):
            original = group[index]
            group[index] = original + step
            above = loss()
            group[index] = original - step
            below = loss()
            group[index] = original
            numerical[index] = (above - below) / (2 * step)
        assert np.allclose(grads, numerical, rtol=1e-5, atol=1e-8), f"parameter group {layer}"


def test_relu_derivative_is_zero_at_exactly_zero():
    # One hidden unit whose pre-activation is exactly 0: no gradient may flow through it.
    model = MLP(hidden_layers=[1])
    coefs = [np.array([[0.0]]), np.array([[1.0]])]
    intercepts = [np.array([0.0]), np.array([0.0])]
    _, coef_grads, intercept_grads = model._loss_and_gradients(
        np.array([[3.0]]), np.array([1.0]), coefs, intercepts)
    assert coef_grads[0][0, 0] == 0.0
    assert intercept_grads[0][0] == 0.0


def test_l2_is_divided_by_the_batch_and_spares_the_intercepts():
    X, y = _problem(rows=10)
    plain, ridge = MLP(hidden_layers=[3], l2=0.0), MLP(hidden_layers=[3], l2=2.0)
    coefs, intercepts = plain._initial_parameters(X.shape[1])
    _, plain_coef, plain_intercept = plain._loss_and_gradients(X, y, coefs, intercepts)
    _, ridge_coef, ridge_intercept = ridge._loss_and_gradients(X, y, coefs, intercepts)
    for weights, without, with_penalty in zip(coefs, plain_coef, ridge_coef):
        assert np.allclose(with_penalty - without, 2.0 * weights / len(X), rtol=1e-12)
    for without, with_penalty in zip(plain_intercept, ridge_intercept):
        assert np.array_equal(without, with_penalty)


def test_initial_weights_follow_glorot_bounds_layer_by_layer():
    model = MLP(hidden_layers=[16, 8], random_state=7)
    coefs, intercepts = model._initial_parameters(8)
    rng = np.random.RandomState(7)
    for (fan_in, fan_out), weights, bias in zip([(8, 16), (16, 8), (8, 1)], coefs, intercepts):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        # Weights then biases, one layer at a time, from a single legacy stream.
        assert np.array_equal(weights, rng.uniform(-bound, bound, (fan_in, fan_out)))
        assert np.array_equal(bias, rng.uniform(-bound, bound, fan_out))


def test_hidden_layer_solves_xor():
    # No linear model can do this; getting every row right needs working backprop.
    X = np.tile(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), (50, 1))
    y = np.tile(np.array([0, 1, 1, 0]), 50)
    model = MLP(hidden_layers=[8], learning_rate=0.05, max_epochs=400).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_linear_hidden_layer_reaches_the_logistic_regression_optimum():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(1)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] - X[:, 1] + rng.normal(size=200) > 0).astype(int)
    # Full-batch steps: a constant-rate mini-batch run hovers around the optimum
    # instead of settling on it, which would test the noise rather than the model.
    model = MLP(hidden_layers=[4], activation="identity", learning_rate=0.05, batch=200)
    model.coefs_, model.intercepts_ = model._descend(X, y.astype(np.float64), 2000)
    logistic = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000).fit(X, y)

    def cross_entropy(probability):
        return -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))

    # Without a nonlinearity the network is a reparametrised logistic regression.
    assert cross_entropy(model.predict_proba(X)) <= cross_entropy(logistic.predict_proba(X)[:, 1]) + 1e-4


def test_early_stopping_watches_the_tail_and_refits_on_everything():
    X, y = _problem(seed=2, rows=200)
    model = MLP(hidden_layers=[8], learning_rate=0.01, max_epochs=60, patience=5).fit(X, y)
    assert model.n_train_rows_ == 200
    assert 1 <= model.best_epoch_ <= model.n_epochs_ <= 60
    assert model.history_["iteration"] == list(range(1, model.n_epochs_ + 1))
    # The stored weights come from a fresh run on all 200 rows for best_epoch_ epochs.
    coefs, intercepts = model._descend(X, y, model.best_epoch_)
    for stored, expected in zip(model.coefs_ + model.intercepts_, coefs + intercepts):
        assert np.array_equal(stored, expected)


def test_patience_stops_training_when_the_tail_stops_improving():
    X, y = _problem(seed=4, rows=200)
    rng = np.random.RandomState(0)
    y[170:] = rng.randint(0, 2, 30)  # the tail is noise, so it cannot keep improving
    model = MLP(hidden_layers=[16], learning_rate=0.05, max_epochs=500, patience=5).fit(X, y)
    assert model.n_epochs_ < 500
    assert model.n_epochs_ - model.best_epoch_ == 5


def test_readonly_input_is_preserved():
    X, y = _problem(rows=60)
    X.setflags(write=False)
    MLP(max_epochs=3).fit(X, y)
    assert not X.flags.writeable


@pytest.mark.parametrize("kwargs", [
    {"hidden_layers": []}, {"hidden_layers": [0]}, {"l2": -1.0}, {"learning_rate": 0.0},
    {"batch": 0}, {"max_epochs": 0}, {"momentum": 1.0}, {"activation": "tanh"},
])
def test_invalid_hyperparameters_fail(kwargs):
    X, y = _problem(rows=40)
    with pytest.raises(ValueError):
        MLP(**kwargs).fit(X, y)


def test_invalid_training_data_fail():
    with pytest.raises(ValueError):
        MLP().fit([[0.0], [1.0]], [0, 2])
    with pytest.raises(ValueError):
        MLP().fit([[0.0], [np.nan]], [0, 1])


def _reference_fit(scratch, X, y):
    with warnings.catch_warnings():
        # max_iter is the epoch count we chose, so "did not converge" is expected.
        warnings.simplefilter("ignore")
        return scratch.reference().fit(X, y)


def _assert_weights_match(scratch, reference):
    assert reference.n_iter_ == scratch.best_epoch_
    for layer, (mine, theirs) in enumerate(zip(scratch.coefs_, reference.coefs_)):
        assert np.allclose(mine, theirs, rtol=1e-7, atol=1e-10), f"layer {layer} weights"
    for layer, (mine, theirs) in enumerate(zip(scratch.intercepts_, reference.intercepts_)):
        assert np.allclose(mine, theirs, rtol=1e-7, atol=1e-10), f"layer {layer} intercepts"


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("hidden_layers,l2,learning_rate", [([16], 0.0, 0.01), ([32], 0.0001, 0.001), ([16, 8], 0.0001, 0.01)])
def test_synthetic_weight_and_strict_parity(seed, hidden_layers, l2, learning_rate):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * rng.normal(size=400) > 0).astype(int)
    scratch = MLP(hidden_layers=hidden_layers, l2=l2, learning_rate=learning_rate,
                  max_epochs=80, random_state=seed).fit(X[:300], y[:300])
    reference = _reference_fit(scratch, X[:300], y[:300])
    _assert_weights_match(scratch, reference)
    assert_parity(scratch, reference, X[:300], y[:300], task="classification")
    assert_parity(scratch, reference, X[300:], y[300:], task="classification")


def test_momentum_free_sgd_also_matches_reference():
    X, y = _problem(seed=6, rows=200)
    scratch = MLP(hidden_layers=[8], momentum=0.0, learning_rate=0.05, max_epochs=40).fit(X, y)
    reference = _reference_fit(scratch, X, y.astype(int))
    _assert_weights_match(scratch, reference)


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
@pytest.mark.parametrize("hidden_layers,l2,learning_rate", [([16], 0, 0.01), ([16, 8], 0.0001, 0.001)])
def test_real_train_and_validation_parity(variant, hidden_layers, l2, learning_rate):
    data = load(variant)
    standardizer = Standardizer().fit(data.train.X)
    train_X = standardizer.transform(data.train.X)
    validation_X = standardizer.transform(data.validation.X)
    scratch = MLP(hidden_layers=hidden_layers, l2=l2, learning_rate=learning_rate).fit(
        train_X, data.train.y_classification)
    reference = _reference_fit(scratch, train_X, data.train.y_classification)
    _assert_weights_match(scratch, reference)
    assert_parity(scratch, reference, train_X, data.train.y_classification, task="classification")
    assert_parity(scratch, reference, validation_X, data.validation.y_classification, task="classification")


def test_save_load_preserves_scores(tmp_path):
    X, y = _problem(seed=2, rows=80)
    original = MLP(hidden_layers=[16, 8], l2=0.0001, max_epochs=10).fit(X, y)
    path = tmp_path / "mlp.npz"
    save_npz(path, original.state_dict(), {"params": original.get_params()})
    arrays, metadata = load_npz(path)
    restored = MLP().load_state(arrays, metadata)
    assert restored.get_params() == original.get_params()
    assert restored.best_epoch_ == original.best_epoch_
    assert np.array_equal(restored.predict_proba(X), original.predict_proba(X))
    assert np.array_equal(restored.predict(X), original.predict(X))


def test_registry_exposes_mlp():
    assert "mlp" in model_names()
    assert isinstance(build("mlp"), MLP)


def test_purge_drops_the_five_trading_days_before_the_tail():
    """config neural.purge_original_trading_days: 5. The target looks 5 days ahead,
    so the last 5 days of sub-train carry labels computed from the tail's own period."""
    X, y = _problem(rows=200)
    model = MLP(max_epochs=3).fit(X, y)
    # 200 rows: the tail is rows 170..199, so sub-train keeps rows 0..164.
    assert model.n_subtrain_rows_ == 165


def test_purge_counts_trading_days_not_rows():
    X, y = _problem(rows=200)
    # A long gap before the tail (as after spike removal): nothing is within 5 days of it.
    positions = np.arange(200)
    positions[170:] += 20
    assert MLP(max_epochs=3).fit(X, y, positions=positions).n_subtrain_rows_ == 170
    # With only rows 168 and 169 inside the window, exactly those two go.
    positions = np.arange(200)
    positions[168:] += 20
    assert MLP(max_epochs=3).fit(X, y, positions=positions).n_subtrain_rows_ == 168


def test_purged_rows_cannot_influence_the_epoch_choice():
    X, y = _problem(seed=1, rows=200)
    base = MLP(hidden_layers=[8], max_epochs=40, patience=5).fit(X, y)
    flipped = y.copy()
    flipped[165:170] = 1 - flipped[165:170]
    probed = MLP(hidden_layers=[8], max_epochs=40, patience=5).fit(X, flipped)
    assert probed.history_ == base.history_
    assert probed.best_epoch_ == base.best_epoch_


def test_invalid_positions_fail():
    X, y = _problem(rows=200)
    with pytest.raises(ValueError):
        MLP(max_epochs=3).fit(X, y, positions=np.arange(200)[::-1])
    with pytest.raises(ValueError):
        MLP(max_epochs=3).fit(X, y, positions=np.arange(199))


def test_data_too_short_for_the_purge_fails_clearly():
    X, y = _problem(rows=6)
    with pytest.raises(ValueError, match="purge window"):
        MLP(max_epochs=3).fit(X, y)
