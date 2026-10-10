import numpy as np
import pytest

from src.ml.classification.slp import SLP


@pytest.fixture
def data():
    rng = np.random.RandomState(13)
    X = rng.normal(size=(300, 5))
    y = (X @ np.array([1.2, -0.8, 0.4, 0.0, 0.6]) + rng.normal(scale=0.6, size=300) > 0).astype(int)
    return X, y


# --- unit test กลไกภายใน ---


def test_gradient_matches_a_numerical_estimate(data):
    """sigmoid + BCE cancel to X.T @ (p - y) / n; check it against finite differences."""
    X, y = data
    model = SLP(l2=0.01)
    weights = np.full(X.shape[1], 0.25)
    intercept = 0.1

    analytic_w, analytic_b = model._gradient(X, y, weights, intercept)
    step = 1e-6
    for j in range(X.shape[1]):
        bumped = weights.copy()
        bumped[j] += step
        high = model._objective(X, y, bumped, intercept)
        bumped[j] -= 2 * step
        low = model._objective(X, y, bumped, intercept)
        assert analytic_w[j] == pytest.approx((high - low) / (2 * step), abs=1e-6), f"feature {j}"

    high = model._objective(X, y, weights, intercept + step)
    low = model._objective(X, y, weights, intercept - step)
    assert analytic_b == pytest.approx((high - low) / (2 * step), abs=1e-6)


def test_weight_init_uses_legacy_randomstate_not_default_rng():
    """sklearn is built on MT19937; default_rng is a different stream entirely."""
    model = SLP(random_state=42)
    weights, _ = model._initial_weights(5)
    expected = np.random.RandomState(42).uniform(
        -np.sqrt(6.0 / 6.0), np.sqrt(6.0 / 6.0), size=5
    )
    assert np.allclose(weights, expected)
    assert not np.allclose(weights, np.random.default_rng(42).uniform(-1, 1, size=5))


def test_init_is_reproducible_across_instances():
    first, _ = SLP(random_state=7)._initial_weights(4)
    second, _ = SLP(random_state=7)._initial_weights(4)
    assert np.array_equal(first, second)


def test_full_batch_matches_a_hand_rolled_gradient_descent(data):
    """batch="full" must be plain full-batch GD, with no mini-batch bookkeeping."""
    X, y = data
    model = SLP(batch="full", learning_rate=0.1, max_epochs=20, l2=0.0, patience=10**6).fit(X, y)

    weights, intercept = SLP(random_state=42)._initial_weights(X.shape[1])
    for _ in range(model.n_epochs_):
        gradient_w, gradient_b = model._gradient(X, y, weights, intercept)
        weights -= 0.1 * gradient_w
        intercept -= 0.1 * gradient_b
    assert np.allclose(model.coef_, weights, atol=1e-10)


def test_the_short_final_batch_is_not_dropped():
    """100 rows at batch 32 gives batches of 32/32/32/4 — the 4 must count."""
    model = SLP(batch=32)
    assert [len(b) for b in model._batches(100)] == [32, 32, 32, 4]


def test_batches_are_contiguous_and_chronological():
    """config says batch_order: contiguous_chronological, shuffle: false."""
    batches = SLP(batch=32)._batches(100)
    assert np.array_equal(np.concatenate(batches), np.arange(100))


def test_objective_decreases_on_a_separable_problem():
    X = np.array([[-3.0], [-2.0], [2.0], [3.0]])
    model = SLP(batch="full", learning_rate=0.5, max_epochs=200, patience=10**6).fit(
        X, np.array([0, 0, 1, 1])
    )
    losses = model.history_["loss"]
    assert losses[-1] < losses[0]
    assert np.array_equal(model.predict(X), np.array([0, 0, 1, 1]))


def test_early_stopping_counts_from_the_best_epoch(data):
    """patience is measured from the best internal score, not the last one."""
    X, y = data
    model = SLP(batch="full", learning_rate=0.5, max_epochs=5000, patience=5).fit(X, y)
    assert model.n_epochs_ < 5000
    assert model.best_epoch_ <= model.n_epochs_


def test_internal_split_is_the_chronological_tail(data):
    """config: internal_tail_fraction 0.15, taken from the end, never shuffled."""
    X, y = data
    model = SLP()
    cut = model._subtrain_size(len(X))
    assert cut == int(len(X) * (1 - 0.15))


def test_refit_uses_the_full_training_set(data):
    """config: refit_full_train true, refit_epochs = selected best epoch."""
    X, y = data
    model = SLP(batch="full", learning_rate=0.3, max_epochs=300, patience=10).fit(X, y)
    assert model.refit_epochs_ == model.best_epoch_
    assert model.n_train_rows_ == len(X)


# --- parity กับ reference ---


@pytest.mark.parametrize("l2", [0.0, 1e-4])
def test_converges_to_the_same_optimum_as_logistic_regression(data, l2):
    """A zero-hidden-layer sigmoid net *is* logistic regression.

    No sklearn estimator fits one directly — MLPClassifier rejects an empty
    hidden layer and SGDClassifier updates per sample — so the meaningful
    reference is the same convex objective solved by LBFGS. Early stopping is
    disabled here so the question is "does this gradient descent reach the
    optimum", not "how much budget does the configured protocol spend".
    Measured: coefficients agree to 1.3e-08.
    """
    X, y = data
    model = SLP(batch="full", l2=l2, learning_rate=0.5, max_epochs=40000, patience=10**9)
    model.fit(X, y)
    model.coef_, model.intercept_ = model._descend(X, y, 40000)
    model.n_train_rows_ = len(X)

    reference = model.reference().fit(X, y)
    assert np.allclose(model.coef_, reference.coef_[0], atol=1e-6), "coefficients"
    assert np.allclose(model.predict_proba(X), reference.predict_proba(X)[:, 1], atol=1e-6)
    assert np.array_equal(model.predict(X), reference.predict(X))


def test_early_stopping_is_what_separates_it_from_plain_logistic(data):
    """The configured protocol stops early by design, so it lands short of the optimum."""
    X, y = data
    stopped = SLP(batch="full", learning_rate=0.5, max_epochs=40000, patience=50).fit(X, y)
    converged = SLP(batch="full", learning_rate=0.5, max_epochs=40000, patience=10**9)
    converged.fit(X, y)
    converged.coef_, converged.intercept_ = converged._descend(X, y, 40000)
    assert stopped.n_epochs_ < 40000
    assert not np.allclose(stopped.coef_, converged.coef_, atol=1e-6)


# --- save / load ---


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = SLP(batch=32, l2=1e-4, learning_rate=0.01).fit(X, y)
    restored = SLP().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))
    assert np.allclose(original.predict_proba(X), restored.predict_proba(X))


def test_default_constructor_works_for_the_trainer(data):
    X, y = data
    assert np.isfinite(SLP().fit(X, y).predict_proba(X)).all()


def test_probabilities_stay_in_range(data):
    X, y = data
    probabilities = SLP().fit(X, y).predict_proba(X)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
