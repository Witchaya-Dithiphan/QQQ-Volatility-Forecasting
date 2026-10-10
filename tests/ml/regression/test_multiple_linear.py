import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.ml.core.compare import assert_parity
from src.ml.regression.multiple_linear import MultipleLinear


@pytest.fixture
def data():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(120, 4))
    y = 2.5 + 1.5 * X[:, 0] - 0.75 * X[:, 1] + 0.25 * X[:, 3] + rng.normal(scale=0.05, size=120)
    return X, y


def test_recovers_known_coefficients(data):
    X, y = data
    model = MultipleLinear().fit(X, y)
    assert np.allclose(model.coef_, [1.5, -0.75, 0.0, 0.25], atol=0.05)
    assert model.intercept_ == pytest.approx(2.5, abs=0.05)


def test_normal_equation_solves_an_exactly_determined_system():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([2.0, 3.0, 5.0])
    model = MultipleLinear(fit_intercept=False).fit(X, y)
    assert np.allclose(model.coef_, [2.0, 3.0])


def test_matches_sklearn_exactly(data):
    X, y = data
    scratch = MultipleLinear().fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert_parity(scratch, reference, X, y, task="regression")


def test_matches_sklearn_without_intercept(data):
    X, y = data
    scratch = MultipleLinear(fit_intercept=False).fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert_parity(scratch, reference, X, y, task="regression")


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = MultipleLinear().fit(X, y)
    restored = MultipleLinear().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))


def test_records_history_for_the_performance_curve(data):
    X, y = data
    model = MultipleLinear().fit(X, y)
    assert model.history_["loss"][0] > 0


def test_reference_is_sklearn_linear_regression():
    assert isinstance(MultipleLinear().reference(), LinearRegression)


def test_handles_collinear_features_at_the_real_conditioning():
    """Two columns correlating at 0.80, like historical_volatility_5d and 20d.

    Deliberately not more collinear than that: a near-singular design makes the
    least-squares solution non-unique, so two SVD solvers legitimately return
    different answers and the test would be measuring rcond cutoffs rather than
    the model. Real train data sits at cond(X) ~ 6.
    """
    rng = np.random.RandomState(0)
    a = rng.normal(size=100)
    b = 0.8 * a + np.sqrt(1 - 0.8**2) * rng.normal(size=100)
    X = np.column_stack([a, b])
    y = 3 * a - 2 * b + rng.normal(scale=0.01, size=100)

    assert np.corrcoef(a, b)[0, 1] == pytest.approx(0.8, abs=0.1)
    scratch = MultipleLinear().fit(X, y)
    assert_parity(scratch, scratch.reference().fit(X, y), X, y, task="regression")
