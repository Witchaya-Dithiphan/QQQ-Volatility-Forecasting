import numpy as np
import pytest
from sklearn.linear_model import ElasticNet as SkElasticNet

from src.ml.core.compare import assert_parity
from src.ml.regression.elastic_net import ElasticNet


@pytest.fixture
def data():
    rng = np.random.RandomState(21)
    X = rng.normal(size=(200, 6))
    y = 1.0 + 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 4] + rng.normal(scale=0.1, size=200)
    return X, y


# --- unit test กลไกภายใน: soft-threshold คือหัวใจของ L1 ---


@pytest.mark.parametrize(
    "value, threshold, expected",
    [(0.5, 0.2, 0.3), (-0.5, 0.2, -0.3), (0.1, 0.2, 0.0), (-0.1, 0.2, 0.0), (0.2, 0.2, 0.0)],
)
def test_soft_threshold_is_sign_times_max_of_abs_minus_gamma(value, threshold, expected):
    assert ElasticNet._soft_threshold(value, threshold) == pytest.approx(expected)


def test_large_alpha_with_pure_l1_drives_every_coefficient_to_exactly_zero(data):
    """Not merely small — exactly 0. That is what distinguishes L1 from ridge."""
    X, y = data
    model = ElasticNet(alpha=10.0, l1_ratio=1.0).fit(X, y)
    assert np.all(model.coef_ == 0.0)


def test_l2_shares_weight_between_duplicated_features_where_lasso_would_pick_one():
    """The grouping effect — the reason Elastic Net suits collinear columns."""
    rng = np.random.RandomState(0)
    base = rng.normal(size=200)
    X = np.column_stack([base, base, rng.normal(size=200)])
    y = 3.0 * base + rng.normal(scale=0.01, size=200)

    lasso = ElasticNet(alpha=0.05, l1_ratio=1.0).fit(X, y)
    elastic = ElasticNet(alpha=0.05, l1_ratio=0.25).fit(X, y)

    lasso_gap = abs(abs(lasso.coef_[0]) - abs(lasso.coef_[1]))
    elastic_gap = abs(abs(elastic.coef_[0]) - abs(elastic.coef_[1]))
    assert elastic_gap < lasso_gap, "L2 term failed to spread weight across the duplicate"


def test_objective_decreases_monotonically(data):
    X, y = data
    model = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X, y)
    losses = model.history_["loss"]
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:]))


def test_objective_uses_the_one_over_two_n_convention(data):
    """sklearn divides the squared error by 2n; using 1/2 rescales alpha by n."""
    X, y = data
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    coef = np.array([0.3, -0.2, 0.0, 0.0, 0.1, 0.0])
    intercept = 0.4
    residual = y - (X @ coef + intercept)
    expected = (
        np.sum(residual**2) / (2 * len(y))
        + 0.1 * 0.5 * np.abs(coef).sum()
        + 0.1 * 0.5 * 0.5 * coef @ coef
    )
    assert model._objective(X, y, coef, intercept) == pytest.approx(expected)


def test_intercept_absorbs_a_shift_in_y_without_touching_the_coefficients(data):
    """The direct statement of "the intercept is not penalised"."""
    X, y = data
    base = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X, y)
    shifted = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X, y + 1000.0)
    assert np.allclose(base.coef_, shifted.coef_, atol=1e-10)
    assert shifted.intercept_ == pytest.approx(base.intercept_ + 1000.0, abs=1e-6)


def test_once_every_coefficient_is_shrunk_away_the_intercept_is_the_mean(data):
    X, y = data
    model = ElasticNet(alpha=10.0, l1_ratio=1.0).fit(X, y)
    assert np.all(model.coef_ == 0.0)
    assert model.intercept_ == pytest.approx(y.mean())


def test_coordinates_are_visited_cyclically(data):
    """config pins selection="cyclic"; a random sweep order gives a different path."""
    X, y = data
    first = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X, y)
    second = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X, y)
    assert np.array_equal(first.coef_, second.coef_)


# --- parity กับ reference ---


@pytest.mark.parametrize("alpha", [0.0001, 0.001, 0.01, 0.1])
@pytest.mark.parametrize("l1_ratio", [0.25, 0.5, 0.75])
def test_matches_sklearn_across_the_whole_grid(data, alpha, l1_ratio):
    """Grid is alpha x l1_ratio = 12 candidates."""
    X, y = data
    scratch = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-12, max_iter=50000).fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert np.allclose(scratch.coef_, reference.coef_, atol=1e-6), "coefficients"
    assert scratch.intercept_ == pytest.approx(reference.intercept_, abs=1e-6)
    assert_parity(scratch, reference, X, y, task="regression")


def test_matches_sklearn_on_collinear_columns_like_the_real_features():
    """historical_volatility_5d and 20d correlate at 0.80 in the real train set."""
    rng = np.random.RandomState(4)
    a = rng.normal(size=200)
    b = 0.8 * a + np.sqrt(1 - 0.8**2) * rng.normal(size=200)
    X = np.column_stack([a, b, rng.normal(size=200)])
    y = 2.0 * a - 1.0 * b + rng.normal(scale=0.05, size=200)
    scratch = ElasticNet(alpha=0.01, l1_ratio=0.5, tol=1e-12, max_iter=50000).fit(X, y)
    assert_parity(scratch, scratch.reference().fit(X, y), X, y, task="regression")


def test_reference_mirrors_the_configured_solver_settings():
    reference = ElasticNet(alpha=0.01, l1_ratio=0.5).reference()
    assert isinstance(reference, SkElasticNet)
    params = reference.get_params()
    assert params["selection"] == "cyclic"
    assert params["fit_intercept"] is True
    assert params["alpha"] == 0.01 and params["l1_ratio"] == 0.5


# --- save / load ---


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = ElasticNet(alpha=0.001, l1_ratio=0.75).fit(X, y)
    restored = ElasticNet().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))


def test_default_constructor_works_for_the_trainer(data):
    X, y = data
    assert np.isfinite(ElasticNet().fit(X, y).predict(X)).all()
