import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.ml.core.compare import assert_parity
from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.regression.polynomial import Polynomial


@pytest.fixture
def data():
    rng = np.random.RandomState(11)
    X = rng.normal(size=(150, 8))
    y = 1.0 + 2.0 * X[:, 3] - 1.5 * X[:, 3] ** 2 + rng.normal(scale=0.05, size=150)
    return X, y


# --- unit test กลไกภายใน ---


def test_recovers_a_known_quadratic_exactly():
    """y = 1 + 2z + 3z^2 with no noise, on already-standardised input."""
    z = np.linspace(-2.0, 2.0, 40)
    X = np.column_stack([z, np.zeros(40)])
    y = 1.0 + 2.0 * z + 3.0 * z**2
    model = Polynomial(feature=0, degree=2, standardize_before_powers=False).fit(X, y)
    assert np.allclose([model.intercept_, *model.coef_], [1.0, 2.0, 3.0], atol=1e-9)


def test_design_matrix_has_no_interaction_terms():
    """Univariate powers only: [z, z^2, z^3], never products of different features."""
    model = Polynomial(feature=0, degree=3, standardize_before_powers=False)
    design = model._design(np.array([[2.0, 99.0], [3.0, 99.0]]))
    assert design.shape == (2, 3)
    assert np.allclose(design[0], [2.0, 4.0, 8.0])
    assert np.allclose(design[1], [3.0, 9.0, 27.0])


def test_standardizes_before_raising_to_powers():
    """(standardised x)^2 is not the standardisation of x^2 — order matters."""
    x = np.array([10.0, 20.0, 30.0, 40.0])
    X = x.reshape(-1, 1)
    model = Polynomial(feature=0, degree=2, standardize_before_powers=True)
    model.fit(X, np.zeros(4))
    design = model._design(X)
    z = (x - x.mean()) / x.std()
    assert np.allclose(design[:, 0], z)
    assert np.allclose(design[:, 1], z**2)
    assert not np.allclose(design[:, 1], (x**2 - (x**2).mean()) / (x**2).std())


def test_degree_one_reduces_to_simple_linear_regression():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 2))
    y = 3.0 * X[:, 0] + 1.0
    model = Polynomial(feature=0, degree=1, standardize_before_powers=False).fit(X, y)
    assert model.coef_[0] == pytest.approx(3.0, abs=1e-9)
    assert model.intercept_ == pytest.approx(1.0, abs=1e-9)


def test_feature_can_be_named_as_a_string():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(50, 8))
    y = X[:, FEATURE_COLUMNS.index("rsi_14")] ** 2
    model = Polynomial(feature="rsi_14", degree=2).fit(X, y)
    assert model.feature_index_ == FEATURE_COLUMNS.index("rsi_14")


def test_constant_feature_does_not_produce_nan():
    X = np.ones((20, 2))
    y = np.linspace(0.0, 19.0, 20)
    model = Polynomial(feature=0, degree=2).fit(X, y)
    assert np.isfinite(model.predict(X)).all()


# --- parity กับ reference ---


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("feature", range(8))
def test_matches_sklearn_across_the_whole_grid(data, degree, feature):
    """Grid is degree x feature = 16 candidates, matching polynomial_max_candidates."""
    X, y = data
    scratch = Polynomial(feature=feature, degree=degree).fit(X, y)
    reference = scratch.reference().fit(scratch._standardized_column(X), y)
    assert np.allclose(
        scratch.predict(X), reference.predict(scratch._standardized_column(X)), rtol=1e-5, atol=1e-8
    )


def test_matches_sklearn_through_the_parity_gate(data):
    X, y = data
    scratch = Polynomial(feature=3, degree=2).fit(X, y)

    class _Wrapped:
        def __init__(self, pipeline, model):
            self.pipeline, self.model = pipeline, model

        def predict(self, X):
            return self.pipeline.predict(self.model._standardized_column(X))

    reference = _Wrapped(scratch.reference().fit(scratch._standardized_column(X), y), scratch)
    assert_parity(scratch, reference, X, y, task="regression")


def test_reference_pipeline_excludes_the_bias_column():
    """PolynomialFeatures adds a bias term by default, duplicating the intercept."""
    reference = Polynomial(degree=2).reference()
    assert isinstance(reference, Pipeline)
    expander = reference.named_steps["polynomialfeatures"]
    assert isinstance(expander, PolynomialFeatures)
    assert expander.include_bias is False
    assert isinstance(reference.named_steps["linearregression"], LinearRegression)


# --- save / load ---


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = Polynomial(feature=3, degree=3).fit(X, y)
    restored = Polynomial().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))


def test_default_constructor_works_for_the_trainer(data):
    X, y = data
    assert np.isfinite(Polynomial().fit(X, y).predict(X)).all()


def test_records_history_for_the_performance_curve(data):
    X, y = data
    assert Polynomial(feature=3, degree=2).fit(X, y).history_["loss"][0] >= 0
