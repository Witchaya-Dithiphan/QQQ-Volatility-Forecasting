import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.ml.core.compare import assert_parity
from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.regression.simple_linear import SimpleLinear


@pytest.fixture
def data():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(120, 8))
    y = 1.25 + 2.5 * X[:, 2] + rng.normal(scale=0.05, size=120)
    return X, y


# --- unit test กลไกภายใน: closed form ที่คำนวณมือได้ ---


def test_recovers_exact_slope_and_intercept_on_a_perfect_line():
    """y = 2x, so b1 = 2 and b0 = 0 exactly."""
    X = np.array([[1.0], [2.0], [3.0]])
    model = SimpleLinear(feature=0).fit(X, np.array([2.0, 4.0, 6.0]))
    assert model.slope_ == pytest.approx(2.0)
    assert model.intercept_ == pytest.approx(0.0, abs=1e-12)


def test_slope_uses_the_sum_of_squares_denominator():
    """b1 = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2) — hand-computed."""
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([2.0, 3.0, 9.0])
    expected = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    model = SimpleLinear(feature=0).fit(x.reshape(-1, 1), y)
    assert model.slope_ == pytest.approx(expected)


def test_uses_only_the_selected_feature():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 3))
    y = 5.0 * X[:, 1]
    chosen = SimpleLinear(feature=1).fit(X, y)
    ignored = SimpleLinear(feature=0).fit(X, y)
    assert chosen.slope_ == pytest.approx(5.0, abs=1e-10)
    assert abs(ignored.slope_) < 1.0


def test_constant_feature_yields_zero_slope_and_the_mean(data):
    """Zero variance means the slope is undefined; predict the mean instead of nan."""
    X = np.ones((20, 2))
    y = np.linspace(0.0, 19.0, 20)
    model = SimpleLinear(feature=0).fit(X, y)
    assert model.slope_ == 0.0
    assert model.intercept_ == pytest.approx(y.mean())
    assert np.isfinite(model.predict(X)).all()


def test_feature_can_be_named_as_a_string():
    """configs/modeling.json expresses the grid as feature names, not indices."""
    rng = np.random.RandomState(1)
    X = rng.normal(size=(40, 8))
    y = 3.0 * X[:, FEATURE_COLUMNS.index("rsi_14")]
    model = SimpleLinear(feature="rsi_14").fit(X, y)
    assert model.feature_index_ == FEATURE_COLUMNS.index("rsi_14")
    assert model.slope_ == pytest.approx(3.0, abs=1e-10)


def test_unknown_feature_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown feature"):
        SimpleLinear(feature="not_a_feature").fit(np.zeros((3, 8)), np.zeros(3))


# --- parity กับ reference ---


def test_matches_sklearn_on_synthetic_data(data):
    X, y = data
    scratch = SimpleLinear(feature=2).fit(X, y)
    reference = scratch.reference().fit(X[:, [2]], y)
    assert np.allclose(scratch.predict(X), reference.predict(X[:, [2]]), rtol=1e-5, atol=1e-8)


def test_matches_sklearn_for_every_feature(data):
    """The grid searches all 8 features, so all 8 have to be right."""
    X, y = data
    for index in range(8):
        scratch = SimpleLinear(feature=index).fit(X, y)
        reference = scratch.reference().fit(X[:, [index]], y)
        assert np.allclose(
            scratch.predict(X), reference.predict(X[:, [index]]), rtol=1e-5, atol=1e-8
        ), f"feature {index}"


def test_matches_sklearn_through_the_parity_gate(data):
    X, y = data
    scratch = SimpleLinear(feature=2).fit(X, y)

    class _Wrapped:
        def __init__(self, estimator, index):
            self.estimator, self.index = estimator, index

        def predict(self, X):
            return self.estimator.predict(X[:, [self.index]])

    reference = _Wrapped(scratch.reference().fit(X[:, [2]], y), 2)
    assert_parity(scratch, reference, X, y, task="regression")


def test_reference_is_sklearn_linear_regression():
    assert isinstance(SimpleLinear().reference(), LinearRegression)


# --- save / load ---


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = SimpleLinear(feature=5).fit(X, y)
    restored = SimpleLinear().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))
    assert restored.get_params() == original.get_params()


def test_default_constructor_works_for_the_trainer(data):
    """trainer builds model_class() with no arguments before load_state()."""
    X, y = data
    model = SimpleLinear()
    assert model.get_params()["feature"] in (0, *FEATURE_COLUMNS)
    assert np.isfinite(model.fit(X, y).predict(X)).all()


def test_records_history_for_the_performance_curve(data):
    X, y = data
    model = SimpleLinear(feature=2).fit(X, y)
    assert model.history_["loss"][0] >= 0
