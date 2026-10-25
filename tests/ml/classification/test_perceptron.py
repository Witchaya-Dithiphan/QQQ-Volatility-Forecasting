import numpy as np
import pytest
from scipy.optimize import linprog
from sklearn.linear_model import Perceptron as SkPerceptron

from src.ml.classification.perceptron import Perceptron
from src.ml.core.compare import assert_parity
from src.ml.core.data import load
from src.ml.core.preprocess import Standardizer


@pytest.fixture
def data():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(200, 5))
    y = (X @ np.array([1.0, -0.5, 0.25, 0.0, 0.75]) > 0).astype(int)
    return X, y


# --- unit test กลไกภายใน ---


def test_a_single_mistake_moves_the_weights_by_exactly_lr_times_y_x():
    """The update rule is w += lr*y*x with y in {-1,+1} — hand-checkable."""
    X = np.array([[2.0, 3.0]])
    model = Perceptron(learning_rate=0.1, max_epochs=1).fit(X, np.array([1]))
    assert np.allclose(model.coef_, [0.2, 0.3])
    assert model.intercept_ == pytest.approx(0.1)


def test_a_correct_prediction_leaves_the_weights_alone():
    """Mistake-driven: no error, no update."""
    X = np.array([[1.0]])
    model = Perceptron(learning_rate=0.5, max_epochs=3)
    model.fit(X, np.array([1]))
    first = model.coef_.copy()
    again = Perceptron(learning_rate=0.5, max_epochs=10).fit(X, np.array([1]))
    assert np.allclose(first, again.coef_), "extra epochs changed an already-correct fit"


def test_separable_data_converges_to_zero_mistakes():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    model = Perceptron(learning_rate=0.1, max_epochs=1000).fit(X, np.array([0, 0, 1, 1]))
    assert np.array_equal(model.predict(X), np.array([0, 0, 1, 1]))
    assert model.mistake_rate_ == 0.0


def test_stops_early_once_an_epoch_is_mistake_free():
    X = np.array([[-2.0], [2.0]])
    model = Perceptron(learning_rate=0.1, max_epochs=1000).fit(X, np.array([0, 1]))
    assert model.n_epochs_ < 1000
    assert model.history_["updates"][-1] == 0


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
def test_qqq_train_cannot_be_separated_by_a_linear_boundary(variant):
    data = load(variant)
    X = Standardizer().fit_transform(data.train.X)
    signed = 2 * data.train.y_classification - 1
    design = np.column_stack((X, np.ones(len(X))))
    result = linprog(
        np.zeros(design.shape[1]),
        A_ub=-signed[:, None] * design,
        b_ub=-np.ones(len(X)),
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    assert result.status == 2, result.message  # infeasible: no strict linear separator


def test_output_is_a_decision_score_not_a_probability(data):
    """config says output: decision_score — there is no calibrated probability here."""
    X, y = data
    model = Perceptron().fit(X, y)
    scores = model.decision_function(X)
    assert scores.min() < 0 < scores.max()
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)


def test_reports_the_mistake_rate(data):
    """config asks for report_mistake_rate: true."""
    X, y = data
    model = Perceptron().fit(X, y)
    assert 0.0 <= model.mistake_rate_ <= 1.0
    assert model.history_["loss"][-1] >= 0


def test_samples_are_visited_in_order(data):
    """shuffle=false in the config; a different order gives a different model."""
    X, y = data
    first = Perceptron(learning_rate=0.1, max_epochs=5).fit(X, y)
    second = Perceptron(learning_rate=0.1, max_epochs=5).fit(X, y)
    assert np.array_equal(first.coef_, second.coef_)


# --- parity กับ reference ---


@pytest.mark.parametrize("learning_rate", [0.001, 0.01, 0.1])
def test_matches_sklearn_across_the_whole_grid(data, learning_rate):
    X, y = data
    scratch = Perceptron(learning_rate=learning_rate).fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert np.allclose(scratch.coef_, reference.coef_[0], atol=1e-10), "coefficients"
    assert_parity(scratch, reference, X, y, task="classification")


def test_reference_disables_the_default_early_stopping():
    """sklearn defaults to tol=1e-3 and would stop on a different epoch than we do."""
    reference = Perceptron().reference()
    assert isinstance(reference, SkPerceptron)
    params = reference.get_params()
    assert params["tol"] is None
    assert params["shuffle"] is False
    assert params["penalty"] is None


# --- save / load ---


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = Perceptron(learning_rate=0.01).fit(X, y)
    restored = Perceptron().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))
    assert np.allclose(original.decision_function(X), restored.decision_function(X))


def test_default_constructor_works_for_the_trainer(data):
    X, y = data
    assert np.isfinite(Perceptron().fit(X, y).decision_function(X)).all()
