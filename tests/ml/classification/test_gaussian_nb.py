import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB as SkGaussianNB

from src.ml.classification.gaussian_nb import GaussianNB
from src.ml.core.compare import assert_parity


@pytest.fixture
def data():
    rng = np.random.RandomState(7)
    X = np.vstack([rng.normal(-1.0, 1.0, size=(120, 4)), rng.normal(1.0, 1.5, size=(80, 4))])
    y = np.array([0] * 120 + [1] * 80)
    return X, y


# --- unit test กลไกภายใน ---


def test_estimates_class_means_and_variances_by_hand():
    X = np.array([[1.0], [3.0], [10.0], [14.0]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNB(var_smoothing=0.0).fit(X, y)
    assert model.theta_[0][0] == pytest.approx(2.0)
    assert model.theta_[1][0] == pytest.approx(12.0)
    # ddof=0, matching sklearn: mean of squared deviations.
    assert model.var_[0][0] == pytest.approx(1.0)
    assert model.var_[1][0] == pytest.approx(4.0)


def test_priors_are_the_empirical_class_frequencies():
    X = np.zeros((10, 2))
    y = np.array([0] * 7 + [1] * 3)
    model = GaussianNB().fit(X, y)
    assert model.class_prior_[0] == pytest.approx(0.7)
    assert model.class_prior_[1] == pytest.approx(0.3)


def test_var_smoothing_scales_with_the_largest_variance_in_the_whole_set():
    """sklearn adds var_smoothing * max(var(X, axis=0)) — the dataset max, not per class."""
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(0, 1, 100), rng.normal(0, 100, 100)])
    y = (np.arange(100) % 2).astype(int)
    smoothing = 1e-3
    model = GaussianNB(var_smoothing=smoothing).fit(X, y)

    expected_epsilon = smoothing * np.var(X, axis=0).max()
    unsmoothed = GaussianNB(var_smoothing=0.0).fit(X, y)
    assert model.epsilon_ == pytest.approx(expected_epsilon)
    assert model.var_[0][0] == pytest.approx(unsmoothed.var_[0][0] + expected_epsilon)


def test_log_likelihood_is_computed_in_log_space_without_underflow():
    """Multiplying densities directly underflows; the joint must stay in logs."""
    rng = np.random.RandomState(1)
    X = rng.normal(size=(50, 60))
    y = (np.arange(50) % 2).astype(int)
    model = GaussianNB().fit(X, y)
    joint = model._joint_log_likelihood(X)
    assert np.isfinite(joint).all()
    assert (joint < 0).all()


def test_separable_classes_are_classified_perfectly():
    X = np.array([[-10.0], [-9.0], [9.0], [10.0]])
    model = GaussianNB().fit(X, np.array([0, 0, 1, 1]))
    assert np.array_equal(model.predict(X), np.array([0, 0, 1, 1]))


def test_zero_variance_feature_does_not_produce_nan():
    X = np.column_stack([np.ones(20), np.linspace(0, 19, 20)])
    y = (np.arange(20) % 2).astype(int)
    model = GaussianNB().fit(X, y)
    assert np.isfinite(model.predict_proba(X)).all()


# --- parity กับ reference ---


@pytest.mark.parametrize("var_smoothing", [1e-9, 1e-8, 1e-7])
def test_matches_sklearn_across_the_whole_grid(data, var_smoothing):
    X, y = data
    scratch = GaussianNB(var_smoothing=var_smoothing).fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert np.allclose(scratch.theta_, reference.theta_, atol=1e-12)
    assert np.allclose(scratch.var_, reference.var_, atol=1e-12)
    assert np.allclose(scratch.predict_proba(X), reference.predict_proba(X)[:, 1], atol=1e-10)
    assert_parity(scratch, reference, X, y, task="classification")


def test_matches_sklearn_on_the_real_feature_count(data):
    """8 features, unscaled — preprocess is "none" for this model in the config."""
    rng = np.random.RandomState(3)
    X = rng.normal(size=(200, 8)) * np.array([1, 10, 100, 0.1, 1, 5, 50, 2])
    y = (X[:, 2] + X[:, 6] > 0).astype(int)
    scratch = GaussianNB().fit(X, y)
    assert_parity(scratch, scratch.reference().fit(X, y), X, y, task="classification")


def test_reference_is_sklearn_gaussian_nb():
    assert isinstance(GaussianNB().reference(), SkGaussianNB)


# --- save / load ---


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = GaussianNB(var_smoothing=1e-8).fit(X, y)
    restored = GaussianNB().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))
    assert np.allclose(original.predict_proba(X), restored.predict_proba(X))


def test_default_constructor_works_for_the_trainer(data):
    X, y = data
    assert np.isfinite(GaussianNB().fit(X, y).predict_proba(X)).all()
