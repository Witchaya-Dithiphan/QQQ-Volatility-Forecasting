"""Tests for Perceptron (M5-11, with-spike)."""

import numpy as np

from src.modeling.classification.perceptron import Perceptron
from src.modeling.datasets import load_train_validation


def _separable():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    s = X[:, 0] + X[:, 1]
    keep = np.abs(s) > 0.5  # margin so the perceptron theorem guarantees convergence
    X, s = X[keep], s[keep]
    return X, (s > 0).astype(int)


def test_fit_synthetic():
    X, y = _separable()
    model = Perceptron(max_iter=50).fit(X, y)
    assert model.coefficients_.shape == (5,)
    assert len(model.mistake_rate_history_) > 0
    loaded = Perceptron.from_dict(model.to_dict())
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_predict_shape():
    X, y = _separable()
    pred = Perceptron(max_iter=50).fit(X, y).predict(X)
    assert pred.shape == (len(X),)
    assert np.all(np.isin(pred, [0, 1]))


def test_converge_linearly_separable():
    X, y = _separable()
    model = Perceptron(max_iter=1000).fit(X, y)
    assert model.mistake_rate_history_[-1] < model.mistake_rate_history_[0]
    assert model.mistake_rate_history_[-1] == 0
    assert np.all(model.predict(X) == y)


def test_decision_function_is_score_not_probability():
    X, y = _separable()
    model = Perceptron(max_iter=50).fit(X, y)
    assert not hasattr(model, "predict_proba")
    assert model.decision_function(X).shape == (len(X),)


def test_load_train_validation_real_data():
    """Frozen M2 target: DatasetSplit.y_classification (Original-Train-Q75), never a recomputed median split."""
    ds = load_train_validation("with_spike")
    model = Perceptron(max_iter=20).fit(ds.train.X, ds.train.y_classification)
    for split in (ds.train, ds.validation):
        pred = model.predict(split.X)
        assert pred.shape == split.y_classification.shape and set(np.unique(pred)) <= {0, 1}
    assert np.mean(model.predict(ds.train.X) == ds.train.y_classification) > 0.5
