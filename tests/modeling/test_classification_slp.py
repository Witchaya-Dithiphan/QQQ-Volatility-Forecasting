"""Tests for Single-Layer Perceptron (sigmoid + BCE) (M5-12, with-spike)."""

import numpy as np

from src.modeling.classification.slp import SLP
from src.modeling.datasets import load_train_validation


def _synthetic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


def test_fit_synthetic():
    X, y = _synthetic()
    model = SLP(max_iter=100).fit(X, y)
    assert model.coefficients_.shape == (5,)
    assert len(model.loss_history_) == 100
    loaded = SLP.from_dict(model.to_dict())
    np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X))


def test_predict_shape():
    X, y = _synthetic()
    pred = SLP(max_iter=100).fit(X, y).predict(X)
    assert pred.shape == (100,)
    assert np.all(np.isin(pred, [0, 1]))


def test_converge_loss_decrease():
    X, y = _synthetic()
    model = SLP(learning_rate=0.1, max_iter=200).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert np.mean(model.predict(X) == y) > 0.9


def test_predict_proba():
    X, y = _synthetic()
    proba = SLP(max_iter=100).fit(X, y).predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_load_train_validation_real_data():
    """Frozen M2 target: DatasetSplit.y_classification (Original-Train-Q75), never a recomputed median split."""
    ds = load_train_validation("with_spike")
    model = SLP(max_iter=200).fit(ds.train.X, ds.train.y_classification)
    for split in (ds.train, ds.validation):
        pred = model.predict(split.X)
        assert pred.shape == split.y_classification.shape and set(np.unique(pred)) <= {0, 1}
    assert np.mean(model.predict(ds.train.X) == ds.train.y_classification) > 0.5
