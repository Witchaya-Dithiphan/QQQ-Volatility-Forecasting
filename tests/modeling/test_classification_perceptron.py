"""Tests for Perceptron (M5-11, with-spike)."""

import json
from pathlib import Path

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
    assert len(model.loss_history_) > 0
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
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert model.loss_history_[-1] == 0
    assert np.all(model.predict(X) == y)


def test_predict_proba():
    X, y = _separable()
    proba = Perceptron(max_iter=50).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(np.isin(proba, [0.0, 1.0]))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_load_train_validation_real_data():
    dataset = load_train_validation("with_spike")
    train_X, train_y_reg = dataset.train.X, dataset.train.y_regression
    val_X, val_y_reg = dataset.validation.X, dataset.validation.y_regression

    median = np.median(train_y_reg)
    train_y = (train_y_reg > median).astype(int)
    val_y = (val_y_reg > median).astype(int)

    model = Perceptron(max_iter=20).fit(train_X, train_y)
    train_pred, val_pred = model.predict(train_X), model.predict(val_X)
    assert not np.any(np.isnan(train_pred)) and not np.any(np.isnan(val_pred))

    artifact_dir = Path("outputs/modeling/with_spike/classification/perceptron/scratch/M5_11")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "train_predictions.npy", train_pred)
    np.save(artifact_dir / "val_predictions.npy", val_pred)
    metadata = {
        "model_class": "Perceptron",
        "variant": "with_spike",
        "task": "M5-11",
        "train_samples": len(train_X),
        "val_samples": len(val_X),
        "epochs_run": len(model.loss_history_),
        "train_accuracy": float(np.mean(train_pred == train_y)),
        "val_accuracy": float(np.mean(val_pred == val_y)),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
