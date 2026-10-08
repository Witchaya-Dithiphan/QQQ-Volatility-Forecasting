"""Tests for Random Forest (M5-09, with-spike)."""

import json
from pathlib import Path

import numpy as np

from src.modeling.datasets import load_train_validation
from src.modeling.ensemble.random_forest import RandomForest


def _synthetic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


def test_fit_synthetic():
    X, y = _synthetic()
    model = RandomForest(n_estimators=10).fit(X, y)
    assert len(model.model_.estimators_) == 10
    # dict round-trip preserves predictions
    loaded = RandomForest.from_dict(model.to_dict())
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_predict_shape():
    X, y = _synthetic()
    pred = RandomForest(n_estimators=10).fit(X, y).predict(X)
    assert pred.shape == (100,)
    assert np.all(np.isin(pred, [0, 1]))


def test_predict_proba():
    X, y = _synthetic()
    proba = RandomForest(n_estimators=10).fit(X, y).predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_load_train_validation_real_data():
    dataset = load_train_validation("with_spike")
    train_X, train_y_reg = dataset.train.X, dataset.train.y_regression
    val_X, val_y_reg = dataset.validation.X, dataset.validation.y_regression

    median = np.median(train_y_reg)
    train_y = (train_y_reg > median).astype(int)
    val_y = (val_y_reg > median).astype(int)

    model = RandomForest(n_estimators=10, max_depth=5).fit(train_X, train_y)
    train_pred, val_pred = model.predict(train_X), model.predict(val_X)
    assert not np.any(np.isnan(train_pred)) and not np.any(np.isnan(val_pred))

    artifact_dir = Path("outputs/modeling/with_spike/ensemble/random_forest/scratch/M5_09")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "train_predictions.npy", train_pred)
    np.save(artifact_dir / "val_predictions.npy", val_pred)
    metadata = {
        "model_class": "RandomForest",
        "variant": "with_spike",
        "task": "M5-09",
        "train_samples": len(train_X),
        "val_samples": len(val_X),
        "n_estimators": 10,
        "train_accuracy": float(np.mean(train_pred == train_y)),
        "val_accuracy": float(np.mean(val_pred == val_y)),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
