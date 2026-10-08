"""Tests for k-Nearest Neighbors (M5-04, reference/sklearn wrapper)."""
import numpy as np

from src.modeling.classification.knn import KNN
from src.modeling.datasets import load_train_validation


def test_fit_predict():
    """Fit synthetic, check predict shape."""
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = KNN(n_neighbors=5).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (50,)
    assert np.all(np.isin(pred, [0, 1]))


def test_proba():
    """Check predict_proba sums to 1."""
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = KNN(n_neighbors=5).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_load_train_validation_real_data():
    """Frozen M2 target: DatasetSplit.y_classification (Original-Train-Q75), never a recomputed median split."""
    ds = load_train_validation("with_spike")
    model = KNN(n_neighbors=5).fit(ds.train.X, ds.train.y_classification)
    for split in (ds.train, ds.validation):
        pred = model.predict(split.X)
        assert pred.shape == split.y_classification.shape and set(np.unique(pred)) <= {0, 1}
    assert np.mean(model.predict(ds.train.X) == ds.train.y_classification) > 0.5
