"""Tests for Decision Tree Classifier (M5-03, with-spike, scratch)."""
import numpy as np
import pytest
from src.modeling.classification.decision_tree import DecisionTreeClassifier
from src.modeling.datasets import load_train_validation

def test_fit_synthetic():
    """RED: Fit on synthetic."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)
    assert model.classes_ is not None
    assert model.tree_ is not None

def test_predict_shape():
    """Predictions match input."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier()
    model.fit(X, y)
    y_pred = model.predict(np.random.randn(10, 3))
    assert y_pred.shape == (10,)
    assert np.all(np.isin(y_pred, [0, 1]))

def test_predict_proba():
    """Proba sum to 1."""
    np.random.seed(42)
    X = np.random.randn(100, 5)  # More data, more features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Ensure both classes present
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (100, 2)
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

def test_unfitted_raises():
    """Unfitted error."""
    with pytest.raises(ValueError, match="not fitted"):
        DecisionTreeClassifier().predict(np.random.randn(10, 3))

def test_load_train_validation_real_data():
    """Frozen M2 target: DatasetSplit.y_classification (Original-Train-Q75), never a recomputed median split."""
    ds = load_train_validation("with_spike")
    model = DecisionTreeClassifier(max_depth=5).fit(ds.train.X, ds.train.y_classification)
    for split in (ds.train, ds.validation):
        pred = model.predict(split.X)
        assert pred.shape == split.y_classification.shape and set(np.unique(pred)) <= {0, 1}
    assert np.mean(model.predict(ds.train.X) == ds.train.y_classification) > 0.5
