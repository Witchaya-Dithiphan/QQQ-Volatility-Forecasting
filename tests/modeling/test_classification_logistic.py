"""
Tests for Logistic Regression (M5-01, with-spike).
"""

import numpy as np
import pytest
from src.modeling.classification.logistic import LogisticRegression
from src.modeling.datasets import load_train_validation

try:
    from sklearn.linear_model import LogisticRegression as SKLogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def test_fit_synthetic():
    """RED: Fit on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    
    model = LogisticRegression(learning_rate=0.01, max_iter=100)
    model.fit(X, y)
    
    assert model.coefficients_ is not None
    assert model.intercept_ is not None
    assert model.coefficients_.shape == (3,)


def test_predict_shape():
    """Predictions should match input."""
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(10, 3)
    y_pred = model.predict(X_test)
    
    assert y_pred.shape == (10,)
    assert np.all(np.isin(y_pred, [0, 1]))


def test_predict_proba_range():
    """Probabilities in [0,1], rows sum to 1."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    proba = model.predict_proba(X)
    assert proba.shape == (50, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)


def test_predict_unfitted_raises():
    """Predict without fitting should raise."""
    model = LogisticRegression()
    X = np.random.randn(10, 3)
    
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)


def test_save_load_equality():
    """Serialize/deserialize preserves predictions."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    d = model.to_dict()
    model_loaded = LogisticRegression.from_dict(d)
    
    X_test = np.random.randn(10, 3)
    np.testing.assert_array_almost_equal(
        model.predict_proba(X_test),
        model_loaded.predict_proba(X_test)
    )


def test_load_train_validation_real_data():
    """Fit on real Train, predict Val."""
    dataset = load_train_validation("with_spike")
    train_X, train_y_regression = dataset.train.X, dataset.train.y_regression
    val_X, val_y_regression = dataset.validation.X, dataset.validation.y_regression
    
    # Binary classification
    median = np.median(train_y_regression)
    train_y = (train_y_regression > median).astype(int)
    val_y = (val_y_regression > median).astype(int)
    
    model = LogisticRegression(learning_rate=0.01, max_iter=1000)
    model.fit(train_X, train_y)
    
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)
    
    train_acc = np.mean(train_pred == train_y)
    val_acc = np.mean(val_pred == val_y)
    
    print(f"  Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
    assert train_acc > 0.5

@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_compare_sklearn():
    """Compare with sklearn LogisticRegression."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    model = LogisticRegression(learning_rate=0.01, max_iter=1000)
    model.fit(X, y)
    y_pred_ours = model.predict(X)
    
    sk_model = SKLogisticRegression(max_iter=1000, random_state=42)
    sk_model.fit(X, y)
    y_pred_sk = sk_model.predict(X)
    
    # Accuracy should be close
    acc_ours = np.mean(y_pred_ours == y)
    acc_sk = np.mean(y_pred_sk == y)
    assert np.abs(acc_ours - acc_sk) < 0.15, f"Ours: {acc_ours:.3f}, sklearn: {acc_sk:.3f}"
