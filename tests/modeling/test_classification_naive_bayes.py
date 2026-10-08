"""
Tests for Gaussian Naive Bayes (M5-07, with-spike).
"""

import numpy as np
import pytest
from pathlib import Path

from src.modeling.classification.naive_bayes import GaussianNaiveBayes
from src.modeling.datasets import load_train_validation

try:
    from sklearn.naive_bayes import GaussianNB
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# RED TEST (1st — should fail initially)
# ============================================================================

def test_fit_synthetic_gaussian_nb():
    """RED: Fit Gaussian NB on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)  # Simple binary rule
    
    model = GaussianNaiveBayes()
    model.fit(X, y)
    
    assert model.classes_ is not None
    assert model.class_priors_ is not None
    assert model.means_.shape == (2, 3)
    assert model.variances_.shape == (2, 3)


# ============================================================================
# GREEN TESTS
# ============================================================================

def test_predict_shape_matches():
    """Predictions should match input."""
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(10, 3)
    y_pred = model.predict(X_test)
    
    assert y_pred.shape == (10,)
    assert np.all(np.isin(y_pred, [0, 1]))


def test_predict_proba_sums_to_one():
    """Probabilities should sum to 1."""
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(10, 3)
    proba = model.predict_proba(X_test)
    
    assert proba.shape == (10, 2)
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_zero_variance_smoothing():
    """Laplace smoothing should prevent zero-variance crash."""
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    y = np.array([0, 0, 0, 1])
    
    # First feature is constant within class 0
    model = GaussianNaiveBayes(var_smoothing=1e-9)
    model.fit(X, y)
    
    # Should not crash
    X_test = np.array([[1.0, 2.5]])
    proba = model.predict_proba(X_test)
    
    assert proba.shape == (1, 2)
    assert not np.any(np.isnan(proba))
    assert not np.any(np.isinf(proba))


def test_predict_unfitted_raises():
    """Predict before fitting should raise."""
    model = GaussianNaiveBayes()
    X = np.random.randn(5, 3)
    
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)


def test_feature_mismatch_raises():
    """Mismatch between train and test features should raise."""
    X_train = np.random.randn(20, 3)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(5, 5)  # Wrong number of features
    
    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict(X_test)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_compare_sklearn_gaussian_nb():
    """Compare predictions with sklearn GaussianNB."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Our model
    model = GaussianNaiveBayes()
    model.fit(X, y)
    y_pred_ours = model.predict(X)
    
    # sklearn's model
    sk_model = GaussianNB()
    sk_model.fit(X, y)
    y_pred_sk = sk_model.predict(X)
    
    # Accuracy should match within 5%
    acc_ours = np.mean(y_pred_ours == y)
    acc_sk = np.mean(y_pred_sk == y)
    
    assert np.abs(acc_ours - acc_sk) < 0.05, f"Ours: {acc_ours:.3f}, sklearn: {acc_sk:.3f}"


def test_save_load_equality(tmp_path):
    """Serialize/deserialize preserves model."""
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    
    # Dict round-trip
    d = model.to_dict()
    model_loaded = GaussianNaiveBayes.from_dict(d)
    
    X_test = np.random.randn(10, 3)
    y_pred_orig = model.predict(X_test)
    y_pred_loaded = model_loaded.predict(X_test)
    
    np.testing.assert_array_equal(y_pred_orig, y_pred_loaded)


def test_load_train_validation_real_data():
    """Frozen M2 target: DatasetSplit.y_classification (Original-Train-Q75), never a recomputed median split."""
    ds = load_train_validation("with_spike")
    model = GaussianNaiveBayes().fit(ds.train.X, ds.train.y_classification)
    for split in (ds.train, ds.validation):
        pred = model.predict(split.X)
        assert pred.shape == split.y_classification.shape and set(np.unique(pred)) <= {0, 1}
    assert np.mean(model.predict(ds.train.X) == ds.train.y_classification) > 0.5
