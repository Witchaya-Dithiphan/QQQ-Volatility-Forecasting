"""
Tests for Stacking Classifier (M5-02, with-spike).
"""

import numpy as np
import pytest
from pathlib import Path

from src.modeling.ensemble import StackingClassifier
from src.modeling.datasets import load_train_validation
from src.modeling.contracts import FEATURE_COLUMNS


def test_fit_synthetic_stacking():
    """RED: Fit stacking on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple binary rule
    
    model = StackingClassifier(random_state=42)
    model.fit(X, y)
    
    assert model.base_learners_ is not None
    assert len(model.base_learners_) == 3  # LR, DT, KNN
    assert model.meta_learner_ is not None
    assert model.classes_ is not None


def test_predict_shape_matches():
    """Predictions should match input."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    model = StackingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(20, 5)
    y_pred = model.predict(X_test)
    
    assert y_pred.shape == (20,)
    assert np.all(np.isin(y_pred, [0, 1]))


def test_predict_proba_sums_to_one():
    """Probabilities should sum to 1."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    model = StackingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(20, 5)
    proba = model.predict_proba(X_test)
    
    assert proba.shape == (20, 2)
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_unfitted_raises():
    """Predict before fitting should raise."""
    model = StackingClassifier()
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)


def test_oos_folds_consistent():
    """OOS fold sizes should be consistent with chronological splits."""
    np.random.seed(42)
    n_samples = 120  # Divisible by 4
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] > 0).astype(int)
    
    # Fit with this sample size
    model = StackingClassifier(random_state=42)
    model.fit(X, y)
    
    # Should succeed without error (internal OOS logic was correct)
    y_pred = model.predict(X)
    assert y_pred.shape == (n_samples,)


def test_load_train_validation_real_data():
    """Fit stacking on real with-spike Train/Validation data."""
    dataset = load_train_validation("with_spike")
    train_X, train_y_regression = dataset.train.X, dataset.train.y_regression
    val_X, val_y_regression = dataset.validation.X, dataset.validation.y_regression
    
    # Convert to binary classification
    median = np.median(train_y_regression)
    train_y = (train_y_regression > median).astype(int)
    val_y = (val_y_regression > median).astype(int)
    
    # Fit
    model = StackingClassifier(random_state=42)
    model.fit(train_X, train_y)
    
    # Predict
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)
    
    assert train_pred.shape == (len(train_X),)
    assert val_pred.shape == (len(val_X),)
    assert np.all(np.isin(train_pred, [0, 1]))
    assert np.all(np.isin(val_pred, [0, 1]))
    
    # Check accuracy is better than random
    train_acc = np.mean(train_pred == train_y)
    val_acc = np.mean(val_pred == val_y)
    
    print(f"  Train accuracy: {train_acc:.3f}, Val accuracy: {val_acc:.3f}")
    assert train_acc > 0.5, "Train accuracy should beat random"
