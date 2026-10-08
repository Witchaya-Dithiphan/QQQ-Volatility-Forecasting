"""
Tests for SimpleLinearRegression (M4-01, with-spike).
"""

import numpy as np
import pytest
from pathlib import Path

from src.modeling.regression.simple_linear import SimpleLinearRegression
from src.modeling.datasets import load_train_validation
from src.modeling.persistence import save_npz, load_npz
from src.modeling.contracts import FEATURE_COLUMNS

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# RED TEST (1st — should pass with implementation)
# ============================================================================

def test_fit_synthetic_univariate():
    """RED: Fit on synthetic 1D data; expect scalar coefficient."""
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * X + 1.0 + np.random.randn(5) * 0.1
    
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    assert hasattr(model, 'coefficient_')
    assert hasattr(model, 'intercept_')
    assert isinstance(model.coefficient_, (float, np.floating))
    assert isinstance(model.intercept_, (float, np.floating))


# ============================================================================
# GREEN TESTS
# ============================================================================

def test_predict_shape_matches():
    """Predict shape should match input."""
    X = np.random.randn(10)
    y = np.random.randn(10)
    
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    X_test = np.random.randn(5)
    y_pred = model.predict(X_test)
    
    assert y_pred.shape == (5,)
    assert isinstance(y_pred, np.ndarray)


def test_predict_unfitted_raises():
    """Predict on unfitted model should raise."""
    model = SimpleLinearRegression()
    X = np.random.randn(5)
    
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)


def test_zero_variance_raises():
    """Fit on zero-variance feature should raise."""
    X = np.ones(10)  # All same value
    y = np.random.randn(10)
    
    model = SimpleLinearRegression()
    
    with pytest.raises(ValueError, match="zero variance"):
        model.fit(X, y)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_coefficients_close_to_sklearn():
    """Compare our coefficients with sklearn's LinearRegression."""
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * X + 1.0 + np.random.randn(5) * 0.1
    
    # Our model
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    # sklearn's model
    sk_model = LinearRegression()
    sk_model.fit(X.reshape(-1, 1), y)
    
    # Compare coefficients
    np.testing.assert_allclose(
        model.coefficient_, sk_model.coef_[0], rtol=1e-3, atol=1e-4
    )
    
    # Compare intercepts
    np.testing.assert_allclose(
        model.intercept_, sk_model.intercept_, rtol=1e-3, atol=1e-4
    )


def test_save_load_equality(tmp_path):
    """Serialize/deserialize should preserve model."""
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * X + 1.0 + np.random.randn(5) * 0.1
    
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    # Test dict round-trip
    d = model.to_dict()
    model_loaded = SimpleLinearRegression.from_dict(d)
    
    X_test = np.array([2.5, 3.5])
    y_pred_orig = model.predict(X_test)
    y_pred_loaded = model_loaded.predict(X_test)
    
    np.testing.assert_array_equal(y_pred_orig, y_pred_loaded)


def test_load_train_validation_return_1d():
    """Load real with-spike Train/Validation and fit on return_1d feature."""
    dataset = load_train_validation("with_spike")
    
    train_X_full, train_y = dataset.train.X, dataset.train.y_regression
    val_X_full, val_y = dataset.validation.X, dataset.validation.y_regression
    
    # Extract return_1d (first column)
    return_1d_idx = FEATURE_COLUMNS.index("return_1d")
    train_X = train_X_full[:, return_1d_idx]
    val_X = val_X_full[:, return_1d_idx]
    
    # Fit on train
    model = SimpleLinearRegression()
    model.fit(train_X, train_y)
    
    # Predict on validation
    val_pred = model.predict(val_X)
    
    assert val_pred.shape == val_y.shape
    assert not np.any(np.isnan(val_pred)), "Predictions contain NaN"
