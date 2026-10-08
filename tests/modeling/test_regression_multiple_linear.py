"""
Tests for MultiLinearRegression (M3-02, with-spike).
"""

import numpy as np
import pytest
from pathlib import Path

from src.modeling.regression.multiple_linear import MultiLinearRegression
from src.modeling.datasets import load_train_validation
from src.modeling.persistence import save_npz, load_npz, verify_reload
from src.modeling.contracts import FEATURE_COLUMNS

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# RED TESTS (1st — should fail initially)
# ============================================================================

def test_fit_synthetic_matrix():
    """RED: Fit on synthetic data; expect coefficients_ shape (3,)."""
    X = np.random.randn(10, 3)
    y = X @ np.array([2.0, -1.0, 0.5]) + np.random.randn(10) * 0.1
    
    model = MultiLinearRegression()
    model.fit(X, y)
    
    assert hasattr(model, 'coefficients_')
    assert model.coefficients_.shape == (3,)
    assert hasattr(model, 'intercept_')
    assert isinstance(model.intercept_, (float, np.floating))


# ============================================================================
# GREEN TESTS (add one-by-one after RED passes)
# ============================================================================

def test_predict_shape_matches():
    """Predict shape should match input."""
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    
    model = MultiLinearRegression()
    model.fit(X, y)
    
    X_test = np.random.randn(5, 3)
    y_pred = model.predict(X_test)
    
    assert y_pred.shape == (5,)
    assert isinstance(y_pred, np.ndarray)


def test_predict_unfitted_raises():
    """Predict on unfitted model should raise."""
    model = MultiLinearRegression()
    X = np.random.randn(5, 3)
    
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)


def test_predict_feature_mismatch_raises():
    """Predict with wrong feature count should raise."""
    X_train = np.random.randn(10, 3)
    y_train = np.random.randn(10)
    
    model = MultiLinearRegression()
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(5, 5)  # Wrong: 5 features instead of 3
    
    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict(X_test)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_coefficients_close_to_sklearn():
    """Compare our coefficients with sklearn's LinearRegression."""
    X = np.random.randn(20, 3)
    y = X @ np.array([2.0, -1.0, 0.5]) + np.random.randn(20) * 0.1
    
    # Our model
    model = MultiLinearRegression()
    model.fit(X, y)
    
    # sklearn's model
    sk_model = LinearRegression()
    sk_model.fit(X, y)
    
    # Compare coefficients (within 1% relative tolerance)
    np.testing.assert_allclose(
        model.coefficients_, sk_model.coef_, rtol=1e-2, atol=1e-3
    )
    
    # Compare intercepts
    np.testing.assert_allclose(
        model.intercept_, sk_model.intercept_, rtol=1e-2, atol=1e-3
    )


def test_save_load_equality(tmp_path):
    """Serialize/deserialize should preserve model."""
    X_train = np.random.randn(20, 3)
    y_train = np.random.randn(20)
    
    model = MultiLinearRegression()
    model.fit(X_train, y_train)
    
    # Test dict round-trip
    d = model.to_dict()
    model_loaded = MultiLinearRegression.from_dict(d)
    
    X_test = np.random.randn(5, 3)
    y_pred_orig = model.predict(X_test)
    y_pred_loaded = model_loaded.predict(X_test)
    
    np.testing.assert_array_equal(y_pred_orig, y_pred_loaded)
    
    # Test NPZ round-trip via persistence
    npz_path = tmp_path / "model.npz"
    save_npz(
        npz_path,
        {"coefficients": model.coefficients_, "intercept": model.intercept_},
        metadata={"feature_order": FEATURE_COLUMNS}
    )
    
    loaded_data, loaded_meta = load_npz(npz_path)
    model_reloaded = MultiLinearRegression.from_dict({
        "coefficients": loaded_data["coefficients"],
        "intercept": loaded_data["intercept"]
    })
    
    y_pred_reloaded = model_reloaded.predict(X_test)
    np.testing.assert_array_equal(y_pred_orig, y_pred_reloaded)


def test_load_train_validation_shape():
    """Load real with-spike Train/Validation and check shapes."""
    dataset = load_train_validation("with_spike")
    
    train_X, train_y = dataset.train.X, dataset.train.y_regression
    val_X, val_y = dataset.validation.X, dataset.validation.y_regression
    
    # Check shapes
    assert train_X.shape[1] == 8, f"Expected 8 features, got {train_X.shape[1]}"
    assert len(train_y) == train_X.shape[0]
    assert val_X.shape[1] == 8
    assert len(val_y) == val_X.shape[0]
    
    # Fit on train, predict on validation
    model = MultiLinearRegression()
    model.fit(train_X, train_y)
    
    val_pred = model.predict(val_X)
    assert val_pred.shape == val_y.shape
    assert not np.any(np.isnan(val_pred)), "Predictions contain NaN"
