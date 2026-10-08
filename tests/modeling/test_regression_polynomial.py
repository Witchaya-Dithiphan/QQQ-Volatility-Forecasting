"""
Tests for PolynomialRegression (M4-03, with-spike).
"""

import numpy as np
import pytest
from pathlib import Path

from src.modeling.regression.polynomial import PolynomialRegression
from src.modeling.datasets import load_train_validation
from src.modeling.persistence import save_npz
from src.modeling.contracts import FEATURE_COLUMNS

try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# RED TEST (1st — should fail initially)
# ============================================================================

def test_fit_synthetic_quadratic():
    """RED: Fit degree-2 polynomial on synthetic data."""
    np.random.seed(42)
    x = np.linspace(-2, 2, 20)
    y = 2 * x**2 - 1 * x + 3 + np.random.randn(20) * 0.5
    
    model = PolynomialRegression(degree=2)
    model.fit(x, y)
    
    assert model.coefficients_ is not None
    assert model.coefficients_.shape == (3,)  # [c0, c1, c2]
    assert model.feature_mean_ is not None
    assert model.target_mean_ is not None


# ============================================================================
# GREEN TESTS
# ============================================================================

def test_fit_cubic():
    """Fit degree-3 polynomial."""
    np.random.seed(42)
    x = np.linspace(-1, 1, 25)
    y = x**3 - 2*x + 1 + np.random.randn(25) * 0.2
    
    model = PolynomialRegression(degree=3)
    model.fit(x, y)
    
    assert model.coefficients_.shape == (4,)  # [c0, c1, c2, c3]


def test_predict_shape_matches():
    """Predictions should match input shape."""
    np.random.seed(42)
    x_train = np.random.randn(20)
    y_train = x_train**2 + np.random.randn(20) * 0.1
    
    model = PolynomialRegression(degree=2)
    model.fit(x_train, y_train)
    
    x_test = np.random.randn(5)
    y_pred = model.predict(x_test)
    
    assert y_pred.shape == (5,)


def test_predict_unfitted_raises():
    """Predict without fitting should raise."""
    model = PolynomialRegression(degree=2)
    x = np.random.randn(5)
    
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(x)


def test_zero_variance_raises():
    """Fit on constant feature should raise."""
    x = np.ones(10)
    y = np.random.randn(10)
    
    model = PolynomialRegression(degree=2)
    
    with pytest.raises(ValueError, match="zero variance"):
        model.fit(x, y)


def test_insufficient_samples_raises():
    """Degree-3 needs ≥4 samples."""
    x = np.array([1.0, 2.0, 3.0])  # 3 samples < 4 needed
    y = np.array([1.0, 4.0, 9.0])
    
    model = PolynomialRegression(degree=3)
    
    with pytest.raises(ValueError, match="at least 4 samples"):
        model.fit(x, y)


def test_invalid_degree_raises():
    """Degree must be 2 or 3."""
    with pytest.raises(ValueError, match="degree must be 2 or 3"):
        PolynomialRegression(degree=4)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_compare_sklearn_polynomial():
    """Compare predictions with sklearn PolynomialFeatures + LinearRegression."""
    np.random.seed(42)
    x = np.linspace(-2, 2, 30)
    y = x**2 - 0.5*x + 2 + np.random.randn(30) * 0.3
    
    # Our model
    model = PolynomialRegression(degree=2)
    model.fit(x, y)
    y_pred_ours = model.predict(x)
    
    # sklearn's model
    poly_feat = PolynomialFeatures(degree=2, include_bias=True)
    x_poly = poly_feat.fit_transform(x.reshape(-1, 1))
    
    sk_model = LinearRegression()
    sk_model.fit(x_poly, y)
    y_pred_sk = sk_model.predict(x_poly)
    
    # Predictions should be close (within 1% RMSE)
    rmse_diff = np.sqrt(np.mean((y_pred_ours - y_pred_sk)**2))
    assert rmse_diff < 0.1, f"RMSE diff {rmse_diff} too large"


def test_save_load_equality(tmp_path):
    """Serialize/deserialize preserves model."""
    np.random.seed(42)
    x_train = np.random.randn(20)
    y_train = x_train**2 + np.random.randn(20) * 0.1
    
    model = PolynomialRegression(degree=2)
    model.fit(x_train, y_train)
    
    # Dict round-trip
    d = model.to_dict()
    model_loaded = PolynomialRegression.from_dict(d)
    
    x_test = np.random.randn(5)
    y_pred_orig = model.predict(x_test)
    y_pred_loaded = model_loaded.predict(x_test)
    
    np.testing.assert_array_almost_equal(y_pred_orig, y_pred_loaded)


def test_load_train_validation_best_degree():
    """Load real data and fit both degree-2 and degree-3, choose best on validation."""
    dataset = load_train_validation("with_spike")
    train_X, train_y = dataset.train.X, dataset.train.y_regression
    val_X, val_y = dataset.validation.X, dataset.validation.y_regression
    
    # Try each feature, each degree
    best_rmse = float('inf')
    best_model = None
    
    for feat_idx in range(min(3, train_X.shape[1])):  # Try first 3 features
        for degree in [2, 3]:
            try:
                model = PolynomialRegression(degree=degree)
                model.fit(train_X[:, feat_idx], train_y)
                
                val_pred = model.predict(val_X[:, feat_idx])
                val_rmse = np.sqrt(np.mean((val_y - val_pred)**2))
                
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_model = model
            except (ValueError, np.linalg.LinAlgError):
                # Skip bad fits
                continue
    
    assert best_model is not None, "Could not fit any polynomial"
    assert best_rmse < float('inf')
    assert not np.any(np.isnan(best_model.predict(val_X[:, 0])))
