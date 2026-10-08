"""
Tests for ElasticNet (M4-04, with-spike).
"""

import numpy as np
import pytest
from pathlib import Path

from src.modeling.regression.elastic_net import ElasticNet
from src.modeling.datasets import load_train_validation
from src.modeling.persistence import save_npz
from src.modeling.contracts import FEATURE_COLUMNS

try:
    from sklearn.linear_model import ElasticNet as SKElasticNet
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# RED TEST
# ============================================================================

def test_fit_synthetic_elastic_net():
    """RED: Fit on synthetic data; expect coefficients shape."""
    X = np.random.randn(20, 3)
    y = X @ np.array([2.0, -1.0, 0.5]) + np.random.randn(20) * 0.1
    
    model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    model.fit(X, y)
    
    assert hasattr(model, 'coefficients_')
    assert model.coefficients_.shape == (3,)
    assert hasattr(model, 'intercept_')


# ============================================================================
# GREEN TESTS
# ============================================================================

def test_predict_shape_matches():
    """Predict shape should match input."""
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    
    model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    model.fit(X, y)
    
    X_test = np.random.randn(5, 3)
    y_pred = model.predict(X_test)
    
    assert y_pred.shape == (5,)
    assert isinstance(y_pred, np.ndarray)


def test_l1_l2_penalties_applied():
    """Verify regularization reduces overfitting."""
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    
    # No regularization
    model_no_reg = ElasticNet(alpha=0.0)
    model_no_reg.fit(X, y)
    
    # Strong regularization
    model_strong = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model_strong.fit(X, y)
    
    # L1 penalty should shrink coefficients
    coef_norm_no_reg = np.linalg.norm(model_no_reg.coefficients_)
    coef_norm_strong = np.linalg.norm(model_strong.coefficients_)
    
    assert coef_norm_strong < coef_norm_no_reg, "Regularization should shrink coefficients"


def test_predict_unfitted_raises():
    """Predict on unfitted model should raise."""
    model = ElasticNet()
    X = np.random.randn(5, 3)
    
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)


def test_feature_mismatch_raises():
    """Predict with wrong feature count should raise."""
    X_train = np.random.randn(20, 3)
    y_train = np.random.randn(20)
    
    model = ElasticNet()
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(5, 5)  # Wrong feature count
    
    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict(X_test)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_compare_sklearn_elastic_net():
    """Compare with sklearn ElasticNet (rough comparison)."""
    X = np.random.randn(50, 4)
    y = X @ np.array([2.0, -1.0, 0.5, 1.5]) + np.random.randn(50) * 0.1
    
    # Our model
    model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    model.fit(X, y)
    
    # sklearn's model
    sk_model = SKElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
    sk_model.fit(X, y)
    
    # Predictions should be reasonably close
    y_pred = model.predict(X)
    y_pred_sk = sk_model.predict(X)
    
    rmse_ours = np.sqrt(np.mean((y - y_pred) ** 2))
    rmse_sk = np.sqrt(np.mean((y - y_pred_sk) ** 2))
    
    # Allow up to 50% difference (our solver is simpler)
    assert rmse_ours < rmse_sk * 1.5, f"Our RMSE {rmse_ours} too much worse than sklearn {rmse_sk}"


def test_save_load_equality(tmp_path):
    """Serialize/deserialize should preserve model."""
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    
    model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    model.fit(X, y)
    
    d = model.to_dict()
    model_loaded = ElasticNet.from_dict(d)
    
    X_test = np.random.randn(5, 3)
    y_pred_orig = model.predict(X_test)
    y_pred_loaded = model_loaded.predict(X_test)
    
    np.testing.assert_array_equal(y_pred_orig, y_pred_loaded)


def test_load_train_validation_all_features():
    """Load real with-spike Train/Validation and fit on all 8 features."""
    dataset = load_train_validation("with_spike")
    
    train_X, train_y = dataset.train.X, dataset.train.y_regression
    val_X, val_y = dataset.validation.X, dataset.validation.y_regression
    
    # Fit on train
    model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    model.fit(train_X, train_y)
    
    # Predict on validation
    val_pred = model.predict(val_X)
    
    assert val_pred.shape == val_y.shape
    assert not np.any(np.isnan(val_pred)), "Predictions contain NaN"
    
    # Check regularization effect
    train_pred = model.predict(train_X)
    train_mse = np.mean((train_y - train_pred) ** 2)
    val_mse = np.mean((val_y - val_pred) ** 2)
    
    assert train_mse > 0, "Train MSE should be positive"
    assert val_mse > 0, "Val MSE should be positive"
