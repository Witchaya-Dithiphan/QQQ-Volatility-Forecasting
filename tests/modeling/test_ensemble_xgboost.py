"""
Tests for XGBoost Classifier (M6, with-spike).
"""

import numpy as np
import pytest

from src.modeling.ensemble.xgboost import XGBoostClassifier

try:
    import xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ============================================================================
# RED TEST (1st — synthetic data, must pass for core to work)
# ============================================================================

def test_fit_predict_shape_synthetic():
    """RED: Fit XGBoost on 100×5 synthetic, predict shape."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    model = XGBoostClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    assert y_pred.shape == (100,)
    assert np.all(np.isin(y_pred, [0, 1]))
    assert len(model.trees_) > 0
    assert len(model.loss_history_) > 0


# ============================================================================
# GREEN TESTS
# ============================================================================

def test_loss_decreases_per_tree():
    """Loss should decrease (or plateau) per tree on separable data."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    # Highly separable
    y = ((X[:, 0] + 2*X[:, 1]) > 0.5).astype(int)
    
    model = XGBoostClassifier(n_estimators=10, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    
    # Loss should not increase (allow tiny numerical fluctuations)
    for i in range(1, len(model.loss_history_)):
        assert model.loss_history_[i] <= model.loss_history_[i-1] + 1e-6, \
            f"Loss increased at tree {i}: {model.loss_history_[i-1]} → {model.loss_history_[i]}"


def test_predict_proba_range():
    """Probabilities should be in [0, 1] and rows sum to 1."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    
    model = XGBoostClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)
    
    proba = model.predict_proba(X)
    
    assert proba.shape == (50, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)


def test_max_depth_respected():
    """Tree depth should respect max_depth."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    
    model = XGBoostClassifier(n_estimators=3, max_depth=2)
    model.fit(X, y)
    
    for tree in model.trees_:
        depth = model._tree_depth(tree)
        assert depth <= 2, f"Tree depth {depth} exceeds max_depth=2"


def test_single_tree_vs_sklearn():
    """Single-tree XGBoost vs sklearn DecisionTree: correlation >0.6."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Our XGBoost with 1 tree
    model = XGBoostClassifier(n_estimators=1, max_depth=3)
    model.fit(X, y)
    our_proba = model.predict_proba(X)[:, 1]
    
    # sklearn DT
    try:
        from sklearn.tree import DecisionTreeClassifier
        sk_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        sk_model.fit(X, y)
        sk_proba = sk_model.predict_proba(X)[:, 1]
        
        # Pearson correlation
        corr = np.corrcoef(our_proba, sk_proba)[0, 1]
        assert corr > 0.6, f"Correlation {corr:.3f} too low"
    except ImportError:
        pytest.skip("sklearn not available")


def test_predict_unfitted_raises():
    """Predict without fitting should raise."""
    model = XGBoostClassifier()
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError, match="not fitted"):
        model.predict_proba(X)


def test_invalid_input_raises():
    """X/y length mismatch should raise."""
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, 8)  # Wrong length
    
    model = XGBoostClassifier()
    
    with pytest.raises(ValueError, match="same number"):
        model.fit(X, y)


def test_save_load_equality():
    """Serialize/deserialize preserves predictions."""
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    model = XGBoostClassifier(n_estimators=3, max_depth=2)
    model.fit(X_train, y_train)
    
    # Dict round-trip
    d = model.to_dict()
    model_loaded = XGBoostClassifier.from_dict(d)
    
    X_test = np.random.randn(10, 3)
    proba_orig = model.predict_proba(X_test)
    proba_loaded = model_loaded.predict_proba(X_test)
    
    np.testing.assert_array_almost_equal(proba_orig, proba_loaded, decimal=6)
