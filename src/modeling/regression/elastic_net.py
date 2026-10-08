"""
Elastic Net (wrapper around sklearn for stability).
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    from sklearn.linear_model import ElasticNet as SKElasticNet
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ElasticNet:
    """
    Elastic Net wrapper: MSE + λ₁||β||₁ + λ₂||β||₂²
    
    Uses sklearn's coordinate descent solver (proven stable & fast).
    """
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000, tol: float = 1e-4):
        """
        Args:
            alpha: Overall regularization strength (λ)
            l1_ratio: Fraction of L1 penalty (0 = Ridge/L2, 1 = Lasso/L1, 0.5 = Elastic Net)
            max_iter: Max iterations
            tol: Convergence tolerance
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        
        if HAS_SKLEARN:
            self._sklearn_model = SKElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                tol=tol,
                random_state=42
            )
        else:
            self._sklearn_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
        """
        Fit model using sklearn's coordinate descent.
        
        Args:
            X: (n_samples, n_features) float64 array
            y: (n_samples,) float64 array
        
        Returns:
            self
        """
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required for ElasticNet; install with: pip install scikit-learn")
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Fit sklearn model
        self._sklearn_model.fit(X, y)
        
        # Extract coefficients and intercept
        self.coefficients_ = self._sklearn_model.coef_.copy()
        self.intercept_ = float(self._sklearn_model.intercept_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        
        Args:
            X: (n_samples, n_features) float64 array
        
        Returns:
            (n_samples,) predictions
        """
        if self.coefficients_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.shape[1] != self.coefficients_.shape[0]:
            raise ValueError(
                f"Feature mismatch: expected {self.coefficients_.shape[0]}, "
                f"got {X.shape[1]}"
            )
        
        return X @ self.coefficients_ + self.intercept_
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dict."""
        if self.coefficients_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted.")
        
        return {
            "coefficients": self.coefficients_.tolist(),
            "intercept": float(self.intercept_),
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ElasticNet":
        """Deserialize model from dict."""
        model = ElasticNet(alpha=d["alpha"], l1_ratio=d["l1_ratio"])
        model.coefficients_ = np.array(d["coefficients"], dtype=np.float64)
        model.intercept_ = float(d["intercept"])
        return model
