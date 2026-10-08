"""
Simple Linear Regression (scratch implementation).
"""

import numpy as np
from typing import Optional, Dict, Any


class SimpleLinearRegression:
    """
    Scratch simple linear regression (univariate).
    
    Fit: β₁ = cov(X, y) / var(X), β₀ = mean(y) - β₁ * mean(X)
    Predict: ŷ = β₀ + β₁ * X
    """
    
    def __init__(self):
        self.coefficient_: Optional[float] = None  # β₁
        self.intercept_: Optional[float] = None    # β₀
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLinearRegression":
        """
        Fit model to training data.
        
        Args:
            X: (n_samples,) or (n_samples, 1) float64 array
            y: (n_samples,) float64 array
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same length: {X.shape[0]} != {y.shape[0]}")
        
        # Compute covariance and variance
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        cov_xy = np.mean((X - X_mean) * (y - y_mean))
        var_x = np.mean((X - X_mean) ** 2)
        
        if var_x == 0:
            raise ValueError("Feature has zero variance; cannot fit")
        
        self.coefficient_ = cov_xy / var_x
        self.intercept_ = y_mean - self.coefficient_ * X_mean
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        
        Args:
            X: (n_samples,) or (n_samples, 1) float64 array
        
        Returns:
            (n_samples,) predictions
        """
        if self.coefficient_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64).ravel()
        return self.intercept_ + self.coefficient_ * X
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dict."""
        if self.coefficient_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted.")
        
        return {
            "coefficient": float(self.coefficient_),
            "intercept": float(self.intercept_),
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SimpleLinearRegression":
        """Deserialize model from dict."""
        model = SimpleLinearRegression()
        model.coefficient_ = float(d["coefficient"])
        model.intercept_ = float(d["intercept"])
        return model
