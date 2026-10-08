"""
Multiple Linear Regression (scratch implementation).
"""

import numpy as np
from typing import Optional, Dict, Any


class MultiLinearRegression:
    """
    Scratch multiple linear regression using normal equation.
    
    Fit: β = (X^T X)^-1 X^T y (via np.linalg.lstsq for stability)
    Predict: ŷ = X β + intercept
    """
    
    def __init__(self):
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiLinearRegression":
        """
        Fit model to training data.
        
        Args:
            X: (n_samples, n_features) float64 array
            y: (n_samples,) float64 array
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Prepend ones column for intercept
        n_samples = X.shape[0]
        X_aug = np.column_stack([np.ones(n_samples), X])
        
        # Solve: X_aug @ [intercept, coef1, coef2, ...] = y
        solution, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        
        self.intercept_ = solution[0]
        self.coefficients_ = solution[1:]
        
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
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MultiLinearRegression":
        """Deserialize model from dict."""
        model = MultiLinearRegression()
        model.coefficients_ = np.array(d["coefficients"], dtype=np.float64)
        model.intercept_ = float(d["intercept"])
        return model
