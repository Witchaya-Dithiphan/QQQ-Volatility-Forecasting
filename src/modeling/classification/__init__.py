"""
Logistic Regression (sklearn wrapper for classification).
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    from sklearn.linear_model import LogisticRegression as SKLogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class LogisticRegression:
    """
    Binary Logistic Regression wrapper (sigmoid + cross-entropy).
    Uses sklearn's LBFGS solver.
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        """
        Args:
            C: Inverse regularization strength (1/λ). Higher = less regularization.
            max_iter: Max iterations
            tol: Convergence tolerance
        """
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        
        if HAS_SKLEARN:
            self._sklearn_model = SKLogisticRegression(
                C=C,
                max_iter=max_iter,
                tol=tol,
                solver='lbfgs',
                random_state=42
            )
        else:
            self._sklearn_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit model to training data.
        
        Args:
            X: (n_samples, n_features) float64 array
            y: (n_samples,) binary labels {0, 1}
        
        Returns:
            self
        """
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        
        self._sklearn_model.fit(X, y)
        
        self.coefficients_ = self._sklearn_model.coef_[0].copy()
        self.intercept_ = float(self._sklearn_model.intercept_[0])
        self.classes_ = self._sklearn_model.classes_.copy()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            (n_samples,) predictions
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        
        return self._sklearn_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates."""
        return self._sklearn_model.predict_proba(X)
    
    def to_dict(self) -> Dict[str, Any]:
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        
        return {
            "coefficients": self.coefficients_.tolist(),
            "intercept": float(self.intercept_),
            "classes": self.classes_.tolist(),
            "C": self.C,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LogisticRegression":
        model = LogisticRegression(C=d["C"])
        model.coefficients_ = np.array(d["coefficients"], dtype=np.float64)
        model.intercept_ = float(d["intercept"])
        model.classes_ = np.array(d["classes"], dtype=np.int32)
        return model
