"""
Polynomial Regression (scratch implementation).
Fit univariate polynomial basis [x, x², x³] via OLS.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple


class PolynomialRegression:
    """
    Scratch polynomial regression: fit degree-2 or degree-3 univariate polynomial.
    
    Approach:
    1. Choose single feature (via external validation loop)
    2. Standardize feature
    3. Generate polynomial basis [x, x², x³ (if degree=3)]
    4. Fit OLS via pseudoinverse
    5. Standardize predictions to original scale
    """
    
    def __init__(self, degree: int = 2):
        """
        Args:
            degree: Polynomial degree (2 or 3). Default 2.
        """
        if degree not in (2, 3):
            raise ValueError(f"degree must be 2 or 3, got {degree}")
        
        self.degree = degree
        self.coefficients_: Optional[np.ndarray] = None  # [c0, c1, c2, ...c_degree]
        self.feature_mean_: Optional[float] = None
        self.feature_std_: Optional[float] = None
        self.target_mean_: Optional[float] = None
        self.target_std_: Optional[float] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        """
        Fit polynomial regression.
        
        Args:
            X: (n_samples,) or (n_samples, 1) univariate feature
            y: (n_samples,) target
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same length")
        
        if X.shape[0] < self.degree + 1:
            raise ValueError(f"Need at least {self.degree + 1} samples for degree {self.degree}")
        
        # Standardize X and y
        self.feature_mean_ = np.mean(X)
        self.feature_std_ = np.std(X)
        if self.feature_std_ == 0:
            raise ValueError("Feature has zero variance")
        
        X_std = (X - self.feature_mean_) / self.feature_std_
        
        self.target_mean_ = np.mean(y)
        self.target_std_ = np.std(y)
        y_std = (y - self.target_mean_) / (self.target_std_ + 1e-10)
        
        # Build polynomial design matrix: [1, x, x², x³, ...]
        n = len(X_std)
        design_matrix = np.column_stack([X_std ** i for i in range(self.degree + 1)])
        
        # OLS via pseudoinverse
        try:
            coeffs = np.linalg.lstsq(design_matrix, y_std, rcond=None)[0]
        except np.linalg.LinAlgError:
            raise ValueError("Cannot solve polynomial regression (singular matrix)")
        
        self.coefficients_ = coeffs
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using fitted polynomial.
        
        Args:
            X: (n_samples,) or (n_samples, 1)
        
        Returns:
            (n_samples,) predictions
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64).ravel()
        
        # Standardize X
        X_std = (X - self.feature_mean_) / self.feature_std_
        
        # Evaluate polynomial
        design_matrix = np.column_stack([X_std ** i for i in range(self.degree + 1)])
        y_std_pred = design_matrix @ self.coefficients_
        
        # Unstandardize y
        y_pred = y_std_pred * self.target_std_ + self.target_mean_
        
        return y_pred
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model."""
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        
        return {
            "degree": self.degree,
            "coefficients": self.coefficients_.tolist(),
            "feature_mean": float(self.feature_mean_),
            "feature_std": float(self.feature_std_),
            "target_mean": float(self.target_mean_),
            "target_std": float(self.target_std_),
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PolynomialRegression":
        """Deserialize model."""
        model = PolynomialRegression(degree=d["degree"])
        model.coefficients_ = np.array(d["coefficients"], dtype=np.float64)
        model.feature_mean_ = float(d["feature_mean"])
        model.feature_std_ = float(d["feature_std"])
        model.target_mean_ = float(d["target_mean"])
        model.target_std_ = float(d["target_std"])
        return model
