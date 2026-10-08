"""
Logistic Regression (scratch implementation).

Binary classification via sigmoid + gradient descent.
"""

import numpy as np
from typing import Optional, Dict, Any


class LogisticRegression:
    """
    Scratch logistic regression: fit via gradient descent.
    
    Model: P(y=1|x) = 1 / (1 + exp(-(w·x + b)))
    Loss: binary cross-entropy
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, random_state: int = 42):
        """
        Args:
            learning_rate: Step size for gradient descent.
            max_iter: Maximum iterations.
            random_state: Random seed.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit logistic regression via gradient descent.
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples,) binary labels (0, 1)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        np.random.seed(self.random_state)
        self.coefficients_ = np.random.randn(n_features) * 0.01
        self.intercept_ = 0.0
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Forward pass
            logits = X @ self.coefficients_ + self.intercept_
            proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))  # Clip for numerical stability
            
            # Compute gradients
            errors = proba - y
            grad_w = (X.T @ errors) / n_samples
            grad_b = errors.mean()
            
            # Update weights
            self.coefficients_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            (n_samples,) predictions (0 or 1)
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.coefficients_ + self.intercept_
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        return (proba >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            (n_samples, 2) probabilities [P(y=0), P(y=1)]
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.coefficients_ + self.intercept_
        proba_1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model."""
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        
        return {
            "coefficients": self.coefficients_.tolist(),
            "intercept": float(self.intercept_),
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LogisticRegression":
        """Deserialize model."""
        model = LogisticRegression()
        model.coefficients_ = np.array(d["coefficients"], dtype=np.float64)
        model.intercept_ = float(d["intercept"])
        return model
