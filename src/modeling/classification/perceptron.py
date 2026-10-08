"""
Perceptron (scratch implementation, binary).

Online update: w += lr * (y - y_hat) * x, stopping after an epoch with no mistakes.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class Perceptron:
    def __init__(self, learning_rate: float = 0.1, max_iter: int = 100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.loss_history_: List[int] = []  # mistakes per epoch

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")

        self.coefficients_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.max_iter):
            errors = 0
            for x_i, y_i in zip(X, y):
                update = self.learning_rate * (y_i - int(x_i @ self.coefficients_ + self.intercept_ > 0))
                if update:
                    self.coefficients_ += update * x_i
                    self.intercept_ += update
                    errors += 1
            self.loss_history_.append(errors)
            if errors == 0:
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)
        return (X @ self.coefficients_ + self.intercept_ > 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Hard 0/1 'probabilities' (a perceptron has no calibrated score)."""
        pred = self.predict(X)
        return np.column_stack([1 - pred, pred]).astype(np.float64)

    def to_dict(self) -> Dict[str, Any]:
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "coefficients": self.coefficients_.tolist(),
            "intercept": float(self.intercept_),
            "loss_history": list(self.loss_history_),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Perceptron":
        model = Perceptron(d["learning_rate"], d["max_iter"])
        model.coefficients_ = np.array(d["coefficients"], dtype=np.float64)
        model.intercept_ = float(d["intercept"])
        model.loss_history_ = list(d["loss_history"])
        return model
