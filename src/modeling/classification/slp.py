"""
Single-Layer Perceptron (scratch implementation, binary).

tanh activation, MSE loss, +/-1 targets, batch gradient descent.
(Contrast with LogisticRegression: sigmoid + cross-entropy, 0/1 targets.)
"""

from typing import Any, Dict, List, Optional

import numpy as np


class SLP:
    def __init__(self, learning_rate: float = 0.1, max_iter: int = 500, random_state: int = 42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SLP":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")

        t = 2.0 * y - 1.0  # {0,1} -> {-1,+1}
        rng = np.random.default_rng(self.random_state)
        self.coefficients_ = rng.standard_normal(X.shape[1]) * 0.01
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.max_iter):
            a = np.tanh(X @ self.coefficients_ + self.intercept_)
            self.loss_history_.append(float(np.mean((t - a) ** 2)))
            # dL/dz = -2 (t - a) (1 - a^2) / n
            delta = -2.0 * (t - a) * (1.0 - a**2) / len(t)
            self.coefficients_ -= self.learning_rate * (X.T @ delta)
            self.intercept_ -= self.learning_rate * delta.sum()
        return self

    def _activation(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        return np.tanh(np.asarray(X, dtype=np.float64) @ self.coefficients_ + self.intercept_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._activation(X) > 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Maps tanh output in [-1, 1] to P(y=1) = (a + 1) / 2."""
        p1 = (self._activation(X) + 1.0) / 2.0
        return np.column_stack([1.0 - p1, p1])

    def to_dict(self) -> Dict[str, Any]:
        if self.coefficients_ is None:
            raise ValueError("Model not fitted")
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "coefficients": self.coefficients_.tolist(),
            "intercept": float(self.intercept_),
            "loss_history": list(self.loss_history_),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SLP":
        model = SLP(d["learning_rate"], d["max_iter"], d["random_state"])
        model.coefficients_ = np.array(d["coefficients"], dtype=np.float64)
        model.intercept_ = float(d["intercept"])
        model.loss_history_ = list(d["loss_history"])
        return model
