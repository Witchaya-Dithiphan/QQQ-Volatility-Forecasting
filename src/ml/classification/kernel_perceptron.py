








"""RBF kernel Perceptron for the separate convergence experiment."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


def _rbf(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    distances = np.maximum(
        (X * X).sum(axis=1)[:, None] + (Y * Y).sum(axis=1)[None, :] - 2 * X @ Y.T,
        0.0,
    )
    return np.exp(-gamma * distances)


class KernelPerceptron(BaseModel):
    name = "kernel_perceptron"
    task = "classification"

    def __init__(self, gamma: float = 0.5, learning_rate: float = 1.0, max_epochs: int = 500):
        if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("gamma and learning_rate must be positive and finite")
        if type(max_epochs) is not int or max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer")
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def get_params(self) -> dict:
        return {"gamma": self.gamma, "learning_rate": self.learning_rate, "max_epochs": self.max_epochs}

    def fit(self, X, y=None) -> "KernelPerceptron":
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
            raise ValueError("Expected nonempty finite 2D matrix")
        if labels.shape != (len(X),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Expected one binary label per row")
        signed = 2 * labels.astype(np.int64) - 1
        kernel = _rbf(X, X, self.gamma)
        alpha = np.zeros(len(X), dtype=np.float64)
        scores = np.zeros(len(X), dtype=np.float64)
        intercept = 0.0
        self.history_ = {"iteration": [], "loss": [], "updates": []}

        for epoch in range(1, self.max_epochs + 1):
            updates = 0
            for i, yi in enumerate(signed):
                if yi * scores[i] <= 0:
                    step = self.learning_rate * yi
                    alpha[i] += step
                    intercept += step
                    scores += step * (kernel[:, i] + 1.0)
                    updates += 1
            margin = signed * scores
            self.history_["iteration"].append(epoch)
            self.history_["loss"].append(float(np.mean(np.maximum(0.0, -margin))))
            self.history_["updates"].append(updates)
            if updates == 0:
                break

        support = alpha != 0
        self.support_vectors_ = X[support].copy()
        self.alpha_ = alpha[support].copy()
        self.intercept_ = intercept
        self.n_epochs_ = epoch
        self.converged_ = updates == 0
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.support_vectors_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Expected finite 2D matrix with the fitted feature count")
        return _rbf(X, self.support_vectors_, self.gamma) @ self.alpha_ + self.intercept_

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "support_vectors": self.support_vectors_,
            "alpha": self.alpha_,
            "intercept": np.array([self.intercept_]),
            "diagnostics": np.array([self.n_epochs_, int(self.converged_)]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "KernelPerceptron":
        self.support_vectors_ = np.asarray(arrays["support_vectors"], dtype=np.float64)
        self.alpha_ = np.asarray(arrays["alpha"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        self.n_epochs_ = int(arrays["diagnostics"][0])
        self.converged_ = bool(arrays["diagnostics"][1])
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self
