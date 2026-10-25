"""Perceptron — online, mistake-driven updates with no loss surface to descend."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class Perceptron(BaseModel):
    """Walks the samples in order and corrects only where it is wrong.

        for each sample, in the original order (shuffle is off):
            yhat = sign(w·x + b)          with labels mapped to {-1, +1}
            if yhat != y:  w += lr*y*x ,  b += lr*y
        stop when an epoch makes no mistakes, or at max_epochs

    Unlike logistic regression there is no differentiable objective being
    minimised, so the reported loss is the perceptron criterion
    mean(max(0, -margin)) that the config names.
    """

    name = "perceptron"
    task = "classification"

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def get_params(self) -> dict:
        return {"learning_rate": self.learning_rate, "max_epochs": self.max_epochs}

    def fit(self, X, y=None) -> "Perceptron":
        X = np.asarray(X, dtype=np.float64)
        signed = 2 * np.asarray(y, dtype=np.int64) - 1

        weights = np.zeros(X.shape[1])
        intercept = 0.0
        self.history_ = {"iteration": [], "loss": [], "updates": []}

        epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            mistakes = 0
            for xi, yi in zip(X, signed):
                if yi * (xi @ weights + intercept) <= 0:
                    weights += self.learning_rate * yi * xi
                    intercept += self.learning_rate * yi
                    mistakes += 1
            margin = signed * (X @ weights + intercept)
            self.history_["iteration"].append(epoch)
            self.history_["loss"].append(float(np.mean(np.maximum(0.0, -margin))))
            self.history_["updates"].append(mistakes)
            if mistakes == 0:
                break

        self.coef_, self.intercept_, self.n_epochs_ = weights, intercept, epoch
        self.mistake_rate_ = float(np.mean(np.sign(X @ weights + intercept) != signed))
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "coef": self.coef_,
            "intercept": np.array([self.intercept_]),
            "diagnostics": np.array([self.n_epochs_, self.mistake_rate_]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "Perceptron":
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        self.n_epochs_, self.mistake_rate_ = (
            int(arrays["diagnostics"][0]),
            float(arrays["diagnostics"][1]),
        )
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        from sklearn.linear_model import Perceptron as SkPerceptron

        # tol=None matters: sklearn defaults to 1e-3 and would stop on a
        # different epoch, leaving the weights somewhere else entirely.
        return SkPerceptron(
            eta0=self.learning_rate,
            max_iter=self.max_epochs,
            tol=None,
            shuffle=False,
            penalty=None,
            random_state=42,
        )
