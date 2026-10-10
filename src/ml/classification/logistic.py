"""Logistic regression by full-batch gradient descent on weighted cross-entropy."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class Logistic(BaseModel):
    """Minimises the weighted mean cross-entropy plus an L2 term on the weights.

        p = sigmoid(Xw + b)
        J = sum(wi * BCE(yi, pi)) / sum(wi)  +  (lambda/2) * ||w||^2

    The intercept is deliberately outside the penalty. The objective is convex,
    so gradient descent and sklearn's LBFGS land on the same optimum even though
    they take different paths — which is why LogisticRegression is the reference
    here rather than SGDClassifier, whose per-sample updates never converge to
    the same iterate.
    """

    name = "logistic"
    task = "classification"

    def __init__(
        self,
        l2: float = 0.0,
        class_weight: str | None = None,
        learning_rate: float = 0.05,
        max_iter: int = 5000,
        tol: float = 1e-6,
    ):
        self.l2 = l2
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def get_params(self) -> dict:
        return {
            "l2": self.l2,
            "class_weight": self.class_weight,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Branch on sign so neither exp overflows for large |z|.
        out = np.empty_like(z, dtype=np.float64)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def _objective(self, X, y, weights, intercept, sample_weight) -> float:
        margin = X @ weights + intercept
        # logaddexp form of BCE: numerically safe where log(sigmoid) would underflow.
        losses = np.logaddexp(0.0, margin) - y * margin
        return float(np.sum(sample_weight * losses) + 0.5 * self.l2 * weights @ weights)

    def _gradient(self, X, y, weights, intercept, sample_weight):
        residual = sample_weight * (self._sigmoid(X @ weights + intercept) - y)
        return X.T @ residual + self.l2 * weights, float(residual.sum())

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones(len(y), dtype=np.float64)
        if self.class_weight != "balanced":
            raise ValueError(f"Unknown class_weight {self.class_weight!r}; expected None or 'balanced'")
        counts = np.bincount(y, minlength=2)
        return (len(y) / (2.0 * counts))[y]

    def fit(self, X, y=None) -> "Logistic":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        raw_weight = self._sample_weights(y.astype(np.int64))
        self.sample_weight_ = raw_weight
        normalised = raw_weight / raw_weight.sum()

        weights = np.zeros(X.shape[1])
        intercept = 0.0
        self.history_ = {"iteration": [], "loss": []}
        for iteration in range(self.max_iter):
            gradient_w, gradient_b = self._gradient(X, y, weights, intercept, normalised)
            weights -= self.learning_rate * gradient_w
            intercept -= self.learning_rate * gradient_b
            if iteration % 50 == 0 or iteration == self.max_iter - 1:
                self.history_["iteration"].append(iteration)
                self.history_["loss"].append(self._objective(X, y, weights, intercept, normalised))
            if max(np.abs(gradient_w).max(), abs(gradient_b)) < self.tol:
                break

        self.coef_, self.intercept_ = weights, intercept
        self.n_samples_ = len(y)
        self.sample_weight_sum_ = float(raw_weight.sum())
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        return self._sigmoid(self.decision_function(X))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"coef": self.coef_, "intercept": np.array([self.intercept_])}

    def load_state(self, arrays: dict, metadata: dict) -> "Logistic":
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        """C scales the loss term, not the penalty, so C = 1/(lambda * sum(sample_weight)).

        Derivation: ours is mean_w(BCE) + (lambda/2)||w||^2; sklearn's is
        C * sum_i w_i * logloss_i + 0.5||w||^2. Dividing ours by lambda and
        matching term by term gives C = 1/(lambda * sum(w)). With class_weight
        None or "balanced" that sum is n either way.
        """
        from sklearn.linear_model import LogisticRegression

        total_weight = getattr(self, "sample_weight_sum_", None) or getattr(self, "n_samples_", 1)
        shared = dict(
            fit_intercept=True,
            tol=1e-12,
            max_iter=200000,
            solver="lbfgs",
            class_weight=self.class_weight,
        )
        if self.l2 == 0:
            return LogisticRegression(C=np.inf, **shared)
        # sklearn 1.9 deprecated penalty=; l1_ratio=0 is how you spell L2 now.
        return LogisticRegression(C=1.0 / (self.l2 * total_weight), l1_ratio=0, **shared)
