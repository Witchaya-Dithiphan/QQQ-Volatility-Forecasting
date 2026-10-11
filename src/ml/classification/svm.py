"""Linear soft-margin SVM solved by subgradient descent on the primal.

    J(w, b) = 0.5 * ||w||^2 + C * sum(max(0, 1 - y_i (w.x_i + b)))      y in {-1, +1}

The intercept is not penalised. A point only pulls on the solution while its margin
is below 1; once it is safely beyond, it contributes nothing. Features are centred
inside the solver (and the intercept shifted back afterwards), so the fit does not
depend on where the caller's origin happens to be.
"""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class SVM(BaseModel):
    name = "svm"
    task = "classification"

    def __init__(self, C: float = 1.0, max_iter: int = 20000, fit_intercept: bool = True,
                 random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "fit_intercept": self.fit_intercept,
            "random_state": self.random_state,
        }

    @staticmethod
    def step_size(iteration: int) -> float:
        """1/t: the norm term makes J 1-strongly convex, and a constant step would orbit the kinks."""
        return 1.0 / iteration

    def fit(self, X, y=None) -> "SVM":
        if (not np.isfinite(self.C) or self.C <= 0
                or not isinstance(self.max_iter, int) or self.max_iter < 1
                or not isinstance(self.fit_intercept, bool)):
            raise ValueError("Invalid SVM hyperparameters")
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or labels.shape != (len(X),) or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array aligned with y")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("SVM requires binary labels 0 and 1")
        signed = 2.0 * labels - 1.0

        # The intercept is free, so w.(x - mean) + b is the same model as w.x + b' and has
        # the same optimum. Solving on centred features matters in practice: an offset of
        # a few units couples the intercept to every weight, and the 1/t steps then crawl.
        offset = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        X = X - offset

        weights, intercept = np.zeros(X.shape[1]), 0.0
        best = (np.inf, weights, intercept)
        stride = max(1, self.max_iter // 500)
        self.history_ = {"iteration": [], "loss": []}

        for iteration in range(1, self.max_iter + 1):
            margin = signed * (X @ weights + intercept)
            # The margins are already in hand, so scoring the current point is free.
            objective = float(0.5 * weights @ weights + self.C * np.maximum(0.0, 1.0 - margin).sum())
            if objective < best[0]:
                # A subgradient step can go uphill, so keep the lowest point visited.
                best = (objective, weights, intercept)
            if iteration == 1 or iteration % stride == 0:
                self.history_["iteration"].append(iteration)
                self.history_["loss"].append(objective)

            inside = margin < 1.0
            pull = signed[inside]
            step = self.step_size(iteration)
            weights = weights - step * (weights - self.C * (pull @ X[inside]))
            if self.fit_intercept:
                intercept = intercept + step * self.C * float(pull.sum())

        margin = signed * (X @ weights + intercept)
        objective = float(0.5 * weights @ weights + self.C * np.maximum(0.0, 1.0 - margin).sum())
        if objective < best[0]:
            best = (objective, weights, intercept)
        self.objective_, self.coef_ = best[0], best[1]
        # Back to the caller's coordinates.
        self.intercept_ = float(best[2] - best[1] @ offset)
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "coef": self.coef_,
            "intercept": np.array([self.intercept_]),
            "objective": np.array([self.objective_]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "SVM":
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        self.objective_ = float(arrays["objective"][0])
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        """libsvm's linear SVC, which minimises exactly J above.

        LinearSVC is the obvious choice but solves a different problem: liblinear
        penalises the intercept, and the intercept_scaling=1000 that hides this
        stops its solver converging (measured: worse objective than ours after
        2,000,000 iterations).
        """
        from sklearn.svm import SVC

        if not self.fit_intercept:
            from sklearn.svm import LinearSVC

            return LinearSVC(C=self.C, loss="hinge", fit_intercept=False, tol=1e-10,
                             max_iter=200000, random_state=self.random_state)
        return SVC(kernel="linear", C=self.C, tol=1e-8, cache_size=1000)
