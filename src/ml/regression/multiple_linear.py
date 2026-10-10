"""Multiple linear regression by least squares."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class MultipleLinear(BaseModel):
    """Minimises ||y - Xb||^2.

    Solved with lstsq rather than the normal equations: the features are collinear
    (condition number 35.6 on this train set) and inv(X'X) squares that conditioning,
    while lstsq's SVD handles it directly — which is also what sklearn does, so parity
    comes for free.
    """

    name = "multiple_linear"
    task = "regression"

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def get_params(self) -> dict:
        return {"fit_intercept": self.fit_intercept}

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y=None) -> "MultipleLinear":
        y = np.asarray(y, dtype=np.float64)
        solution, *_ = np.linalg.lstsq(self._design(X), y, rcond=None)
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(solution[0]), solution[1:]
        else:
            self.intercept_, self.coef_ = 0.0, solution
        self.history_ = {"iteration": [0], "loss": [float(np.mean((y - self.predict(X)) ** 2))]}
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"coef": np.asarray(self.coef_), "intercept": np.array([self.intercept_])}

    def load_state(self, arrays: dict, metadata: dict) -> "MultipleLinear":
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        self.fit_intercept = metadata["params"]["fit_intercept"]
        return self

    def reference(self):
        from sklearn.linear_model import LinearRegression

        return LinearRegression(fit_intercept=self.fit_intercept)
