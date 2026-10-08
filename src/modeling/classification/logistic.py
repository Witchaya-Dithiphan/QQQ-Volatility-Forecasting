"""
Logistic Regression (scratch, NumPy only).

Objective (plan Section 14): weighted mean BCE + 0.5 * l2 * ||w||^2, intercept not penalized,
full-batch gradient descent, stop when the gradient infinity-norm falls below `tol`.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from ._common import NOT_FITTED, apply_scaler, check_param, check_X, check_Xy, fit_scaler, scaler_from_dict, scaler_to_dict, sigmoid


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.05, max_iter: int = 5000, tol: float = 1e-6, l2: float = 0.0,
                 class_weight: Optional[str] = None, standardize: bool = True, random_state: int = 42):
        """`random_state` is accepted for runner/stability API compatibility; the convex fit starts at zero and is deterministic."""
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2
        self.class_weight = class_weight
        self.standardize = standardize
        self.random_state = random_state

        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.scaler_ = None
        self.classes_ = np.array([0, 1])
        self.loss_history_: List[float] = []
        self.n_iter_ = 0
        self.converged_ = False

    def _validate_params(self):
        check_param("learning_rate", self.learning_rate)
        check_param("max_iter", self.max_iter, integer=True)
        check_param("tol", self.tol)
        check_param("l2", self.l2, allow_zero=True)
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit full-batch descent; raise ValueError on nonfinite loss, gradient or candidate state."""
        self._validate_params()
        X, y = check_Xy(X, y)
        self.scaler_, X = fit_scaler(X, self.standardize)
        n = len(y)
        # balanced: n / (2 * class_count), as in the frozen config
        sample_w = np.ones(n) if self.class_weight is None else (n / (2.0 * np.bincount(y)))[y]
        total = sample_w.sum()

        w, b = np.zeros(X.shape[1]), 0.0
        self.loss_history_, self.converged_, self.n_iter_ = [], False, 0
        for it in range(1, self.max_iter + 1):
            with np.errstate(all="ignore"):  # overflow is reported below through explicit finite checks
                p = sigmoid(X @ w + b)
                loss = float(sample_w @ -(y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))) / total
                             + 0.5 * self.l2 * w @ w)
                residual = sample_w * (p - y) / total
                grad_w, grad_b = X.T @ residual + self.l2 * w, residual.sum()
            if not (np.isfinite(loss) and np.isfinite(grad_w).all() and np.isfinite(grad_b)):
                raise ValueError("Gradient descent diverged; reduce learning_rate")
            self.loss_history_.append(loss)
            self.n_iter_ = it
            if max(np.abs(grad_w).max(), abs(grad_b)) < self.tol:
                self.converged_ = True
                break
            with np.errstate(all="ignore"):
                candidate_w = w - self.learning_rate * grad_w
                candidate_b = b - self.learning_rate * grad_b
            if not (np.isfinite(candidate_w).all() and np.isfinite(candidate_b)):
                raise ValueError("Gradient descent diverged; reduce learning_rate")
            w, b = candidate_w, candidate_b
        self.coefficients_, self.intercept_ = w, float(b)
        return self

    def _fitted(self, X):
        if self.coefficients_ is None:
            raise ValueError(NOT_FITTED)
        X = check_X(X, len(self.coefficients_))
        return apply_scaler(self.scaler_, X) if self.scaler_ is not None else X

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._fitted(X) @ self.coefficients_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def to_dict(self) -> Dict[str, Any]:
        if self.coefficients_ is None:
            raise ValueError(NOT_FITTED)
        return {"learning_rate": self.learning_rate, "max_iter": self.max_iter, "tol": self.tol, "l2": self.l2,
                "class_weight": self.class_weight, "standardize": self.standardize, "random_state": self.random_state,
                "coefficients": self.coefficients_.tolist(), "intercept": self.intercept_, "scaler": scaler_to_dict(self.scaler_),
                "loss_history": list(self.loss_history_), "n_iter": self.n_iter_, "converged": self.converged_}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LogisticRegression":
        model = LogisticRegression(d["learning_rate"], d["max_iter"], d["tol"], d["l2"], d["class_weight"], d["standardize"], d["random_state"])
        model.coefficients_, model.intercept_ = np.array(d["coefficients"], dtype=np.float64), float(d["intercept"])
        model.scaler_ = scaler_from_dict(d["scaler"])
        model.loss_history_, model.n_iter_, model.converged_ = list(d["loss_history"]), d["n_iter"], d["converged"]
        return model
