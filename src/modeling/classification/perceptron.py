"""
Perceptron (scratch, NumPy only): linear decision score, hard step output, online mistake-driven update.

Update on every sample in fixed order (no shuffle): if t * score <= 0 with t = +/-1, then w += lr * t * x, b += lr * t.
The output is a decision score, NOT a probability; there is intentionally no predict_proba.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from ._common import NOT_FITTED, apply_scaler, check_param, check_X, check_Xy, fit_scaler, scaler_from_dict, scaler_to_dict


class Perceptron:
    def __init__(self, learning_rate: float = 0.1, max_iter: int = 1000, standardize: bool = True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter  # max epochs
        self.standardize = standardize

        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.scaler_ = None
        self.mistake_rate_history_: List[float] = []  # online updates / n_samples, per epoch

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        check_param("learning_rate", self.learning_rate)
        check_param("max_iter", self.max_iter, integer=True)
        X, y = check_Xy(X, y)
        self.scaler_, X = fit_scaler(X, self.standardize)
        t = 2.0 * y - 1.0
        w, b = np.zeros(X.shape[1]), 0.0
        self.mistake_rate_history_ = []
        for _ in range(self.max_iter):
            mistakes = 0
            for x_i, t_i in zip(X, t):
                if t_i * (x_i @ w + b) <= 0:
                    w, b = w + self.learning_rate * t_i * x_i, b + self.learning_rate * t_i
                    mistakes += 1
            self.mistake_rate_history_.append(mistakes / len(y))
            if mistakes == 0:
                break
        self.coefficients_, self.intercept_ = w, float(b)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise ValueError(NOT_FITTED)
        X = check_X(X, len(self.coefficients_))
        if self.scaler_ is not None:
            X = apply_scaler(self.scaler_, X)
        return X @ self.coefficients_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)  # M2 convention: score >= threshold (0.0)

    def to_dict(self) -> Dict[str, Any]:
        if self.coefficients_ is None:
            raise ValueError(NOT_FITTED)
        return {"learning_rate": self.learning_rate, "max_iter": self.max_iter, "standardize": self.standardize,
                "coefficients": self.coefficients_.tolist(), "intercept": self.intercept_, "scaler": scaler_to_dict(self.scaler_),
                "mistake_rate_history": list(self.mistake_rate_history_)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Perceptron":
        model = Perceptron(d["learning_rate"], d["max_iter"], d["standardize"])
        model.coefficients_, model.intercept_ = np.array(d["coefficients"], dtype=np.float64), float(d["intercept"])
        model.scaler_ = scaler_from_dict(d["scaler"])
        model.mistake_rate_history_ = list(d["mistake_rate_history"])
        return model
