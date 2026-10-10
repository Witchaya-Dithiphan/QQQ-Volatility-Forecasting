"""Polynomial regression on a single feature — univariate powers, no interactions."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from ..core.contracts import FEATURE_COLUMNS


class Polynomial(BaseModel):
    """Fits y = b0 + b1*z + b2*z^2 + ... + bd*z^d for one standardised column z.

    The config asks for standardize_before_powers, and the order is not cosmetic:
    (standardised x)^2 and the standardisation of x^2 are different columns, so
    swapping them changes the fit. Solved with lstsq because a degree-3 design on
    a heavy-tailed feature is poorly conditioned.
    """

    name = "polynomial"
    task = "regression"

    def __init__(
        self,
        feature: int | str = 0,
        degree: int = 2,
        standardize_before_powers: bool = True,
    ):
        self.feature = feature
        self.degree = degree
        self.standardize_before_powers = standardize_before_powers

    def get_params(self) -> dict:
        return {
            "feature": self.feature,
            "degree": self.degree,
            "standardize_before_powers": self.standardize_before_powers,
        }

    @property
    def feature_index_(self) -> int:
        if isinstance(self.feature, str):
            if self.feature not in FEATURE_COLUMNS:
                raise ValueError(f"Unknown feature {self.feature!r}; expected one of {list(FEATURE_COLUMNS)}")
            return FEATURE_COLUMNS.index(self.feature)
        return int(self.feature)

    def _standardized_column(self, X) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)[:, self.feature_index_]
        if not self.standardize_before_powers:
            return x.reshape(-1, 1)
        scale = self.scale_ if self.scale_ > 0 else 1.0
        return ((x - self.mean_) / scale).reshape(-1, 1)

    def _design(self, X) -> np.ndarray:
        z = self._standardized_column(X).ravel()
        return np.column_stack([z**power for power in range(1, self.degree + 1)])

    def fit(self, X, y=None) -> "Polynomial":
        x = np.asarray(X, dtype=np.float64)[:, self.feature_index_]
        y = np.asarray(y, dtype=np.float64)
        self.mean_, self.scale_ = float(x.mean()), float(x.std())

        design = self._design(X)
        solution, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(design)), design]), y, rcond=None)
        self.intercept_, self.coef_ = float(solution[0]), solution[1:]
        self.history_ = {"iteration": [0], "loss": [float(np.mean((y - self.predict(X)) ** 2))]}
        return self

    def predict(self, X) -> np.ndarray:
        return self._design(X) @ self.coef_ + self.intercept_

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "coef": np.asarray(self.coef_),
            "intercept": np.array([self.intercept_]),
            "scaler": np.array([self.mean_, self.scale_]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "Polynomial":
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        self.mean_, self.scale_ = (float(v) for v in arrays["scaler"])
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        """Fit on _standardized_column(X) — the pipeline expands that single column."""
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        # include_bias=False: LinearRegression already fits an intercept, and the
        # duplicate column makes the design rank-deficient.
        return make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=False), LinearRegression()
        )
