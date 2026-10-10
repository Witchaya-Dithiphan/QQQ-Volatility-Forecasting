"""Simple linear regression — ordinary least squares on one feature."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from ..core.contracts import FEATURE_COLUMNS


class SimpleLinear(BaseModel):
    """Fits y = b0 + b1*x on a single column, chosen by the grid.

        b1 = sum((xi - xbar)(yi - ybar)) / sum((xi - xbar)^2)
        b0 = ybar - b1*xbar

    The denominator is the raw sum of squares, not a variance: using var(ddof=1)
    on one side and a plain sum on the other is the classic way this comes out
    scaled by n/(n-1).
    """

    name = "simple_linear"
    task = "regression"

    def __init__(self, feature: int | str = 0):
        self.feature = feature

    def get_params(self) -> dict:
        return {"feature": self.feature}

    @property
    def feature_index_(self) -> int:
        if isinstance(self.feature, str):
            if self.feature not in FEATURE_COLUMNS:
                raise ValueError(f"Unknown feature {self.feature!r}; expected one of {list(FEATURE_COLUMNS)}")
            return FEATURE_COLUMNS.index(self.feature)
        return int(self.feature)

    def fit(self, X, y=None) -> "SimpleLinear":
        x = np.asarray(X, dtype=np.float64)[:, self.feature_index_]
        y = np.asarray(y, dtype=np.float64)
        x_centered, y_centered = x - x.mean(), y - y.mean()
        denominator = np.sum(x_centered**2)
        # A constant feature leaves the slope undefined; predicting the mean is the
        # least-squares answer for that case and keeps the metric finite.
        self.slope_ = float(np.sum(x_centered * y_centered) / denominator) if denominator > 0 else 0.0
        self.intercept_ = float(y.mean() - self.slope_ * x.mean())
        self.history_ = {"iteration": [0], "loss": [float(np.mean((y - self.predict(X)) ** 2))]}
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)[:, self.feature_index_] * self.slope_ + self.intercept_

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"coefficients": np.array([self.slope_, self.intercept_])}

    def load_state(self, arrays: dict, metadata: dict) -> "SimpleLinear":
        self.slope_, self.intercept_ = (float(v) for v in arrays["coefficients"])
        self.feature = metadata["params"]["feature"]
        return self

    def reference(self):
        """Fit this on X[:, [feature_index_]] — it takes the single column, not all 8."""
        from sklearn.linear_model import LinearRegression

        return LinearRegression(fit_intercept=True)
