"""
Elastic Net from scratch (cyclic coordinate descent, numpy only).
"""

import numpy as np
from typing import Optional, Dict, Any


class ElasticNet:
    """
    Minimizes (1/2n)||y - Xb - b0||² + alpha*l1_ratio*||b||₁ + alpha*(1-l1_ratio)/2*||b||₂²

    (sklearn's scaling, so alpha is comparable across implementations.)
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000, tol: float = 1e-4):
        """
        Args:
            alpha: Overall regularization strength (λ)
            l1_ratio: Fraction of L1 penalty (0 = Ridge/L2, 1 = Lasso/L1, 0.5 = Elastic Net)
            max_iter: Max full sweeps over the features
            tol: Stop when the largest coefficient change in a sweep is below this
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape

        # Center (not scale) so coefficients stay in original units and the
        # solution matches the objective above; the intercept is recovered after.
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - x_mean
        r = y - y_mean  # residual for beta = 0
        col_sq = (Xc ** 2).sum(axis=0) / n

        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)
        beta = np.zeros(p)

        for it in range(1, self.max_iter + 1):
            max_change = 0.0
            for j in range(p):
                denom = col_sq[j] + l2
                if denom <= 0.0:  # constant column and no ridge term
                    continue
                old = beta[j]
                rho = Xc[:, j] @ r / n + col_sq[j] * old  # partial-residual correlation
                new = np.sign(rho) * max(abs(rho) - l1, 0.0) / denom
                if new != old:
                    r -= Xc[:, j] * (new - old)
                    beta[j] = new
                    max_change = max(max_change, abs(new - old))
            self.n_iter_ = it
            if max_change < self.tol:
                break

        self.coefficients_ = beta
        self.intercept_ = float(y_mean - x_mean @ beta)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)

        if X.shape[1] != self.coefficients_.shape[0]:
            raise ValueError(
                f"Feature mismatch: expected {self.coefficients_.shape[0]}, "
                f"got {X.shape[1]}"
            )

        return X @ self.coefficients_ + self.intercept_

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dict."""
        if self.coefficients_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted.")

        return {
            "coefficients": self.coefficients_.tolist(),
            "intercept": float(self.intercept_),
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ElasticNet":
        """Deserialize model from dict."""
        model = ElasticNet(alpha=d["alpha"], l1_ratio=d["l1_ratio"])
        model.coefficients_ = np.array(d["coefficients"], dtype=np.float64)
        model.intercept_ = float(d["intercept"])
        return model
