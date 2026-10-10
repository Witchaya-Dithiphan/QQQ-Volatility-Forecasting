"""Elastic Net by coordinate descent — the extracurricular regression model.

Chosen for this dataset on evidence rather than convenience: the training
features carry a covariance condition number of 35.6, with
historical_volatility_5d and historical_volatility_20d correlating at 0.804.
That collinearity is exactly what makes plain least squares produce unstable
coefficients, and what the combined L1/L2 penalty is designed to handle — L1
discards redundant columns outright, while L2 spreads weight across a correlated
group instead of letting Lasso pick one of them arbitrarily.
"""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class ElasticNet(BaseModel):
    """Minimises

        J(b) = ||y - Xb||^2 / (2n)  +  alpha*rho*||b||_1  +  (alpha*(1-rho)/2)*||b||^2

    with rho = l1_ratio. The L1 term is not differentiable at zero, so gradient
    descent cannot be used; coordinate descent instead fixes every coefficient
    but one and solves that one in closed form:

        r_j = y - Xb + X[:,j]*b_j          residual without feature j
        rho_j = X[:,j] . r_j / n           its correlation with that residual
        z_j = X[:,j] . X[:,j] / n
        b_j = S(rho_j, alpha*rho) / (z_j + alpha*(1-rho))

    S is the soft-threshold, sign(v)*max(|v| - gamma, 0). It is what sets
    coefficients to exactly zero rather than merely small: a feature whose
    correlation with the residual falls below the threshold is dropped outright.
    The denominator is where L2 acts, shrinking every surviving coefficient
    proportionally.
    """

    name = "elastic_net"
    task = "regression"

    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
        tol: float = 1e-6,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def get_params(self) -> dict:
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    @staticmethod
    def _soft_threshold(value: float, gamma: float) -> float:
        return float(np.sign(value) * max(abs(value) - gamma, 0.0))

    def _objective(self, X, y, coef, intercept) -> float:
        residual = y - (X @ coef + intercept)
        return float(
            residual @ residual / (2 * len(y))
            + self.alpha * self.l1_ratio * np.abs(coef).sum()
            + 0.5 * self.alpha * (1 - self.l1_ratio) * coef @ coef
        )

    def fit(self, X, y=None) -> "ElasticNet":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        # Centering lets the intercept stay out of the penalty: fit on centred
        # data, then recover it as ybar - xbar.b at the end.
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean

        coef = np.zeros(n_features)
        norms = np.einsum("ij,ij->j", Xc, Xc) / n_samples
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1 - self.l1_ratio)
        residual = yc - Xc @ coef

        self.history_ = {"iteration": [], "loss": []}
        for iteration in range(self.max_iter):
            largest_step = 0.0
            for j in range(n_features):  # cyclic order, per the config
                if norms[j] == 0.0:
                    continue
                previous = coef[j]
                # Add feature j back into the residual before solving for it.
                rho_j = Xc[:, j] @ (residual + Xc[:, j] * previous) / n_samples
                coef[j] = self._soft_threshold(rho_j, l1_penalty) / (norms[j] + l2_penalty)
                if coef[j] != previous:
                    residual -= Xc[:, j] * (coef[j] - previous)
                largest_step = max(largest_step, abs(coef[j] - previous))

            intercept = float(y_mean - x_mean @ coef)
            if iteration % 10 == 0 or largest_step < self.tol:
                self.history_["iteration"].append(iteration)
                self.history_["loss"].append(self._objective(X, y, coef, intercept))
            if largest_step < self.tol:
                break

        self.coef_ = coef
        self.intercept_ = float(y_mean - x_mean @ coef)
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"coef": self.coef_, "intercept": np.array([self.intercept_])}

    def load_state(self, arrays: dict, metadata: dict) -> "ElasticNet":
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        """The ready-to-use half of this model, which the assignment requires.

        sklearn uses the same 1/(2n) convention on the squared error, so alpha
        transfers directly — had we written 1/2 instead, the same alpha would
        mean something n times stronger.
        """
        from sklearn.linear_model import ElasticNet as SkElasticNet

        return SkElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=True,
            max_iter=self.max_iter,
            tol=self.tol,
            selection="cyclic",
        )
