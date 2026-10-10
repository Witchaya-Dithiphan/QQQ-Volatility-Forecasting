"""Gaussian Naive Bayes — closed form, no iteration and no RNG."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class GaussianNB(BaseModel):
    """Assumes each feature is independent and Gaussian within a class.

        mu_c, var_c = per-class mean and variance (ddof=0)
        var_c      += var_smoothing * max(var(X))      # dataset-wide max, not per class
        log p(x|c)  = -0.5 * sum( log(2*pi*var_cj) + (xj - mu_cj)^2 / var_cj )

    Everything stays in log space; multiplying the per-feature densities directly
    underflows once there are more than a handful of features.
    """

    name = "gaussian_nb"
    task = "classification"

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def get_params(self) -> dict:
        return {"var_smoothing": self.var_smoothing}

    def fit(self, X, y=None) -> "GaussianNB":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        # sklearn scales the floor by the largest variance in the whole dataset,
        # not by each class's own — getting this wrong shifts every density slightly.
        self.epsilon_ = float(self.var_smoothing * np.var(X, axis=0).max())

        self.classes_ = np.array([0, 1])
        self.theta_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.vstack([X[y == c].var(axis=0) for c in self.classes_]) + self.epsilon_
        self.class_prior_ = np.array([(y == c).mean() for c in self.classes_])
        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        joint = []
        for index in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[index])
            deviation = X - self.theta_[index]
            log_density = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.var_[index]) + deviation**2 / self.var_[index], axis=1
            )
            joint.append(log_prior + log_density)
        return np.column_stack(joint)

    def predict_proba(self, X) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        # log-sum-exp normalisation, shifted by the row max to stay finite.
        shifted = joint - joint.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities[:, 1] / probabilities.sum(axis=1)

    def decision_function(self, X) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        return joint[:, 1] - joint[:, 0]

    def predict(self, X) -> np.ndarray:
        return (self._joint_log_likelihood(X).argmax(axis=1)).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "theta": self.theta_,
            "var": self.var_,
            "prior": self.class_prior_,
            "epsilon": np.array([self.epsilon_]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "GaussianNB":
        self.theta_ = np.asarray(arrays["theta"], dtype=np.float64)
        self.var_ = np.asarray(arrays["var"], dtype=np.float64)
        self.class_prior_ = np.asarray(arrays["prior"], dtype=np.float64)
        self.epsilon_ = float(arrays["epsilon"][0])
        self.classes_ = np.array([0, 1])
        self.var_smoothing = metadata["params"]["var_smoothing"]
        return self

    def reference(self):
        from sklearn.naive_bayes import GaussianNB as SkGaussianNB

        return SkGaussianNB(var_smoothing=self.var_smoothing)
