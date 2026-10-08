"""
Gaussian Naive Bayes (scratch, NumPy only), log-space posteriors with empirical Train priors.

var_smoothing follows the plan/sklearn convention: epsilon = var_smoothing * max_j Var(X_j) over all Train rows.
"""
from typing import Any, Dict, Optional

import numpy as np

from ._common import NOT_FITTED, check_param, check_X, check_Xy


class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

        self.classes_ = np.array([0, 1])
        self.class_priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.variances_: Optional[np.ndarray] = None  # smoothed
        self.epsilon_: Optional[float] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        check_param("var_smoothing", self.var_smoothing, allow_zero=True)
        X, y = check_Xy(X, y)
        with np.errstate(all="ignore"):
            peak = X.var(axis=0).max()
            self.epsilon_ = float(self.var_smoothing * (peak if peak > 0 else 1.0))
            groups = [X[y == c] for c in (0, 1)]
            self.class_priors_ = np.array([len(g) / len(X) for g in groups])
            self.means_ = np.array([g.mean(axis=0) for g in groups])
            self.variances_ = np.array([g.var(axis=0) for g in groups]) + self.epsilon_
        if not (np.isfinite(self.means_).all() and np.isfinite(self.variances_).all() and np.isfinite(self.epsilon_)):
            self.means_ = None
            raise ValueError("Train moments overflow float64 (|x| beyond ~1e154); rescale the features")
        self.n_features_ = X.shape[1]
        if (self.variances_ <= 0).any():
            raise ValueError("Zero variance feature; increase var_smoothing")
        return self

    def _log_posterior(self, X) -> np.ndarray:
        if self.means_ is None:
            raise ValueError(NOT_FITTED)
        d = check_X(X, self.n_features_)[:, None, :] - self.means_  # finite: Train means are bounded by the finite variances
        with np.errstate(over="ignore", invalid="ignore"):
            quad = (d ** 2 / self.variances_).sum(axis=2)
        offset = np.log(self.class_priors_) - 0.5 * np.log(2 * np.pi * self.variances_).sum(axis=1)
        log_post = offset - 0.5 * quad
        bad = ~np.isfinite(log_post).all(axis=1)
        if bad.any():
            log_post[bad] = self._saturated(d[bad], offset)
        return log_post

    def _saturated(self, d, offset) -> np.ndarray:
        """Log posteriors for rows whose quadratic form exceeds float64.

        Honest limit: such a form dwarfs every finite term, so the class with the smaller exact log-quadratic form
        (computed as logsumexp of 2*log|z|, which cannot overflow) takes all the mass (0 vs -inf in log space). Only when
        both forms are equal in float64 (the queries are indistinguishable) do the finite prior/normalizer terms decide.
        """
        with np.errstate(divide="ignore"):
            log_z2 = 2.0 * (np.log(np.abs(d)) - 0.5 * np.log(self.variances_))
            peak = log_z2.max(axis=2, keepdims=True)
            peak = np.where(np.isfinite(peak), peak, 0.0)
            log_quad = peak[..., 0] + np.log(np.exp(log_z2 - peak).sum(axis=2))
        out = np.where(np.arange(2) == log_quad.argmin(axis=1)[:, None], 0.0, -np.inf)
        tie = log_quad[:, 0] == log_quad[:, 1]
        out[tie] = offset
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_post = self._log_posterior(X)
        log_post -= log_post.max(axis=1, keepdims=True)
        proba = np.exp(log_post)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]  # M2 convention: probability >= 0.5 is positive

    def to_dict(self) -> Dict[str, Any]:
        if self.means_ is None:
            raise ValueError(NOT_FITTED)
        return {"var_smoothing": self.var_smoothing, "epsilon": self.epsilon_, "classes": self.classes_.tolist(),
                "class_priors": self.class_priors_.tolist(), "means": self.means_.tolist(), "variances": self.variances_.tolist()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GaussianNaiveBayes":
        model = GaussianNaiveBayes(d["var_smoothing"])
        model.epsilon_, model.classes_ = d["epsilon"], np.array(d["classes"])
        model.class_priors_, model.means_, model.variances_ = (np.array(d[k], dtype=np.float64) for k in ("class_priors", "means", "variances"))
        model.n_features_ = model.means_.shape[1]
        return model
