"""Scratch Gradient Boosting for binary logistic loss (Friedman 2001, LogitBoost-style Newton leaves).

F0 = prior log-odds. Each round fits a regression tree to the negative gradient r = y - sigmoid(F) (MSE split),
then replaces each leaf value with the Newton step sum(r) / sum(p(1-p)) and updates F += learning_rate * leaf.
"""
from typing import Any, Dict, Optional

import numpy as np

from src.modeling.ensemble._tree import (
    apply_tree, build_tree, check_X, check_binary_y, check_int, check_positive, validate_tree,
)


def _sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _logistic_loss(y, F):
    return float(np.mean(np.logaddexp(0.0, F) - y * F))


class GradientBoosting:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 2,
                 tol: float = 1e-6, random_state: int = 42):
        self.n_estimators = check_int("n_estimators", n_estimators)
        self.learning_rate = check_positive("learning_rate", learning_rate)
        self.max_depth = check_int("max_depth", max_depth)
        self.tol = float(tol)
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be a non-negative finite number")
        self.random_state = check_int("random_state", random_state, 0)  # trees are deterministic; kept for the run contract
        self.trees_: Optional[list] = None
        self.init_score_: Optional[float] = None
        self.loss_history_: list = []
        self.converged_: Optional[bool] = None
        self.n_features_: Optional[int] = None

    def fit(self, X, y) -> "GradientBoosting":
        X = check_X(X)
        y = check_binary_y(y, len(X), both_classes=True)
        prior = y.mean()
        self.n_features_, self.init_score_ = X.shape[1], float(np.log(prior / (1 - prior)))
        F = np.full(len(y), self.init_score_)
        trees, history = [], []
        for _ in range(self.n_estimators):
            p = _sigmoid(F)
            tree = build_tree(X, y - p, criterion="mse", max_depth=self.max_depth, min_samples_leaf=1)
            leaf = apply_tree(tree, X)
            num = np.bincount(leaf, weights=y - p, minlength=len(tree["value"]))
            den = np.bincount(leaf, weights=p * (1 - p), minlength=len(tree["value"]))
            is_leaf = np.asarray(tree["feature"]) < 0
            tree["value"] = [float(v) for v in np.where(is_leaf & (den > 1e-300), num / np.where(den > 1e-300, den, 1.0), 0.0)]
            F = F + self.learning_rate * np.asarray(tree["value"])[leaf]
            trees.append(tree)
            history.append(_logistic_loss(y, F))
        self.trees_, self.loss_history_ = trees, history
        self.converged_ = bool(len(history) > 1 and history[-2] - history[-1] < self.tol)
        return self

    def _raw(self, X) -> np.ndarray:
        if self.trees_ is None:
            raise ValueError("Model not fitted")
        X = check_X(X, self.n_features_)
        F = np.full(len(X), self.init_score_)
        for tree in self.trees_:
            F += self.learning_rate * np.asarray(tree["value"])[apply_tree(tree, X)]
        return F

    def decision_function(self, X) -> np.ndarray:
        return self._raw(X)

    def predict_proba(self, X) -> np.ndarray:
        p1 = _sigmoid(self._raw(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self._raw(X) >= 0).astype(np.int64)

    def to_dict(self) -> Dict[str, Any]:
        if self.trees_ is None:
            raise ValueError("Model not fitted")
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "max_depth": self.max_depth,
                "tol": self.tol, "random_state": self.random_state, "n_features": self.n_features_,
                "init_score": self.init_score_, "loss_history": self.loss_history_, "converged": self.converged_,
                "trees": self.trees_}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GradientBoosting":
        model = cls(d["n_estimators"], d["learning_rate"], d["max_depth"], d["tol"], d["random_state"])
        n_features = check_int("n_features", d["n_features"])
        if len(d["trees"]) != model.n_estimators or len(d["loss_history"]) != model.n_estimators or not np.isfinite(d["init_score"]):
            raise ValueError("Malformed boosting state")
        for tree in d["trees"]:
            validate_tree(tree, n_features)
        model.n_features_, model.init_score_ = n_features, float(d["init_score"])
        model.loss_history_, model.converged_ = [float(v) for v in d["loss_history"]], bool(d["converged"])
        model.trees_ = [{k: list(v) for k, v in t.items()} for t in d["trees"]]
        return model
