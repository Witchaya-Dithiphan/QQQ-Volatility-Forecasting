"""
Decision Tree Classifier (scratch, NumPy only): greedy binary CART with Gini impurity.

Splits are `x[feature] <= threshold` at midpoints between consecutive distinct sorted values. Ties between equally good
splits go to the lowest feature index, then the lowest threshold. A node splits whenever a valid split exists (even with
zero impurity gain, as XOR needs) until it is pure or hits max_depth / min_samples_split / min_samples_leaf.
Leaves store class counts for classes [0, 1] so a one-class leaf maps to the right probability column.
"""
from typing import Any, Dict, Optional

import numpy as np

from ._common import NOT_FITTED, check_param, check_X, check_Xy

_TIE = 1e-12  # relative tolerance on weighted child impurity when comparing splits


class DecisionTreeNode:
    """Internal tree node. A leaf has `value` (counts of class 0 and 1); an inner node has feature/threshold/left/right."""

    def __init__(self):
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["DecisionTreeNode"] = None
        self.right: Optional["DecisionTreeNode"] = None
        self.value: Optional[np.ndarray] = None

    def is_leaf(self) -> bool:
        return self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        if self.is_leaf():
            return {"value": self.value.tolist()}
        return {"feature": int(self.feature), "threshold": float(self.threshold), "left": self.left.to_dict(), "right": self.right.to_dict()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DecisionTreeNode":
        node = DecisionTreeNode()
        if "value" in d:
            node.value = np.array(d["value"], dtype=np.int64)
        else:
            node.feature, node.threshold = int(d["feature"]), float(d["threshold"])
            node.left, node.right = DecisionTreeNode.from_dict(d["left"]), DecisionTreeNode.from_dict(d["right"])
        return node


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, min_samples_leaf: int = 1, random_state: int = 42):
        """`random_state` is accepted for runner/stability API compatibility; the algorithm is fully deterministic."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.tree_: Optional[DecisionTreeNode] = None
        self.classes_ = np.array([0, 1])
        self.n_features_: Optional[int] = None

    def _best_split(self, X, y):
        n, msl = len(y), self.min_samples_leaf
        best = None  # (cost, feature, threshold)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="stable")
            xs, ys = X[order, j], y[order]
            n_left = np.arange(1, n)
            ones_left = np.cumsum(ys)[:-1]
            ones_right = ys.sum() - ones_left
            left = np.stack([n_left - ones_left, ones_left])
            right = np.stack([(n - n_left) - ones_right, ones_right])
            valid = (xs[:-1] < xs[1:]) & (n_left >= msl) & (n - n_left >= msl)
            if not valid.any():
                continue
            # n_c * gini_c = n_c - sum_k count_k^2 / n_c
            cost = (n_left - (left ** 2).sum(axis=0) / n_left) + ((n - n_left) - (right ** 2).sum(axis=0) / (n - n_left))
            cost = np.where(valid, cost, np.inf)
            i = int(np.flatnonzero(cost <= cost.min() + _TIE * n)[0])
            if best is None or cost[i] < best[0] - _TIE * n:
                threshold = xs[i] / 2.0 + xs[i + 1] / 2.0  # halve first: a + b overflows near +-float64 max
                best = (cost[i], j, threshold if xs[i] <= threshold < xs[i + 1] else xs[i])
        return best

    def _build(self, X, y, depth) -> DecisionTreeNode:
        node = DecisionTreeNode()
        counts = np.bincount(y, minlength=2)
        split = None
        if depth < self.max_depth and counts.min() > 0 and len(y) >= self.min_samples_split and len(y) >= 2 * self.min_samples_leaf:
            split = self._best_split(X, y)
        if split is None:
            node.value = counts
            return node
        node.feature, node.threshold = split[1], float(split[2])
        mask = X[:, node.feature] <= node.threshold
        node.left, node.right = self._build(X[mask], y[mask], depth + 1), self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        check_param("max_depth", self.max_depth, integer=True)
        check_param("min_samples_split", self.min_samples_split, integer=True)
        check_param("min_samples_leaf", self.min_samples_leaf, integer=True)
        X, y = check_Xy(X, y)
        self.n_features_ = X.shape[1]
        self.tree_ = self._build(X, y, 0)
        return self

    def _leaf_counts(self, X) -> np.ndarray:
        if self.tree_ is None:
            raise ValueError(NOT_FITTED)
        X = check_X(X, self.n_features_)
        out = np.empty((len(X), 2))
        for i, x in enumerate(X):
            node = self.tree_
            while not node.is_leaf():
                node = node.left if x[node.feature] <= node.threshold else node.right
            out[i] = node.value
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        counts = self._leaf_counts(X)
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]  # M2 convention: probability >= 0.5 is positive

    def to_dict(self) -> Dict[str, Any]:
        if self.tree_ is None:
            raise ValueError(NOT_FITTED)
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state, "n_features": self.n_features_, "classes": self.classes_.tolist(),
                "tree": self.tree_.to_dict()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DecisionTreeClassifier":
        model = DecisionTreeClassifier(d["max_depth"], d["min_samples_split"], d["min_samples_leaf"], d["random_state"])
        model.n_features_, model.classes_ = d["n_features"], np.array(d["classes"])
        model.tree_ = DecisionTreeNode.from_dict(d["tree"])
        return model
