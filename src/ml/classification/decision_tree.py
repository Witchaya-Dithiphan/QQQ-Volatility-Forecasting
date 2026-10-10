"""Binary CART classifier with weighted Gini splits."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..core.base import BaseModel


def _best_split(X, y, sample_weight, min_samples_leaf):
    """Return the highest-gain (feature, midpoint, gain), or None."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    n = len(y)
    if n < 2 * min_samples_leaf:
        return None

    total = weights.sum()
    positive = weights[y == 1].sum()
    if total <= 0 or positive <= 0 or positive >= total:
        return None
    parent_gini = 2 * positive * (total - positive) / total**2
    best = None
    best_gain = 0.0

    for feature in range(X.shape[1]):
        order = np.argsort(X[:, feature], kind="stable")
        values = X[order, feature]
        sorted_weights = weights[order]
        left_weight = np.cumsum(sorted_weights)[:-1]
        left_positive = np.cumsum(sorted_weights * y[order])[:-1]
        right_weight = total - left_weight
        right_positive = positive - left_positive
        positions = np.arange(1, n)
        valid = (
            (values[:-1] < values[1:])
            & (positions >= min_samples_leaf)
            & (n - positions >= min_samples_leaf)
            & (left_weight > 0)
            & (right_weight > 0)
        )
        for i in np.flatnonzero(valid):
            left_gini = 2 * left_positive[i] * (left_weight[i] - left_positive[i]) / left_weight[i] ** 2
            right_gini = 2 * right_positive[i] * (right_weight[i] - right_positive[i]) / right_weight[i] ** 2
            gain = parent_gini - (left_weight[i] * left_gini + right_weight[i] * right_gini) / total
            if gain > best_gain:
                threshold = float(np.float64(values[i] / 2) + np.float64(values[i + 1] / 2))
                best = feature, threshold, float(gain)
                best_gain = gain
    return best


class DecisionTree(BaseModel):
    name = "decision_tree"
    task = "classification"

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 5, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }

    def fit(self, X, y=None, sample_weight=None) -> "DecisionTree":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] == 0 or y.ndim != 1 or len(X) != len(y) or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array aligned with y")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("DecisionTree requires binary labels 0 and 1")
        y = y.astype(np.int64)
        if self.max_depth < 1 or self.min_samples_leaf < 1:
            raise ValueError("max_depth and min_samples_leaf must be positive")
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (len(y),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("sample_weight must be finite, nonnegative and have positive sum")

        self.n_features_in_ = X.shape[1]
        feature, threshold, left, right, probability = [], [], [], [], []
        features = list(range(self.n_features_in_))
        rng_state = int(np.random.RandomState(self.random_state).randint(0, 2**31 - 1))

        def feature_order():
            nonlocal rng_state
            order = []
            for end in range(len(features), 0, -1):
                # sklearn permutes features even with splitter="best"; ties follow this seeded order.
                if rng_state == 0:
                    rng_state = 1
                rng_state ^= (rng_state << 13) & 0xFFFFFFFF
                rng_state ^= rng_state >> 17
                rng_state ^= (rng_state << 5) & 0xFFFFFFFF
                selected = (rng_state & 0x7FFFFFFF) % end
                order.append(features[selected])
                features[end - 1], features[selected] = features[selected], features[end - 1]
            return order

        def grow(indices, depth):
            node = len(feature)
            feature.append(-2)
            threshold.append(-2.0)
            left.append(-1)
            right.append(-1)
            node_weight = weights[indices]
            positive_weight = node_weight @ y[indices]
            probability.append(float(positive_weight / node_weight.sum()))
            if (depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf
                    or positive_weight <= 0 or positive_weight >= node_weight.sum()):
                return node
            order = feature_order()
            split = _best_split(X[indices][:, order], y[indices], node_weight, self.min_samples_leaf)
            if split is None:
                return node
            column, cutoff, _ = split
            column = order[column]
            feature[node], threshold[node] = column, cutoff
            mask = X[indices, column] <= cutoff
            left[node] = grow(indices[mask], depth + 1)
            right[node] = grow(indices[~mask], depth + 1)
            return node

        grow(np.arange(len(y)), 0)
        self.tree_ = SimpleNamespace(
            feature=np.asarray(feature, dtype=np.int64),
            threshold=np.asarray(threshold, dtype=np.float64),
            children_left=np.asarray(left, dtype=np.int64),
            children_right=np.asarray(right, dtype=np.int64),
            probability=np.asarray(probability, dtype=np.float64),
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X has an invalid shape or nonfinite values")
        result = np.empty(len(X), dtype=np.float64)
        tree = self.tree_
        for row_index, row in enumerate(X):
            node = 0
            while tree.feature[node] >= 0:
                node = tree.children_left[node] if row[tree.feature[node]] <= tree.threshold[node] else tree.children_right[node]
            result[row_index] = tree.probability[node]
        return result

    def decision_function(self, X) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "feature": self.tree_.feature,
            "threshold": self.tree_.threshold,
            "children_left": self.tree_.children_left,
            "children_right": self.tree_.children_right,
            "probability": self.tree_.probability,
            "n_features_in": np.array([self.n_features_in_]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "DecisionTree":
        self.tree_ = SimpleNamespace(**{key: np.asarray(arrays[key]) for key in (
            "feature", "threshold", "children_left", "children_right", "probability"
        )})
        self.n_features_in_ = int(arrays["n_features_in"][0])
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=None,
            random_state=self.random_state,
        )
