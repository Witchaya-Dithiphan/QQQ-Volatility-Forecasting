"""Binary CART classifier with weighted Gini splits."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..core.base import BaseModel


def _best_split(X, y, sample_weight, min_samples_leaf, allow_zero_gain=False):
    """Return the highest-gain (feature, midpoint, gain), or None."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    active = weights > 0
    X, y, weights = X[active], y[active], weights[active]
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
            if best is None:
                better = gain > 0 or (allow_zero_gain and gain == 0.0)
            else:
                # shortcut: roundoff-level gain ties keep feature order; revisit if weighted parity differs.
                better = gain > best_gain + (1e-15 if allow_zero_gain else 0.0)
            if better:
                threshold = float(np.float64(values[i] / 2) + np.float64(values[i + 1] / 2))
                best = feature, threshold, float(gain)
                best_gain = gain
    return best


class DecisionTree(BaseModel):
    name = "decision_tree"
    task = "classification"

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 5, random_state: int = 42,
                 max_features: str | None = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_features = max_features

    def get_params(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "max_features": self.max_features,
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
        if self.max_features not in (None, "sqrt"):
            raise ValueError("max_features must be None or 'sqrt'")
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (len(y),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("sample_weight must be finite, nonnegative and have positive sum")

        self.n_features_in_ = X.shape[1]
        feature, threshold, left, right, probability = [], [], [], [], []
        features = list(range(self.n_features_in_))
        constant_features = features.copy()
        max_features = self.n_features_in_ if self.max_features is None else max(1, int(np.sqrt(self.n_features_in_)))
        rng_state = int(np.random.RandomState(self.random_state).randint(0, 2**31 - 1))

        def feature_order(node_X, n_known_constants):
            nonlocal rng_state
            order = []
            end = len(features)
            visited = found_constants = drawn_constants = 0
            total_constants = n_known_constants
            while end > total_constants and (visited < max_features or visited <= found_constants + drawn_constants):
                visited += 1
                # sklearn permutes features even with splitter="best"; ties follow this seeded order.
                if rng_state == 0:
                    rng_state = 1
                rng_state ^= (rng_state << 13) & 0xFFFFFFFF
                rng_state ^= rng_state >> 17
                rng_state ^= (rng_state << 5) & 0xFFFFFFFF
                selected = drawn_constants + (rng_state & 0x7FFFFFFF) % (end - found_constants - drawn_constants)
                if selected < n_known_constants:
                    features[drawn_constants], features[selected] = features[selected], features[drawn_constants]
                    drawn_constants += 1
                    continue
                selected += found_constants
                column = features[selected]
                values = node_X[:, column]
                if values.max() <= values.min() + np.float32(1e-7):
                    features[selected], features[total_constants] = features[total_constants], features[selected]
                    found_constants += 1
                    total_constants += 1
                    continue
                order.append(column)
                end -= 1
                features[end], features[selected] = features[selected], features[end]
            # Ancestor constants keep their order for both children, including after a sibling grows.
            features[:n_known_constants] = constant_features[:n_known_constants]
            constant_features[n_known_constants:total_constants] = features[n_known_constants:total_constants]
            return order, total_constants

        def grow(indices, depth, n_known_constants=0):
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
            order, n_constants = feature_order(X[indices], n_known_constants)
            split = _best_split(X[indices][:, order], y[indices], node_weight, self.min_samples_leaf,
                                allow_zero_gain=self.max_features == "sqrt")
            if split is None:
                return node
            column, cutoff, _ = split
            column = order[column]
            feature[node], threshold[node] = column, cutoff
            mask = X[indices, column] <= cutoff
            left[node] = grow(indices[mask], depth + 1, n_constants)
            right[node] = grow(indices[~mask], depth + 1, n_constants)
            return node

        grow(np.flatnonzero(weights > 0), 0)
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
            max_features=self.max_features,
            random_state=self.random_state,
        )
