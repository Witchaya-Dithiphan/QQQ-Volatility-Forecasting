"""XGBoost's exact greedy method: second-order gain with L2 on leaf weights, then gamma pruning.

The library works in float32 for gradients, gains, thresholds, leaf values and margins,
and accumulates gradient sums in float64. This file keeps those widths on purpose:
a float64 rewrite is mathematically cleaner but picks different splits on near-ties.
"""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel

_F32 = np.float32
# xgboost's kRtEps: a split must gain more than this to be made at all.
_MIN_GAIN = _F32(1e-6)
_LEAF = -1


def _sigmoid(margin):
    """xgboost's float32 sigmoid, clamp included.

    exp is taken in float64 and then rounded: numpy's float32 exp is off by one ulp from
    the C library's expf on about 40% of inputs, which is enough to change leaf values.
    """
    exponent = np.exp(np.minimum(-margin, _F32(88.7)).astype(np.float64)).astype(_F32)
    return _F32(1.0) / (exponent + _F32(1.0) + _F32(1e-16))


def _running_sum(values) -> float:
    """Left-to-right float64 sum, the order a single xgboost thread accumulates in."""
    return float(np.cumsum(values, dtype=np.float64)[-1]) if len(values) else 0.0


def _weight(gradient, hessian, reg_lambda):
    """Optimal leaf weight w* = -G / (H + lambda), rounded to float32."""
    gradient, hessian = np.asarray(gradient, dtype=np.float64), np.asarray(hessian, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = -gradient / (hessian + np.float64(reg_lambda))
    return np.where(hessian > 0, weight, 0.0).astype(_F32)


def _score(gradient, hessian, reg_lambda):
    """A node's score G^2 / (H + lambda), evaluated the way xgboost 3.4 does.

    The library plugs the float32 weight back into -(2 G w + (H + lambda) w^2), with w^2
    multiplied in float32. Algebraically that is G^2 / (H + lambda); numerically it
    rounds differently, and the difference decides near-tied splits.
    """
    gradient, hessian = np.asarray(gradient, dtype=np.float64), np.asarray(hessian, dtype=np.float64)
    weight = _weight(gradient, hessian, reg_lambda)
    square = (weight * weight).astype(np.float64)
    score = -(2.0 * gradient * weight.astype(np.float64) + (hessian + np.float64(reg_lambda)) * square)
    return np.where(hessian > 0, score, 0.0).astype(_F32)


class XGBoost(BaseModel):
    name = "xgboost"
    task = "classification"

    def __init__(self, n_estimators: int = 100, max_depth: int = 2, learning_rate: float = 0.1,
                 reg_lambda: float = 1.0, gamma: float = 0.0, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "random_state": self.random_state,
        }

    def fit(self, X, y=None) -> "XGBoost":
        if (not isinstance(self.n_estimators, int) or self.n_estimators < 1
                or not isinstance(self.max_depth, int) or self.max_depth < 1
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not np.isfinite(self.reg_lambda) or self.reg_lambda < 0
                or not np.isfinite(self.gamma) or self.gamma < 0
                or not isinstance(self.random_state, int) or not 0 <= self.random_state < 2**32):
            raise ValueError("Invalid XGBoost hyperparameters")
        X = np.asarray(X, dtype=_F32)
        labels = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or labels.shape != (len(X),) or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array aligned with y")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("XGBoost requires binary labels 0 and 1")
        y = labels.astype(_F32)

        self.n_features_in_ = X.shape[1]
        # Largest value first: with no missing data the library only scans in this direction.
        descending = [np.argsort(X[:, column], kind="stable")[::-1] for column in range(X.shape[1])]
        # base_score = 0.5 means F0 = 0, whatever the class balance.
        margin = np.zeros(len(y), dtype=_F32)
        trees = []
        self.history_ = {"iteration": [], "loss": []}

        for iteration in range(1, self.n_estimators + 1):
            probability = _sigmoid(margin)
            gradient = probability - y
            hessian = np.maximum(probability * (_F32(1.0) - probability), _F32(1e-16))
            tree, leaf_of_row = self._grow(X, descending, gradient, hessian)
            margin = margin + tree["value"][leaf_of_row]
            trees.append(tree)
            raw = margin.astype(np.float64)
            self.history_["iteration"].append(iteration)
            self.history_["loss"].append(float(np.mean(np.logaddexp(0.0, raw) - labels * raw)))

        self.tree_sizes_ = np.array([len(tree["feature"]) for tree in trees], dtype=np.int64)
        for key in ("feature", "threshold", "children_left", "children_right", "value", "loss_change"):
            setattr(self, f"{key}_", np.concatenate([tree[key] for tree in trees]))
        return self

    def _best_split(self, X, descending, gradient, hessian, in_node, node_gradient, node_hessian):
        """Return (gain, feature, threshold) of the best split, or None if nothing gains.

        A later candidate has to beat the best strictly, so ties go to the lower feature
        index and, within a feature, to the higher threshold.
        """
        parent_score = _score(node_gradient, node_hessian, self.reg_lambda)
        best, best_gain = None, _F32(0.0)
        for column, order in enumerate(descending):
            rows = order[in_node[order]]
            values = X[rows, column]
            right_gradient = np.cumsum(gradient[rows], dtype=np.float64)
            right_hessian = np.cumsum(hessian[rows], dtype=np.float64)
            left_gradient = node_gradient - right_gradient
            left_hessian = node_hessian - right_hessian
            gain = (_score(left_gradient, left_hessian, self.reg_lambda)
                    + _score(right_gradient, right_hessian, self.reg_lambda)) - parent_score
            # A cut is only possible between two different values, and both sides need
            # positive hessian. The last entry is the library's extra candidate that sends
            # every row to the right; it only qualifies if roundoff leaves the left side > 0.
            allowed = (np.append(values[:-1] != values[1:], True) & (left_hessian > 0)
                       & (right_hessian > 0) & np.isfinite(gain))
            if not allowed.any():
                continue
            i = int(np.argmax(np.where(allowed, gain, -np.inf)))
            if gain[i] > best_gain:
                if i == len(rows) - 1:
                    threshold = values[i] - (np.abs(values[i]) + _MIN_GAIN)
                else:
                    threshold = (values[i + 1] + values[i]) * _F32(0.5)
                    if threshold == values[i + 1]:
                        threshold = values[i]
                best, best_gain = (gain[i], column, threshold), gain[i]
        return best

    def _grow(self, X, descending, gradient, hessian):
        """Grow one tree level by level, then prune splits that gain less than gamma."""
        rate = _F32(self.learning_rate)
        position = np.zeros(len(gradient), dtype=np.int64)
        feature, threshold, left, right, parent, weight, loss_change = [], [], [], [], [], [], []

        def add_node(parent_id):
            rows = position == len(feature)
            node_gradient, node_hessian = _running_sum(gradient[rows]), _running_sum(hessian[rows])
            feature.append(_LEAF)
            threshold.append(_F32(0.0))
            left.append(-1)
            right.append(-1)
            parent.append(parent_id)
            weight.append(_F32(_weight(node_gradient, node_hessian, self.reg_lambda)))
            loss_change.append(_F32(0.0))
            return node_gradient, node_hessian

        stats = {0: add_node(-1)}
        frontier = [0]
        for _ in range(self.max_depth):
            grown = []
            for node in frontier:
                in_node = position == node
                split = self._best_split(X, descending, gradient, hessian, in_node, *stats[node])
                if split is None or not split[0] > _MIN_GAIN:
                    continue
                loss_change[node], feature[node], threshold[node] = split
                left[node], right[node] = len(feature), len(feature) + 1
                goes_left = in_node & (X[:, feature[node]] < threshold[node])
                position[goes_left] = left[node]
                position[in_node & ~goes_left] = right[node]
                stats[left[node]] = add_node(node)
                stats[right[node]] = add_node(node)
                grown += [left[node], right[node]]
            frontier = grown

        # Pruning runs after growth and only bottom-up: a weak split survives when
        # something below it is worth keeping.
        gamma = _F32(self.gamma)
        for node in range(len(feature)):
            while feature[node] == _LEAF and parent[node] >= 0:
                above = parent[node]
                if feature[above] == _LEAF or feature[left[above]] != _LEAF or feature[right[above]] != _LEAF:
                    break
                if not loss_change[above] < gamma:
                    break
                position[np.isin(position, (left[above], right[above]))] = above
                feature[above], left[above], right[above] = _LEAF, -1, -1
                node = above

        # Renumber the surviving nodes breadth-first so pruned ones leave no gaps.
        keep, queue = [], [0]
        while queue:
            node = queue.pop(0)
            keep.append(node)
            if feature[node] != _LEAF:
                queue += [left[node], right[node]]
        renumber = {old: new for new, old in enumerate(keep)}
        is_leaf = np.array([feature[node] == _LEAF for node in keep])
        tree = {
            "feature": np.array([feature[node] for node in keep], dtype=np.int64),
            "threshold": np.array([threshold[node] for node in keep], dtype=_F32),
            "children_left": np.array([renumber.get(left[node], -1) for node in keep], dtype=np.int64),
            "children_right": np.array([renumber.get(right[node], -1) for node in keep], dtype=np.int64),
            # Shrinkage is folded into the stored leaf value, once.
            "value": np.where(is_leaf, np.array([weight[node] for node in keep], dtype=_F32) * rate, _F32(0.0)).astype(_F32),
            "loss_change": np.array([loss_change[node] for node in keep], dtype=_F32),
        }
        return tree, np.array([renumber[node] for node in position], dtype=np.int64)

    def staged_decision_function(self, X):
        """Yield the float32 margin after each tree, for round-by-round comparison."""
        X = np.asarray(X, dtype=_F32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X has an invalid shape or nonfinite values")
        rows = np.arange(len(X))
        margin = np.zeros(len(X), dtype=_F32)
        offset = 0
        for size in self.tree_sizes_:
            nodes = slice(offset, offset + size)
            feature, threshold = self.feature_[nodes], self.threshold_[nodes]
            left, right = self.children_left_[nodes], self.children_right_[nodes]
            node = np.zeros(len(X), dtype=np.int64)
            while True:
                internal = feature[node] >= 0
                if not internal.any():
                    break
                goes_left = X[rows, np.where(internal, feature[node], 0)] < threshold[node]
                node = np.where(internal, np.where(goes_left, left[node], right[node]), node)
            margin = margin + self.value_[nodes][node]
            offset += size
            yield margin

    def _margin(self, X):
        for margin in self.staged_decision_function(X):
            pass
        return margin

    def decision_function(self, X) -> np.ndarray:
        return self._margin(X).astype(np.float64)

    def predict_proba(self, X) -> np.ndarray:
        return _sigmoid(self._margin(X)).astype(np.float64)

    def predict(self, X) -> np.ndarray:
        # The library thresholds the float32 probability, so a tiny positive margin
        # whose sigmoid rounds to exactly 0.5 is still class 0.
        return (_sigmoid(self._margin(X)) > _F32(0.5)).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "feature": self.feature_,
            "threshold": self.threshold_,
            "children_left": self.children_left_,
            "children_right": self.children_right_,
            "value": self.value_,
            "loss_change": self.loss_change_,
            "tree_sizes": self.tree_sizes_,
            "n_features_in": np.array([self.n_features_in_]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "XGBoost":
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        for key in ("feature", "threshold", "children_left", "children_right", "value",
                    "loss_change", "tree_sizes"):
            setattr(self, f"{key}_", np.asarray(arrays[key]))
        self.n_features_in_ = int(arrays["n_features_in"][0])
        return self

    def reference(self):
        from xgboost import XGBClassifier

        return XGBClassifier(
            tree_method="exact",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            reg_alpha=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            colsample_bynode=1,
            min_child_weight=0,
            base_score=0.5,
            random_state=self.random_state,
            # One thread fixes the order gradient sums are accumulated in.
            n_jobs=1,
        )
