"""Gradient boosting on binary log loss: squared-error trees with Newton-step leaves."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel

# sklearn treats feature values closer than this as equal when placing a split.
_FEATURE_THRESHOLD = np.float32(1e-7)
_LEAF = -2


def _running_sum(values) -> float:
    """Left-to-right sum, the order sklearn's criterion accumulates in (np.sum is pairwise)."""
    return float(np.cumsum(values)[-1])


def _left_sums(sorted_target, positions, total):
    """sum(sorted_target[:p]) for each candidate p, rounded the way sklearn rounds it."""
    n = len(sorted_target)
    steps = np.diff(positions, prepend=0)
    if not (steps > n - positions).any():
        return np.cumsum(sorted_target)[positions - 1]
    # Rare: when the jump is longer than what remains, sklearn restarts from the right end.
    sums, running, at = [], 0.0, 0
    for p in positions.tolist():
        if p - at <= n - p:
            running = _running_sum(np.concatenate(([running], sorted_target[at:p])))
        else:
            running = _running_sum(np.concatenate(([total], -sorted_target[p:][::-1])))
        sums.append(running)
        at = p
    return np.asarray(sums)


def _best_split(X, target, rows, columns, min_samples_leaf, total):
    """Return ((column, threshold, n_left) or None, rows sorted by the last column tried).

    Maximising sum_left^2 / n_left + sum_right^2 / n_right is the same as minimising
    the children's squared error. Ties keep the earlier column, then the lower cut.
    """
    n = len(rows)
    best, best_score = None, -np.inf
    for column in columns:
        rows = rows[np.argsort(X[rows, column], kind="stable")]
        values = X[rows, column]
        positions = np.flatnonzero(values[1:] > values[:-1] + _FEATURE_THRESHOLD) + 1
        positions = positions[(positions >= min_samples_leaf) & (n - positions >= min_samples_leaf)]
        if len(positions) == 0:
            continue
        left_sum = _left_sums(target[rows], positions, total)
        right_sum = total - left_sum
        score = left_sum * left_sum / positions + right_sum * right_sum / (n - positions)
        i = int(np.argmax(score))
        if score[i] > best_score:
            cut = int(positions[i])
            threshold = float(np.float64(values[cut - 1]) / 2.0 + np.float64(values[cut]) / 2.0)
            best, best_score = (column, threshold, cut), score[i]
    return best, rows


def _partition(rows, goes_left):
    """sklearn's in-place two-pointer partition; the resulting order feeds the children's sums."""
    rows, goes_left = rows.tolist(), goes_left.tolist()
    start, end = 0, len(rows)
    while start < end:
        if goes_left[start]:
            start += 1
        else:
            end -= 1
            rows[start], rows[end] = rows[end], rows[start]
            goes_left[start], goes_left[end] = goes_left[end], goes_left[start]
    return np.asarray(rows, dtype=np.int64)


def _log_odds(p: float) -> float:
    """log(p / (1 - p)), evaluated the way scipy's logit does so F0 agrees to the last bit.

    One ulp here looks harmless but shifts every residual, and that is enough to flip
    which of two exactly tied features a later tree picks.
    """
    if p < 0.3 or p > 0.65:
        return float(np.log(p / (1.0 - p)))
    centred = 2.0 * (p - 0.5)
    return float(np.log1p(centred) - np.log1p(-centred))


class _FeatureOrder:
    """The order sklearn's splitter visits features at each node, which decides ties.

    Same xorshift walk as the one inside decision_tree.fit; kept here so this model
    stays in one file.
    """

    def __init__(self, n_features: int, state: int):
        self.features = list(range(n_features))
        self.constant_features = self.features.copy()
        self.state = state

    def __call__(self, node_X, n_known_constants):
        features = self.features
        order = []
        end = len(features)
        found_constants = drawn_constants = 0
        total_constants = n_known_constants
        while end > total_constants:
            if self.state == 0:
                self.state = 1
            self.state ^= (self.state << 13) & 0xFFFFFFFF
            self.state ^= self.state >> 17
            self.state ^= (self.state << 5) & 0xFFFFFFFF
            selected = drawn_constants + (self.state & 0x7FFFFFFF) % (end - found_constants - drawn_constants)
            if selected < n_known_constants:
                features[drawn_constants], features[selected] = features[selected], features[drawn_constants]
                drawn_constants += 1
                continue
            selected += found_constants
            column = features[selected]
            values = node_X[:, column]
            if values.max() <= values.min() + _FEATURE_THRESHOLD:
                features[selected], features[total_constants] = features[total_constants], features[selected]
                found_constants += 1
                total_constants += 1
                continue
            order.append(column)
            end -= 1
            features[end], features[selected] = features[selected], features[end]
        # Ancestor constants keep their order for both children, including after a sibling grows.
        features[:n_known_constants] = self.constant_features[:n_known_constants]
        self.constant_features[n_known_constants:total_constants] = features[n_known_constants:total_constants]
        return order, total_constants


class GradientBoosting(BaseModel):
    name = "gradient_boosting"
    task = "classification"

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 1,
                 min_samples_leaf: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }

    def fit(self, X, y=None) -> "GradientBoosting":
        if (not isinstance(self.n_estimators, int) or self.n_estimators < 1
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not isinstance(self.max_depth, int) or self.max_depth < 1
                or not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1
                or not isinstance(self.random_state, int) or not 0 <= self.random_state < 2**32):
            raise ValueError("Invalid gradient boosting hyperparameters")
        # sklearn grows its trees on float32 features, so thresholds only match on float32.
        X = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or labels.shape != (len(X),) or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array aligned with y")
        if not np.isin(labels, [0, 1]).all() or labels.min() == labels.max():
            raise ValueError("GradientBoosting requires both binary labels 0 and 1")
        y = labels.astype(np.float64)

        self.n_features_in_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        # F0 is the log-odds of the prior, the constant that minimises log loss.
        self.init_ = _log_odds(y.mean())
        raw = np.full(len(y), self.init_)
        trees = []
        self.history_ = {"iteration": [], "loss": []}

        for iteration in range(1, self.n_estimators + 1):
            # residual = y - sigmoid(F), written the way sklearn evaluates it.
            decay = np.exp(-raw)
            residual = -(((1.0 - y) - y * decay) / (1.0 + decay))
            tree, leaf_of_row = self._grow(X, residual, int(rng.randint(0, 2**31 - 1)))
            for leaf in np.flatnonzero(tree["feature"] == _LEAF):
                rows = leaf_of_row == leaf
                probability = y[rows] - residual[rows]
                hessian = np.mean(probability * (1.0 - probability))
                # One Newton step per leaf: sum(residual) / sum(p(1-p)), not the mean residual.
                tree["value"][leaf] = 0.0 if abs(hessian) < 1e-150 else np.mean(residual[rows]) / hessian
            raw += self.learning_rate * tree["value"][leaf_of_row]
            trees.append(tree)
            self.history_["iteration"].append(iteration)
            self.history_["loss"].append(float(np.mean(np.logaddexp(0.0, raw) - y * raw)))

        self.tree_sizes_ = np.array([len(tree["feature"]) for tree in trees], dtype=np.int64)
        for key in ("feature", "threshold", "children_left", "children_right", "value"):
            setattr(self, f"{key}_", np.concatenate([tree[key] for tree in trees]))
        return self

    def _grow(self, X, target, state):
        """Grow one squared-error tree, keeping sklearn's row order so roundoff agrees too."""
        feature, threshold, left, right = [], [], [], []
        leaf_of_row = np.empty(len(target), dtype=np.int64)
        feature_order = _FeatureOrder(X.shape[1], state)
        eps = np.finfo(np.float64).eps
        n_total = len(target)

        def grow(rows, depth, impurity, n_known_constants=0):
            node = len(feature)
            feature.append(_LEAF)
            threshold.append(-2.0)
            left.append(-1)
            right.append(-1)
            n = len(rows)
            leaf_of_row[rows] = node
            if depth >= self.max_depth or n < 2 or n < 2 * self.min_samples_leaf or impurity <= eps:
                return node
            total = _running_sum(target[rows])
            square_total = _running_sum(target[rows] * target[rows])
            order, n_constants = feature_order(X[rows], n_known_constants)
            split, rows = _best_split(X, target, rows, order, self.min_samples_leaf, total)
            if split is None:
                return node
            column, cutoff, n_left = split
            rows = _partition(rows, X[rows, column] <= cutoff)
            n_right = n - n_left
            left_target, right_target = target[rows[:n_left]], target[rows[n_left:]]
            if n_left <= n_right:
                left_sum = _running_sum(left_target)
            else:
                left_sum = _running_sum(np.concatenate(([total], -right_target[::-1])))
            right_sum = total - left_sum
            left_square = _running_sum(left_target * left_target)
            left_impurity = left_square / n_left - (left_sum / n_left) ** 2
            right_impurity = (square_total - left_square) / n_right - (right_sum / n_right) ** 2
            improvement = n / n_total * (impurity - n_right / n * right_impurity - n_left / n * left_impurity)
            if improvement + eps < 0:
                return node
            feature[node], threshold[node] = column, cutoff
            left[node] = grow(rows[:n_left], depth + 1, left_impurity, n_constants)
            right[node] = grow(rows[n_left:], depth + 1, right_impurity, n_constants)
            return node

        rows = np.arange(n_total)
        root_impurity = _running_sum(target * target) / n_total - (_running_sum(target) / n_total) ** 2
        grow(rows, 0, root_impurity)
        return {
            "feature": np.asarray(feature, dtype=np.int64),
            "threshold": np.asarray(threshold, dtype=np.float64),
            "children_left": np.asarray(left, dtype=np.int64),
            "children_right": np.asarray(right, dtype=np.int64),
            "value": np.zeros(len(feature), dtype=np.float64),
        }, leaf_of_row

    def staged_decision_function(self, X):
        """Yield the raw score after each tree, for stage-by-stage comparison."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X has an invalid shape or nonfinite values")
        rows = np.arange(len(X))
        raw = np.full(len(X), self.init_)
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
                goes_left = X[rows, np.where(internal, feature[node], 0)] <= threshold[node]
                node = np.where(internal, np.where(goes_left, left[node], right[node]), node)
            raw = raw + self.learning_rate * self.value_[nodes][node]
            offset += size
            yield raw

    def decision_function(self, X) -> np.ndarray:
        for raw in self.staged_decision_function(X):
            pass
        return raw

    def predict_proba(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))

    def predict(self, X) -> np.ndarray:
        # sklearn's gradient boosting gives a zero score to class 1, unlike its argmax models.
        return (self.decision_function(X) >= 0).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "feature": self.feature_,
            "threshold": self.threshold_,
            "children_left": self.children_left_,
            "children_right": self.children_right_,
            "value": self.value_,
            "tree_sizes": self.tree_sizes_,
            "init": np.array([self.init_]),
            "n_features_in": np.array([self.n_features_in_]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "GradientBoosting":
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        for key in ("feature", "threshold", "children_left", "children_right", "value", "tree_sizes"):
            setattr(self, f"{key}_", np.asarray(arrays[key]))
        self.init_ = float(arrays["init"][0])
        self.n_features_in_ = int(arrays["n_features_in"][0])
        return self

    def reference(self):
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            loss="log_loss",
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=None,
            subsample=1.0,
            random_state=self.random_state,
        )
