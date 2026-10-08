"""NumPy-only CART primitive shared by the scratch Random Forest and Gradient Boosting.

A tree is a plain dict of parallel lists (strict-JSON ready): node i is a leaf when feature[i] == -1;
otherwise samples with x[feature] <= threshold go to left[i], the rest to right[i].
"""
import numbers

import numpy as np

_TIE = 1e-12  # a later candidate must beat the incumbent by more than this (lowest feature, then threshold, wins)


def check_X(X, n_features=None):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.bool_):
        raise ValueError("X must be a non-empty 2D numeric array")
    X = X.astype(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("X must be finite")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Feature mismatch: expected {n_features} features, got {X.shape[1]}")
    return X


def check_binary_y(y, n_samples, *, both_classes=False):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n_samples:
        raise ValueError("y must be 1D with one label per row of X")
    if y.dtype.kind not in "biuf" or not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain binary labels {0, 1}")
    y = y.astype(np.int64)
    if both_classes and len(np.unique(y)) < 2:
        raise ValueError("y must contain both classes")
    return y


def check_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def check_positive(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _best_split(X, t, features, min_leaf, gini):
    n = len(t)
    k = np.arange(1, n)
    best = None  # (cost, feature, threshold)
    for f in features:
        order = np.argsort(X[:, f], kind="stable")
        x, cum = X[order, f], np.cumsum(t[order])
        ok = (x[1:] > x[:-1]) & (k >= min_leaf) & (n - k >= min_leaf)
        if not ok.any():
            continue
        sl, nl, nr = cum[:-1], k, n - k
        sr = cum[-1] - sl
        cost = sl * (nl - sl) / nl + sr * (nr - sr) / nr if gini else -(sl ** 2 / nl + sr ** 2 / nr)
        i = int(np.argmin(np.where(ok, cost, np.inf)))  # first minimum -> lowest threshold
        if best is None or cost[i] < best[0] - _TIE:
            thr = (x[i] + x[i + 1]) / 2
            best = (cost[i], int(f), float(thr if thr < x[i + 1] else x[i]))
    return best


def build_tree(X, t, *, criterion, max_depth, min_samples_leaf, max_features=None, rng=None):
    """Greedy CART. criterion 'gini' needs t in {0,1} (leaf = P(class 1)); 'mse' fits t (leaf = mean).

    max_features draws that many candidate features per split from rng (without replacement, sorted).
    """
    X, t = np.asarray(X, dtype=np.float64), np.asarray(t, dtype=np.float64)
    if criterion not in ("gini", "mse"):
        raise ValueError("criterion must be 'gini' or 'mse'")
    tree = {"feature": [], "threshold": [], "left": [], "right": [], "value": []}

    def grow(idx, depth):
        node = len(tree["feature"])
        for key, v in (("feature", -1), ("threshold", 0.0), ("left", -1), ("right", -1), ("value", float(t[idx].mean()))):
            tree[key].append(v)
        ts = t[idx]
        if depth >= max_depth or len(idx) < 2 * min_samples_leaf or ts.min() == ts.max():
            return node
        d = X.shape[1]
        feats = np.arange(d) if max_features is None or max_features >= d else np.sort(rng.choice(d, max_features, replace=False))
        split = _best_split(X[idx], ts, feats, min_samples_leaf, criterion == "gini")
        if split is None:
            return node
        _, f, thr = split
        go_left = X[idx, f] <= thr
        tree["feature"][node], tree["threshold"][node] = f, thr
        tree["left"][node] = grow(idx[go_left], depth + 1)
        tree["right"][node] = grow(idx[~go_left], depth + 1)
        return node

    grow(np.arange(len(t)), 0)
    return tree


def apply_tree(tree, X):
    """Leaf node id for every row of X."""
    feature, threshold = np.asarray(tree["feature"]), np.asarray(tree["threshold"])
    left, right = np.asarray(tree["left"]), np.asarray(tree["right"])
    node = np.zeros(len(X), dtype=np.int64)
    while True:
        f = feature[node]
        active = f >= 0
        if not active.any():
            return node
        rows = np.flatnonzero(active)
        go_left = X[rows, f[rows]] <= threshold[node[rows]]
        node[rows] = np.where(go_left, left[node[rows]], right[node[rows]])


def predict_tree(tree, X):
    return np.asarray(tree["value"])[apply_tree(tree, X)]


def tree_depth(tree):
    def depth(i):
        return 0 if tree["feature"][i] < 0 else 1 + max(depth(tree["left"][i]), depth(tree["right"][i]))
    return depth(0)


def validate_tree(tree, n_features):
    """Structural check for deserialized trees (fail closed on malformed state)."""
    sizes = {len(tree[k]) for k in ("feature", "threshold", "left", "right", "value")}
    n = sizes.pop() if len(sizes) == 1 else 0
    if n == 0:
        raise ValueError("Malformed tree state")
    for i in range(n):
        f = tree["feature"][i]
        if f == -1:
            continue
        if not 0 <= f < n_features or not (i < tree["left"][i] < n and i < tree["right"][i] < n):
            raise ValueError("Malformed tree state")
