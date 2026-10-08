"""
XGBoost Classifier (scratch implementation, binary logistic, NumPy only).

Exact second-order regularized trees (Chen & Guestrin 2016) matching the frozen M6 scope
`exact_second_order_regularized_trees` with tuple fields
[n_estimators, max_depth, learning_rate, reg_lambda, gamma].
"""

import numbers
from typing import Any, Dict, List, Optional

import numpy as np


class XGBoostNode:
    """Decision tree node for XGBoost."""
    def __init__(self):
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["XGBoostNode"] = None
        self.right: Optional["XGBoostNode"] = None
        self.leaf_weight: Optional[float] = None

    def is_leaf(self) -> bool:
        return self.leaf_weight is not None


def _int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be a finite number {'> 0' if positive else '>= 0'}")
    return float(value)


class XGBoostClassifier:
    """
    Scratch XGBoost classifier: binary logistic, exact greedy splits, base score 0.5 (margin 0).

    Per tree:
    - Gradient g = p - y, Hessian h = p(1-p)
    - Gain = 0.5*[GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)] - γ
    - Leaf weight w* = -G / (H + λ), scaled by the learning rate (shrinkage)
    - Split candidates are midpoints between adjacent unique sorted feature values; ties keep the
      lowest feature index, then the lowest threshold. A child must have Hessian sum >= min_child_weight.
    All n_estimators trees are built (no early stopping); the procedure is deterministic.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 2,
        learning_rate: float = 0.1,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
    ):
        self.n_estimators = _int("n_estimators", n_estimators, 1)
        self.max_depth = _int("max_depth", max_depth, 1)
        self.learning_rate = _real("learning_rate", learning_rate, positive=True)
        self.reg_lambda = _real("reg_lambda", reg_lambda)
        self.gamma = _real("gamma", gamma)
        self.min_child_weight = _real("min_child_weight", min_child_weight)

        self._reset()

    def _reset(self) -> None:
        self.trees_: List[XGBoostNode] = []
        self.loss_history_: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Overflow-safe sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _log_loss(y: np.ndarray, scores: np.ndarray) -> float:
        """Mean binary cross-entropy from margins: log(1+e^s) - y*s."""
        return float(np.mean(np.logaddexp(0.0, scores) - y * scores))

    @staticmethod
    def _check_X(X, n_features: Optional[int] = None) -> np.ndarray:
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be numeric") from exc
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.isfinite(X).all():
            raise ValueError("X must be finite")
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError(f"Feature mismatch: expected {n_features}, got {X.shape[1]}")
        return X

    def _leaf(self, G: float, H: float) -> XGBoostNode:
        node = XGBoostNode()
        denom = H + self.reg_lambda
        node.leaf_weight = float(-G / denom) if denom > 0 else 0.0
        return node

    def _best_split(self, X: np.ndarray, g: np.ndarray, h: np.ndarray):
        lam, mcw = self.reg_lambda, self.min_child_weight
        best_gain, best = 0.0, None  # a split must have strictly positive gain after gamma
        G_tot, H_tot = g.sum(), h.sum()
        parent = G_tot**2 / (H_tot + lam) if H_tot + lam > 0 else 0.0
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f], kind="stable")
            xs = X[order, f]
            GL, HL = np.cumsum(g[order])[:-1], np.cumsum(h[order])[:-1]
            GR, HR = G_tot - GL, H_tot - HL
            ok = (xs[:-1] < xs[1:]) & (HL >= mcw) & (HR >= mcw) & (HL + lam > 0) & (HR + lam > 0)
            if not ok.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - parent) - self.gamma
            gain = np.where(ok, gain, -np.inf)
            i = int(np.argmax(gain))  # first maximum -> lowest threshold
            if gain[i] > best_gain:
                lo, hi = xs[i], xs[i + 1]
                mid = lo * 0.5 + hi * 0.5  # no overflow for extreme finite values
                best_gain, best = float(gain[i]), (f, float(mid if lo <= mid < hi else lo))
        return best

    def _build_tree(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int = 0) -> XGBoostNode:
        G, H = float(np.sum(g)), float(np.sum(h))
        if depth >= self.max_depth or len(X) < 2:
            return self._leaf(G, H)
        split = self._best_split(X, g, h)
        if split is None:
            return self._leaf(G, H)
        feature, threshold = split
        mask = X[:, feature] <= threshold
        node = XGBoostNode()
        node.feature, node.threshold = feature, threshold
        node.left = self._build_tree(X[mask], g[mask], h[mask], depth + 1)
        node.right = self._build_tree(X[~mask], g[~mask], h[~mask], depth + 1)
        return node

    def _tree_predict(self, X: np.ndarray, node: XGBoostNode) -> np.ndarray:
        """Leaf weights for every row of X."""
        out = np.empty(len(X))
        stack = [(node, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf():
                out[idx] = node.leaf_weight
                continue
            left = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[left]))
            stack.append((node.right, idx[~left]))
        return out

    def _scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(X))
        for tree in self.trees_:
            scores += self.learning_rate * self._tree_predict(X, tree)
        return scores

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        """Fit on X (n, d) finite and y (n,) in {0, 1}. Refitting discards all previous state."""
        X = self._check_X(X)
        try:
            y = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must be numeric") from exc
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("X and y must have same number of samples (y 1D)")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y must contain only binary labels 0 and 1")

        self._reset()
        self.classes_ = np.array([0, 1])
        self.n_features_ = X.shape[1]

        scores = np.zeros(len(X))  # base score 0.5 -> margin 0
        for _ in range(self.n_estimators):
            p = self._sigmoid(scores)
            tree = self._build_tree(X, p - y, p * (1.0 - p))
            self.trees_.append(tree)
            scores += self.learning_rate * self._tree_predict(X, tree)
            self.loss_history_.append(self._log_loss(y, scores))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, 2) class probabilities."""
        if not self.trees_:
            raise ValueError("Model not fitted")
        p1 = self._sigmoid(self._scores(self._check_X(X, self.n_features_)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class labels with the M2 default rule probability >= 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _tree_depth(self, node: XGBoostNode, depth: int = 0) -> int:
        if node.is_leaf():
            return depth
        return max(self._tree_depth(node.left, depth + 1), self._tree_depth(node.right, depth + 1))

    def to_dict(self) -> Dict[str, Any]:
        """Complete strict-JSON state (json.dumps(..., allow_nan=False) safe)."""
        if not self.trees_:
            raise ValueError("Model not fitted")

        def tree_to_dict(node: XGBoostNode) -> Dict[str, Any]:
            if node.is_leaf():
                return {"leaf": float(node.leaf_weight)}
            return {
                "feature": int(node.feature),
                "threshold": float(node.threshold),
                "left": tree_to_dict(node.left),
                "right": tree_to_dict(node.right),
            }

        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "n_features": int(self.n_features_),
            "classes": [int(c) for c in self.classes_],
            "trees": [tree_to_dict(tree) for tree in self.trees_],
            "loss_history": [float(v) for v in self.loss_history_],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "XGBoostClassifier":
        """Rebuild a fitted model from to_dict() output (validates structure)."""
        required = ("n_estimators", "max_depth", "learning_rate", "reg_lambda", "gamma", "min_child_weight",
                    "n_features", "classes", "trees", "loss_history")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Missing XGBoost state keys: {missing}")
        n_features = _int("n_features", d["n_features"], 1)

        def dict_to_tree(t: Dict[str, Any]) -> XGBoostNode:
            node = XGBoostNode()
            if "leaf" in t:
                node.leaf_weight = float(t["leaf"])
                if not np.isfinite(node.leaf_weight):
                    raise ValueError("Non-finite leaf weight")
                return node
            node.feature = _int("feature", t["feature"], 0)
            node.threshold = float(t["threshold"])
            if node.feature >= n_features or not np.isfinite(node.threshold):
                raise ValueError("Invalid split node")
            node.left, node.right = dict_to_tree(t["left"]), dict_to_tree(t["right"])
            return node

        model = XGBoostClassifier(*(d[k] for k in ("n_estimators", "max_depth", "learning_rate", "reg_lambda", "gamma", "min_child_weight")))
        if list(d["classes"]) != [0, 1]:
            raise ValueError("classes must be [0, 1]")
        if len(d["trees"]) != model.n_estimators or len(d["loss_history"]) != model.n_estimators:
            raise ValueError("trees/loss_history length must equal n_estimators")
        model.trees_ = [dict_to_tree(t) for t in d["trees"]]
        model.loss_history_ = [float(v) for v in d["loss_history"]]
        model.classes_ = np.array([0, 1])
        model.n_features_ = n_features
        return model
