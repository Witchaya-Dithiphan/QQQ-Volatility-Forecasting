"""Scratch Random Forest: bootstrap CART/Gini trees with sqrt feature subsampling and hard-vote probabilities.

Randomness: tree i owns seed tree_seeds_[i]; default_rng(seed) first draws the bootstrap rows, then the
per-split feature subsets. Serializing the seeds therefore reproduces bootstrap_indices_ and every tree.
"""
from typing import Any, Dict, Optional

import numpy as np

from src.modeling.ensemble._tree import (
    build_tree, check_X, check_binary_y, check_int, predict_tree, validate_tree,
)


class RandomForest:
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, max_features: str = "sqrt",
                 min_samples_leaf: int = 5, random_state: int = 42):
        self.n_estimators = check_int("n_estimators", n_estimators)
        self.max_depth = check_int("max_depth", max_depth)
        self.min_samples_leaf = check_int("min_samples_leaf", min_samples_leaf)
        self.random_state = check_int("random_state", random_state, 0)
        if max_features != "sqrt":
            raise ValueError("max_features must be 'sqrt'")
        self.max_features = max_features
        self.trees_: Optional[list] = None
        self.tree_seeds_: Optional[np.ndarray] = None
        self.bootstrap_indices_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.max_features_: Optional[int] = None

    @staticmethod
    def _bootstrap(seed: int, n_samples: int):
        rng = np.random.default_rng(seed)
        return rng, rng.integers(0, n_samples, n_samples)

    def fit(self, X, y) -> "RandomForest":
        X = check_X(X)
        y = check_binary_y(y, len(X))
        self.n_features_ = X.shape[1]
        self.max_features_ = max(1, int(np.sqrt(self.n_features_)))
        self.tree_seeds_ = np.random.default_rng(self.random_state).integers(0, 2 ** 32, self.n_estimators)
        trees, boots = [], []
        for seed in self.tree_seeds_:
            rng, idx = self._bootstrap(int(seed), len(X))
            trees.append(build_tree(X[idx], y[idx], criterion="gini", max_depth=self.max_depth,
                                    min_samples_leaf=self.min_samples_leaf, max_features=self.max_features_, rng=rng))
            boots.append(idx)
        self.bootstrap_indices_, self.trees_ = np.array(boots), trees
        return self

    def _fitted(self, X) -> np.ndarray:
        if self.trees_ is None:
            raise ValueError("Model not fitted")
        return check_X(X, self.n_features_)

    def predict_tree_votes(self, i: int, X) -> np.ndarray:
        """Hard class vote (0/1) of tree i; a leaf votes 1 only on a strict class-1 majority."""
        return (predict_tree(self.trees_[i], self._fitted(X)) > 0.5).astype(np.int64)

    def predict_proba(self, X) -> np.ndarray:
        X = self._fitted(X)
        votes = np.sum([predict_tree(t, X) > 0.5 for t in self.trees_], axis=0)
        p1 = votes / len(self.trees_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def to_dict(self) -> Dict[str, Any]:
        if self.trees_ is None:
            raise ValueError("Model not fitted")
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth, "max_features": self.max_features,
                "min_samples_leaf": self.min_samples_leaf, "random_state": self.random_state,
                "n_features": self.n_features_, "n_samples": int(self.bootstrap_indices_.shape[1]),
                "tree_seeds": [int(s) for s in self.tree_seeds_], "trees": self.trees_}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RandomForest":
        model = cls(d["n_estimators"], d["max_depth"], d["max_features"], d["min_samples_leaf"], d["random_state"])
        n_features, n_samples = check_int("n_features", d["n_features"]), check_int("n_samples", d["n_samples"])
        seeds, trees = d["tree_seeds"], d["trees"]
        if len(seeds) != len(trees) or len(trees) != model.n_estimators:
            raise ValueError("Malformed forest state")
        for tree in trees:
            validate_tree(tree, n_features)
        model.n_features_, model.max_features_ = n_features, max(1, int(np.sqrt(n_features)))
        model.tree_seeds_ = np.array(seeds, dtype=np.int64)
        model.bootstrap_indices_ = np.array([cls._bootstrap(int(s), n_samples)[1] for s in seeds])
        model.trees_ = [{k: list(v) for k, v in t.items()} for t in trees]
        return model
