"""Bootstrap ensemble of scratch Gini decision trees."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from .decision_tree import DecisionTree


class RandomForest(BaseModel):
    name = "random_forest"
    task = "classification"

    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 min_samples_leaf: int = 5, max_features: str | None = "sqrt",
                 bootstrap: bool = True, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }

    def fit(self, X, y=None) -> "RandomForest":
        if (not isinstance(self.n_estimators, int) or self.n_estimators < 1
                or not isinstance(self.max_depth, int) or self.max_depth < 1
                or not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1
                or self.max_features not in (None, "sqrt")
                or not isinstance(self.bootstrap, bool)
                or not isinstance(self.random_state, int) or not 0 <= self.random_state < 2**32):
            raise ValueError("Invalid random forest hyperparameters")
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or y.shape != (len(X),):
            raise ValueError("X and y must be aligned nonempty arrays")
        self.n_features_in_ = X.shape[1]
        seeds = np.random.RandomState(self.random_state).randint(
            np.iinfo(np.int32).max, size=self.n_estimators)
        self.trees_ = []
        for seed in seeds:
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=self.max_features,
                                random_state=int(seed))
            weights = None
            if self.bootstrap:
                indices = np.random.RandomState(int(seed)).randint(0, len(X), len(X))
                weights = np.bincount(indices, minlength=len(X))
            self.trees_.append(tree.fit(X, y, sample_weight=weights))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([tree.predict_proba(X) for tree in self.trees_], axis=0)

    def decision_function(self, X) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        arrays = {"n_features_in": np.array([self.n_features_in_]),
                  "n_trees": np.array([len(self.trees_)])}
        for i, tree in enumerate(self.trees_):
            arrays.update({f"tree_{i}_{key}": value for key, value in tree.state_dict().items()})
        return arrays

    def load_state(self, arrays: dict, metadata: dict) -> "RandomForest":
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        self.n_features_in_ = int(arrays["n_features_in"][0])
        self.trees_ = []
        for i in range(int(arrays["n_trees"][0])):
            prefix = f"tree_{i}_"
            state = {key[len(prefix):]: value for key, value in arrays.items()
                     if key.startswith(prefix)}
            self.trees_.append(DecisionTree().load_state(state, {"params": {}}))
        return self

    def reference(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
        )
