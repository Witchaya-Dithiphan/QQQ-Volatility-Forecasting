"""
Random Forest classifier (thin wrapper over sklearn.ensemble.RandomForestClassifier).
"""

import base64
import pickle
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    """sklearn RandomForestClassifier with fit/predict/predict_proba/to_dict/from_dict."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_: Optional[RandomForestClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def _fitted(self) -> RandomForestClassifier:
        if self.model_ is None:
            raise ValueError("Model not fitted")
        return self.model_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict_proba(X)

    def to_dict(self) -> Dict[str, Any]:
        # ponytail: pickled trees; from_dict must only be fed trusted data.
        blob = base64.b64encode(pickle.dumps(self._fitted())).decode("ascii")
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "model_pickle_b64": blob,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RandomForest":
        model = RandomForest(d["n_estimators"], d["max_depth"], d["random_state"])
        model.model_ = pickle.loads(base64.b64decode(d["model_pickle_b64"]))
        return model
