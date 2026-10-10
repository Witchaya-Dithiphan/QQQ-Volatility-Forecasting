"""k-Nearest Neighbours — no training, the data is the model."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class KNN(BaseModel):
    """Classifies by the k closest training points under Euclidean distance.

        uniform  : vote  = count of each class among the k nearest
        distance : vote  = sum of 1/d, so nearer neighbours count for more

    Ties in distance are broken by original index, matching sklearn, which is why
    the sort below is stable.
    """

    name = "knn"
    task = "classification"

    def __init__(self, k: int = 5, weights: str = "uniform"):
        self.k = k
        self.weights = weights

    def get_params(self) -> dict:
        return {"k": self.k, "weights": self.weights}

    def fit(self, X, y=None) -> "KNN":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if self.weights not in ("uniform", "distance"):
            raise ValueError(f"Unknown weights {self.weights!r}; expected uniform or distance")
        if not 1 <= self.k <= len(X):
            raise ValueError(f"k={self.k} must be between 1 and the training size {len(X)}")
        self.X_, self.y_ = X.copy(), y.copy()
        return self

    def _neighbours(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(X[:, None, :] - self.X_[None, :, :], axis=2)
        # Stable sort so equidistant neighbours resolve by training index, like sklearn.
        order = np.argsort(distances, axis=1, kind="stable")[:, : self.k]
        return np.take_along_axis(distances, order, axis=1), self.y_[order]

    def predict_proba(self, X) -> np.ndarray:
        distances, labels = self._neighbours(X)
        if self.weights == "uniform":
            return labels.mean(axis=1)

        probabilities = np.empty(len(distances))
        for row, (d, label) in enumerate(zip(distances, labels)):
            exact = d == 0.0
            # sklearn gives zero-distance neighbours all the weight rather than inf.
            weight = exact.astype(np.float64) if exact.any() else 1.0 / d
            probabilities[row] = weight[label == 1].sum() / weight.sum()
        return probabilities

    def decision_function(self, X) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        # Strict >: sklearn takes argmax over class probabilities, so an exact 0.5
        # tie resolves to the lower class index, not the positive one.
        return (self.predict_proba(X) > 0.5).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"X": self.X_, "y": self.y_}

    def load_state(self, arrays: dict, metadata: dict) -> "KNN":
        self.X_ = np.asarray(arrays["X"], dtype=np.float64)
        self.y_ = np.asarray(arrays["y"], dtype=np.int64)
        self.k = metadata["params"]["k"]
        self.weights = metadata["params"]["weights"]
        return self

    def reference(self):
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(
            n_neighbors=self.k, weights=self.weights, algorithm="brute", metric="euclidean"
        )
