"""
k-Nearest Neighbors classifier (scratch implementation, not sklearn wrapper).
M5-04: With-Spike classification, binary classification only.

Algorithm:
  1. Store training data (no model learning)
  2. Predict: find k nearest neighbors by Euclidean distance
  3. Majority vote among k neighbors
  4. predict_proba: fraction of neighbors in each class
"""

from typing import Optional

import numpy as np


class KNN:
    """k-Nearest Neighbors classifier (scratch)."""

    def __init__(self, n_neighbors: int = 5):
        """
        Args:
            n_neighbors: Number of neighbors to use for voting.
        """
        self.n_neighbors = n_neighbors
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """
        Store training data (no model learning for k-NN).

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,), binary {0, 1}

        Returns:
            self
        """
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y, dtype=np.int32)
        return self

    def _fitted(self):
        """Check if model is fitted."""
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("Model not fitted")
        return self.X_train_, self.y_train_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,), binary {0, 1}
        """
        X_train, y_train = self._fitted()
        X = np.asarray(X, dtype=np.float32)

        if X.shape[1] != X_train.shape[1]:
            raise ValueError(
                f"Feature mismatch: expected {X_train.shape[1]}, got {X.shape[1]}"
            )

        # Compute Euclidean distances: shape (n_test, n_train)
        # distance[i, j] = ||X[i] - X_train[j]||
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2))

        # Find k nearest neighbors for each test sample
        k = min(self.n_neighbors, len(y_train))
        predictions = []

        for i in range(len(X)):
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(distances[i])[:k]
            nearest_labels = y_train[nearest_indices]

            # Majority vote
            prediction = np.bincount(nearest_labels).argmax()
            predictions.append(prediction)

        return np.array(predictions, dtype=np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, 2).
            proba[:, 0] = P(y=0), proba[:, 1] = P(y=1)
        """
        X_train, y_train = self._fitted()
        X = np.asarray(X, dtype=np.float32)

        if X.shape[1] != X_train.shape[1]:
            raise ValueError(
                f"Feature mismatch: expected {X_train.shape[1]}, got {X.shape[1]}"
            )

        # Compute distances
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2))

        # Find k nearest neighbors and compute class fractions
        k = min(self.n_neighbors, len(y_train))
        proba_list = []

        for i in range(len(X)):
            nearest_indices = np.argsort(distances[i])[:k]
            nearest_labels = y_train[nearest_indices]

            # Count each class
            count_0 = np.sum(nearest_labels == 0)
            count_1 = np.sum(nearest_labels == 1)

            # Probability = fraction of each class
            p_0 = count_0 / k
            p_1 = count_1 / k

            proba_list.append([p_0, p_1])

        return np.array(proba_list, dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialize model to dictionary."""
        X_train, y_train = self._fitted()
        return {
            "n_neighbors": self.n_neighbors,
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KNN":
        """Deserialize model from dictionary."""
        model = cls(n_neighbors=d["n_neighbors"])
        model.X_train_ = np.array(d["X_train"], dtype=np.float32)
        model.y_train_ = np.array(d["y_train"], dtype=np.int32)
        return model
