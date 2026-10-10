"""Agglomerative clustering — bottom-up merging via Lance-Williams updates."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class Agglomerative(BaseModel):
    """Starts with every point alone and merges the cheapest pair until k remain.

        ward    : cost = (|A||B| / (|A|+|B|)) * ||centroid_A - centroid_B||^2
        average : cost = mean distance over all cross-pairs

    Costs are maintained with the Lance-Williams recurrence rather than
    recomputed from scratch after each merge, which is both how scipy and
    sklearn do it and the only way the numbers stay identical to theirs:

        d(A u B, C) = alpha_a*d(A,C) + alpha_b*d(B,C) + beta*d(A,B) + gamma*|d(A,C) - d(B,C)|

    **This algorithm has no native predict.** It produces a labelling of the
    training set, not a rule for new points. The config settles that with
    oos_extension: frozen_nearest_centroid — freeze each cluster's centroid
    after fitting and send unseen points to the closest one. That extension is
    an addition of ours and has to be declared as a limitation in the report.
    """

    name = "agglomerative"
    task = "clustering"

    def __init__(self, k: int = 3, linkage: str = "ward"):
        self.k = k
        self.linkage = linkage

    def get_params(self) -> dict:
        return {"k": self.k, "linkage": self.linkage}

    def _initial_cost(self, X: np.ndarray) -> np.ndarray:
        difference = X[:, None, :] - X[None, :, :]
        squared = np.einsum("ijk,ijk->ij", difference, difference)
        if self.linkage == "ward":
            # Two singletons: the merge raises within-cluster variance by d^2/2.
            return 0.5 * squared
        return np.sqrt(squared)

    def _lance_williams(self, cost, sizes, a, b, alive):
        size_a, size_b = sizes[a], sizes[b]
        if self.linkage == "ward":
            size_c = sizes[alive]
            total = size_a + size_b + size_c
            updated = (
                (size_a + size_c) * cost[a, alive]
                + (size_b + size_c) * cost[b, alive]
                - size_c * cost[a, b]
            ) / total
        else:
            updated = (size_a * cost[a, alive] + size_b * cost[b, alive]) / (size_a + size_b)
        return updated

    def fit(self, X, y=None) -> "Agglomerative":
        X = np.asarray(X, dtype=np.float64)
        if self.linkage not in ("ward", "average"):
            raise ValueError(f"Unknown linkage {self.linkage!r}; expected ward or average")
        n_samples = len(X)
        if not 1 <= self.k <= n_samples:
            raise ValueError(f"k={self.k} must be between 1 and the sample count {n_samples}")

        cost = self._initial_cost(X)
        np.fill_diagonal(cost, np.inf)
        sizes = np.ones(n_samples)
        members = {index: [index] for index in range(n_samples)}
        active = np.ones(n_samples, dtype=bool)

        while active.sum() > self.k:
            masked = np.where(active[:, None] & active[None, :], cost, np.inf)
            # argmin scans in row-major order, so the lowest index pair wins a tie.
            a, b = np.unravel_index(np.argmin(masked), masked.shape)
            a, b = (a, b) if a < b else (b, a)

            others = active.copy()
            others[a] = others[b] = False
            if others.any():
                updated = self._lance_williams(cost, sizes, a, b, others)
                cost[a, others] = updated
                cost[others, a] = updated

            members[a] = members[a] + members[b]
            sizes[a] += sizes[b]
            active[b] = False
            cost[a, a] = np.inf

        labels = np.empty(n_samples, dtype=np.int64)
        for new_id, root in enumerate(np.flatnonzero(active)):
            labels[members[root]] = new_id

        self.labels_ = labels
        self.n_clusters_ = int(active.sum())
        self.cluster_centers_ = np.vstack(
            [X[labels == index].mean(axis=0) for index in range(self.n_clusters_)]
        )
        self.train_X_ = X.copy()
        self.history_ = {"iteration": [0], "loss": [float(self._within_cluster_sum(X, labels))]}
        return self

    @staticmethod
    def _within_cluster_sum(X, labels) -> float:
        return float(
            sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels))
        )

    def predict(self, X) -> np.ndarray:
        """Training rows keep their fitted labels; anything else gets the nearest centroid."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape == self.train_X_.shape and np.array_equal(X, self.train_X_):
            return self.labels_
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return distances.argmin(axis=1).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "centers": self.cluster_centers_,
            "labels": self.labels_,
            "train_X": self.train_X_,
        }

    def load_state(self, arrays: dict, metadata: dict) -> "Agglomerative":
        self.cluster_centers_ = np.asarray(arrays["centers"], dtype=np.float64)
        self.labels_ = np.asarray(arrays["labels"], dtype=np.int64)
        self.train_X_ = np.asarray(arrays["train_X"], dtype=np.float64)
        self.n_clusters_ = len(self.cluster_centers_)
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        from sklearn.cluster import AgglomerativeClustering as SkAgglomerative

        return SkAgglomerative(n_clusters=self.k, linkage=self.linkage, metric="euclidean")
