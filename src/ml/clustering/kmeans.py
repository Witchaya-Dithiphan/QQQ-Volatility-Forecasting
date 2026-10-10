"""k-Means by Lloyd's algorithm with k-means++ initialisation."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class KMeans(BaseModel):
    """Alternates assignment and update until the centroids stop moving.

        assign : each point joins its nearest centroid
        update : each centroid becomes the mean of its members
        repeat until the shift falls below tol, or max_iter

    Initialisation is k-means++, which seeds centroids far apart by sampling
    with probability proportional to D(x)^2. sklearn tries 2 + log(k) candidates
    per round and keeps whichever lowers inertia most, so a single draw per round
    would diverge from it immediately.
    """

    name = "kmeans"
    task = "clustering"

    def __init__(
        self,
        k: int = 3,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "k": self.k,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    def _n_local_trials(self) -> int:
        return 2 + int(np.log(self.k))

    @staticmethod
    def _closest_distance_squared(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).min(axis=1)

    def _kmeans_plus_plus(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Seeds centers far apart, following sklearn's RNG calls exactly.

        Three details decide whether this reproduces sklearn or merely resembles
        it: the first center comes from choice(n, p=...) rather than randint,
        the per-round draws come from uniform(size=...), and the candidate index
        is searchsorted over a plain cumsum then clipped. Change any one and the
        stream diverges from the first center onward.
        """
        n_samples = len(X)
        weight = np.ones(n_samples)
        first = rng.choice(n_samples, p=weight / weight.sum())
        centers = [X[first]]
        closest = ((X - X[first]) ** 2).sum(axis=1)
        potential = closest @ weight

        for _ in range(1, self.k):
            draws = rng.uniform(size=self._n_local_trials()) * potential
            candidate_ids = np.searchsorted(np.cumsum(weight * closest), draws)
            np.clip(candidate_ids, None, closest.size - 1, out=candidate_ids)

            to_candidates = ((X[candidate_ids][:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
            np.minimum(closest, to_candidates, out=to_candidates)
            potentials = to_candidates @ weight.reshape(-1, 1)

            best = int(np.argmin(potentials))
            potential = potentials[best]
            closest = to_candidates[best]
            centers.append(X[candidate_ids[best]])
        return np.array(centers)

    def _lloyd(self, X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
        history = []
        for _ in range(self.max_iter):
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            updated = np.array(
                [X[labels == index].mean(axis=0) if np.any(labels == index) else centers[index]
                 for index in range(self.k)]
            )
            history.append(float(distances[np.arange(len(X)), labels].sum()))
            shift = np.linalg.norm(updated - centers)
            centers = updated
            if shift < self.tol:
                break
        labels = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        return centers, labels, history

    def fit(self, X, y=None) -> "KMeans":
        X = np.asarray(X, dtype=np.float64)
        if not 1 <= self.k <= len(X):
            raise ValueError(f"k={self.k} must be between 1 and the sample count {len(X)}")
        # Legacy MT19937, the stream sklearn draws from.
        rng = np.random.RandomState(self.random_state)

        # sklearn subtracts the column means before clustering for distance
        # accuracy and adds them back at the end. The partition is unchanged, but
        # the floating point is not, and neither is the RNG-sensitive seeding.
        offset = X.mean(axis=0)
        centered = X - offset

        best = None
        for _ in range(self.n_init):
            centers, labels, history = self._lloyd(centered, self._kmeans_plus_plus(centered, rng))
            inertia = float(((centered - centers[labels]) ** 2).sum())
            if best is None or inertia < best[0]:
                best = (inertia, centers, labels, history)

        self.inertia_, centers, labels, history = best
        self.cluster_centers_ = centers + offset
        self.history_ = {"iteration": list(range(len(history))), "loss": history}
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"centers": self.cluster_centers_, "inertia": np.array([self.inertia_])}

    def load_state(self, arrays: dict, metadata: dict) -> "KMeans":
        self.cluster_centers_ = np.asarray(arrays["centers"], dtype=np.float64)
        self.inertia_ = float(arrays["inertia"][0])
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        from sklearn.cluster import KMeans as SkKMeans

        return SkKMeans(
            n_clusters=self.k,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
