"""
k-Nearest Neighbors (scratch, NumPy only): Euclidean distance, uniform or inverse-distance votes.

Determinism: neighbors are ranked by (distance, training row index), so ties always favor the earlier training row;
an exactly tied vote (probability 0.5) predicts class 1, as in the M2 `score >= threshold` convention. With
weights="distance", exact-distance-0 matches alone vote (sklearn convention).

Overflow safety: distances are max-scaled (m * sqrt(sum((d/m)^2))) so squaring neither overflows nor underflows, and when
values exceed 1e150 all coordinates are first multiplied by an exact power of two (a common positive factor, so neighbor
order and normalized votes are unchanged). Standardization that leaves float64 range is rejected (see `apply_scaler`).
"""
from typing import Optional

import numpy as np

from ._common import NOT_FITTED, apply_scaler, check_param, check_X, check_Xy, fit_scaler, scaler_from_dict, scaler_to_dict

_CHUNK = 32  # test rows per distance block; bounds the (chunk, n_train, d) broadcast
_SAFE = 1e150  # beyond this, coordinate differences times sqrt(d) could leave float64 range


class KNN:
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", standardize: bool = True):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.standardize = standardize
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.scaler_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        check_param("n_neighbors", self.n_neighbors, integer=True)
        if self.weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        X, y = check_Xy(X, y)
        if self.n_neighbors > len(y):
            raise ValueError("n_neighbors exceeds the number of training samples")
        self.scaler_, self.X_train_ = fit_scaler(X, self.standardize)
        self.y_train_ = y
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise ValueError(NOT_FITTED)
        X = check_X(X, self.X_train_.shape[1])
        if self.scaler_ is not None:
            X = apply_scaler(self.scaler_, X)
        train = self.X_train_
        peak = max(np.abs(X).max(), np.abs(train).max())
        if peak > _SAFE:
            factor = 2.0 ** -np.frexp(peak)[1]  # exact scaling into [0.5, 1)
            X, train = X * factor, train * factor
        out = np.empty((len(X), 2))
        for lo in range(0, len(X), _CHUNK):
            diff = X[lo:lo + _CHUNK, None, :] - train[None, :, :]
            m = np.abs(diff).max(axis=2)
            dist = m * np.sqrt(((diff / np.where(m > 0, m, 1.0)[..., None]) ** 2).sum(axis=2))
            for i, d in enumerate(dist):
                nearest = np.argsort(d, kind="stable")[:self.n_neighbors]
                labels, d = self.y_train_[nearest], d[nearest]
                if self.weights == "uniform":
                    w = np.ones(len(nearest))
                elif (d == 0).any():
                    w = (d == 0).astype(np.float64)
                else:
                    w = d.min() / d  # 1/d rescaled by a common factor: avoids inf for tiny d, same normalized votes
                votes = np.array([w[labels == 0].sum(), w[labels == 1].sum()])
                out[lo + i] = votes / votes.sum()
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def to_dict(self) -> dict:
        if self.X_train_ is None:
            raise ValueError(NOT_FITTED)
        return {"n_neighbors": self.n_neighbors, "weights": self.weights, "standardize": self.standardize,
                "scaler": scaler_to_dict(self.scaler_), "X_train": self.X_train_.tolist(), "y_train": self.y_train_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "KNN":
        model = cls(d["n_neighbors"], d["weights"], d["standardize"])
        model.scaler_ = scaler_from_dict(d["scaler"])
        model.X_train_, model.y_train_ = np.array(d["X_train"], dtype=np.float64), np.array(d["y_train"], dtype=np.int64)
        return model
