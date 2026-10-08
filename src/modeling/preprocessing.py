"""Train-fitted float64 scaling and deterministic covariance PCA."""
from __future__ import annotations
import numpy as np


def _matrix(X) -> np.ndarray:
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2 or not all(data.shape) or not np.isfinite(data).all():
        raise ValueError("Expected nonempty finite 2D matrix")
    return data


class Standardizer:
    def __init__(self, variance_floor: float = 1e-12):
        if not np.isfinite(variance_floor) or variance_floor < 0:
            raise ValueError("Invalid variance floor")
        self.variance_floor = variance_floor

    def fit(self, X) -> Standardizer:
        data = _matrix(X)
        self.mean_ = data.mean(axis=0)
        self.variance_ = data.var(axis=0, ddof=0)
        self.constant_ = self.variance_ <= self.variance_floor
        self.scale_ = np.sqrt(self.variance_)
        self.scale_[self.constant_] = 1.0
        self.warnings_ = [f"Constant feature {index}: variance <= {self.variance_floor}" for index in np.flatnonzero(self.constant_)]
        return self

    def transform(self, X) -> np.ndarray:
        data = _matrix(X)
        if not hasattr(self, "mean_") or data.shape[1] != len(self.mean_):
            raise ValueError("Unfitted Standardizer or feature count mismatch")
        transformed = (data - self.mean_) / self.scale_
        transformed[:, self.constant_] = 0.0
        return transformed

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


def _projector_basis(vectors: np.ndarray) -> np.ndarray:
    projector = vectors @ vectors.T
    basis = []
    for index in range(projector.shape[0]):
        candidate = projector[:, index].copy()
        # Reorthogonalized modified Gram-Schmidt protects near-dependent columns.
        for _ in range(2):
            for vector in basis:
                candidate -= np.dot(vector, candidate) * vector
        norm = np.linalg.norm(candidate)
        if norm > 1e-12:
            basis.append(candidate / norm)
        if len(basis) == vectors.shape[1]:
            return np.column_stack(basis)
    raise ValueError("Could not construct degenerate eigenspace basis")


class PCA:
    """Covariance PCA on standardized Train data; never center internally.

    Each column mean must be within 1e-10 of zero (absolute tolerance
    on the standardized scale), allowing float64 standardization noise.
    """
    def __init__(self, n_components: int, *, rank_floor: float = 1e-12, degenerate_rtol: float = 1e-10, negative_tolerance: float = 1e-12):
        self.n_components = n_components
        self.rank_floor = rank_floor
        self.degenerate_rtol = degenerate_rtol
        self.negative_tolerance = negative_tolerance

    def fit(self, X) -> PCA:
        data = _matrix(X)
        if data.shape[0] < 2 or type(self.n_components) is not int or not 1 <= self.n_components <= data.shape[1]:
            raise ValueError("Invalid PCA samples/component count")
        if (np.abs(data.mean(axis=0)) > 1e-10).any():
            raise ValueError("PCA requires centered standardized Train input (mean tolerance 1e-10)")
        covariance = data.T @ data / (data.shape[0] - 1)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        numerical_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if (eigenvalues < -self.negative_tolerance * numerical_scale).any():
            raise ValueError("Significantly negative covariance eigenvalue")
        eigenvalues = np.maximum(eigenvalues, 0.0)
        order = np.argsort(-eigenvalues, kind="stable")
        eigenvalues, vectors = eigenvalues[order], vectors[:, order]
        self.rank_ = int((eigenvalues > self.rank_floor).sum())
        if self.n_components > self.rank_:
            raise ValueError("PCA component count exceeds numerical rank")
        start = 0
        while start < len(eigenvalues):
            stop = start + 1
            while stop < len(eigenvalues) and abs(eigenvalues[stop] - eigenvalues[start]) <= self.degenerate_rtol * max(abs(eigenvalues[start]), abs(eigenvalues[stop])):
                stop += 1
            if stop - start > 1:
                vectors[:, start:stop] = _projector_basis(vectors[:, start:stop])
            start = stop
        for column in range(vectors.shape[1]):
            vector = vectors[:, column]
            magnitudes = np.abs(vector)
            # Lowest index among relative near-ties, matching degeneracy policy.
            pivot = np.flatnonzero(magnitudes.max() - magnitudes <= self.degenerate_rtol * magnitudes.max())[0]
            if vector[pivot] < 0:
                vectors[:, column] *= -1
        self.eigenvalues_ = eigenvalues
        self.components_ = vectors[:, :self.n_components].T.copy()
        self.n_features_ = data.shape[1]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()
        return self

    def transform(self, X) -> np.ndarray:
        data = _matrix(X)
        if not hasattr(self, "components_") or data.shape[1] != self.n_features_:
            raise ValueError("Unfitted PCA or feature count mismatch")
        return data @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
