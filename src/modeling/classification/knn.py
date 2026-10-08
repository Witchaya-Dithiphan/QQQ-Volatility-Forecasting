"""
k-Nearest Neighbors classifier (thin wrapper over sklearn.neighbors.KNeighborsClassifier).
M5-04: With-Spike classification reference implementation.
"""

from sklearn.neighbors import KNeighborsClassifier

from src.modeling.ensemble.sklearn_wrapper import SklearnEnsemble


class KNN(SklearnEnsemble):
    """k-NN wrapper (reference, not scratch)."""

    _cls = KNeighborsClassifier
    _defaults = {"n_neighbors": 5, "metric": "euclidean"}
