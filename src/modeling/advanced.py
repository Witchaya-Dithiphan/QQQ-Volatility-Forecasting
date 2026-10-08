"""Compatibility aliases: the scratch M6 SVM/MLP classifiers (no sklearn wrappers)."""
from .classification.mlp import MLPClassifier as MLP  # noqa: F401
from .classification.svm import LinearSVM as SVM  # noqa: F401
from .classification.knn import KNN  # noqa: F401
