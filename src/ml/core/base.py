"""The contract every scratch model implements. See AGENTS.md section 4.

Abstract rather than duck-typed on purpose: the two model families are written in
parallel by different developers on different branches, so a missing method has to
fail at construction, not after a training run.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

TASKS = ("regression", "classification", "clustering")


class BaseModel(ABC):
    name: str = ""
    task: str = ""

    @abstractmethod
    def get_params(self) -> dict:
        """Every hyperparameter, JSON-serializable, enough to rebuild this model.

        All of them must have defaults in __init__: the trainer rebuilds a model with
        model_class() and then load_state() to prove a reloaded model still predicts
        the same, and that step breaks if the constructor demands arguments.
        """

    @abstractmethod
    def fit(self, X, y=None) -> "BaseModel":
        """Clustering models receive y=None."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """regression: values | classification: 0/1 labels | clustering: cluster ids."""

    @abstractmethod
    def state_dict(self) -> dict[str, np.ndarray]:
        """Only learned parameters, as NumPy arrays."""

    @abstractmethod
    def load_state(self, arrays: dict, metadata: dict) -> "BaseModel":
        """Rebuild from state_dict() output plus {"params": get_params()}."""

    def decision_function(self, X) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not produce decision scores")

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not produce calibrated probabilities")

    def reference(self):
        raise NotImplementedError(f"{type(self).__name__} has no reference implementation yet")
