"""
Shared thin wrapper for sklearn ensemble classifiers (fit/predict/predict_proba/to_dict/from_dict).
"""

import base64
import pickle
from typing import Any, Dict, Optional

import numpy as np


class SklearnEnsemble:
    """Subclasses set `_cls` (sklearn estimator class) and `_defaults` (hyperparameters)."""

    _cls: type
    _defaults: Dict[str, Any] = {}

    def __init__(self, **params: Any):
        self.params = {**self._defaults, **params}
        self.model_: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model_ = self._cls(**self.params).fit(X, y)
        return self

    def _fitted(self):
        if self.model_ is None:
            raise ValueError("Model not fitted")
        return self.model_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict_proba(X)

    def to_dict(self) -> Dict[str, Any]:
        # ponytail: pickled estimator; from_dict must only be fed trusted data.
        blob = base64.b64encode(pickle.dumps(self._fitted())).decode("ascii")
        return {"params": self.params, "model_pickle_b64": blob}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        model = cls(**d["params"])
        model.model_ = pickle.loads(base64.b64decode(d["model_pickle_b64"]))
        return model
