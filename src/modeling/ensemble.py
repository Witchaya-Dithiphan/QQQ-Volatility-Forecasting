"""
Ensemble methods (RandomForest, GradientBoosting, AdaBoost).
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class RandomForest:
    """Random Forest wrapper."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        if HAS_SKLEARN:
            self._model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            self._model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class GradientBoosting:
    """Gradient Boosting wrapper."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        if HAS_SKLEARN:
            self._model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        else:
            self._model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class AdaBoost:
    """AdaBoost wrapper."""
    
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        if HAS_SKLEARN:
            self._model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
        else:
            self._model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoost":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
