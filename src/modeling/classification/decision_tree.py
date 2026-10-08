"""
Decision Tree Classifier (sklearn wrapper).
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DecisionTreeClassifier:
    """Decision Tree wrapper (CART algorithm)."""
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, max_features: str = "sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        
        self._sklearn_model = None
        self.classes_: Optional[np.ndarray] = None
        
        if HAS_SKLEARN:
            self._sklearn_model = SKDecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                random_state=42
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        
        self._sklearn_model.fit(X, y)
        self.classes_ = self._sklearn_model.classes_.copy()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._sklearn_model is None:
            raise ValueError("Model not fitted")
        return self._sklearn_model.predict(X)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
        }
