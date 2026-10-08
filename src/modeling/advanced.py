"""
Advanced models (SVM, k-NN, MLP).
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SVM:
    """Support Vector Machine wrapper."""
    
    def __init__(self, kernel: str = "rbf", C: float = 1.0, task: str = "classification"):
        self.kernel = kernel
        self.C = C
        self.task = task
        
        if HAS_SKLEARN:
            if task == "classification":
                self._model = SVC(kernel=kernel, C=C, random_state=42)
            else:
                self._model = SVR(kernel=kernel, C=C)
        else:
            self._model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class KNN:
    """k-Nearest Neighbors wrapper."""
    
    def __init__(self, n_neighbors: int = 5, task: str = "classification"):
        self.n_neighbors = n_neighbors
        self.task = task
        
        if HAS_SKLEARN:
            if task == "classification":
                self._model = KNeighborsClassifier(n_neighbors=n_neighbors)
            else:
                self._model = KNeighborsRegressor(n_neighbors=n_neighbors)
        else:
            self._model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class MLP:
    """Multi-Layer Perceptron wrapper."""
    
    def __init__(self, hidden_layer_sizes: tuple = (100,), max_iter: int = 1000, task: str = "classification"):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.task = task
        
        if HAS_SKLEARN:
            if task == "classification":
                self._model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    random_state=42
                )
            else:
                self._model = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    random_state=42
                )
        else:
            self._model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
