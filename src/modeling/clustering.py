"""
Clustering methods (k-Means, Agglomerative, PCA).
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class KMeansClustering:
    """k-Means wrapper."""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
        if HAS_SKLEARN:
            self._model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        else:
            self._model = None
    
    def fit(self, X: np.ndarray) -> "KMeansClustering":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class AgglomerativeClustering:
    """Hierarchical clustering wrapper."""
    
    def __init__(self, n_clusters: int = 3, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        
        if HAS_SKLEARN:
            self._model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        else:
            self._model = None
    
    def fit(self, X: np.ndarray) -> "AgglomerativeClustering":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X)
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        return self._model.fit_predict(X)


class PCADecomposition:
    """PCA wrapper."""
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        
        if HAS_SKLEARN:
            self._model = PCA(n_components=n_components, random_state=42)
        else:
            self._model = None
    
    def fit(self, X: np.ndarray) -> "PCADecomposition":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        self._model.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        return self._model.fit_transform(X)
