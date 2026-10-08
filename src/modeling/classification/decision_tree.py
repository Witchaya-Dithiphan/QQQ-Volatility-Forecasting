"""
Decision Tree Classifier (scratch implementation, ID3-style with Gini impurity).
"""

import numpy as np
from typing import Optional, Dict, Any


class DecisionTreeNode:
    """Internal tree node."""
    def __init__(self):
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["DecisionTreeNode"] = None
        self.right: Optional["DecisionTreeNode"] = None
        self.value: Optional[np.ndarray] = None  # class counts (for leaf)
    
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """
    Scratch decision tree classifier using Gini impurity.
    Greedy ID3-style recursive splitting. No pruning.
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, random_state: int = 42):
        """
        Args:
            max_depth: Maximum tree depth.
            min_samples_split: Minimum samples required to split.
            random_state: Random seed.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.tree_: Optional[DecisionTreeNode] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
    
    def _gini(self, y: np.ndarray) -> float:
        """Compute Gini impurity."""
        _, counts = np.unique(y, return_counts=True)
        proba = counts / len(y)
        return 1.0 - np.sum(proba ** 2)
    
    def _split_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """Compute information gain from split."""
        n = len(parent)
        n_left = len(left)
        n_right = len(right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        gini_parent = self._gini(parent)
        gini_left = self._gini(left)
        gini_right = self._gini(right)
        
        weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        return gini_parent - weighted_gini
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """Recursively build tree."""
        node = DecisionTreeNode()
        
        # Base cases: leaf
        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) < self.min_samples_split):
            _, counts = np.unique(y, return_counts=True)
            node.value = counts
            return node
        
        # Find best split
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        for feat_idx in range(X.shape[1]):
            feature_values = X[:, feat_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self._split_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold
        
        # No good split found
        if best_feature is None:
            _, counts = np.unique(y, return_counts=True)
            node.value = counts
            return node
        
        # Split and recurse
        mask = X[:, best_feature] <= best_threshold
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Fit decision tree.
        Args:
            X: (n_samples, n_features)
            y: (n_samples,) class labels
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Predict single sample by traversing tree."""
        if node.is_leaf():
            return self.classes_[np.argmax(node.value)]
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        Args:
            X: (n_samples, n_features)
        Returns:
            (n_samples,) predictions
        """
        if self.tree_ is None:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Feature mismatch: expected {self.n_features_}, got {X.shape[1]}")
        
        predictions = np.array([self._predict_sample(x, self.tree_) for x in X])
        return predictions
    
    def _predict_proba_sample(self, x: np.ndarray, node: DecisionTreeNode) -> np.ndarray:
        """Get class probabilities for single sample."""
        if node.is_leaf():
            # Return probabilities for ALL classes, not just those in leaf
            proba = np.zeros(len(self.classes_))
            for i, c in enumerate(self.classes_):
                if c < len(node.value):
                    proba[i] = node.value[i] / np.sum(node.value)
            return proba
        
        if x[node.feature] <= node.threshold:
            return self._predict_proba_sample(x, node.left)
        else:
            return self._predict_proba_sample(x, node.right)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        Args:
            X: (n_samples, n_features)
        Returns:
            (n_samples, n_classes) probabilities
        """
        if self.tree_ is None:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Feature mismatch: expected {self.n_features_}, got {X.shape[1]}")
        
        n_classes = len(self.classes_)
        probas = np.zeros((len(X), n_classes))
        
        for i, x in enumerate(X):
            proba = self._predict_proba_sample(x, self.tree_)
            probas[i] = proba
        
        return probas
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
        }
