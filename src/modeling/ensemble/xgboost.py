"""
XGBoost Classifier (scratch implementation, binary logistic).

From-scratch XGBoost following Chen & Guestrin 2016.
Greedy exact splits, binary logistic loss, L2 regularization.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple


class XGBoostNode:
    """Decision tree node for XGBoost."""
    def __init__(self):
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["XGBoostNode"] = None
        self.right: Optional["XGBoostNode"] = None
        self.leaf_weight: Optional[float] = None
    
    def is_leaf(self) -> bool:
        return self.leaf_weight is not None


class XGBoostClassifier:
    """
    Scratch XGBoost classifier: binary logistic with exact greedy splits.
    
    Per tree:
    - Gradient: g = p - y
    - Hessian: h = p(1-p)
    - Gain = 0.5*[GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)] - γ
    - Leaf weight: w* = -G / (H + λ)
    """
    
    def __init__(
        self,
        max_trees: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.1,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        """
        Args:
            max_trees: Maximum number of trees.
            max_depth: Maximum tree depth.
            learning_rate: Shrinkage (eta).
            reg_lambda: L2 regularization coefficient.
            min_child_weight: Minimum hessian sum for split.
            gamma: Minimum gain for split.
            tol: Loss improvement tolerance for early stopping.
            random_state: Random seed.
        """
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.tol = tol
        self.random_state = random_state
        
        self.trees_: List[XGBoostNode] = []
        self.scores_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _log_loss(self, y: np.ndarray, scores: np.ndarray) -> float:
        """Binary cross-entropy in log-space (stable)."""
        # L = -[y*log(p) + (1-y)*log(1-p)]
        # = -y*log(1/(1+exp(-s))) - (1-y)*log(exp(-s)/(1+exp(-s)))
        # = y*log(1+exp(-s)) + (1-y)*(-s + log(1+exp(-s)))
        # = y*log(1+exp(-s)) + (1-y)*log(1+exp(-s)) - (1-y)*s
        # = log(1+exp(-s)) - (1-y)*s (via logaddexp)
        # Simplified: mean(np.logaddexp(0, -scores) + (1-y)*scores)
        
        clipped_scores = np.clip(scores, -500, 500)
        loss = np.mean(np.logaddexp(0, clipped_scores) - y * clipped_scores)
        return loss
    
    def _build_tree(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        depth: int = 0,
    ) -> XGBoostNode:
        """Build single tree using exact greedy splits."""
        node = XGBoostNode()
        n_samples = len(X)
        
        # Base case: leaf
        G = np.sum(g)
        H = np.sum(h)
        
        if (depth >= self.max_depth or 
            n_samples < 2 or 
            H < self.min_child_weight):
            node.leaf_weight = -G / (H + self.reg_lambda)
            return node
        
        # Find best split
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        for feat_idx in range(X.shape[1]):
            feature_vals = X[:, feat_idx]
            thresholds = np.unique(feature_vals)
            
            for threshold in thresholds:
                mask = feature_vals <= threshold
                if np.sum(mask) == 0 or np.sum(~mask) == 0:
                    continue
                
                GL = np.sum(g[mask])
                GR = np.sum(g[~mask])
                HL = np.sum(h[mask])
                HR = np.sum(h[~mask])
                
                # Check min_child_weight
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue
                
                # Gain = 0.5*[GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)] - γ
                gain = (
                    0.5 * (
                        GL**2 / (HL + self.reg_lambda) +
                        GR**2 / (HR + self.reg_lambda) -
                        G**2 / (H + self.reg_lambda)
                    ) - self.gamma
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold
        
        # No good split
        if best_feature is None:
            node.leaf_weight = -G / (H + self.reg_lambda)
            return node
        
        # Recurse
        mask = X[:, best_feature] <= best_threshold
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[mask], g[mask], h[mask], depth + 1)
        node.right = self._build_tree(X[~mask], g[~mask], h[~mask], depth + 1)
        
        return node
    
    def _tree_predict(self, x: np.ndarray, node: XGBoostNode) -> float:
        """Predict leaf weight for single sample."""
        if node.is_leaf():
            return node.leaf_weight
        
        if x[node.feature] <= node.threshold:
            return self._tree_predict(x, node.left)
        else:
            return self._tree_predict(x, node.right)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        """
        Fit XGBoost classifier.
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples,) binary labels (0, 1)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        
        # Initialize score to 0 (p=0.5)
        self.scores_ = np.zeros(len(X))
        self.loss_history_ = []
        self.trees_ = []
        
        # Boosting
        for tree_idx in range(self.max_trees):
            # Compute p, gradients, hessians
            proba = self._sigmoid(self.scores_)
            g = proba - y
            h = proba * (1.0 - proba)
            
            # Build tree
            tree = self._build_tree(X, g, h)
            self.trees_.append(tree)
            
            # Update scores
            tree_pred = np.array([self._tree_predict(x, tree) for x in X])
            self.scores_ += self.learning_rate * tree_pred
            
            # Compute loss
            loss = self._log_loss(y, self.scores_)
            self.loss_history_.append(loss)
            
            # Early stopping
            if len(self.loss_history_) > 1:
                if self.loss_history_[-2] - self.loss_history_[-1] < self.tol:
                    break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            (n_samples, 2) probabilities
        """
        if len(self.trees_) == 0:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Feature mismatch: expected {self.n_features_}, got {X.shape[1]}")
        
        scores = np.zeros(len(X))
        for tree in self.trees_:
            tree_pred = np.array([self._tree_predict(x, tree) for x in X])
            scores += self.learning_rate * tree_pred
        
        proba_1 = self._sigmoid(scores)
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            (n_samples,) predictions
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def _tree_depth(self, node: XGBoostNode, depth: int = 0) -> int:
        """Compute tree depth."""
        if node.is_leaf():
            return depth
        return max(
            self._tree_depth(node.left, depth + 1),
            self._tree_depth(node.right, depth + 1),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model."""
        if len(self.trees_) == 0:
            raise ValueError("Model not fitted")
        
        def tree_to_dict(node: XGBoostNode) -> Dict[str, Any]:
            if node.is_leaf():
                return {"leaf": float(node.leaf_weight)}
            return {
                "feature": int(node.feature),
                "threshold": float(node.threshold),
                "left": tree_to_dict(node.left),
                "right": tree_to_dict(node.right),
            }
        
        return {
            "max_trees": self.max_trees,
            "max_depth": self.max_depth,
            "learning_rate": float(self.learning_rate),
            "reg_lambda": float(self.reg_lambda),
            "min_child_weight": float(self.min_child_weight),
            "gamma": float(self.gamma),
            "n_features": int(self.n_features_),
            "trees": [tree_to_dict(tree) for tree in self.trees_],
            "loss_history": [float(l) for l in self.loss_history_],
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "XGBoostClassifier":
        """Deserialize model."""
        def dict_to_tree(tree_dict: Dict[str, Any]) -> XGBoostNode:
            node = XGBoostNode()
            if "leaf" in tree_dict:
                node.leaf_weight = float(tree_dict["leaf"])
            else:
                node.feature = int(tree_dict["feature"])
                node.threshold = float(tree_dict["threshold"])
                node.left = dict_to_tree(tree_dict["left"])
                node.right = dict_to_tree(tree_dict["right"])
            return node
        
        model = XGBoostClassifier(
            max_trees=d["max_trees"],
            max_depth=d["max_depth"],
            learning_rate=d["learning_rate"],
            reg_lambda=d["reg_lambda"],
            min_child_weight=d["min_child_weight"],
            gamma=d["gamma"],
        )
        model.trees_ = [dict_to_tree(t) for t in d["trees"]]
        model.loss_history_ = d["loss_history"]
        model.n_features_ = d["n_features"]
        return model
