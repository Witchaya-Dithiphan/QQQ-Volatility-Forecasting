"""
Stacking Classifier (scratch implementation).

Architecture:
- Base learners: Logistic Regression, Decision Tree, k-NN (trained on OOS folds)
- Meta-learner: Logistic Regression (trained on OOS predictions from base learners)
- Protocol: 4-fold chronological expanding window, no shuffle
"""

import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta


class StackingClassifier:
    """
    Scratch stacking classifier with chronological OOS folds.
    
    4-fold expanding window protocol:
    - Fold 1: Train on [0:25%], predict on [25%:50%]
    - Fold 2: Train on [0:50%], predict on [50%:75%]
    - Fold 3: Train on [0:75%], predict on [75%:100%]
    - Fold 4: Train on [0:100%], predict on full data (for final fit)
    
    Then fit meta-learner on OOS predictions.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Base learners (will be sklearn for simplicity; can swap to scratch)
        self.base_learners_: Optional[List] = None
        self.meta_learner_: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingClassifier":
        """
        Fit stacking classifier using 4-fold chronological OOS protocol.
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples,) binary labels
        
        Returns:
            self
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        n_samples = len(X)
        self.classes_ = np.unique(y)
        
        # 4-fold chronological expanding splits
        fold_sizes = [n_samples // 4, n_samples // 2, (3 * n_samples) // 4, n_samples]
        
        # Collect OOS predictions for meta-learner training
        oos_predictions = []
        oos_labels = []
        
        for fold_idx, test_size in enumerate(fold_sizes[:-1]):
            train_idx = np.arange(test_size)
            val_idx = np.arange(test_size, fold_sizes[fold_idx + 1])
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train base learners
            lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
            dt = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
            knn = KNeighborsClassifier(n_neighbors=5)
            
            lr.fit(X_train, y_train)
            dt.fit(X_train, y_train)
            knn.fit(X_train, y_train)
            
            # Get OOS predictions (probabilities)
            lr_proba = lr.predict_proba(X_val)[:, 1]  # P(class=1)
            dt_proba = dt.predict_proba(X_val)[:, 1]
            knn_proba = knn.predict_proba(X_val)[:, 1]
            
            # Stack predictions
            oos_meta_features = np.column_stack([lr_proba, dt_proba, knn_proba])
            oos_predictions.append(oos_meta_features)
            oos_labels.append(y_val)
        
        # Combine all OOS folds
        X_meta = np.vstack(oos_predictions)
        y_meta = np.hstack(oos_labels)
        
        # Fit meta-learner on OOS predictions
        self.meta_learner_ = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.meta_learner_.fit(X_meta, y_meta)
        
        # Store base learners trained on full data
        lr_full = LogisticRegression(max_iter=1000, random_state=self.random_state)
        dt_full = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
        knn_full = KNeighborsClassifier(n_neighbors=5)
        
        lr_full.fit(X, y)
        dt_full.fit(X, y)
        knn_full.fit(X, y)
        
        self.base_learners_ = [lr_full, dt_full, knn_full]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            (n_samples,) predictions
        """
        if self.meta_learner_ is None or self.base_learners_ is None:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Get base learner predictions
        base_probas = [
            self.base_learners_[0].predict_proba(X)[:, 1],
            self.base_learners_[1].predict_proba(X)[:, 1],
            self.base_learners_[2].predict_proba(X)[:, 1],
        ]
        X_meta = np.column_stack(base_probas)
        
        # Meta-learner prediction
        return self.meta_learner_.predict(X_meta)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates."""
        X = np.asarray(X, dtype=np.float64)
        
        base_probas = [
            self.base_learners_[0].predict_proba(X)[:, 1],
            self.base_learners_[1].predict_proba(X)[:, 1],
            self.base_learners_[2].predict_proba(X)[:, 1],
        ]
        X_meta = np.column_stack(base_probas)
        
        return self.meta_learner_.predict_proba(X_meta)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize (simplified)."""
        return {"type": "StackingClassifier", "classes": self.classes_.tolist()}
