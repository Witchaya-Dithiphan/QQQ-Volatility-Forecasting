"""
Gaussian Naive Bayes (scratch implementation).

Binary classifier using Gaussian likelihood per class.
Fit: compute class priors, per-feature means, per-feature variances.
Predict: argmax_c [ log(P(c)) + Σ_j log N(x_j | μ_c,j, σ²_c,j + ε) ]
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple


class GaussianNaiveBayes:
    """
    Scratch Gaussian Naive Bayes for binary classification.
    
    Assumes feature independence given class label (naive assumption).
    Fit per-class Gaussian distributions.
    """
    
    def __init__(self, epsilon: float = 1e-9):
        """
        Args:
            epsilon: Laplace smoothing for zero-variance features.
        """
        self.epsilon = epsilon
        
        # Fitted parameters
        self.classes_: Optional[np.ndarray] = None
        self.class_priors_: Optional[np.ndarray] = None  # P(c)
        self.means_: Optional[np.ndarray] = None  # (n_classes, n_features)
        self.variances_: Optional[np.ndarray] = None  # (n_classes, n_features)
        self.n_features_: Optional[int] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """
        Fit Gaussian Naive Bayes.
        
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
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.n_features_ = n_features
        
        # Initialize storage
        self.class_priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        
        # Compute per-class statistics
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[idx] = len(X_c) / len(X)
            self.means_[idx] = X_c.mean(axis=0)
            
            # Variance with Laplace smoothing
            var_c = X_c.var(axis=0)
            self.variances_[idx] = var_c + self.epsilon
        
        return self
    
    def _gaussian_likelihood(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """
        Compute log-likelihood of x under Gaussian(mean, var).
        log N(x|μ,σ²) = -0.5 * log(2π*σ²) - 0.5 * (x-μ)²/σ²
        
        Args:
            x: (n_features,) sample
            mean: (n_features,) class mean
            var: (n_features,) class variance
        
        Returns:
            scalar log-likelihood
        """
        numerator = (x - mean) ** 2
        denominator = 2 * var
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var) + numerator / denominator)
        return log_prob
    
    def _log_posterior(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log-posterior for each class.
        log P(c|x) ∝ log P(c) + Σ_j log P(x_j|c)
        
        Args:
            x: (n_features,) single sample
        
        Returns:
            (n_classes,) log-posteriors
        """
        posteriors = np.zeros(len(self.classes_))
        for idx in range(len(self.classes_)):
            posteriors[idx] = np.log(self.class_priors_[idx])
            posteriors[idx] += self._gaussian_likelihood(x, self.means_[idx], self.variances_[idx])
        return posteriors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            (n_samples,) predicted class labels
        """
        if self.classes_ is None:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Feature mismatch: expected {self.n_features_}, got {X.shape[1]}")
        
        predictions = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            posteriors = self._log_posterior(x)
            predictions[i] = self.classes_[np.argmax(posteriors)]
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            (n_samples, n_classes) probabilities
        """
        if self.classes_ is None:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Feature mismatch: expected {self.n_features_}, got {X.shape[1]}")
        
        n_samples = len(X)
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            # Log-posteriors
            log_posteriors = self._log_posterior(x)
            
            # Convert to probabilities via log-sum-exp (numerical stability)
            log_posteriors_max = np.max(log_posteriors)
            log_sum_exp = log_posteriors_max + np.log(
                np.sum(np.exp(log_posteriors - log_posteriors_max))
            )
            probabilities[i] = np.exp(log_posteriors - log_sum_exp)
        
        return probabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model."""
        if self.classes_ is None:
            raise ValueError("Model not fitted")
        
        return {
            "classes": self.classes_.tolist(),
            "class_priors": self.class_priors_.tolist(),
            "means": self.means_.tolist(),
            "variances": self.variances_.tolist(),
            "epsilon": self.epsilon,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GaussianNaiveBayes":
        """Deserialize model."""
        model = GaussianNaiveBayes(epsilon=d.get("epsilon", 1e-9))
        model.classes_ = np.array(d["classes"], dtype=np.int32)
        model.class_priors_ = np.array(d["class_priors"], dtype=np.float64)
        model.means_ = np.array(d["means"], dtype=np.float64)
        model.variances_ = np.array(d["variances"], dtype=np.float64)
        model.n_features_ = model.means_.shape[1]
        return model
