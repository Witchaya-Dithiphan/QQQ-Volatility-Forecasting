"""
Single-Layer Perceptron (scratch, NumPy only): one sigmoid unit trained with BCE (+ 0.5*l2*||w||^2, intercept free).

Training contract (differs from LogisticRegression, which is full-batch with gradient-tolerance stopping and class weights):
epoch-based full/mini-batch updates over contiguous, unshuffled batches, with patience early stopping on the monitored loss
(eval_set if given, else Train) and best-epoch weights restored. Initial weights are small normal draws from `random_state`.
`min_delta` is the minimum monitored-loss improvement that counts as a new best. `early_stopping=False` runs exactly
`max_iter` epochs and keeps the final-epoch weights (used for the frozen refit-on-full-Train step); the chronological
selection/refit protocol itself lives in `src/modeling/classification_training.py`.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._common import NOT_FITTED, apply_scaler, check_param, check_X, check_Xy, fit_scaler, scaler_from_dict, scaler_to_dict, sigmoid


def _bce(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


class SLP:
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, batch_size: Optional[int] = 32, l2: float = 0.0,
                 patience: int = 50, standardize: bool = True, random_state: int = 42,
                 min_delta: float = 0.0, early_stopping: bool = True):
        """batch_size: positive int, or None / "full" for full-batch. max_iter counts epochs."""
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.patience = patience
        self.standardize = standardize
        self.random_state = random_state
        self.min_delta = min_delta
        self.early_stopping = early_stopping

        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.scaler_ = None
        self.loss_history_: List[float] = []      # mean Train BCE after each epoch
        self.val_loss_history_: List[float] = []  # mean eval_set BCE after each epoch (empty without eval_set)
        self.best_epoch_ = 0                      # 1-based epoch whose weights are kept
        self.epochs_run_ = 0

    @staticmethod
    def _init(n_features: int, random_state: int):
        return np.random.default_rng(random_state).standard_normal(n_features) * 0.01, 0.0

    @staticmethod
    def _loss_grad(X, y, w, b, l2) -> Tuple[float, np.ndarray, float]:
        p = sigmoid(X @ w + b)
        residual = (p - y) / len(y)
        return _bce(p, y) + 0.5 * l2 * w @ w, X.T @ residual + l2 * w, float(residual.sum())

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> "SLP":
        check_param("learning_rate", self.learning_rate)
        check_param("max_iter", self.max_iter, integer=True)
        check_param("l2", self.l2, allow_zero=True)
        check_param("patience", self.patience, integer=True)
        check_param("min_delta", self.min_delta, allow_zero=True)
        full = self.batch_size is None or self.batch_size == "full"
        if not full:
            check_param("batch_size", self.batch_size, integer=True)
        X, y = check_Xy(X, y)
        self.scaler_, X = fit_scaler(X, self.standardize)
        if eval_set is not None:
            Xv, yv = check_X(eval_set[0], X.shape[1]), np.asarray(eval_set[1], dtype=np.float64).ravel()
            if len(yv) != len(Xv) or not np.isin(yv, (0, 1)).all():
                raise ValueError("eval_set must hold binary labels {0, 1} matching X")
            if self.scaler_ is not None:
                Xv = apply_scaler(self.scaler_, Xv)

        step = len(y) if full else self.batch_size
        w, b = self._init(X.shape[1], self.random_state)
        self.loss_history_, self.val_loss_history_ = [], []
        best, best_wb, self.best_epoch_, self.epochs_run_ = np.inf, (w, b), 0, 0
        for epoch in range(1, self.max_iter + 1):
            for lo in range(0, len(y), step):
                _, grad_w, grad_b = self._loss_grad(X[lo:lo + step], y[lo:lo + step], w, b, self.l2)
                w, b = w - self.learning_rate * grad_w, b - self.learning_rate * grad_b
            self.epochs_run_ = epoch
            self.loss_history_.append(_bce(sigmoid(X @ w + b), y))
            monitored = self.loss_history_[-1]
            if eval_set is not None:
                self.val_loss_history_.append(_bce(sigmoid(Xv @ w + b), yv))
                monitored = self.val_loss_history_[-1]
            if not np.isfinite(monitored):
                raise ValueError("Gradient descent diverged; reduce learning_rate")
            if not self.early_stopping:
                continue
            if monitored < best - self.min_delta:
                best, best_wb, self.best_epoch_ = monitored, (w, b), epoch
            elif epoch - self.best_epoch_ >= self.patience:
                break
        if not self.early_stopping:
            best_wb, self.best_epoch_ = (w, b), self.epochs_run_
        self.coefficients_, self.intercept_ = best_wb[0], float(best_wb[1])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise ValueError(NOT_FITTED)
        X = check_X(X, len(self.coefficients_))
        if self.scaler_ is not None:
            X = apply_scaler(self.scaler_, X)
        return X @ self.coefficients_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def to_dict(self) -> Dict[str, Any]:
        if self.coefficients_ is None:
            raise ValueError(NOT_FITTED)
        return {"learning_rate": self.learning_rate, "max_iter": self.max_iter, "batch_size": self.batch_size, "l2": self.l2,
                "patience": self.patience, "standardize": self.standardize, "random_state": self.random_state,
                "min_delta": self.min_delta, "early_stopping": self.early_stopping,
                "coefficients": self.coefficients_.tolist(), "intercept": self.intercept_, "scaler": scaler_to_dict(self.scaler_),
                "loss_history": list(self.loss_history_), "val_loss_history": list(self.val_loss_history_),
                "best_epoch": self.best_epoch_, "epochs_run": self.epochs_run_}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SLP":
        model = SLP(d["learning_rate"], d["max_iter"], d["batch_size"], d["l2"], d["patience"], d["standardize"], d["random_state"],
                    d.get("min_delta", 0.0), d.get("early_stopping", True))
        model.coefficients_, model.intercept_ = np.array(d["coefficients"], dtype=np.float64), float(d["intercept"])
        model.scaler_ = scaler_from_dict(d["scaler"])
        model.loss_history_, model.val_loss_history_ = list(d["loss_history"]), list(d["val_loss_history"])
        model.best_epoch_, model.epochs_run_ = d["best_epoch"], d["epochs_run"]
        return model
