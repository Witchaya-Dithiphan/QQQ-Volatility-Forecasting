"""Single-layer perceptron — one sigmoid unit trained by mini-batch gradient descent.

Different from `perceptron` in the way that matters: this one has a differentiable
loss, so it converges on data that is not linearly separable instead of stopping
wherever the last correction left it.
"""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class SLP(BaseModel):
    """Minimises mean binary cross-entropy plus L2 on the weights.

        p  = sigmoid(Xw + b)
        J  = mean BCE(y, p) + (l2/2)*||w||^2
        dw = X.T @ (p - y) / batch + l2*w      # sigmoid and BCE cancel to this

    Training follows the config's neural protocol: chronological contiguous
    batches with no shuffling, early stopping watched on the last 15% of train
    (with the 5 trading days before it purged from the fitting rows), then a refit on the full training set for the epoch count that scored best.
    """

    name = "slp"
    task = "classification"

    INTERNAL_TAIL_FRACTION = 0.15
    # configs/modeling.json -> neural.purge_original_trading_days
    PURGE_TRADING_DAYS = 5

    def __init__(
        self,
        batch: int | str = 32,
        l2: float = 0.0,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        patience: int = 50,
        min_delta: float = 1e-6,
        random_state: int = 42,
    ):
        self.batch = batch
        self.l2 = l2
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "batch": self.batch,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "random_state": self.random_state,
        }

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=np.float64)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def _initial_weights(self, n_features: int) -> tuple[np.ndarray, float]:
        """Glorot-uniform from legacy MT19937, the stream sklearn draws from."""
        rng = np.random.RandomState(self.random_state)
        bound = np.sqrt(6.0 / (n_features + 1))
        return rng.uniform(-bound, bound, size=n_features), 0.0

    def _batches(self, n_samples: int) -> list[np.ndarray]:
        if self.batch == "full":
            return [np.arange(n_samples)]
        size = int(self.batch)
        # The trailing partial batch is kept: config sets last_short_batch true.
        return [np.arange(start, min(start + size, n_samples)) for start in range(0, n_samples, size)]

    def _subtrain_size(self, n_samples: int) -> int:
        return int(n_samples * (1 - self.INTERNAL_TAIL_FRACTION))

    def _objective(self, X, y, weights, intercept) -> float:
        margin = X @ weights + intercept
        losses = np.logaddexp(0.0, margin) - y * margin
        return float(losses.mean() + 0.5 * self.l2 * weights @ weights)

    def _gradient(self, X, y, weights, intercept):
        residual = self._sigmoid(X @ weights + intercept) - y
        return X.T @ residual / len(X) + self.l2 * weights, float(residual.mean())

    def _descend(self, X, y, epochs: int) -> tuple[np.ndarray, float]:
        weights, intercept = self._initial_weights(X.shape[1])
        for _ in range(epochs):
            for index in self._batches(len(X)):
                gradient_w, gradient_b = self._gradient(X[index], y[index], weights, intercept)
                weights -= self.learning_rate * gradient_w
                intercept -= self.learning_rate * gradient_b
        return weights, intercept

    def _subtrain_rows(self, positions, n_samples: int) -> np.ndarray:
        """Rows that train the early-stopping run: everything before the tail, minus the purge.

        The target looks 5 trading days ahead, so a row within 5 days of the tail has a
        label computed from the tail's own period. Days are counted on the original
        timeline because non_spike has rows removed.
        """
        positions = np.arange(n_samples) if positions is None else np.asarray(positions)
        if positions.shape != (n_samples,) or (np.diff(positions) <= 0).any():
            raise ValueError("positions must be strictly increasing, one per row")
        cut = int(n_samples * (1 - self.INTERNAL_TAIL_FRACTION))
        rows = np.arange(cut)
        if cut < n_samples:
            rows = rows[positions[:cut] < positions[cut] - self.PURGE_TRADING_DAYS]
        if len(rows) == 0:
            raise ValueError("No training rows are left once the purge window before the tail is removed")
        return rows

    def fit(self, X, y=None, positions=None, n_positions=None) -> "SLP":
        """`positions` are trading-day indices on the original timeline; rows are assumed
        to be consecutive days when it is omitted. `n_positions` is accepted because the
        trainer passes it, and is not needed here."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        cut = self._subtrain_size(len(X))
        rows = self._subtrain_rows(positions, len(X))
        sub_X, sub_y = X[rows], y[rows]
        tail_X, tail_y = X[cut:], y[cut:]

        weights, intercept = self._initial_weights(X.shape[1])
        best_score, best_epoch, since_best = np.inf, 0, 0
        self.history_ = {"iteration": [], "loss": []}

        epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            for index in self._batches(len(sub_X)):
                gradient_w, gradient_b = self._gradient(
                    sub_X[index], sub_y[index], weights, intercept
                )
                weights -= self.learning_rate * gradient_w
                intercept -= self.learning_rate * gradient_b

            watched = self._objective(tail_X, tail_y, weights, intercept) if len(tail_X) else np.inf
            self.history_["iteration"].append(epoch)
            self.history_["loss"].append(self._objective(sub_X, sub_y, weights, intercept))

            if watched < best_score - self.min_delta:
                # Patience counts from the best epoch, and ties keep the earliest.
                best_score, best_epoch, since_best = watched, epoch, 0
            else:
                since_best += 1
                if since_best >= self.patience:
                    break

        self.n_epochs_ = epoch
        self.best_epoch_ = best_epoch or epoch
        self.refit_epochs_ = self.best_epoch_
        self.n_train_rows_ = len(X)
        self.n_subtrain_rows_ = len(rows)
        # Refit on everything for the epoch count the tail liked best.
        self.coef_, self.intercept_ = self._descend(X, y, self.refit_epochs_)
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        return self._sigmoid(self.decision_function(X))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "coef": self.coef_,
            "intercept": np.array([self.intercept_]),
            "schedule": np.array([self.n_epochs_, self.best_epoch_, self.n_train_rows_]),
        }

    def load_state(self, arrays: dict, metadata: dict) -> "SLP":
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        self.n_epochs_, self.best_epoch_, self.n_train_rows_ = (int(v) for v in arrays["schedule"])
        self.refit_epochs_ = self.best_epoch_
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        """No sklearn estimator fits a zero-hidden-layer MLP, so this is a behavioural check.

        MLPClassifier refuses hidden_layer_sizes=(), and SGDClassifier updates per
        sample rather than per batch. The comparison that means something here is
        against logistic regression on the same convex objective: with l2=0 and
        enough epochs both minimise mean cross-entropy, so they should agree.
        """
        from sklearn.linear_model import LogisticRegression

        if self.l2 == 0:
            return LogisticRegression(C=np.inf, fit_intercept=True, tol=1e-12, max_iter=200000)
        return LogisticRegression(
            C=1.0 / (self.l2 * max(self.n_train_rows_, 1)),
            l1_ratio=0,
            fit_intercept=True,
            tol=1e-12,
            max_iter=200000,
        )
