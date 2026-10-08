"""MLP binary classifier (scratch, NumPy only): ReLU hidden layers, sigmoid output, BCE + 0.5*l2*||W||^2, plain SGD.

Minibatches are contiguous and chronological (never shuffled). `fit` implements the frozen neural protocol:
choose the epoch count on an internal chronological tail (purged by original trading days, scaler fit on the
subtrain), then reinitialize with the seed and refit on the full Train for exactly that many epochs.
"""
import math

import numpy as np

from . import _state
from ..preprocessing import Standardizer


def init_parameters(sizes, seed):
    """He-normal weights (fan_in, fan_out) and zero biases, drawn layer by layer from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal((i, o)) * math.sqrt(2.0 / i) for i, o in zip(sizes[:-1], sizes[1:])]
    return weights, [np.zeros(o) for o in sizes[1:]]


def forward(weights, biases, X):
    """Return (layer activations incl. input and hidden outputs, output logits of shape (n,))."""
    activations = [X]
    for W, b in zip(weights[:-1], biases[:-1]):
        activations.append(np.maximum(0.0, activations[-1] @ W + b))
    return activations, (activations[-1] @ weights[-1] + biases[-1]).ravel()


def _bce(logits, y):
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits))


def _sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def loss_and_gradients(weights, biases, X, y, l2):
    activations, logits = forward(weights, biases, X)
    value = _bce(logits, y) + 0.5 * l2 * sum(float(np.sum(W * W)) for W in weights)
    delta = ((_sigmoid(logits) - y) / len(y))[:, None]
    gW, gb = [None] * len(weights), [None] * len(weights)
    for layer in range(len(weights) - 1, -1, -1):
        gW[layer] = activations[layer].T @ delta + l2 * weights[layer]
        gb[layer] = delta.sum(axis=0)
        if layer:
            delta = (delta @ weights[layer].T) * (activations[layer] > 0)
    return value, gW, gb


def batch_slices(n, size):
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


def train_epoch(weights, biases, X, y, lr, l2, batch_size):
    """One in-order pass of minibatch SGD, updating parameters in place."""
    for lo, hi in batch_slices(len(X), batch_size):
        _, gW, gb = loss_and_gradients(weights, biases, X[lo:hi], y[lo:hi], l2)
        for p, g in zip(weights + biases, gW + gb):
            p -= lr * g


class EpochMonitor:
    """Patience monitor: an epoch improves only if loss < best - min_delta; the earliest best epoch is kept."""

    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.epoch = self.best_epoch = self.bad = 0
        self.best_loss = math.inf

    def update(self, value) -> bool:
        self.epoch += 1
        if value < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.bad = value, self.epoch, 0
        else:
            self.bad += 1
        return self.bad >= self.patience


def _int(value):
    return type(value) is int


class MLPClassifier:
    def __init__(self, hidden_layers=(16,), learning_rate=0.01, l2=0.0, batch_size=32, max_epochs=1000, patience=50,
                 min_delta=1e-6, seed=42, tail_fraction=0.15, purge=5):
        layers = list(hidden_layers)
        if not layers or not all(_int(h) and h > 0 for h in layers):
            raise ValueError("hidden_layers must be a nonempty sequence of positive integers")
        if not (np.isfinite(learning_rate) and learning_rate > 0 and np.isfinite(l2) and l2 >= 0 and np.isfinite(min_delta) and min_delta >= 0):
            raise ValueError("learning_rate must be positive; l2 and min_delta nonnegative; all finite")
        if not all(_int(v) and v >= 1 for v in (batch_size, max_epochs, patience)) or not _int(seed) or not _int(purge) or purge < 0:
            raise ValueError("batch_size, max_epochs, patience must be positive ints; seed int; purge nonnegative int")
        if not 0 < tail_fraction < 1:
            raise ValueError("tail_fraction must be in (0, 1)")
        self.hidden_layers, self.learning_rate, self.l2 = layers, float(learning_rate), float(l2)
        self.batch_size, self.max_epochs, self.patience, self.min_delta = batch_size, max_epochs, patience, float(min_delta)
        self.seed, self.tail_fraction, self.purge = seed, float(tail_fraction), purge
        self.weights_ = None

    @staticmethod
    def _check(X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D matrix")
        if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
            raise ValueError("y must be 1D {0, 1} labels aligned with X")
        return X, y.astype(np.float64)

    def _train(self, X, y, epochs, tail=None):
        """Fresh seeded network on X (scaler fit on X); returns (scaler, weights, biases, monitor, tail_bce)."""
        scaler = Standardizer().fit(X)
        Z = scaler.transform(X)
        W, b = init_parameters([Z.shape[1], *self.hidden_layers, 1], self.seed)
        monitor, curve = EpochMonitor(self.patience, self.min_delta), []
        for _ in range(epochs):
            train_epoch(W, b, Z, y, self.learning_rate, self.l2, self.batch_size)
            if tail is not None:
                curve.append(_bce(forward(W, b, scaler.transform(tail[0]))[1], tail[1]))
                if monitor.update(curve[-1]):
                    break
        return scaler, W, b, monitor, curve

    def fit_epochs(self, X, y, epochs) -> "MLPClassifier":
        """Train on all of X for exactly `epochs` epochs, no early stopping."""
        X, y = self._check(X, y)
        if not _int(epochs) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        self.scaler_, self.weights_, self.biases_, _, _ = self._train(X, y, epochs)
        self.epochs_trained_, self.selection_ = epochs, None
        return self

    def fit(self, X, y, original_positions=None) -> "MLPClassifier":
        """Frozen protocol: chronological tail selects the epoch count; no external Validation is accepted."""
        X, y = self._check(X, y)
        n = len(X)
        positions = np.arange(n) if original_positions is None else np.asarray(original_positions)
        if positions.shape != (n,) or positions.dtype.kind not in "iu" or (np.diff(positions) <= 0).any():
            raise ValueError("original_positions must be strictly increasing integers, one per row")
        n_tail = math.ceil(self.tail_fraction * n)
        tail_start = n - n_tail
        boundary = int(positions[tail_start]) - self.purge
        n_sub = int(np.searchsorted(positions, boundary, side="left"))  # rows strictly before the purge gap
        if n_sub < 2:
            raise ValueError("Too few rows for subtrain + purge + internal tail")
        scaler, _, _, monitor, curve = self._train(X[:n_sub], y[:n_sub], self.max_epochs, tail=(X[tail_start:], y[tail_start:]))
        selection = {"n_subtrain": n_sub, "n_tail": n_tail, "tail_start": tail_start, "purge_boundary_position": boundary,
                     "best_epoch": monitor.best_epoch, "stopped_epoch": monitor.epoch,
                     "status": "early_stopped" if monitor.bad >= self.patience else "max_epochs",
                     "tail_bce": curve, "subtrain_scaler_mean": scaler.mean_.tolist()}
        self.scaler_, self.weights_, self.biases_, _, _ = self._train(X, y, monitor.best_epoch)  # reinit seed, full Train
        self.epochs_trained_, self.selection_ = monitor.best_epoch, selection
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Model not fitted")
        p = _sigmoid(forward(self.weights_, self.biases_, self.scaler_.transform(X))[1])
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def to_dict(self) -> dict:
        if self.weights_ is None:
            raise ValueError("Model not fitted")
        return {"kind": "mlp_classifier", "schema_version": 1, "hidden_layers": self.hidden_layers, "learning_rate": self.learning_rate,
                "l2": self.l2, "batch_size": self.batch_size, "max_epochs": self.max_epochs, "patience": self.patience,
                "min_delta": self.min_delta, "seed": self.seed, "tail_fraction": self.tail_fraction, "purge": self.purge,
                "weights": [W.tolist() for W in self.weights_], "biases": [b.tolist() for b in self.biases_],
                "epochs_trained": self.epochs_trained_, "selection": self.selection_, "scaler": _state.scaler_state(self.scaler_)}

    def to_json(self) -> str:
        return _state.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, state: dict) -> "MLPClassifier":
        if state.get("kind") != "mlp_classifier" or state.get("schema_version") != 1:
            raise ValueError("Not an mlp_classifier state")
        keys = ("hidden_layers", "learning_rate", "l2", "batch_size", "max_epochs", "patience", "min_delta", "seed", "tail_fraction", "purge")
        model = cls(**{k: state[k] for k in keys})
        model.weights_ = [np.array(W, dtype=np.float64) for W in state["weights"]]
        model.biases_ = [np.array(b, dtype=np.float64) for b in state["biases"]]
        model.epochs_trained_, model.selection_ = state["epochs_trained"], state["selection"]
        model.scaler_ = _state.scaler_from_state(state["scaler"])
        return model

    @classmethod
    def from_json(cls, text: str) -> "MLPClassifier":
        return cls.from_dict(_state.loads(text))
