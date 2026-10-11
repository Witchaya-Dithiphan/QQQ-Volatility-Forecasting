"""Multi-layer perceptron: ReLU hidden layers, one sigmoid output, trained by backpropagation.

    forward :  a0 = X ;  z_l = a_(l-1) W_l + b_l ;  a_l = relu(z_l)   (hidden layers)
               p = sigmoid(z_last)
    backward:  d_last = p - y                       # sigmoid and cross-entropy cancel to this
               d_l    = (d_(l+1) W_(l+1)^T) * 1[z_l > 0]
               dW_l   = (a_(l-1)^T d_l + l2 * W_l) / batch ;   db_l = mean(d_l)

Training follows the config's neural protocol, the same one `slp` uses: chronological
contiguous batches with no shuffling, early stopping watched on the last 15% of train,
then a refit on the full training set for the epoch count that scored best.
"""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class MLP(BaseModel):
    name = "mlp"
    task = "classification"

    INTERNAL_TAIL_FRACTION = 0.15

    def __init__(self, hidden_layers=(16,), l2: float = 0.0, learning_rate: float = 0.01,
                 batch: int = 32, max_epochs: int = 1000, patience: int = 50,
                 min_delta: float = 1e-6, momentum: float = 0.9, activation: str = "relu",
                 random_state: int = 42):
        self.hidden_layers = list(hidden_layers)
        self.l2 = l2
        self.learning_rate = learning_rate
        self.batch = batch
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.momentum = momentum
        self.activation = activation
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "hidden_layers": list(self.hidden_layers),
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "batch": self.batch,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "momentum": self.momentum,
            "activation": self.activation,
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

    def _initial_parameters(self, n_features: int):
        """Glorot-uniform, weights then biases layer by layer, from legacy MT19937."""
        rng = np.random.RandomState(self.random_state)
        sizes = [n_features, *self.hidden_layers, 1]
        coefs, intercepts = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            coefs.append(rng.uniform(-bound, bound, (fan_in, fan_out)))
            intercepts.append(rng.uniform(-bound, bound, fan_out))
        return coefs, intercepts

    def _forward(self, X, coefs, intercepts):
        """Return every layer's activation; the last entry is the output probability."""
        activations = [X]
        for layer, (weights, bias) in enumerate(zip(coefs, intercepts)):
            z = activations[-1] @ weights + bias
            if layer == len(coefs) - 1:
                activations.append(self._sigmoid(z))
            else:
                activations.append(np.maximum(z, 0.0) if self.activation == "relu" else z)
        return activations

    def _loss_and_gradients(self, X, y, coefs, intercepts):
        """Mean cross-entropy plus L2 on one batch, and its gradient for every parameter."""
        n = len(X)
        activations = self._forward(X, coefs, intercepts)
        probability = np.clip(activations[-1][:, 0], 1e-15, 1 - 1e-15)
        penalty = sum(float(np.sum(weights * weights)) for weights in coefs)
        loss = float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))
                     + 0.5 * self.l2 * penalty / n)

        coef_grads, intercept_grads = [None] * len(coefs), [None] * len(coefs)
        delta = activations[-1] - y[:, None]
        for layer in range(len(coefs) - 1, -1, -1):
            # The penalty is divided by the batch like the data term, and never touches a bias.
            coef_grads[layer] = (activations[layer].T @ delta + self.l2 * coefs[layer]) / n
            intercept_grads[layer] = delta.sum(axis=0) / n
            if layer > 0:
                delta = delta @ coefs[layer].T
                if self.activation == "relu":
                    # relu'(0) is taken as 0: a unit sitting exactly at zero passes nothing back.
                    delta = delta * (activations[layer] > 0)
        return loss, coef_grads, intercept_grads

    def _run_epoch(self, X, y, parameters, velocities) -> None:
        """One pass of mini-batch SGD with Nesterov momentum, updating in place."""
        n_layers = len(parameters) // 2
        for start in range(0, len(X), self.batch):
            # The trailing partial batch is kept: config sets last_short_batch true.
            stop = min(start + self.batch, len(X))
            _, coef_grads, intercept_grads = self._loss_and_gradients(
                X[start:stop], y[start:stop], parameters[:n_layers], parameters[n_layers:])
            for index, gradient in enumerate(coef_grads + intercept_grads):
                velocities[index] = self.momentum * velocities[index] - self.learning_rate * gradient
                # Nesterov: step along the velocity as it will be after this update.
                parameters[index] += self.momentum * velocities[index] - self.learning_rate * gradient

    def _start(self, n_features: int):
        coefs, intercepts = self._initial_parameters(n_features)
        parameters = coefs + intercepts
        return parameters, [np.zeros_like(parameter) for parameter in parameters]

    def _descend(self, X, y, epochs: int):
        parameters, velocities = self._start(X.shape[1])
        for _ in range(epochs):
            self._run_epoch(X, y, parameters, velocities)
        n_layers = len(parameters) // 2
        return parameters[:n_layers], parameters[n_layers:]

    def _cross_entropy(self, X, y, parameters) -> float:
        n_layers = len(parameters) // 2
        z = X
        for layer in range(n_layers - 1):
            z = z @ parameters[layer] + parameters[n_layers + layer]
            z = np.maximum(z, 0.0) if self.activation == "relu" else z
        margin = (z @ parameters[n_layers - 1] + parameters[-1])[:, 0]
        return float(np.mean(np.logaddexp(0.0, margin) - y * margin))

    def fit(self, X, y=None) -> "MLP":
        if (not self.hidden_layers
                or any(not isinstance(size, (int, np.integer)) or size < 1 for size in self.hidden_layers)
                or not np.isfinite(self.l2) or self.l2 < 0
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not isinstance(self.batch, int) or self.batch < 1
                or not isinstance(self.max_epochs, int) or self.max_epochs < 1
                or not isinstance(self.patience, int) or self.patience < 1
                or not 0 <= self.momentum < 1
                or self.activation not in ("relu", "identity")):
            raise ValueError("Invalid MLP hyperparameters")
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or labels.shape != (len(X),) or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array aligned with y")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("MLP requires binary labels 0 and 1")
        y = labels.astype(np.float64)

        cut = int(len(X) * (1 - self.INTERNAL_TAIL_FRACTION))
        sub_X, sub_y, tail_X, tail_y = X[:cut], y[:cut], X[cut:], y[cut:]

        parameters, velocities = self._start(X.shape[1])
        best_score, best_epoch, since_best = np.inf, 0, 0
        self.history_ = {"iteration": [], "loss": []}

        epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            self._run_epoch(sub_X, sub_y, parameters, velocities)
            watched = self._cross_entropy(tail_X, tail_y, parameters) if len(tail_X) else np.inf
            self.history_["iteration"].append(epoch)
            self.history_["loss"].append(self._cross_entropy(sub_X, sub_y, parameters))
            if watched < best_score - self.min_delta:
                # Patience counts from the best epoch, and ties keep the earliest.
                best_score, best_epoch, since_best = watched, epoch, 0
            else:
                since_best += 1
                if since_best >= self.patience:
                    break

        self.n_epochs_ = epoch
        self.best_epoch_ = best_epoch or epoch
        self.n_train_rows_ = len(X)
        # Refit on everything for the epoch count the tail liked best.
        self.coefs_, self.intercepts_ = self._descend(X, y, self.best_epoch_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._forward(X, self.coefs_, self.intercepts_)[-1][:, 0]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        arrays = {"schedule": np.array([self.n_epochs_, self.best_epoch_, self.n_train_rows_, len(self.coefs_)])}
        for layer, (weights, bias) in enumerate(zip(self.coefs_, self.intercepts_)):
            arrays[f"coef_{layer}"] = weights
            arrays[f"intercept_{layer}"] = bias
        return arrays

    def load_state(self, arrays: dict, metadata: dict) -> "MLP":
        self.n_epochs_, self.best_epoch_, self.n_train_rows_, n_layers = (int(v) for v in arrays["schedule"])
        self.coefs_ = [np.asarray(arrays[f"coef_{layer}"], dtype=np.float64) for layer in range(n_layers)]
        self.intercepts_ = [np.asarray(arrays[f"intercept_{layer}"], dtype=np.float64) for layer in range(n_layers)]
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        return self

    def reference(self):
        """sklearn's SGD MLP run for the epoch count our early stopping chose.

        Its own stopping rule is switched off (tol=0 and a patience longer than the
        run), so the two models take the same number of identical steps from the same
        initial weights.
        """
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layers),
            activation=self.activation,
            solver="sgd",
            alpha=self.l2,
            batch_size=self.batch,
            learning_rate_init=self.learning_rate,
            momentum=self.momentum,
            nesterovs_momentum=True,
            shuffle=False,
            early_stopping=False,
            tol=0,
            max_iter=self.best_epoch_,
            n_iter_no_change=self.best_epoch_ + 1,
            random_state=self.random_state,
        )
