"""Discrete AdaBoost (SAMME) over weighted Gini stumps."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from .decision_tree import DecisionTree

# configs/modeling.json -> adaboost.epsilon
EPSILON = 1e-12


class AdaBoost(BaseModel):
    name = "adaboost"
    task = "classification"

    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }

    def fit(self, X, y=None) -> "AdaBoost":
        if (not isinstance(self.n_estimators, int) or self.n_estimators < 1
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not isinstance(self.random_state, int) or not 0 <= self.random_state < 2**32):
            raise ValueError("Invalid AdaBoost hyperparameters")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or y.shape != (len(X),):
            raise ValueError("X and y must be aligned nonempty arrays")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("AdaBoost requires binary labels 0 and 1")
        y = y.astype(np.int64)
        signed = 2 * y - 1

        self.n_features_in_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        weights = np.full(len(y), 1.0 / len(y))
        margin = np.zeros(len(y))
        self.stumps_, alphas, errors = [], [], []
        self.history_ = {"iteration": [], "loss": []}
        self.status_ = "completed"

        for iteration in range(1, self.n_estimators + 1):
            # sklearn floors the weights each round so log(w) below stays finite.
            weights = np.clip(weights, np.finfo(np.float64).eps, None)
            # One seed per round, drawn in order, is what sklearn hands each stump.
            seed = int(rng.randint(np.iinfo(np.int32).max))
            stump = DecisionTree(max_depth=1, min_samples_leaf=1, max_features=None,
                                 random_state=seed).fit(X, y, sample_weight=weights)
            prediction = stump.predict(X)
            incorrect = prediction != y
            error = float((incorrect * weights).sum() / weights.sum())

            if error <= EPSILON:
                # log((1 - err) / err) would be infinite; sklearn records weight 1 and stops.
                self.stumps_.append(stump)
                alphas.append(1.0)
                errors.append(error)
                margin += 2 * prediction - 1
                self._record(iteration, signed, margin)
                self.status_ = "completed_perfect"
                break
            if error >= 0.5:
                if not self.stumps_:
                    raise ValueError("First stump is worse than random; AdaBoost cannot be fit")
                self.status_ = "completed_early"
                break

            # The learning rate enters here once and never again at prediction time.
            alpha = self.learning_rate * np.log((1.0 - error) / error)
            self.stumps_.append(stump)
            alphas.append(alpha)
            errors.append(error)
            margin += alpha * (2 * prediction - 1)
            self._record(iteration, signed, margin)

            weights = np.exp(np.log(weights) + alpha * incorrect)
            weights /= weights.sum()

        self.estimator_weights_ = np.asarray(alphas, dtype=np.float64)
        self.estimator_errors_ = np.asarray(errors, dtype=np.float64)
        self.sample_weight_ = weights / weights.sum()
        return self

    def _record(self, iteration, signed, margin) -> None:
        # SAMME's alpha is twice the textbook one, so the loss it descends is exp(-yF/2);
        # at learning_rate=1 this equals the product of the round normalisers Z_m.
        self.history_["iteration"].append(iteration)
        self.history_["loss"].append(float(np.mean(np.exp(-0.5 * signed * margin))))

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        votes = sum(alpha * (2 * stump.predict(X) - 1)
                    for stump, alpha in zip(self.stumps_, self.estimator_weights_))
        # Same scale as sklearn: 2 * sum(alpha * h) / sum(alpha), so scores lie in [-2, 2].
        return 2.0 * votes / self.estimator_weights_.sum()

    def predict_proba(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        arrays = {
            "n_features_in": np.array([self.n_features_in_]),
            "estimator_weights": self.estimator_weights_,
            "estimator_errors": self.estimator_errors_,
        }
        for i, stump in enumerate(self.stumps_):
            arrays.update({f"stump_{i}_{key}": value for key, value in stump.state_dict().items()})
        return arrays

    def load_state(self, arrays: dict, metadata: dict) -> "AdaBoost":
        for key, value in metadata["params"].items():
            setattr(self, key, value)
        self.n_features_in_ = int(arrays["n_features_in"][0])
        self.estimator_weights_ = np.asarray(arrays["estimator_weights"], dtype=np.float64)
        self.estimator_errors_ = np.asarray(arrays["estimator_errors"], dtype=np.float64)
        self.stumps_ = []
        for i in range(len(self.estimator_weights_)):
            prefix = f"stump_{i}_"
            state = {key[len(prefix):]: value for key, value in arrays.items()
                     if key.startswith(prefix)}
            self.stumps_.append(DecisionTree().load_state(state, {"params": {}}))
        return self

    def reference(self):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        # sklearn 1.9 dropped the `algorithm` argument; SAMME is the only behaviour left.
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, criterion="gini"),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
