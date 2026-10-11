"""Stacking: a logistic meta-learner over out-of-sample probabilities of three base models.

    bases = logistic, decision tree, k-NN          meta = logistic

The meta-learner must never see a probability that a base model produced for a row it
was trained on, so the meta features come from expanding chronological folds:

    1. the first 40% of the original timeline only ever serves as training history
    2. the remaining 60% is cut into 4 consecutive blocks
    3. for each block, the bases are fitted on everything more than `purge` trading
       days before the block starts, and predict the block
    4. the meta-learner is fitted on those out-of-sample predictions alone
    5. the bases are refitted on the full training set for use at prediction time

Positions are counted on the original trading-day timeline, not in rows: the
non_spike variant has rows removed, and its gaps must not be mistaken for elapsed days.
"""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from ..core.preprocess import Standardizer
from .decision_tree import DecisionTree
from .knn import KNN
from .logistic import Logistic

# configs/modeling.json -> stacking.meta_features. Each base brings its own
# preprocessing: the tree splits on raw values, the other two need standardised ones.
# The logistic base is run to convergence. Its configured budget (rate 0.05, 5,000
# steps) stops well short of the optimum on these folds, which would make the meta
# features depend on where gradient descent happened to be standing.
LOGISTIC_PARAMS = {"learning_rate": 0.5, "max_iter": 60000, "tol": 1e-9}
BASES = (
    ("logistic_probability", Logistic, LOGISTIC_PARAMS, True),
    ("tree_probability", DecisionTree, {}, False),
    ("knn_probability", KNN, {}, True),
)
META_FEATURES = tuple(base[0] for base in BASES)
# The meta features are three correlated probabilities in [0, 1], a badly scaled
# problem for plain gradient descent, so the meta-learner gets a larger step and budget.
META_PARAMS = {"learning_rate": 1.0, "max_iter": 200000, "tol": 1e-8}


def fold_plan(positions, n_positions: int, folds: int, initial_fraction: float, purge: int):
    """Return [(train_mask, block_mask, block_start, block_stop)] for each expanding fold."""
    positions = np.asarray(positions)
    first = int(n_positions * initial_fraction)
    plan = []
    for block in np.array_split(np.arange(first, n_positions), folds):
        start, stop = int(block[0]), int(block[-1]) + 1
        plan.append((positions < start - purge, (positions >= start) & (positions < stop), start, stop))
    return plan


class Stacking(BaseModel):
    name = "stacking"
    task = "classification"

    def __init__(self, folds: int = 4, initial_fraction: float = 0.4, purge: int = 5,
                 random_state: int = 42):
        self.folds = folds
        self.initial_fraction = initial_fraction
        self.purge = purge
        self.random_state = random_state

    def get_params(self) -> dict:
        return {
            "folds": self.folds,
            "initial_fraction": self.initial_fraction,
            "purge": self.purge,
            "random_state": self.random_state,
        }

    @staticmethod
    def _fit_bases(X, y):
        """Fit the three bases on one set of rows, each behind its own preprocessing."""
        fitted = []
        for _, model_class, params, standardize in BASES:
            scaler = Standardizer().fit(X) if standardize else None
            model = model_class(**params).fit(scaler.transform(X) if scaler else X, y)
            fitted.append((scaler, model))
        return fitted

    @staticmethod
    def _base_probabilities(fitted, X) -> np.ndarray:
        return np.column_stack([
            model.predict_proba(scaler.transform(X) if scaler else X) for scaler, model in fitted
        ])

    def fit(self, X, y=None, positions=None, n_positions=None) -> "Stacking":
        if (not isinstance(self.folds, int) or self.folds < 1
                or not 0 < self.initial_fraction < 1
                or not isinstance(self.purge, int) or self.purge < 0):
            raise ValueError("Invalid stacking hyperparameters")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or y.shape != (len(X),) or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array aligned with y")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Stacking requires binary labels 0 and 1")
        y = y.astype(np.int64)
        # Without positions every row is taken to be one trading day after the last.
        positions = np.arange(len(X)) if positions is None else np.asarray(positions, dtype=np.int64)
        n_positions = int(positions.max()) + 1 if n_positions is None else int(n_positions)
        if (positions.shape != (len(X),) or (np.diff(positions) <= 0).any()
                or positions[0] < 0 or positions[-1] >= n_positions):
            raise ValueError("positions must be strictly increasing trading-day indices within the timeline")

        self.oos_features_ = np.full((len(X), len(BASES)), np.nan)
        self.folds_ = []
        for train, block, start, stop in fold_plan(positions, n_positions, self.folds,
                                                   self.initial_fraction, self.purge):
            if not block.any() or len(np.unique(y[train])) < 2:
                raise ValueError(f"Fold for positions [{start}, {stop}) has no rows to predict "
                                 "or a single-class history")
            self.oos_features_[block] = self._base_probabilities(self._fit_bases(X[train], y[train]), X[block])
            self.folds_.append({"start": start, "stop": stop, "train_rows": int(train.sum()),
                                "predicted_rows": int(block.sum()),
                                "last_train_position": int(positions[train][-1])})

        self.meta_rows_ = ~np.isnan(self.oos_features_).any(axis=1)
        self.meta_ = Logistic(**META_PARAMS).fit(self.oos_features_[self.meta_rows_], y[self.meta_rows_])
        self.bases_ = self._fit_bases(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def base_probabilities(self, X) -> np.ndarray:
        """The three meta features for new rows, from the bases fitted on all of train."""
        return self._base_probabilities(self.bases_, np.asarray(X, dtype=np.float64))

    def predict_proba(self, X) -> np.ndarray:
        return self.meta_.predict_proba(self.base_probabilities(X))

    def decision_function(self, X) -> np.ndarray:
        return self.meta_.decision_function(self.base_probabilities(X))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(np.int64)

    def state_dict(self) -> dict[str, np.ndarray]:
        arrays = {f"meta_{key}": value for key, value in self.meta_.state_dict().items()}
        arrays["n_features_in"] = np.array([self.n_features_in_])
        arrays["fold_bounds"] = np.array([[fold["start"], fold["stop"], fold["train_rows"],
                                           fold["predicted_rows"], fold["last_train_position"]]
                                          for fold in self.folds_])
        for (name, _, _, _), (scaler, model) in zip(BASES, self.bases_):
            arrays.update({f"{name}_{key}": value for key, value in model.state_dict().items()})
            if scaler is not None:
                arrays[f"{name}_scaler_mean"] = scaler.mean_
                arrays[f"{name}_scaler_scale"] = scaler.scale_
                arrays[f"{name}_scaler_constant"] = scaler.constant_
        return arrays

    def load_state(self, arrays: dict, metadata: dict) -> "Stacking":
        for key, value in metadata["params"].items():
            setattr(self, key, value)

        def part(prefix):
            return {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}

        self.meta_ = Logistic(**META_PARAMS).load_state(part("meta_"), {"params": {}})
        self.bases_ = []
        for name, model_class, params, standardize in BASES:
            state = part(f"{name}_")
            scaler = None
            if standardize:
                scaler = Standardizer()
                scaler.mean_ = np.asarray(state.pop("scaler_mean"))
                scaler.scale_ = np.asarray(state.pop("scaler_scale"))
                scaler.constant_ = np.asarray(state.pop("scaler_constant")).astype(bool)
            base = model_class(**params)
            self.bases_.append((scaler, base.load_state(state, {"params": base.get_params()})))
        self.n_features_in_ = int(arrays["n_features_in"][0])
        self.folds_ = [dict(zip(("start", "stop", "train_rows", "predicted_rows", "last_train_position"),
                                (int(v) for v in row))) for row in arrays["fold_bounds"]]
        return self

    def reference(self):
        """sklearn's stack of the same three learners, as a behavioural comparison only.

        StackingClassifier cannot express this protocol: it rejects TimeSeriesSplit
        because every row must be predicted by some fold, so it has no past-only
        folds and no purge. With contiguous KFold its early blocks are predicted by
        models trained on the future, and its numbers are expected to differ. The
        exact check is in the tests: the same folds rebuilt from each base's own
        sklearn reference.
        """
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        return StackingClassifier(
            estimators=[
                ("logistic", make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=10000))),
                ("tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=self.random_state)),
                ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, algorithm="brute"))),
            ],
            final_estimator=LogisticRegression(C=np.inf, max_iter=10000),
            cv=KFold(self.folds, shuffle=False),
            stack_method="predict_proba",
            passthrough=False,
        )
