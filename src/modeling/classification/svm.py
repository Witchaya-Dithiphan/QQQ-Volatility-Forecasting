"""Linear soft-margin SVM (scratch, NumPy only): min 0.5||w||^2 + C * sum(hinge), intercept unpenalized.

Solved by deterministic full-batch subgradient descent with step eta0/sqrt(t) on the objective divided by C*n,
returning the best-objective iterate. Outputs are decision scores (>= 0 -> class 1), never probabilities.
"""
import numpy as np

from . import _state
from ..preprocessing import PCA, Standardizer


def objective_and_subgradient(w, b, X, y, C):
    """Objective and a subgradient w.r.t. (w, b); y in {0, 1}. Margin exactly 1 contributes nothing."""
    signed = 2.0 * np.asarray(y) - 1.0
    margin = signed * (X @ w + b)
    violated = margin < 1.0
    objective = 0.5 * w @ w + C * np.maximum(0.0, 1.0 - margin).sum()
    return float(objective), w - C * (signed[violated] @ X[violated]), float(-C * signed[violated].sum())


class LinearSVM:
    def __init__(self, C: float = 1.0, pca_components: int | None = None, max_iter: int = 5000, tol: float = 1e-8,
                 patience: int = 200, eta0: float = 1.0):
        if not (np.isfinite(C) and C > 0 and np.isfinite(eta0) and eta0 > 0 and np.isfinite(tol) and tol >= 0):
            raise ValueError("C and eta0 must be positive finite; tol nonnegative finite")
        if type(max_iter) is not int or max_iter < 1 or type(patience) is not int or patience < 1:
            raise ValueError("max_iter and patience must be positive integers")
        if pca_components is not None and (type(pca_components) is not int or pca_components < 1):
            raise ValueError("pca_components must be a positive integer or None")
        self.C, self.pca_components, self.max_iter, self.tol, self.patience, self.eta0 = float(C), pca_components, max_iter, tol, patience, eta0
        self.coef_ = None

    def fit(self, X, y) -> "LinearSVM":
        y = np.asarray(y)
        scaler = Standardizer()
        Z = scaler.fit_transform(X)  # Train-only preprocessing; validates 2D/finite
        if y.shape != (len(Z),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
            raise ValueError("y must be 1D {0, 1} labels, aligned with X, containing both classes")
        pca = None
        if self.pca_components is not None:
            pca = PCA(self.pca_components).fit(Z)
            Z = pca.transform(Z)
        y = y.astype(np.int64)
        n, w, b = len(Z), np.zeros(Z.shape[1]), 0.0
        best = (np.inf, w, b)
        stall = 0
        for t in range(1, self.max_iter + 1):
            obj, gw, gb = objective_and_subgradient(w, b, Z, y, self.C)
            if obj < best[0] - self.tol * max(1.0, abs(best[0])):
                stall = 0
            else:
                stall += 1
            if obj < best[0]:
                best = (obj, w.copy(), b)
            self.n_iter_ = t
            if stall >= self.patience:
                break
            step = self.eta0 / np.sqrt(t) / (self.C * n)
            w, b = w - step * gw, b - step * gb
        else:
            obj = objective_and_subgradient(w, b, Z, y, self.C)[0]
            if obj < best[0]:
                best = (obj, w, b)
        self.status_ = "converged" if stall >= self.patience else "max_iter"
        self.objective_, self.coef_, self.intercept_ = best[0], best[1].copy(), float(best[2])
        self.scaler_, self.pca_ = scaler, pca
        return self

    def decision_function(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model not fitted")
        Z = self.scaler_.transform(X)
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return Z @ self.coef_ + self.intercept_

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)

    def to_dict(self) -> dict:
        if self.coef_ is None:
            raise ValueError("Model not fitted")
        return {"kind": "linear_svm", "schema_version": 1, "C": self.C, "pca_components": self.pca_components, "max_iter": self.max_iter,
                "tol": self.tol, "patience": self.patience, "eta0": self.eta0, "coef": self.coef_.tolist(), "intercept": self.intercept_,
                "status": self.status_, "n_iter": self.n_iter_, "objective": self.objective_,
                "scaler": _state.scaler_state(self.scaler_), "pca": _state.pca_state(self.pca_)}

    def to_json(self) -> str:
        return _state.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, state: dict) -> "LinearSVM":
        if state.get("kind") != "linear_svm" or state.get("schema_version") != 1:
            raise ValueError("Not a linear_svm state")
        model = cls(state["C"], state["pca_components"], state["max_iter"], state["tol"], state["patience"], state["eta0"])
        model.coef_, model.intercept_ = np.array(state["coef"], dtype=np.float64), float(state["intercept"])
        model.status_, model.n_iter_, model.objective_ = state["status"], state["n_iter"], state["objective"]
        model.scaler_, model.pca_ = _state.scaler_from_state(state["scaler"]), _state.pca_from_state(state["pca"])
        return model

    @classmethod
    def from_json(cls, text: str) -> "LinearSVM":
        return cls.from_dict(_state.loads(text))
