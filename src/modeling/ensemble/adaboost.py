"""Scratch discrete AdaBoost (AdaBoost.M1) with weighted decision stumps.

Inputs are {0,1}; internally labels are y_s = 2y - 1 in {-1,+1}. Stump: h(x) = polarity * (+1 if x[f] > thr else -1),
searched over both polarities with ties resolved by lowest feature, then lowest threshold, then normal (+1) polarity.
alpha = alpha_factor * learning_rate * ln((1-err)/err); sample weights live in log space and are renormalized with
logsumexp every round. Statuses: completed | completed_early | completed_perfect | failed (see configs/modeling.json).
predict_proba is the Friedman logistic link sigmoid(2F) of the vote sum F (a monotone score, not a calibrated probability).
"""
from typing import Any, Dict, Optional

import numpy as np

from src.modeling.ensemble._tree import check_X, check_binary_y, check_int, check_positive

_TIE = 1e-12


class AdaBoostFailure(RuntimeError):
    """The first weak learner is no better than chance; the model is not usable."""


def _logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def _sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _stump_votes(stump, X):
    return stump["polarity"] * np.where(X[:, stump["feature"]] > stump["threshold"], 1.0, -1.0)


def _best_stump(X, ys, w):
    """Minimum weighted-error stump -> (error, stump) or None when no feature is available."""
    total, pos, neg = w.sum(), np.where(ys == 1, w, 0.0), np.where(ys == 1, 0.0, w)
    best = None
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="stable")
        x = X[order, f]
        valid = x[1:] > x[:-1]
        err_normal = np.cumsum(pos[order])[:-1] + (neg.sum() - np.cumsum(neg[order])[:-1])
        err_normal = np.append(err_normal, pos.sum())  # both constant predictions at the upper boundary
        errs = np.stack([err_normal, total - err_normal], axis=1).ravel() / total  # per threshold: normal, then flipped
        ok = np.repeat(np.append(valid, True), 2)
        k = int(np.flatnonzero(ok & (errs <= errs[ok].min() + _TIE))[0])
        if best is None or errs[k] < best[0] - _TIE:
            i = k // 2
            thr = x[-1] if i == len(x) - 1 else (x[i] + x[i + 1]) / 2
            best = (float(errs[k]), {"feature": f, "threshold": float(thr if i == len(x) - 1 or thr < x[i + 1] else x[i]),
                                     "polarity": 1 if k % 2 == 0 else -1})
    return best


class AdaBoost:
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0, alpha_factor: float = 0.5,
                 epsilon: float = 1e-12, random_state: int = 42):
        self.n_estimators = check_int("n_estimators", n_estimators)
        self.learning_rate = check_positive("learning_rate", learning_rate)
        self.alpha_factor = check_positive("alpha_factor", alpha_factor)
        self.epsilon = self._check_epsilon(epsilon)
        self.random_state = check_int("random_state", random_state, 0)  # fitting is deterministic; kept for the run contract
        self.status_: Optional[str] = None
        self.stumps_: list = []
        self.estimator_weights_: list = []
        self.estimator_errors_: list = []
        self.loss_history_: list = []
        self.stop_error_: Optional[float] = None
        self.log_sample_weights_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    @staticmethod
    def _check_epsilon(epsilon):
        epsilon = check_positive("epsilon", epsilon)
        if epsilon >= 0.5:
            raise ValueError("epsilon must be < 0.5")
        return epsilon

    def fit(self, X, y) -> "AdaBoost":
        self.epsilon = self._check_epsilon(self.epsilon)
        X = check_X(X)
        ys = 2 * check_binary_y(y, len(X), both_classes=True) - 1
        n = len(ys)
        self.status_, self.stumps_, self.estimator_weights_, self.estimator_errors_ = None, [], [], []
        self.loss_history_, self.stop_error_, self.n_features_ = [], None, X.shape[1]
        log_w, F = np.full(n, -np.log(n)), np.zeros(n)
        self.status_ = "completed"
        for t in range(self.n_estimators):
            found = _best_stump(X, ys, np.exp(log_w))
            err, stump = found if found is not None else (0.5, None)
            if found is None or err >= 0.5:
                self.stop_error_ = float(err)
                if t == 0:
                    self.status_ = "failed"
                    self.log_sample_weights_ = log_w
                    raise AdaBoostFailure(f"first weak learner error {err:.6g} >= 0.5")
                self.status_ = "completed_early"
                break
            clipped = max(err, self.epsilon)
            alpha = self.alpha_factor * self.learning_rate * np.log((1.0 - clipped) / clipped)
            vote = _stump_votes(stump, X)
            self.stumps_.append(stump)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(float(err))
            F += alpha * vote
            self.loss_history_.append(float(np.exp(min(_logsumexp(-ys * F) - np.log(n), 700.0))))
            if err <= self.epsilon:
                self.status_ = "completed_perfect"
                break
            log_w = log_w - alpha * ys * vote
            log_w -= _logsumexp(log_w)
        self.log_sample_weights_ = log_w
        return self

    def decision_function(self, X) -> np.ndarray:
        if self.status_ not in ("completed", "completed_early", "completed_perfect"):
            raise ValueError("Model not fitted")
        X = check_X(X, self.n_features_)
        F = np.zeros(len(X))
        for stump, alpha in zip(self.stumps_, self.estimator_weights_):
            F += alpha * _stump_votes(stump, X)
        return F

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(np.int64)  # zero vote -> label +1

    def predict_proba(self, X) -> np.ndarray:
        p1 = _sigmoid(2.0 * self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def to_dict(self) -> Dict[str, Any]:
        if self.status_ not in ("completed", "completed_early", "completed_perfect"):
            raise ValueError("Model not fitted")
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "alpha_factor": self.alpha_factor,
                "epsilon": self.epsilon, "random_state": self.random_state, "n_features": self.n_features_,
                "status": self.status_, "stop_error": self.stop_error_, "stumps": self.stumps_,
                "estimator_weights": [float(a) for a in self.estimator_weights_],
                "estimator_errors": self.estimator_errors_, "loss_history": self.loss_history_,
                "log_sample_weights": [float(v) for v in self.log_sample_weights_]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdaBoost":
        model = cls(d["n_estimators"], d["learning_rate"], d["alpha_factor"], d["epsilon"], d["random_state"])
        n_features = check_int("n_features", d["n_features"])
        stumps, k = d["stumps"], len(d["stumps"])
        if d["status"] not in ("completed", "completed_early", "completed_perfect") or not 1 <= k <= model.n_estimators \
                or any(len(d[key]) != k for key in ("estimator_weights", "estimator_errors", "loss_history")):
            raise ValueError("Malformed AdaBoost state")
        for s in stumps:
            if not (0 <= s["feature"] < n_features and s["polarity"] in (-1, 1) and np.isfinite(s["threshold"])):
                raise ValueError("Malformed AdaBoost state")
        model.n_features_, model.status_, model.stop_error_ = n_features, d["status"], d["stop_error"]
        model.stumps_ = [{"feature": int(s["feature"]), "threshold": float(s["threshold"]), "polarity": int(s["polarity"])} for s in stumps]
        model.estimator_weights_ = [float(v) for v in d["estimator_weights"]]
        model.estimator_errors_ = [float(v) for v in d["estimator_errors"]]
        model.loss_history_ = [float(v) for v in d["loss_history"]]
        model.log_sample_weights_ = np.asarray(d["log_sample_weights"], dtype=np.float64)
        return model
