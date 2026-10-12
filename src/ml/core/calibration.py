"""Probability calibration and precision-constrained threshold selection.

All fitting inputs must come from Train (normally out-of-fold predictions).  The
held-out Validation split is evaluation-only for the calibration experiment.
"""
from __future__ import annotations

import numpy as np

from .metrics import classification_metrics


def _binary_inputs(y, scores) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y, dtype=np.float64)
    probability = np.asarray(scores, dtype=np.float64)
    if target.ndim != 1 or probability.ndim != 1 or len(target) != len(probability) or not len(target):
        raise ValueError("Expected equally sized nonempty 1D arrays")
    if not np.isfinite(target).all() or not np.isfinite(probability).all():
        raise ValueError("Calibration inputs must be finite")
    if not np.isin(target, (0.0, 1.0)).all() or len(np.unique(target)) < 2:
        raise ValueError("Calibration target must contain both binary classes")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("Calibration scores must be probabilities")
    return target, probability


def _sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def probability_metrics(y, probability, *, threshold: float) -> dict:
    """Classification metrics plus proper scoring rules for calibration."""
    target, score = _binary_inputs(y, probability)
    clipped = np.clip(score, 1e-15, 1.0 - 1e-15)
    metrics = classification_metrics(target, score, threshold=threshold)
    metrics["brier"] = {"value": float(np.mean((score - target) ** 2)), "reason": None}
    metrics["log_loss"] = {
        "value": float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))),
        "reason": None,
    }
    return metrics


class PlattCalibrator:
    """Fit ``sigmoid(slope * logit(p) + intercept)`` with damped Newton steps.

    A positive slope is enforced so calibration cannot reverse the model ranking;
    therefore ROC-AUC and average precision remain unchanged by construction.
    """

    def __init__(self, *, l2: float = 1e-6, max_iter: int = 200, tol: float = 1e-10):
        if not np.isfinite(l2) or l2 < 0 or type(max_iter) is not int or max_iter < 1:
            raise ValueError("Invalid Platt calibration parameters")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("Invalid convergence tolerance")
        self.l2, self.max_iter, self.tol = float(l2), max_iter, float(tol)

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        return np.log(clipped) - np.log1p(-clipped)

    def _loss(self, feature: np.ndarray, target: np.ndarray, slope: float, intercept: float) -> float:
        margin = slope * feature + intercept
        return float(np.mean(np.logaddexp(0.0, margin) - target * margin) + 0.5 * self.l2 * slope**2)

    def fit(self, probability, y) -> "PlattCalibrator":
        target, score = _binary_inputs(y, probability)
        feature = self._logit(score)
        slope, intercept = 1.0, 0.0
        self.history_ = {"iteration": [], "loss": []}

        for iteration in range(self.max_iter):
            margin = slope * feature + intercept
            calibrated = _sigmoid(margin)
            residual = calibrated - target
            weight = calibrated * (1.0 - calibrated)
            gradient = np.array([
                np.mean(residual * feature) + self.l2 * slope,
                np.mean(residual),
            ])
            hessian = np.array([
                [np.mean(weight * feature**2) + self.l2, np.mean(weight * feature)],
                [np.mean(weight * feature), np.mean(weight) + 1e-12],
            ])
            step = np.linalg.solve(hessian, gradient)
            loss = self._loss(feature, target, slope, intercept)
            self.history_["iteration"].append(iteration)
            self.history_["loss"].append(loss)

            scale = 1.0
            accepted = False
            while scale >= 2.0**-20:
                candidate_slope = max(1e-8, slope - scale * step[0])
                candidate_intercept = intercept - scale * step[1]
                candidate_loss = self._loss(feature, target, candidate_slope, candidate_intercept)
                if candidate_loss <= loss:
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break
            change = max(abs(candidate_slope - slope), abs(candidate_intercept - intercept))
            slope, intercept = candidate_slope, candidate_intercept
            if change <= self.tol:
                break

        self.slope_, self.intercept_ = float(slope), float(intercept)
        return self

    def predict_proba(self, probability) -> np.ndarray:
        if not hasattr(self, "slope_"):
            raise ValueError("Unfitted PlattCalibrator")
        score = np.asarray(probability, dtype=np.float64)
        if score.ndim != 1 or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError("Expected finite probabilities")
        return _sigmoid(self.slope_ * self._logit(score) + self.intercept_)

    def state_dict(self) -> dict:
        if not hasattr(self, "slope_"):
            raise ValueError("Unfitted PlattCalibrator")
        return {
            "method": "platt_logit",
            "slope": self.slope_,
            "intercept": self.intercept_,
            "l2": self.l2,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }


def precision_threshold(y, probability, *, min_recall: float = 0.4) -> dict:
    """Maximize precision subject to a minimum recall, using only supplied data."""
    target, score = _binary_inputs(y, probability)
    if not np.isfinite(min_recall) or not 0.0 < min_recall <= 1.0:
        raise ValueError("min_recall must be in (0, 1]")
    choices = []
    for threshold in np.unique(score):
        metrics = probability_metrics(target, score, threshold=float(threshold))
        recall = metrics["recall"]["value"]
        precision = metrics["precision"]["value"]
        if recall is not None and precision is not None and recall >= min_recall:
            choices.append((precision, recall, metrics["f1"]["value"], float(threshold), metrics))
    if not choices:
        raise ValueError(f"No threshold satisfies minimum recall {min_recall}")
    best = max(choices, key=lambda item: item[:4])
    return {
        "threshold": best[3],
        "min_recall": float(min_recall),
        "selection": "max_precision_then_recall_then_f1_then_higher_threshold",
        "metrics": best[4],
        "candidate_count": len(choices),
    }
