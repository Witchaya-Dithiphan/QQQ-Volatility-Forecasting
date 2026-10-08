"""Task metrics and selection policies, including explicit undefined reasons."""
from __future__ import annotations
import numpy as np


def measure(value=None, reason: str | None = None) -> dict:
    if value is None:
        return {"value": None, "reason": reason or "undefined"}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return {"value": None, "reason": reason or "nonfinite numerical result"}
    return {"value": float(value) if np.isscalar(value) else value, "reason": None}


def _paired(y, prediction, *, binary: bool = False):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or prediction.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Expected equally sized finite nonempty vectors")
    if binary and not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary 0/1 labels")
    return y, prediction


def _ratio(numerator, denominator):
    return measure(numerator / denominator) if denominator else measure(reason="zero denominator")


def regression_metrics(y, prediction) -> dict:
    y, prediction = _paired(y, prediction)
    residual = y - prediction
    mse = np.mean(residual ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return {"mae": measure(np.mean(np.abs(residual))), "mse": measure(mse), "rmse": measure(np.sqrt(mse)),
            "r2": measure(1 - np.sum(residual ** 2) / total) if len(y) > 1 and total > 0 else measure(reason="constant target or fewer than two samples")}


def regression_outputs(y, raw_prediction, *, clip_min: float = 0.0) -> dict:
    y, raw = _paired(y, raw_prediction)
    final = np.maximum(raw, clip_min)
    return {"raw_prediction": raw.copy(), "final_prediction": final, "negative_count": int((raw < 0).sum()),
            "clipping_indicator": raw < clip_min, "raw_metrics": regression_metrics(y, raw), "clipped_metrics": regression_metrics(y, final)}


def classification_metrics(y, scores, *, threshold: float = 0.5) -> dict:
    y, scores = _paired(y, scores, binary=True)
    if not np.isfinite(threshold): raise ValueError("Nonfinite threshold")
    predicted = scores >= threshold
    positive = y == 1
    tp, fp = int((predicted & positive).sum()), int((predicted & ~positive).sum())
    fn, tn = int((~predicted & positive).sum()), int((~predicted & ~positive).sum())
    result = {"confusion_matrix": [[tn, fp], [fn, tp]], "accuracy": _ratio(tp + tn, len(y)),
              "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn), "sensitivity": _ratio(tp, tp + fn),
              "specificity": _ratio(tn, tn + fp), "tnr": _ratio(tn, tn + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}
    if not positive.any() or positive.all():
        for name in ("roc_auc", "average_precision", "roc_curve", "pr_curve"):
            result[name] = measure(reason="one-class subset")
        return result
    order = np.argsort(-scores, kind="stable")
    sorted_score, sorted_y = scores[order], y[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_score)), len(y) - 1]
    cumulative_tp = np.cumsum(sorted_y)[endpoints]
    cumulative_fp = endpoints + 1 - cumulative_tp
    recall = cumulative_tp / positive.sum()
    precision = cumulative_tp / (endpoints + 1)
    tpr, fpr = np.r_[0., recall], np.r_[0., cumulative_fp / (~positive).sum()]
    result.update(roc_auc=measure(np.trapezoid(tpr, fpr)), average_precision=measure(np.sum(np.diff(np.r_[0., recall]) * precision)),
                  roc_curve=measure({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": [None, *sorted_score[endpoints].tolist()]}),
                  pr_curve=measure({"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": sorted_score[endpoints].tolist()}))
    return result


def loss(name: str, y, *, scores=None, probabilities=None, epsilon: float = 1e-15) -> dict:
    if name in ("binary_cross_entropy", "log_loss") or (name == "binary_logistic_loss" and scores is None):
        if probabilities is None: return measure(reason="requires genuine probabilities")
        y, probability = _paired(y, probabilities, binary=True)
        if not 0 < epsilon < .5 or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Invalid probability or clipping epsilon")
        probability = np.clip(probability, epsilon, 1 - epsilon)
        return measure(np.mean(-y * np.log(probability) - (1 - y) * np.log1p(-probability)))
    if scores is None: return measure(reason="requires decision scores")
    y, score = _paired(y, scores, binary=True)
    margin = (2 * y - 1) * score
    if name == "binary_logistic_loss": return measure(np.mean(np.logaddexp(0, -margin)))
    if name == "hinge_loss": return measure(np.mean(np.maximum(0, 1 - margin)))
    if name == "perceptron_criterion":
        return {**measure(np.mean(np.maximum(0, -margin))), "mistake_rate": measure(np.mean((score >= 0) != y))}
    if name == "exponential_loss":
        with np.errstate(over="ignore"):
            return measure(np.mean(np.exp(-margin)), reason="exponential loss overflow")
    return measure(reason=f"No meaningful loss: {name}")


def select_threshold(y, scores, *, score_kind: str = "probability", split: str = "validation") -> dict:
    if split != "validation" or score_kind not in ("probability", "decision"):
        raise ValueError("Threshold selection requires Validation and a supported score kind")
    y, scores = _paired(y, scores, binary=True)
    default = .5 if score_kind == "probability" else 0.
    if score_kind == "probability" and ((scores < 0) | (scores > 1)).any(): raise ValueError("Invalid probabilities")
    baseline = classification_metrics(y, scores, threshold=default)
    if len(np.unique(y)) < 2:
        return {"threshold": None, "reason": "one-class Validation", "default_threshold": default, "baseline": baseline, "selection_split": split}
    choices = []
    for threshold in np.unique(scores):
        metrics = classification_metrics(y, scores, threshold=float(threshold))
        choices.append((metrics["f1"]["value"], metrics["recall"]["value"], -float(threshold), metrics))
    best = max(choices, key=lambda item: item[:3])
    return {"threshold": -best[2], "reason": None, "metrics": best[3], "default_threshold": default, "baseline": baseline, "selection_split": split}


def select_candidate(candidates: list[dict], *, task: str) -> dict:
    """Select nested regression_outputs metrics; final tie uses ID, then input order."""
    policies = {"regression": (["rmse", "mae"], 1), "classification": (["average_precision", "roc_auc"], -1), "clustering": (["silhouette"], -1)}
    if task not in policies: raise ValueError("Unknown selection task")
    fields, direction = policies[task]
    valid = []
    for index, candidate in enumerate(candidates):
        metrics = candidate["metrics"]["clipped_metrics"] if task == "regression" else candidate["metrics"]
        values = [metrics[field]["value"] for field in fields]
        if all(value is not None for value in values):
            key = (*[direction * value for value in values], str(candidate.get("id", "")), index)
            valid.append((key, candidate))
    if not valid: raise ValueError("No valid candidates")
    return min(valid, key=lambda item: item[0])[1]


def clustering_metrics(X, labels) -> dict:
    X, labels = np.asarray(X, dtype=np.float64), np.asarray(labels)
    if X.ndim != 2 or not all(X.shape) or labels.shape != (len(X),) or not np.isfinite(X).all():
        raise ValueError("Invalid clustering inputs")
    clusters = np.unique(labels)
    sizes = {str(k): int((labels == k).sum()) for k in clusters}
    inertia = sum(np.sum((X[labels == k] - X[labels == k].mean(axis=0)) ** 2) for k in clusters)
    result = {"inertia": measure(inertia), "cluster_sizes": sizes}
    if not 1 < len(clusters) < len(X):
        result["silhouette"] = measure(reason="requires between 2 and n-1 clusters")
        return result
    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    silhouettes = []
    for index, label in enumerate(labels):
        same = labels == label
        if same.sum() == 1:
            silhouettes.append(0.)
            continue
        a = distances[index, same].sum() / (same.sum() - 1)
        b = min(distances[index, labels == other].mean() for other in clusters if other != label)
        silhouettes.append((b - a) / max(a, b) if max(a, b) else 0.)
    result["silhouette"] = measure(np.mean(silhouettes))
    return result


def train_cluster_mapping(cluster_ids, labels, *, split: str = "train") -> dict:
    if split != "train": raise ValueError("Cluster majority mapping must use Train only")
    y, clusters = _paired(labels, cluster_ids, binary=True)
    return {int(k): int(y[clusters == k].mean() >= .5) for k in np.unique(clusters)}
