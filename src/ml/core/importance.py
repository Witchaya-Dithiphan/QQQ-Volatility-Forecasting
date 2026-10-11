# src/ml/core/importance.py
"""Model-agnostic feature importance on Validation, plus linear-model coefficients.

A feature column is reordered in whole blocks of consecutive days rather than row by
row: the features are strongly autocorrelated, and a row-wise reordering would also
destroy that serial structure and overstate how much the model relies on the feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import load_model, read_runs, run_directory
from .bootstrap import BLOCK, SEED
from .contracts import FEATURE_COLUMNS
from .data import load
from .persist import load_npz
from .trainer import OUTPUT_ROOT, evaluate

LINEAR_MODELS = ("multiple_linear", "elastic_net", "logistic", "slp", "perceptron", "svm")


def block_shuffled(column, *, block: int, rng) -> np.ndarray:
    """Reorder whole blocks of consecutive rows, keeping each block's inner order."""
    column = np.asarray(column)
    pieces = np.array_split(np.arange(len(column)), max(1, -(-len(column) // block)))
    order = np.argsort(rng.random_sample(len(pieces)), kind="stable")
    return column[np.concatenate([pieces[index] for index in order])]


def block_permutation_importance(score, X, *, block: int = BLOCK, repeats: int = 30, seed: int = SEED) -> dict:
    """Drop in score(X) when one column is block-reordered; larger means more important."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.RandomState(seed)
    baseline = float(score(X))
    drops = np.empty((repeats, X.shape[1]))
    for repeat in range(repeats):
        for feature in range(X.shape[1]):
            shuffled = X.copy()
            shuffled[:, feature] = block_shuffled(X[:, feature], block=block, rng=rng)
            drops[repeat, feature] = baseline - float(score(shuffled))
    return {"baseline": baseline, "mean": drops.mean(axis=0),
            "std": drops.std(axis=0, ddof=1) if repeats > 1 else np.zeros(X.shape[1])}


def feature_importance(variant: str, *, artifact_root=None, repeats: int = 30, require_all: bool = True) -> list[dict]:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    validation = load(variant).validation
    rows = []
    for run in read_runs(artifact_root, variant, require_all=require_all):
        if run.task == "clustering":
            continue
        model, transform = load_model(run.directory)

        def score(matrix, model=model, transform=transform, task=run.task):
            metrics, _, _ = evaluate(model, transform(matrix), validation.y_regression,
                                     validation.y_classification, task)
            if task == "regression":
                return -metrics["clipped_metrics"]["rmse"]["value"]
            return metrics["average_precision"]["value"]

        result = block_permutation_importance(score, validation.X, repeats=repeats)
        metric = "negative_rmse" if run.task == "regression" else "average_precision"
        for index, feature in enumerate(FEATURE_COLUMNS):
            rows.append({"model": run.name, "variant": variant, "task": run.task, "feature": feature,
                         "metric": metric, "baseline": result["baseline"],
                         "importance_mean": float(result["mean"][index]),
                         "importance_std": float(result["std"][index])})
    return rows


def linear_coefficients(variant: str, *, artifact_root=None) -> list[dict]:
    """Weights on the standardised features; svm's are mapped back from its PCA basis."""
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    rows = []
    for name in LINEAR_MODELS:
        directory = run_directory(artifact_root, variant, name)
        arrays, _ = load_npz(directory / "model.npz")
        scaler, metadata = load_npz(directory / "preprocessor.npz")
        coefficient = np.asarray(arrays["coef"], dtype=np.float64)
        if metadata["pca_components"]:
            coefficient = scaler["components"].T @ coefficient
        for feature, value in zip(FEATURE_COLUMNS, coefficient):
            rows.append({"model": name, "variant": variant, "feature": feature, "coefficient": float(value)})
    return rows
