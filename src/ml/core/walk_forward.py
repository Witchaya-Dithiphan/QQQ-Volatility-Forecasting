# src/ml/core/walk_forward.py
"""Expanding-window check inside Train, with each model's winning hyperparameters fixed.

This is a robustness table, not a second model selection: Validation is one calm
regime with 34 positive days, while these four blocks span 2019-2023 and hold about
ten times as many. Nothing chosen here feeds back into which model or setting is used.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from ..classification.stacking import fold_plan
from ..registry import MODELS, models_for_task
from .artifacts import run_directory
from .bootstrap import REPLICATES, block_indices, interval, replicate
from .data import load
from .preprocess import PCA, Standardizer
from .report import classification_scores
from .trainer import OUTPUT_ROOT, UNSCALED_MODES, evaluate, trading_day_positions

FIELDS = ("average_precision", "roc_auc", "f1", "recall")


def spearman(a, b) -> float:
    """Rank correlation without tie handling; the inputs here are distinct floats."""
    rank_a = np.argsort(np.argsort(np.asarray(a, dtype=np.float64)))
    rank_b = np.argsort(np.argsort(np.asarray(b, dtype=np.float64)))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def walk_forward(name: str, *, variant: str, artifact_root=None, model_class=None, folds: int = 4,
                 initial_fraction: float = 0.4, purge: int = 5, replicates: int = REPLICATES) -> list[dict]:
    directory = run_directory(Path(artifact_root or OUTPUT_ROOT), variant, name)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model_class = model_class or MODELS[name]
    params = config["params"] if model_class is MODELS[name] else {}
    train = load(variant).train
    positions, n_positions = trading_day_positions(variant, train.dates)
    scaled = config["preprocess"] not in UNSCALED_MODES or bool(config["pca_components"])
    wants_positions = "positions" in inspect.signature(model_class.fit).parameters

    rows, pooled = [], {"y": [], "score": [], "label": []}
    for fold, (history, block, start, stop) in enumerate(
        fold_plan(positions, n_positions, folds, initial_fraction, purge)
    ):
        history_X, block_X = train.X[history], train.X[block]
        if scaled:
            scaler = Standardizer().fit(history_X)
            history_X, block_X = scaler.transform(history_X), scaler.transform(block_X)
        if config["pca_components"]:
            pca = PCA(config["pca_components"]).fit(history_X)
            history_X, block_X = pca.transform(history_X), pca.transform(block_X)
        extras = {}
        if wants_positions:
            # The model's own inner folds must end where this fold's history ends.
            extras = {"positions": positions[history], "n_positions": int(positions[history][-1]) + 1}
        located = {"model": name, "variant": variant, "fold": fold,
                   "block_start": str(train.dates[block][0])[:10], "block_end": str(train.dates[block][-1])[:10],
                   "block_start_position": start, "block_stop_position": stop,
                   "train_rows": int(history.sum()), "block_rows": int(block.sum()),
                   "positives": int(train.y_classification[block].sum())}
        try:
            model = model_class(**params).fit(history_X, train.y_classification[history], **extras)
        except ValueError as exc:
            # The first 275 trading days of Train contain no high-volatility day, so a
            # model that needs both classes in a sub-window of the shortest history
            # (stacking) cannot be fitted there. Say so instead of losing the table.
            rows.append({**located, **{field: None for field in FIELDS}, "reason": str(exc)})
            continue
        _, label, score = evaluate(model, block_X, train.y_regression[block],
                                   train.y_classification[block], "classification")
        y = train.y_classification[block].astype(np.float64)
        label, score = np.asarray(label, dtype=np.float64), np.asarray(score, dtype=np.float64)
        scores = classification_scores(y, score, label)
        rows.append({**located, **{field: scores[field] for field in FIELDS}, "reason": None})
        for key, values in (("y", y), ("score", score), ("label", label)):
            pooled[key].append(values)

    scored = [row for row in rows if row["reason"] is None]
    if not scored:
        raise ValueError(f"{variant}/{name} could not be fitted on any walk-forward fold: {rows[0]['reason']}")
    y, score, label = (np.concatenate(pooled[key]) for key in ("y", "score", "label"))
    point = classification_scores(y, score, label)
    summary = interval(point["average_precision"], replicate(
        classification_scores, (y, score, label), block_indices(len(y), replicates=replicates)
    )["average_precision"])
    rows.append({"model": name, "variant": variant, "fold": "pooled",
                 "block_start": scored[0]["block_start"], "block_end": scored[-1]["block_end"],
                 "block_start_position": scored[0]["block_start_position"],
                 "block_stop_position": scored[-1]["block_stop_position"],
                 "train_rows": None, "block_rows": int(len(y)), "positives": int(y.sum()),
                 **{field: point[field] for field in FIELDS}, "reason": None,
                 "folds_used": len(pooled["y"]),
                 "average_precision_low": summary["low"], "average_precision_high": summary["high"]})
    return rows


def walk_forward_table(variant: str, *, artifact_root=None, models: list[str] | None = None,
                       replicates: int = REPLICATES) -> list[dict]:
    rows = []
    for name in models or models_for_task("classification"):
        rows += walk_forward(name, variant=variant, artifact_root=artifact_root, replicates=replicates)
    return rows
