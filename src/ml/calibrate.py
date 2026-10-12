"""Train-only probability calibration experiment for an existing classifier run.

The calibrator and operating threshold are fitted on expanding-window out-of-fold
predictions inside Train.  Official Validation is read only after both are fixed.
Test is never opened by this module.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from .classification.stacking import fold_plan
from .core.artifacts import read_run, run_directory
from .core.calibration import PlattCalibrator, precision_threshold, probability_metrics
from .core.data import load
from .core.preprocess import PCA, Standardizer
from .core.trainer import OUTPUT_ROOT, UNSCALED_MODES, trading_day_positions
from .registry import MODELS

SUMMARY_FIELDS = (
    "accuracy", "precision", "recall", "f1", "specificity",
    "roc_auc", "average_precision", "brier", "log_loss",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probability(model, X) -> np.ndarray:
    try:
        score = np.asarray(model.predict_proba(X), dtype=np.float64)
    except NotImplementedError as exc:
        raise ValueError(f"{model.name} does not expose probabilities for calibration") from exc
    if score.ndim == 2 and score.shape[1] == 2:
        score = score[:, 1]
    if score.ndim != 1 or ((score < 0) | (score > 1)).any():
        raise ValueError(f"{model.name} returned invalid probabilities")
    return score


def _summary(metrics: dict) -> dict:
    result = {name: metrics[name]["value"] for name in SUMMARY_FIELDS}
    result["confusion_matrix"] = metrics["confusion_matrix"]
    return result


def _walk_forward_predictions(name: str, variant: str, artifact_root: Path, *, folds: int,
                              initial_fraction: float, purge: int) -> tuple[list[dict], dict]:
    directory = run_directory(artifact_root, variant, name)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model_class = MODELS[name]
    if model_class.task != "classification":
        raise ValueError("Calibration requires a classification model")
    train = load(variant).train
    positions, n_positions = trading_day_positions(variant, train.dates)
    scaled = config["preprocess"] not in UNSCALED_MODES or bool(config["pca_components"])
    wants_positions = "positions" in inspect.signature(model_class.fit).parameters

    rows, fold_rows = [], []
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
            extras = {"positions": positions[history], "n_positions": int(positions[history][-1]) + 1}
        model = model_class(**config["params"]).fit(
            history_X, train.y_classification[history], **extras
        )
        score = _probability(model, block_X)
        target = train.y_classification[block].astype(np.float64)
        dates = train.dates[block]
        metrics = probability_metrics(target, score, threshold=0.5)
        fold_rows.append({
            "fold": fold,
            "history_rows": int(history.sum()),
            "block_rows": int(block.sum()),
            "block_start_position": start,
            "block_stop_position": stop,
            "block_start": str(dates[0])[:10],
            "block_end": str(dates[-1])[:10],
            "positives": int(target.sum()),
            "default_metrics": _summary(metrics),
        })
        rows.extend(
            {"Date": str(date)[:10], "fold": fold, "target": int(label), "raw_score": float(value)}
            for date, label, value in zip(dates, target, score)
        )
    return rows, {"folds": fold_rows, "config": config}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_calibration_experiment(*, name: str = "mlp", variant: str = "with_spike",
                               artifact_root: Path | None = None, output_root: Path | None = None,
                               folds: int = 4, initial_fraction: float = 0.4, purge: int = 5,
                               min_recall: float = 0.4, l2: float = 1e-6) -> dict:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    directory = run_directory(artifact_root, variant, name)
    destination = Path(output_root or artifact_root / "experiments" / "calibration") / variant / name
    oof_rows, metadata = _walk_forward_predictions(
        name, variant, artifact_root, folds=folds, initial_fraction=initial_fraction, purge=purge
    )
    oof_target = np.array([row["target"] for row in oof_rows], dtype=np.float64)
    oof_raw = np.array([row["raw_score"] for row in oof_rows], dtype=np.float64)
    calibrator = PlattCalibrator(l2=l2).fit(oof_raw, oof_target)
    oof_calibrated = calibrator.predict_proba(oof_raw)
    raw_policy = precision_threshold(oof_target, oof_raw, min_recall=min_recall)
    calibrated_policy = precision_threshold(oof_target, oof_calibrated, min_recall=min_recall)

    # Only now read the already-saved Validation predictions.  No Test loader is imported or called.
    run = read_run(artifact_root, variant, name)
    validation_raw = run.score.astype(np.float64)
    validation_target = run.target.astype(np.float64)
    validation_calibrated = calibrator.predict_proba(validation_raw)
    policies = {
        "raw_default": (validation_raw, 0.5),
        "raw_precision_threshold": (validation_raw, raw_policy["threshold"]),
        "platt_default": (validation_calibrated, 0.5),
        "platt_precision_threshold": (validation_calibrated, calibrated_policy["threshold"]),
    }
    validation = {
        key: {"threshold": threshold, "metrics": _summary(probability_metrics(
            validation_target, score, threshold=threshold
        ))}
        for key, (score, threshold) in policies.items()
    }
    threshold_sweep = []
    for method, fit_score, evaluation_score in (
        ("raw", oof_raw, validation_raw),
        ("platt", oof_calibrated, validation_calibrated),
    ):
        for recall_floor in (0.2, 0.3, 0.4, 0.5, 0.6):
            policy = precision_threshold(oof_target, fit_score, min_recall=recall_floor)
            evaluation = _summary(probability_metrics(
                validation_target, evaluation_score, threshold=policy["threshold"]
            ))
            threshold_sweep.append({
                "method": method,
                "min_recall_oof": recall_floor,
                "threshold": policy["threshold"],
                "oof_precision": policy["metrics"]["precision"]["value"],
                "oof_recall": policy["metrics"]["recall"]["value"],
                "validation_precision": evaluation["precision"],
                "validation_recall": evaluation["recall"],
                "validation_f1": evaluation["f1"],
            })
    for row, calibrated in zip(oof_rows, oof_calibrated):
        row["calibrated_score"] = float(calibrated)
        row["raw_selected_prediction"] = int(row["raw_score"] >= raw_policy["threshold"])
        row["calibrated_selected_prediction"] = int(calibrated >= calibrated_policy["threshold"])
    validation_rows = []
    for date, target, raw, calibrated in zip(run.dates, validation_target, validation_raw, validation_calibrated):
        validation_rows.append({
            "Date": str(date)[:10], "target": int(target), "raw_score": float(raw),
            "calibrated_score": float(calibrated),
            "raw_default_prediction": int(raw >= 0.5),
            "raw_selected_prediction": int(raw >= raw_policy["threshold"]),
            "platt_default_prediction": int(calibrated >= 0.5),
            "platt_selected_prediction": int(calibrated >= calibrated_policy["threshold"]),
        })

    configuration = {
        "schema_version": 1,
        "model": name,
        "variant": variant,
        "calibration_method": "platt_logit",
        "calibration_source": "expanding-window out-of-fold predictions inside Train",
        "evaluation_split": "Validation",
        "test_accessed": False,
        "folds": folds,
        "initial_fraction": initial_fraction,
        "purge_trading_days": purge,
        "min_recall": min_recall,
        "l2": l2,
        "seed": metadata["config"]["seed"],
        "source_run": str(directory),
        "source_validation_predictions_sha256": _digest(directory / "validation_predictions.csv"),
        "limitation": "The base model and its hyperparameters were selected previously; this is a diagnostic calibration experiment, not a new untouched Test result.",
    }
    result = {
        "config": configuration,
        "calibrator": {**calibrator.state_dict(), "history": calibrator.history_},
        "oof": {
            "rows": len(oof_rows), "positives": int(oof_target.sum()),
            "raw_default": _summary(probability_metrics(oof_target, oof_raw, threshold=0.5)),
            "platt_default": _summary(probability_metrics(oof_target, oof_calibrated, threshold=0.5)),
            "raw_threshold_selection": {**raw_policy, "metrics": _summary(raw_policy["metrics"])},
            "platt_threshold_selection": {**calibrated_policy, "metrics": _summary(calibrated_policy["metrics"])},
            "folds": metadata["folds"],
        },
        "validation": validation,
        "threshold_sweep": threshold_sweep,
    }
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": destination / "config.json",
        "results": destination / "results.json",
        "calibrator": destination / "calibrator.json",
        "oof_predictions": destination / "oof_predictions.csv",
        "validation_predictions": destination / "validation_predictions.csv",
        "threshold_sweep": destination / "threshold_sweep.csv",
        "summary": destination / "README.md",
    }
    _write_json(paths["config"], configuration)
    _write_json(paths["calibrator"], result["calibrator"])
    _write_json(paths["results"], {
        "oof": result["oof"], "validation": validation, "threshold_sweep": threshold_sweep
    })
    _write_csv(paths["oof_predictions"], oof_rows)
    _write_csv(paths["validation_predictions"], validation_rows)
    _write_csv(paths["threshold_sweep"], threshold_sweep)
    lines = [
        f"# Calibration diagnostic — {name} / {variant}", "",
        "Calibration and threshold selection use Train walk-forward predictions only. Test was not read.", "",
        "| policy | threshold | precision | recall | f1 | AP | Brier | log loss |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, policy in validation.items():
        metric = policy["metrics"]
        lines.append(
            f"| {key} | {policy['threshold']:.6f} | {metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{metric['f1']:.4f} | {metric['average_precision']:.4f} | {metric['brier']:.4f} | {metric['log_loss']:.4f} |"
        )
    lines += [
        "", f"Minimum recall constraint selected on Train OOF: `{min_recall:.2f}`.", "",
        "## Threshold sensitivity", "",
        "Every threshold below was selected on Train OOF; the right-hand metrics are diagnostics on Validation.", "",
        "| method | minimum OOF recall | threshold | Validation precision | Validation recall | Validation F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in threshold_sweep:
        lines.append(
            f"| {row['method']} | {row['min_recall_oof']:.2f} | {row['threshold']:.6f} | "
            f"{row['validation_precision']:.4f} | {row['validation_recall']:.4f} | {row['validation_f1']:.4f} |"
        )
    lines += ["", configuration["limitation"]]
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {**result, "written": [str(path) for path in paths.values()]}
