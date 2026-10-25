"""Post-submission diagnostics for the saved Kernel Perceptron experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

from config import PROJECT_ROOT

from .classification.kernel_perceptron import KernelPerceptron
from .classification.stacking import fold_plan
from .core.artifacts import preprocess, sha256_file
from .core.bootstrap import REPLICATES, block_indices, difference, interval, replicate
from .core.contracts import FEATURE_COLUMNS
from .core.data import load
from .core.importance import block_permutation_importance
from .core.metrics import loss
from .core.persist import load_npz
from .core.preprocess import Standardizer
from .core.report import classification_scores
from .core.trainer import trading_day_positions, write_json
from .finalize import segment_metrics


ROOT = PROJECT_ROOT / "outputs" / "experiments" / "kernel_perceptron"
LOCKED = PROJECT_ROOT / "outputs" / "modeling"
FIELDS = ("average_precision", "roc_auc", "f1", "recall")


def predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Date", "target", "raw_score", "final_prediction"}
    if not required <= set(frame) or frame["Date"].duplicated().any():
        raise ValueError(f"Invalid predictions: {path}")
    return frame


def aligned(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if (not reference["Date"].equals(candidate["Date"])
            or not np.array_equal(reference["target"].to_numpy(), candidate["target"].to_numpy())):
        raise ValueError("Prediction dates or targets differ")


def as_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(frame[column].to_numpy(dtype=np.float64)
                 for column in ("target", "raw_score", "final_prediction"))


def bootstrap_comparison(variant: str, split: str, kernel: pd.DataFrame) -> dict:
    selected = json.loads((LOCKED / "selection_lock.json").read_text(encoding="utf-8"))
    leader = selected["headline"][f"{variant}/classification"]
    references = {"linear_perceptron": "perceptron", "locked_headline": leader}
    frames = {"kernel_perceptron": kernel}
    for label, name in references.items():
        source = LOCKED / variant / "classification" / name / f"{split}_predictions.csv"
        frames[label] = predictions(source)
        aligned(kernel, frames[label])
    indices = block_indices(len(kernel), replicates=REPLICATES)
    scores = {name: classification_scores(*as_arrays(frame)) for name, frame in frames.items()}
    sampled = {name: replicate(classification_scores, as_arrays(frame), indices)
               for name, frame in frames.items()}
    return {
        "split": split, "variant": variant, "replicates": REPLICATES,
        "block_days": 20, "reference_models": references,
        "intervals": {name: {field: interval(scores[name][field], sampled[name][field])
                             for field in FIELDS} for name in frames},
        "paired_delta_vs_kernel": {
            name: {field: difference(scores[name][field], sampled[name][field],
                                     scores["kernel_perceptron"][field],
                                     sampled["kernel_perceptron"][field])
                   for field in FIELDS}
            for name in references
        },
    }


def walk_forward(variant: str, params: dict, expected_train_hash: str) -> list[dict]:
    train = load(variant).train
    if train.sha256 != expected_train_hash:
        raise ValueError(f"Train input changed: {variant}")
    positions, n_positions = trading_day_positions(variant, train.dates)
    rows, pooled = [], {"target": [], "score": [], "prediction": []}
    for fold, (history, block, start, stop) in enumerate(fold_plan(positions, n_positions, 4, 0.4, 5)):
        scaler = Standardizer().fit(train.X[history])
        model = KernelPerceptron(**params).fit(scaler.transform(train.X[history]),
                                                train.y_classification[history])
        score = model.decision_function(scaler.transform(train.X[block]))
        predicted = (score > 0).astype(np.int64)
        target = train.y_classification[block]
        measured = classification_scores(target, score, predicted)
        rows.append({"variant": variant, "model": "kernel_perceptron", "fold": fold,
                     "block_start": str(train.dates[block][0])[:10],
                     "block_end": str(train.dates[block][-1])[:10],
                     "block_start_position": start, "block_stop_position": stop,
                     "train_rows": int(history.sum()), "block_rows": int(block.sum()),
                     "positives": int(target.sum()), "epochs": model.n_epochs_,
                     "converged": model.converged_, **{field: measured[field] for field in FIELDS}})
        for name, value in (("target", target), ("score", score), ("prediction", predicted)):
            pooled[name].append(value)
    target, score, predicted = (np.concatenate(pooled[name])
                                for name in ("target", "score", "prediction"))
    measured = classification_scores(target, score, predicted)
    ci = interval(measured["average_precision"], replicate(
        classification_scores, (target, score, predicted), block_indices(len(target))
    )["average_precision"])
    rows.append({"variant": variant, "model": "kernel_perceptron", "fold": "pooled",
                 "block_start": rows[0]["block_start"], "block_end": rows[-1]["block_end"],
                 "train_rows": None, "block_rows": len(target), "positives": int(target.sum()),
                 "folds_used": len(rows), "epochs": None, "converged": None,
                 **{field: measured[field] for field in FIELDS},
                 "average_precision_low": ci["low"], "average_precision_high": ci["high"]})
    return rows


def feature_importance(variant: str, directory: Path, model: KernelPerceptron,
                       scaler: dict, scaler_metadata: dict) -> list[dict]:
    validation = load(variant).validation

    def score(X):
        transformed = preprocess(X, scaler, scaler_metadata)
        decision = model.decision_function(transformed)
        return classification_scores(validation.y_classification, decision,
                                     (decision > 0).astype(np.int64))["average_precision"]

    result = block_permutation_importance(score, validation.X)
    saved = predictions(directory / "validation_predictions.csv")
    if not np.isclose(result["baseline"], classification_scores(*as_arrays(saved))["average_precision"]):
        raise ValueError("Feature-importance baseline differs from saved Validation AP")
    return [{"variant": variant, "model": "kernel_perceptron", "task": "classification",
             "feature": feature, "metric": "average_precision", "baseline": result["baseline"],
             "importance_mean": float(result["mean"][index]),
             "importance_std": float(result["std"][index])}
            for index, feature in enumerate(FEATURE_COLUMNS)]


def reference_parity(variant: str, model: KernelPerceptron, scaler: dict,
                     scaler_metadata: dict, saved: pd.DataFrame) -> dict:
    validation = load(variant).validation
    X = preprocess(validation.X, scaler, scaler_metadata)
    reference = rbf_kernel(X, model.support_vectors_, gamma=model.gamma) @ model.alpha_ + model.intercept_
    actual = saved["raw_score"].to_numpy(dtype=np.float64)
    if (not np.array_equal(saved["Date"].to_numpy(dtype="datetime64[D]"), validation.dates)
            or not np.allclose(reference, actual, rtol=1e-5, atol=1e-8)
            or not np.array_equal((reference > 0).astype(int), saved["final_prediction"].to_numpy())):
        raise ValueError(f"sklearn RBF reference disagrees with saved predictions: {variant}")
    return {"variant": variant, "reference": "sklearn.metrics.pairwise.rbf_kernel",
            "samples": len(X), "support_vectors": len(model.alpha_),
            "max_abs_score_difference": float(np.max(np.abs(reference - actual))),
            "prediction_agreement": 1.0, "passed": True,
            "scope": "saved model inference; sklearn has no equivalent online Kernel Perceptron class"}


def training_reference(variant: str, model: KernelPerceptron, scaler: dict,
                       scaler_metadata: dict, history: dict, validation: pd.DataFrame) -> dict:
    data = load(variant)
    X = preprocess(data.train.X, scaler, scaler_metadata)
    labels = 2 * data.train.y_classification - 1
    kernel = rbf_kernel(X, gamma=model.gamma)
    alpha = np.zeros(len(X))
    intercept = 0.0
    counts = []
    for epoch in range(1, model.max_epochs + 1):
        updates = 0
        for index, label in enumerate(labels):
            # Recompute each decision from dual weights, independently of cached scores.
            if label * (kernel[index] @ alpha + intercept) <= 0:
                step = model.learning_rate * label
                alpha[index] += step
                intercept += step
                updates += 1
        counts.append(updates)
        if updates == 0:
            break
    val_X = preprocess(data.validation.X, scaler, scaler_metadata)
    scores = rbf_kernel(val_X, X, gamma=model.gamma) @ alpha + intercept
    saved = validation["raw_score"].to_numpy(dtype=np.float64)
    mismatches = [i + 1 for i, (original, reference) in enumerate(zip(history["updates"], counts))
                  if original != reference]
    return {"variant": variant, "reference": "sklearn RBF matrix + recomputed dual decision at each step",
            "epochs": epoch, "converged": updates == 0,
            "train_accuracy": float(np.mean((kernel @ alpha + intercept > 0) == data.train.y_classification)),
            "saved_epochs": model.n_epochs_, "same_update_history": counts == history["updates"],
            "first_update_count_mismatch_epoch": mismatches[0] if mismatches else None,
            "validation_average_precision": classification_scores(
                data.validation.y_classification, scores, (scores > 0).astype(int)
            )["average_precision"],
            "saved_validation_average_precision": classification_scores(*as_arrays(validation))["average_precision"],
            "validation_label_agreement": float(np.mean((scores > 0) == (saved > 0))),
            "interpretation": "floating-point order can change updates near zero; both runs still converge"}


def test_segments(variant: str, saved: pd.DataFrame) -> list[dict]:
    source = predictions(LOCKED / variant / "classification" / "perceptron" / "test_predictions.csv")
    aligned(saved, source)
    segments = source["diagnostic_segment"].to_numpy(dtype=str)
    if not np.isin(segments, ["spike_affected", "non_spike"]).all():
        raise ValueError("Missing diagnostic Test segments")
    values = segment_metrics("classification", *as_arrays(saved), segments)
    return [{"variant": variant, "model": "kernel_perceptron", "task": "classification",
             "segment": segment,
             **{key: item["value"] if isinstance(item, dict) else item
                for key, item in metrics.items() if key != "confusion_matrix"},
             "confusion_matrix": metrics["confusion_matrix"]}
            for segment, metrics in values.items()]


def supplemental_leaderboard(variant: str, split: str, directory: Path, report: Path) -> None:
    source = (LOCKED / "report" if split == "validation" else LOCKED / "final") / variant
    original = pd.read_csv(source / "leaderboard_classification.csv")
    original["study"] = "locked study"
    metrics = json.loads((directory / f"{split}_metrics.json").read_text(encoding="utf-8"))
    bootstrap = json.loads((report / f"bootstrap_{split}_{variant}.json").read_text(encoding="utf-8"))
    row = {"model": "kernel_perceptron", "kind": "model", "variant": variant,
           "study": "post-submission", "score_kind": "decision"}
    for field in ("average_precision", "roc_auc", "f1", "recall", "precision", "specificity", "accuracy"):
        row[field] = metrics[field]["value"]
    for field in FIELDS:
        ci = bootstrap["intervals"]["kernel_perceptron"][field]
        row[f"{field}_low"], row[f"{field}_high"] = ci["low"], ci["high"]
        row[f"{field}_valid_replicates"] = ci["n_valid"]
        row[f"{field}_ci"] = f"{row[field]:.4f} [{ci['low']:.4f}, {ci['high']:.4f}]"
    (row["tn"], row["fp"]), (row["fn"], row["tp"]) = metrics["confusion_matrix"]
    row["sensitivity"], row["tnr"] = row["recall"], row["specificity"]
    gap = bootstrap["paired_delta_vs_kernel"]["locked_headline"]["average_precision"]
    row["gap_to_leader"], row["gap_low"], row["gap_high"] = gap["point"], gap["low"], gap["high"]
    row["tied_with_leader"] = gap["excludes_zero"] is not True
    row["best_baseline"] = original.iloc[0]["best_baseline"]
    baseline = original.set_index("model").loc[row["best_baseline"], "average_precision"]
    row["beats_best_baseline"] = bool(row["average_precision"] > baseline)
    if split == "validation":
        frame = predictions(directory / "validation_predictions.csv")
        row["loss_name"] = "perceptron_criterion"
        row["loss"] = loss("perceptron_criterion", frame["target"], scores=frame["raw_score"])["value"]
    if split == "test":
        row["validation_average_precision"] = json.loads(
            (directory / "validation_metrics.json").read_text(encoding="utf-8")
        )["average_precision"]["value"]
        row["headline"] = False
    combined = pd.concat([original, pd.DataFrame([row])], ignore_index=True)
    combined = combined.sort_values("average_precision", ascending=False).reset_index(drop=True)
    combined.insert(0, "ap_rank", combined["average_precision"].rank(method="min", ascending=False).astype(int))
    combined.to_csv(report / f"leaderboard_{split}_{variant}.csv", index=False)


def main() -> None:
    receipt = json.loads((ROOT / "test_receipt.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    if {row["variant"] for row in summary} != {"with_spike", "non_spike"}:
        raise ValueError("Expected selected Kernel Perceptron runs for both variants")
    report = ROOT / "report"
    report.mkdir(exist_ok=True)
    all_walk, all_importance, all_segments, all_parity, all_training_reference = [], [], [], [], []
    for row in summary:
        variant = row["variant"]
        directory = ROOT / variant / "classification" / "kernel_perceptron"
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["selected_candidate"] != row["selected"]:
            raise ValueError(f"Selected candidate changed: {variant}")
        sealed = next(run for run in receipt["runs"] if run["variant"] == variant)
        if (sha256_file(directory / "config.json") != sealed["config_sha256"]
                or sha256_file(directory / "model.npz") != sealed["model_sha256"]):
            raise ValueError(f"Selected model differs from Test receipt: {variant}")
        arrays, metadata = load_npz(directory / "model.npz")
        model = KernelPerceptron().load_state(arrays, metadata)
        scaler, scaler_metadata = load_npz(directory / "preprocessor.npz")
        validation = predictions(directory / "validation_predictions.csv")
        test = predictions(directory / "test_predictions.csv")
        if len(test) != receipt["test_rows"]:
            raise ValueError(f"Test rows differ from receipt: {variant}")
        all_walk.extend(walk_forward(variant, config["params"], config["train_sha256"]))
        all_importance.extend(feature_importance(variant, directory, model, scaler, scaler_metadata))
        all_segments.extend(test_segments(variant, test))
        all_parity.append(reference_parity(variant, model, scaler, scaler_metadata, validation))
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        all_training_reference.append(training_reference(
            variant, model, scaler, scaler_metadata, history, validation
        ))
        for split, frame in (("validation", validation), ("test", test)):
            write_json(report / f"bootstrap_{split}_{variant}.json",
                       bootstrap_comparison(variant, split, frame))
            supplemental_leaderboard(variant, split, directory, report)
        print(f"{variant}: diagnostics completed")
    pd.DataFrame(all_walk).to_csv(report / "walk_forward.csv", index=False)
    pd.DataFrame(all_importance).to_csv(report / "feature_importance.csv", index=False)
    pd.DataFrame(all_segments).to_csv(report / "segmented_metrics.csv", index=False)
    write_json(report / "reference_parity.json", all_parity)
    write_json(report / "training_reference.json", all_training_reference)
    write_json(report / "provenance.json", {
        "test_receipt_sha256": sha256_file(ROOT / "test_receipt.json"),
        "source": "saved predictions only for Test; no Test refit or hyperparameter selection",
        "bootstrap_replicates": REPLICATES,
    })


if __name__ == "__main__":
    main()
