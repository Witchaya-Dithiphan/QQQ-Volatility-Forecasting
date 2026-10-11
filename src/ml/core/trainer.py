"""Grid search on Validation, then save, reload and prove the model survived.

The Test split never appears here: it is read once, from `src.ml.run finalize`,
after every model's hyperparameters are locked.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT

from . import metrics as metrics_module
from .data import load
from .persist import load_npz, save_npz, verify_reload
from .preprocess import PCA, Standardizer

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "modeling"


def config_preprocess(name: str) -> str:
    """Which preprocessing configs/modeling.json specifies for this model.

    Seven models declare "none" (trees, gaussian_nb, simple_linear). Standardizing
    them anyway does not break parity — both sides see the same matrix — but it
    makes the report describe preprocessing incorrectly, and it rescales
    gaussian_nb's var_smoothing, which is defined relative to the largest feature
    variance.
    """
    config = json.loads((PROJECT_ROOT / "configs" / "modeling.json").read_text(encoding="utf-8"))
    return config.get("models", {}).get(name, {}).get("preprocess", "standardize")


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product in sorted key order so candidate ids are reproducible.

    A grid given as {"tuples": [...], "tuple_fields": [...]} lists whole candidates
    instead: each tuple becomes one candidate, in the order written.
    """
    if not grid:
        return [{}]
    if "tuples" in grid or "tuple_fields" in grid:
        if set(grid) != {"tuples", "tuple_fields"}:
            raise ValueError("A tuple grid needs exactly the keys 'tuples' and 'tuple_fields'")
        fields = grid["tuple_fields"]
        if any(len(values) != len(fields) for values in grid["tuples"]):
            raise ValueError(f"Every tuple must have one value per field: {fields}")
        return [dict(zip(fields, values)) for values in grid["tuples"]]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def write_json(path: Path, payload) -> None:
    """No `default=` on purpose: an unserializable value must fail, not be stringified.

    regression_outputs() returns NumPy arrays next to its metrics, and silently
    writing "[0.11 0.42 ...]" into validation_metrics.json would poison the report
    that every number in the paper traces back to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def evaluate(model, X, y_regression, y_classification, task: str):
    """Return (json-safe metrics, final prediction, score used for metrics)."""
    if task == "regression":
        outputs = metrics_module.regression_outputs(y_regression, model.predict(X))
        metrics = {
            "raw_metrics": outputs["raw_metrics"],
            "clipped_metrics": outputs["clipped_metrics"],
            "negative_count": outputs["negative_count"],
        }
        return metrics, outputs["final_prediction"], outputs["raw_prediction"]

    if task == "classification":
        try:
            scores = np.asarray(model.predict_proba(X), dtype=np.float64)
            kind = "probability"
        except NotImplementedError:
            scores = np.asarray(model.decision_function(X), dtype=np.float64)
            kind = "decision"
        if scores.ndim == 2 and scores.shape[1] == 2:
            scores = scores[:, 1]
        threshold = 0.5 if kind == "probability" else 0.0
        prediction = np.asarray(model.predict(X))
        metrics = metrics_module.classification_metrics(
            y_classification, scores, threshold=threshold, predicted_labels=prediction
        )
        return {**metrics, "score_kind": kind}, prediction, scores

    labels = np.asarray(model.predict(X))
    return metrics_module.clustering_metrics(X, labels), labels, labels


def train_model(
    model_class,
    *,
    grid: dict | None = None,
    variant: str = "with_spike",
    output_root: Path | None = None,
    pca_components: int | None = None,
    preprocess: str | None = None,
    seed: int = 42,
) -> dict:
    grid = grid or {}
    data = load(variant)
    task = model_class.task
    directory = Path(output_root or OUTPUT_ROOT) / variant / task / model_class.name
    directory.mkdir(parents=True, exist_ok=True)

    mode = preprocess or config_preprocess(model_class.name)
    # PCA needs centered input, so it implies standardising whatever the model asked for.
    scaled = mode != "none" or bool(pca_components)

    if scaled:
        standardizer = Standardizer().fit(data.train.X)
        train_X = standardizer.transform(data.train.X)
        validation_X = standardizer.transform(data.validation.X)
        preprocessor_arrays = {"mean": standardizer.mean_, "scale": standardizer.scale_,
                               "constant": standardizer.constant_}
    else:
        train_X, validation_X = data.train.X, data.validation.X
        n_features = data.train.X.shape[1]
        # Identity transform, stored so every run has the same artifact shape.
        preprocessor_arrays = {"mean": np.zeros(n_features), "scale": np.ones(n_features),
                               "constant": np.zeros(n_features, dtype=bool)}

    if pca_components:
        pca = PCA(pca_components).fit(train_X)
        train_X, validation_X = pca.transform(train_X), pca.transform(validation_X)
        preprocessor_arrays["components"] = pca.components_

    y_train = data.train.y_regression if task == "regression" else data.train.y_classification
    search_results = []
    def add_candidate(params, candidate_train_X, candidate_validation_X, arrays, components):
        candidate = model_class(**params).fit(
            candidate_train_X, None if task == "clustering" else y_train
        )
        candidate_metrics, prediction, scores = evaluate(
            candidate, candidate_validation_X,
            data.validation.y_regression, data.validation.y_classification, task
        )
        search_results.append(
            {
                "id": f"{model_class.name}-{len(search_results):02d}",
                "params": params,
                "metrics": candidate_metrics,
                "model": candidate,
                "prediction": prediction,
                "scores": scores,
                "pca_components": components,
                "validation_X": candidate_validation_X,
                "preprocessor_arrays": arrays,
                "status": "valid",
            }
        )

    for params in expand_grid(grid):
        add_candidate(params, train_X, validation_X, preprocessor_arrays, pca_components)

    if model_class.name == "random_forest" and pca_components is None:
        raw_best = metrics_module.select_candidate(
            [{"id": c["id"], "metrics": c["metrics"]} for c in search_results], task=task
        )
        raw_params = next(c["params"] for c in search_results if c["id"] == raw_best["id"])
        standardizer = Standardizer().fit(data.train.X)
        standardized_train = standardizer.transform(data.train.X)
        standardized_validation = standardizer.transform(data.validation.X)
        for components in (2, 4, 6):
            try:
                pca = PCA(components).fit(standardized_train)
            except ValueError as exc:
                if str(exc) != "PCA component count exceeds numerical rank":
                    raise
                search_results.append({
                    "id": f"{model_class.name}-{len(search_results):02d}",
                    "params": raw_params,
                    "metrics": {name: metrics_module.measure(reason=str(exc))
                                for name in ("average_precision", "roc_auc")},
                    "pca_components": components,
                    "status": "invalid",
                })
                continue
            arrays = {"mean": standardizer.mean_, "scale": standardizer.scale_,
                      "constant": standardizer.constant_, "components": pca.components_}
            add_candidate(raw_params, pca.transform(standardized_train),
                          pca.transform(standardized_validation), arrays, components)

    best = metrics_module.select_candidate(
        [{"id": c["id"], "metrics": c["metrics"]} for c in search_results], task=task
    )
    winner = next(c for c in search_results if c["id"] == best["id"])
    model = winner["model"]
    validation_X = winner["validation_X"]
    preprocessor_arrays = winner["preprocessor_arrays"]
    pca_components = winner["pca_components"]
    scaled = mode != "none" or bool(pca_components)

    save_npz(directory / "model.npz", model.state_dict(), {"params": model.get_params(), "seed": seed})
    save_npz(
        directory / "preprocessor.npz",
        preprocessor_arrays,
        {"preprocess": mode, "standardized": scaled, "pca_components": pca_components,
         "feature_names": data.feature_names},
    )

    arrays, metadata = load_npz(directory / "model.npz")
    reloaded = model_class().load_state(arrays, metadata)
    saved_preprocessor, saved_metadata = load_npz(directory / "preprocessor.npz")
    reloaded_validation_X = data.validation.X
    if saved_metadata["standardized"]:
        reloaded_validation_X = (reloaded_validation_X - saved_preprocessor["mean"]) / saved_preprocessor["scale"]
        reloaded_validation_X[:, saved_preprocessor["constant"].astype(bool)] = 0.0
    if saved_metadata["pca_components"]:
        reloaded_validation_X = reloaded_validation_X @ saved_preprocessor["components"].T
    verification = verify_reload(
        model.predict(validation_X), reloaded.predict(reloaded_validation_X), labels=task != "regression"
    )

    write_json(
        directory / "config.json",
        {
            "model": model_class.name,
            "task": task,
            "variant": variant,
            "seed": seed,
            "params": model.get_params(),
            "pca_components": pca_components,
            "preprocess": mode,
            "selected_candidate": winner["id"],
            "train_rows": int(len(data.train.X)),
            "train_sha256": data.train.sha256,
            "validation_sha256": data.validation.sha256,
        },
    )
    search_fields = ("id", "params", "metrics")
    if model_class.name == "random_forest":
        search_fields += ("pca_components", "status")
    write_json(directory / "search_results.json",
               [{k: c[k] for k in search_fields} for c in search_results])
    write_json(directory / "validation_metrics.json", winner["metrics"])
    write_json(directory / "load_verification.json", verification)
    if getattr(model, "history_", None):
        write_json(directory / "history.json", model.history_)

    target = data.validation.y_regression if task == "regression" else data.validation.y_classification
    with (directory / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Date", "target", "raw_score", "final_prediction"])
        for date, actual, score, prediction in zip(
            data.validation.dates, target, winner["scores"], winner["prediction"]
        ):
            writer.writerow([str(date)[:10], actual, score, prediction])

    return {
        "directory": directory,
        "model": model,
        "search_results": search_results,
        "validation_metrics": winner["metrics"],
        "load_verification": verification,
        "selected": winner["id"],
    }
