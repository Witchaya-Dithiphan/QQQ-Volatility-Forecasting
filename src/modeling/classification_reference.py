"""Reference classifiers for M5 verification only: never used inside the scratch implementations or for selection."""
from __future__ import annotations
import traceback
from pathlib import Path
import numpy as np
from .datasets import DatasetSplit
from .artifacts import file_sha256, write_json, write_predictions
from .metrics import select_threshold
from .persistence import (_git_state, capture_artifacts, create_run_directory, load_npz, load_reference,
                          make_manifest, save_npz, save_reference, transition, verify_reload)
from .preprocessing import Standardizer

_NOTES = {"slp": "SGDClassifier(log_loss) is a per-sample (batch size 1) SGD reference for the scratch mini-batch SLP: behavioral "
                 "comparison only, not weight equality.",
          "logistic": "Constant-step per-sample SGD reference for the full-batch scratch objective: objective-behavioral comparison only."}


def _estimator(name: str, params: dict, config: dict, protocol: dict | None):
    from sklearn.linear_model import Perceptron as SKPerceptron, SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    seed, fixed = config["experiment"]["seed"], config["models"][name]["parameters"]
    if name == "logistic":
        l2 = params["l2"]
        return SGDClassifier(**config["references"]["logistic"]["parameters"], penalty=None if l2 == 0 else "l2", alpha=l2 or 0.0001,
                             class_weight=params["class_weight"], max_iter=fixed["max_iter"], tol=None)
    if name == "slp":
        l2 = params["l2"]
        return SGDClassifier(**{**config["references"]["slp"]["parameters"], "eta0": params["learning_rate"]}, penalty=None if l2 == 0 else "l2",
                             alpha=l2 or 0.0001, max_iter=protocol["selected_epochs"])
    if name == "gaussian_nb":
        return GaussianNB(var_smoothing=params["var_smoothing"])
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=params["k"], weights=params["weights"])
    if name == "perceptron":
        return SKPerceptron(eta0=params["learning_rate"], shuffle=False, max_iter=fixed["max_epochs"], tol=None, penalty=None, random_state=seed)
    return DecisionTreeClassifier(criterion="gini", max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"], random_state=seed)


def run_reference(name: str, params: dict, config: dict, train: DatasetSplit, validation: DatasetSplit, scratch_score, protocol: dict | None,
                  *, output_root, variant: str, run_id: str, hashes: dict) -> dict:
    """Fit the reference on the same features/preprocessing/seed and compare it with the scratch Validation scores."""
    from .classification_training import _identity

    path = create_run_directory(Path(output_root), variant, "classification", name, "reference", run_id)
    manifest = make_manifest(run_id=run_id, variant=variant, task="classification", model=name, implementation="reference",
                             lifecycle="supervised_tuned", hashes=hashes, role="finalized_reference", required=True)
    manifest["config_id"] = config["config_id"]
    manifest["git_dirty"], manifest["git_revision"] = _git_state()
    manifest["input_identity"] = {"train": _identity(train), "validation": _identity(validation)}
    transition(manifest, "running", stage="reference_fit")
    try:
        return _persist_reference(path, manifest, name, params, config, train, validation, scratch_score, protocol)
    except Exception as error:
        (path / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        transition(manifest, "failed", stage=manifest["stage"],
                   error={"exception_type": type(error).__name__, "message": str(error), "traceback_path": "traceback.txt"})
        write_json(path / "manifest.json", manifest)
        raise


def _persist_reference(path, manifest, name, params, config, train, validation, scratch_score, protocol):
    from .classification_training import _preprocessor, validation_metrics
    from .classification._common import scaler_from_dict

    scaler = None
    X, Xv = train.X, validation.X
    if config["models"][name]["preprocess"] == "standardize":
        scaler = Standardizer(config["preprocessing"]["standardizer"]["variance_floor"]).fit(X)
        X, Xv = scaler.transform(X), scaler.transform(Xv)
    estimator = _estimator(name, params, config, protocol).fit(X, train.y_classification)
    decision = name == "perceptron"
    score = estimator.decision_function(Xv) if decision else estimator.predict_proba(Xv)[:, 1]
    default = 0.0 if decision else 0.5
    y = validation.y_classification
    reference = validation_metrics(name, config, y, score, default)
    scratch = validation_metrics(name, config, y, scratch_score, default)
    result = {"verification_only": True, "constructor": type(estimator).__name__, "parameters": estimator.get_params(),
            "score_kind": "decision" if decision else "probability", "default_threshold": default,
            "scope_note": _NOTES.get(name, "Behavioral reference on identical features, preprocessing and seed."),
            "max_abs_score_difference": float(np.max(np.abs(score - scratch_score))),
            "label_agreement": float(np.mean((score >= default) == (scratch_score >= default))),
            "reference_metrics": {k: reference[k] for k in ("accuracy", "f1", "roc_auc", "average_precision")},
            "scratch_metrics": {k: scratch[k] for k in ("accuracy", "f1", "roc_auc", "average_precision")}}

    threshold = select_threshold(y, score, score_kind=result["score_kind"])
    if threshold["threshold"] is None:
        raise ValueError(f"Threshold selection failed: {threshold['reason']}")
    chosen = float(threshold["threshold"])
    manifest["stage"] = "persisting"
    write_json(path / "config.json", config)
    save_reference(path / "model.joblib", estimator)
    # The same M2 standardizer state format is used by scratch and reference runs.
    from types import SimpleNamespace
    arrays, state = _preprocessor(SimpleNamespace(scaler_=scaler), train.X.shape[1])
    save_npz(path / "preprocessor.npz", arrays, state)
    write_json(path / "search_results.json", {"verification_only": True, "scope_note": result["scope_note"],
               "selection_source": "finalized_scratch", "selected": {"params": params, "protocol": protocol},
               "candidates": [{"params": params, "metrics": reference}]})
    write_json(path / "validation_metrics.json", {"threshold": chosen, "score_kind": result["score_kind"],
               "default_threshold": default, "selection_split": "validation",
               "metrics": validation_metrics(name, config, y, score, chosen), "baseline": reference})
    write_predictions(path / "validation_predictions.csv", dates=validation.dates, target=y, raw=score,
                      task="classification", threshold=chosen)
    write_json(path / "reference_result.json", result)
    restored_estimator = load_reference(path / "model.joblib", expected_sha256=file_sha256(path / "model.joblib"))
    stored_arrays, stored_state = load_npz(path / "preprocessor.npz")
    restored_X = validation.X
    if stored_state["kind"] == "standardizer":
        restored_scaler = scaler_from_dict({k: v for k, v in stored_state.items() if k not in ("kind", "warnings")})
        restored_X = restored_scaler.transform(restored_X)
    after = restored_estimator.decision_function(restored_X) if decision else restored_estimator.predict_proba(restored_X)[:, 1]
    rtol, atol = config["persistence"]["reload_rtol"], config["persistence"]["reload_atol"]
    receipt = {"scores": verify_reload(score, after, rtol=rtol, atol=atol),
               "labels": verify_reload(score >= chosen, after >= chosen, labels=True),
               "model_predict": verify_reload(estimator.predict(Xv), restored_estimator.predict(restored_X), labels=True),
               "preprocessor": verify_reload(Xv, restored_X, rtol=rtol, atol=atol),
               "state_arrays": {"passed": set(arrays) == set(stored_arrays) and
                                all(np.array_equal(arrays[k], stored_arrays[k]) for k in arrays)}}
    receipt["passed"] = all(part["passed"] for part in receipt.values())
    write_json(path / "load_verification.json", receipt)
    manifest["load_verification"] = receipt
    manifest["selected_hyperparameters"] = params
    manifest["output_policy"] = {"threshold": chosen, "score_kind": result["score_kind"], "default_threshold": default,
                                 "selection_split": "validation", "rule": config["selection"]["threshold"]}
    capture_artifacts(manifest, path)
    transition(manifest, "completed", run_dir=path, stage="validation_complete")
    write_json(path / "manifest.json", manifest)
    return result
