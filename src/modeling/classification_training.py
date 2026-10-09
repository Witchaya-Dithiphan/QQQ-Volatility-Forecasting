"""M5 executable Train/Validation workflow for the six scratch classifiers (Test is never loaded).

Per model and variant: read the frozen grid from configs/modeling.json, fit every candidate on Train, score the full
Validation split, select the candidate by Average Precision then ROC-AUC (M2 `select_candidate`), pick the Validation
threshold (M2 `select_threshold`), then persist a `supervised_tuned` run directory through the M2 persistence primitives
and verify that a reload reproduces raw scores and labels.

SLP follows the frozen chronological protocol (`config["neural"]`): a chronological tail of Train selects the epoch
count (scaler fit on the sub-train only, five original-trading-day purge, BCE monitor with min_delta/patience, earliest
best epoch), then the model is re-initialised with the refit seed and refit on the full variant Train for exactly that
many epochs without early stopping. External Validation never reaches this selection.
"""
from __future__ import annotations
import traceback
from pathlib import Path
import numpy as np
from config import PROJECT_ROOT
from .artifacts import write_json, write_predictions
from .classification import DecisionTreeClassifier, GaussianNaiveBayes, KNN, LogisticRegression, Perceptron, SLP
from .classification._common import scaler_from_dict, scaler_to_dict
from .configuration import load_config
from .datasets import DatasetSplit, TrainValidation, load_train_validation
from .expected_runs import _candidates
from .metrics import classification_metrics, loss, select_candidate, select_threshold
from .persistence import _git_state, capture_artifacts, classify_run_directory, create_run_directory, load_npz, make_manifest, read_manifest, resume_check, save_npz, transition, verify_reload

M5_MODELS = ("logistic", "gaussian_nb", "knn", "perceptron", "slp", "decision_tree")
_CLASSES = {"logistic": LogisticRegression, "gaussian_nb": GaussianNaiveBayes, "knn": KNN, "perceptron": Perceptron,
            "slp": SLP, "decision_tree": DecisionTreeClassifier}
_METRIC_FIELDS = ("confusion_matrix", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "loss")


# ---- SLP chronological selection / refit protocol ----------------------------------------------

def slp_selection_split(dates, original_dates, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (sub-train, tail) of the variant Train; the purge is counted on the ORIGINAL trading timeline."""
    neural = config["neural"]
    dates, original = np.asarray(dates), np.asarray(original_dates)
    position = np.searchsorted(original, dates)
    if (position >= len(original)).any() or not np.array_equal(original[position], dates):
        raise ValueError("Train dates must lie on the original trading-day timeline")
    n_tail = int(np.ceil(neural["internal_tail_fraction"] * len(dates) - 1e-9))
    if not 1 <= n_tail < len(dates):
        raise ValueError("Invalid internal tail size")
    tail = np.arange(len(dates) - n_tail, len(dates))
    subtrain = np.flatnonzero(position < position[tail[0]] - int(neural["purge_original_trading_days"]))
    if not len(subtrain):
        raise ValueError("No sub-train rows remain after the tail and purge")
    return subtrain, tail


def _slp(params: dict, config: dict, *, random_state: int, max_iter: int | None = None, early_stopping: bool = True) -> SLP:
    model, neural = config["models"]["slp"], config["neural"]
    return SLP(learning_rate=params["learning_rate"], max_iter=max_iter or model["parameters"]["max_epochs"],
               batch_size=None if params["batch"] == "full" else params["batch"], l2=params["l2"], patience=neural["patience"],
               min_delta=neural["min_delta"], standardize=model["preprocess"] == "standardize", random_state=random_state,
               early_stopping=early_stopping)


def select_slp_epochs(params: dict, train: DatasetSplit, original_dates, config: dict) -> dict:
    """Epoch count from the internal chronological tail. Takes no Validation argument by design."""
    subtrain, tail = slp_selection_split(train.dates, original_dates, config)
    model = _slp(params, config, random_state=config["experiment"]["seed"])
    y = train.y_classification
    model.fit(train.X[subtrain], y[subtrain], eval_set=(train.X[tail], y[tail]))  # scaler is fit on the sub-train only
    return {"selected_epochs": model.best_epoch_, "selection_model": model, "subtrain_idx": subtrain, "tail_idx": tail}


def m5_status(out_root, variant, name, run_id, *, hashes=None, implementation='scratch'):
    import re
    if name not in M5_MODELS or variant not in ('with_spike', 'non_spike') or implementation not in ('scratch', 'reference'):
        raise ValueError('Invalid M5 run identity')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run ID')
    path = Path(out_root) / variant / 'classification' / name / implementation / run_id
    if not path.exists():
        return {'can_skip': False, 'status': 'missing', 'mismatches': {}}
    cl = classify_run_directory(path)
    if cl['status'] != 'completed':
        return {'can_skip': False, 'status': 'legacy_incompatible', 'mismatches': {'classification': cl}}
    manifest = read_manifest(path)
    if hashes is None:
        from .runner import current_hashes
        hashes = current_hashes(variant)
    return resume_check(manifest, hashes, path)


# ---- candidate fitting ---------------------------------------------------------------------------

def build_model(name: str, params: dict, config: dict):
    model, seed = config["models"][name], config["experiment"]["seed"]
    fixed, standardize = model["parameters"], model["preprocess"] == "standardize"
    if name == "logistic":
        return LogisticRegression(fixed["learning_rate"], fixed["max_iter"], fixed["tol"], params["l2"], params["class_weight"], standardize, seed)
    if name == "gaussian_nb":
        return GaussianNaiveBayes(params["var_smoothing"])
    if name == "knn":
        return KNN(params["k"], params["weights"], standardize)
    if name == "perceptron":
        return Perceptron(params["learning_rate"], fixed["max_epochs"], standardize)
    if name == "decision_tree":
        return DecisionTreeClassifier(params["max_depth"], min_samples_leaf=params["min_samples_leaf"], random_state=seed)
    raise ValueError(f"Unsupported M5 classifier: {name}")


def fit_candidate(name: str, params: dict, train: DatasetSplit, original_dates, config: dict):
    """Fit one frozen candidate on Train only; returns (fitted model, protocol info)."""
    y = train.y_classification
    if name != "slp":
        return build_model(name, params, config).fit(train.X, y), None
    neural = config["neural"]
    selection = select_slp_epochs(params, train, original_dates, config)
    epochs = selection["selected_epochs"]
    model = _slp(params, config, random_state=neural["refit_seed"], max_iter=epochs, early_stopping=neural["refit_early_stopping"])
    model.fit(train.X, y)  # fresh scaler and weights on the full variant Train
    info = {"selected_epochs": epochs, "epochs_run_selection": selection["selection_model"].epochs_run_,
            "subtrain_rows": len(selection["subtrain_idx"]), "tail_rows": len(selection["tail_idx"]),
            "purged_rows": len(train.X) - len(selection["subtrain_idx"]) - len(selection["tail_idx"]),
            "refit_rows": len(train.X), "refit_seed": neural["refit_seed"], "refit_epochs_run": model.epochs_run_}
    return model, info


def score_of(name: str, model, X) -> np.ndarray:
    """Raw score on the M2 scale: decision score for the Perceptron, otherwise P(class 1)."""
    return model.decision_function(X) if name == "perceptron" else model.predict_proba(X)[:, 1]


def validation_metrics(name: str, config: dict, y, score, threshold: float) -> dict:
    """Full Validation metrics with the mandatory configured loss on its genuine output scale."""
    metrics = classification_metrics(y, score, threshold=threshold)
    loss_name = config["losses"][name]
    kwargs = {"scores": score} if name == "perceptron" else {"probabilities": score}
    metrics["loss"] = {"name": loss_name, **loss(loss_name, y, **kwargs)}
    return metrics


def _compact(metrics: dict) -> dict:
    return {key: metrics[key] for key in _METRIC_FIELDS}


# ---- persistence helpers -------------------------------------------------------------------------

def _tree_arrays(root) -> dict:
    rows = []
    def walk(node):
        index = len(rows)
        rows.append(None)
        if node.is_leaf():
            rows[index] = (-1, 0.0, -1, -1, *node.value)
        else:
            left = walk(node.left)
            rows[index] = (node.feature, node.threshold, left, walk(node.right), 0, 0)
        return index
    walk(root)
    table = np.array(rows, dtype=np.float64)
    return {"node_feature": table[:, 0].astype(np.int64), "node_threshold": table[:, 1], "node_left": table[:, 2].astype(np.int64),
            "node_right": table[:, 3].astype(np.int64), "leaf_counts": table[:, 4:].astype(np.int64)}


def _model_arrays(model) -> dict:
    if isinstance(model, DecisionTreeClassifier):
        return _tree_arrays(model.tree_)
    arrays = {}
    for key, value in model.to_dict().items():
        if isinstance(value, dict) or value is None or isinstance(value, (bool, str)):
            continue
        array = np.asarray(value)
        if array.dtype.kind in "fiu":
            arrays[key] = array
    return arrays


def _preprocessor(model, n_features: int) -> tuple[dict, dict]:
    scaler = getattr(model, "scaler_", None)
    if scaler is None:
        return {"feature_count": np.array([n_features])}, {"kind": "none"}
    arrays = {"mean": scaler.mean_, "variance": scaler.variance_, "scale": scaler.scale_, "constant": scaler.constant_}
    return arrays, {"kind": "standardizer", "warnings": list(scaler.warnings_), **scaler_to_dict(scaler)}


def _identity(split: DatasetSplit) -> dict:
    source = Path(split.source)
    try:
        source = source.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    return {"path": source.as_posix(), "sha256": split.sha256, "rows": len(split.X),
            "date_range": {"start": str(split.dates[0])[:10], "end": str(split.dates[-1])[:10]}}


def _verify_reload(path: Path, name: str, model, inputs: TrainValidation, config: dict, score: np.ndarray, threshold: float) -> dict:
    stored, state = load_npz(path / "model.npz")
    reloaded = _CLASSES[name].from_dict(state)
    after = score_of(name, reloaded, inputs.validation.X)
    rtol, atol = config["persistence"]["reload_rtol"], config["persistence"]["reload_atol"]
    expected = _model_arrays(reloaded)
    arrays_ok = set(stored) == set(expected) and all(np.array_equal(stored[key], expected[key]) for key in stored)
    receipt = {"scores": verify_reload(score, after, rtol=rtol, atol=atol),
               "labels": verify_reload(score >= threshold, after >= threshold, labels=True),
               "model_predict": verify_reload(model.predict(inputs.validation.X), reloaded.predict(inputs.validation.X), labels=True),
               "state_arrays": {"passed": bool(arrays_ok), "names": sorted(stored)}}
    scaler_arrays, scaler_state = load_npz(path / "preprocessor.npz")
    if scaler_state["kind"] == "standardizer":
        restored = scaler_from_dict({key: value for key, value in scaler_state.items() if key not in ("kind", "warnings")})
        if not all(np.array_equal(scaler_arrays[k], getattr(restored, a)) for k, a in (("mean", "mean_"), ("variance", "variance_"), ("scale", "scale_"), ("constant", "constant_"))):
            receipt["state_arrays"]["passed"] = False
        receipt["preprocessor"] = verify_reload(model.scaler_.transform(inputs.validation.X), restored.transform(inputs.validation.X), rtol=rtol, atol=atol)
    receipt["passed"] = all(part["passed"] for part in receipt.values())
    return receipt


# ---- orchestration ---------------------------------------------------------------------------------

def train_classifier(name: str, variant: str, output_root, run_id: str, *, config: dict | None = None, inputs: TrainValidation | None = None,
                     original_dates=None, hashes: dict | None = None, reference: bool = True) -> Path:
    """Run the frozen candidate search for one M5 classifier and persist a verified `supervised_tuned` run."""
    if name not in M5_MODELS:
        raise ValueError(f"Unsupported M5 classifier: {name}")
    config = config or load_config()
    inputs = inputs or load_train_validation(variant)
    if inputs.variant != variant:
        raise ValueError("Inputs do not match the requested variant")
    if original_dates is None:
        original_dates = inputs.train.dates if variant == "with_spike" else load_train_validation("with_spike").train.dates
    if hashes is None:
        from .runner import current_hashes
        hashes = current_hashes(variant)
    train, validation = inputs.train, inputs.validation
    path = create_run_directory(Path(output_root), variant, "classification", name, "scratch", run_id)
    manifest = make_manifest(run_id=run_id, variant=variant, task="classification", model=name, implementation="scratch",
                             lifecycle="supervised_tuned", hashes=hashes, role="finalized_scratch", required=True)
    manifest["config_id"] = config["config_id"]
    manifest["git_dirty"], manifest["git_revision"] = _git_state()
    manifest["input_identity"] = {"train": _identity(train), "validation": _identity(validation)}
    transition(manifest, "running", stage="candidate_search")
    try:
        kind, default = ("decision", 0.0) if name == "perceptron" else ("probability", 0.5)
        records, best_fit = [], {}
        for index, params in enumerate(_candidates(config["models"][name])):
            record = {"id": f"c{index:02d}", "params": params, "metrics": None, "protocol": None, "error": None}
            try:
                model, record["protocol"] = fit_candidate(name, params, train, original_dates, config)
                score = score_of(name, model, validation.X)
                record["metrics"] = _compact(validation_metrics(name, config, validation.y_classification, score, default))
                best_fit[record["id"]] = (model, score)
            except ValueError as error:
                record["error"] = str(error)
            records.append(record)
        selected = select_candidate([{"id": r["id"], "metrics": r["metrics"]} for r in records if r["error"] is None], task="classification")
        record = next(r for r in records if r["id"] == selected["id"])
        model, score = best_fit[record["id"]]
        threshold = select_threshold(validation.y_classification, score, score_kind=kind)
        if threshold["threshold"] is None:
            raise ValueError(f"Threshold selection failed: {threshold['reason']}")
        chosen = float(threshold["threshold"])

        manifest["stage"] = "persisting"
        write_json(path / "config.json", config)
        write_json(path / "search_results.json", {
            "selection_rule": config["selection"]["classification"], "threshold_rule": config["selection"]["threshold"], "score_kind": kind,
            "selected": {"id": record["id"], "params": record["params"], "protocol": record["protocol"]}, "candidates": records})
        save_npz(path / "model.npz", _model_arrays(model), model.to_dict())  # model.json is the strict-JSON state sidecar
        arrays, state = _preprocessor(model, train.X.shape[1])
        save_npz(path / "preprocessor.npz", arrays, state)
        write_json(path / "validation_metrics.json", {"threshold": chosen, "score_kind": kind, "default_threshold": default,
                                                      "selection_split": threshold["selection_split"],
                                                      "metrics": validation_metrics(name, config, validation.y_classification, score, chosen),
                                                      "baseline": validation_metrics(name, config, validation.y_classification, score, default)})
        write_predictions(path / "validation_predictions.csv", dates=validation.dates, target=validation.y_classification, raw=score,
                          task="classification", threshold=chosen)
        if reference:
            from .classification_reference import run_reference
            write_json(path / "reference_result.json", run_reference(name, record["params"], config, train, validation, score, record["protocol"],
                                                                             output_root=output_root, variant=variant, run_id=run_id, hashes=hashes))
        receipt = _verify_reload(path, name, model, inputs, config, score, chosen)
        manifest["load_verification"] = receipt
        write_json(path / "load_verification.json", receipt)
        manifest["selected_hyperparameters"] = record["params"]
        manifest["output_policy"] = {"threshold": chosen, "score_kind": kind, "default_threshold": default,
                                     "selection_split": threshold["selection_split"], "rule": config["selection"]["threshold"]}
        capture_artifacts(manifest, path)
        transition(manifest, "completed", run_dir=path, stage="validation_complete")
    except Exception as error:
        (path / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        transition(manifest, "failed", stage=manifest["stage"], error={"exception_type": type(error).__name__, "message": str(error), "traceback_path": "traceback.txt"})
        write_json(path / "manifest.json", manifest)
        raise
    write_json(path / "manifest.json", manifest)
    return path
