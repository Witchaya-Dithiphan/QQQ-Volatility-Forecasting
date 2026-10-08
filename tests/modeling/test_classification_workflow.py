"""M5 frozen SLP early-stopping/refit protocol and the executable Train/Validation workflow (no Test access)."""
import copy
import dataclasses
import inspect
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.modeling.classification_training as training
from src.modeling.classification import SLP
from src.modeling.classification_training import M5_MODELS, fit_candidate, select_slp_epochs, slp_selection_split, train_classifier
from src.modeling.configuration import load_config
from src.modeling.datasets import DatasetSplit, TrainValidation, load_train_validation
from src.modeling.expected_runs import _candidates
from src.modeling.metrics import classification_metrics, select_candidate, select_threshold
from src.modeling.persistence import read_manifest, resume_check

HASHES = {"config": "c", "inputs": "i", "code": "k", "dependency_lock": "d", "runtime": "r"}
CONFIG = load_config()


def reduced_config(max_epochs=40):
    """Frozen config with grids cut to <= 2 values per axis and short epoch caps (production code is unchanged)."""
    config = copy.deepcopy(CONFIG)
    for name in M5_MODELS:
        model = config["models"][name]
        model["grid"] = {k: v[:2] for k, v in model["grid"].items()}
        for key in ("max_epochs", "max_iter"):
            if key in model["parameters"]:
                model["parameters"][key] = min(model["parameters"][key], max_epochs)
    return config


def split(n, seed, start="2020-01-01", d=4, positions=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0.3).astype(np.int64)
    days = pd.bdate_range(start, periods=n if positions is None else int(positions.max()) + 1).to_numpy()
    dates = days if positions is None else days[positions]
    return DatasetSplit(dates, X, np.abs(X[:, 0]), y, "0" * 64, "synthetic")


def synthetic(n_train=160, n_val=70, seed=0):
    train = split(n_train, seed)
    validation = split(n_val, seed + 1, start=str(pd.Timestamp(train.dates[-1]) + pd.Timedelta(days=30))[:10])
    return TrainValidation("with_spike", train, validation)


# ---- chronological tail + purge on the ORIGINAL trading timeline ------------------------------

def test_split_with_spike_tail_and_exact_five_day_purge():
    dates = pd.bdate_range("2020-01-01", periods=100).to_numpy()
    subtrain, tail = slp_selection_split(dates, dates, CONFIG)
    np.testing.assert_array_equal(tail, np.arange(85, 100))      # ceil(15% of 100)
    np.testing.assert_array_equal(subtrain, np.arange(0, 80))    # positions 80..84 purged
    assert CONFIG["neural"]["internal_tail_fraction"] == 0.15 and CONFIG["neural"]["purge_original_trading_days"] == 5


def test_split_purges_original_days_not_variant_rows():
    original = pd.bdate_range("2020-01-01", periods=90).to_numpy()
    kept = np.r_[0:68, 78:90]                                    # a removed spike window immediately before the tail
    dates = original[kept]
    subtrain, tail = slp_selection_split(dates, original, CONFIG)
    assert len(tail) == 12 and kept[tail[0]] == 78
    assert (kept[subtrain] < 73).all() and len(subtrain) == 68   # row-gap purging would have dropped 5 more rows
    assert len(np.intersect1d(subtrain, tail)) == 0


def test_split_requires_variant_dates_inside_original_timeline():
    original = pd.bdate_range("2020-01-01", periods=50).to_numpy()
    with pytest.raises(ValueError, match="original"):
        slp_selection_split(original[[0, 1, 2]] + np.timedelta64(1, "D"), original, CONFIG)


# ---- selection fit: subtrain-only scaler, BCE monitor, min_delta, patience, earliest epoch -----

PARAMS = {"batch": 32, "l2": 0.0, "learning_rate": 0.01}


def test_selection_scaler_and_weights_ignore_purged_and_tail_rows():
    train = synthetic().train
    base = select_slp_epochs(PARAMS, train, train.dates, CONFIG)
    sub = base["subtrain_idx"]
    np.testing.assert_allclose(base["selection_model"].scaler_.mean_, train.X[sub].mean(axis=0))
    assert not np.allclose(base["selection_model"].scaler_.mean_, train.X.mean(axis=0))
    purged = np.setdiff1d(np.arange(len(train.X)), np.r_[sub, base["tail_idx"]])
    assert len(purged) == 5
    X2 = train.X.copy()
    X2[purged] += 1e3                                            # purged rows must be invisible
    again = select_slp_epochs(PARAMS, dataclasses.replace(train, X=X2), train.dates, CONFIG)
    np.testing.assert_array_equal(again["selection_model"].coefficients_, base["selection_model"].coefficients_)
    X3 = train.X.copy()
    X3[base["tail_idx"]] += 1e3                                  # the tail only monitors; it never fits the scaler
    tailed = select_slp_epochs(PARAMS, dataclasses.replace(train, X=X3), train.dates, CONFIG)
    np.testing.assert_array_equal(tailed["selection_model"].scaler_.mean_, base["selection_model"].scaler_.mean_)
    assert tailed["selection_model"].loss_history_[0] == base["selection_model"].loss_history_[0]


def test_selected_epoch_replays_frozen_min_delta_patience_rule():
    train = synthetic().train
    result = select_slp_epochs(PARAMS, train, train.dates, CONFIG)
    neural = CONFIG["neural"]
    history = result["selection_model"].val_loss_history_
    best, best_epoch = np.inf, 0
    for epoch, value in enumerate(history, 1):
        if value < best - neural["min_delta"]:
            best, best_epoch = value, epoch
        elif epoch - best_epoch >= neural["patience"]:
            break
    assert result["selected_epochs"] == best_epoch >= 1
    assert result["selection_model"].epochs_run_ in (len(history), best_epoch + neural["patience"])


def test_config_patience_and_min_delta_control_selection():
    train = synthetic().train
    config = copy.deepcopy(CONFIG)
    config["neural"]["patience"] = 2
    flipped = dataclasses.replace(train, y_classification=train.y_classification.copy())
    _, tail = slp_selection_split(train.dates, train.dates, config)
    flipped.y_classification[tail] = 1 - flipped.y_classification[tail]   # tail loss rises immediately
    result = select_slp_epochs(PARAMS, flipped, train.dates, config)
    assert result["selected_epochs"] == 1 and result["selection_model"].epochs_run_ == 3
    config["neural"]["min_delta"] = 10.0                                  # nothing improves by 10 nats after epoch 1
    assert select_slp_epochs(PARAMS, train, train.dates, config)["selected_epochs"] == 1


def test_slp_primitive_min_delta_and_disabled_early_stopping():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] > 0).astype(int)
    strict = SLP(learning_rate=0.01, max_iter=100, patience=3, min_delta=10.0).fit(X, y, eval_set=(X, y))
    assert strict.best_epoch_ == 1 and strict.epochs_run_ == 4
    fixed = SLP(learning_rate=0.01, max_iter=7, patience=1, early_stopping=False).fit(X, y, eval_set=(X, 1 - y))
    assert fixed.epochs_run_ == 7 and fixed.best_epoch_ == 7       # no stop, final-epoch weights kept


# ---- refit on full variant Train for exactly the selected epochs -------------------------------

def test_refit_uses_full_train_scaler_seed_and_exact_epochs_without_early_stopping():
    train = synthetic().train
    config = copy.deepcopy(CONFIG)
    config["neural"]["refit_seed"] = 7
    model, info = fit_candidate("slp", PARAMS, train, train.dates, config)
    epochs = select_slp_epochs(PARAMS, train, train.dates, config)["selected_epochs"]
    assert info["selected_epochs"] == epochs and model.epochs_run_ == epochs and model.best_epoch_ == epochs
    np.testing.assert_allclose(model.scaler_.mean_, train.X.mean(axis=0))          # refit scaler sees full Train
    manual = SLP(learning_rate=0.01, max_iter=epochs, batch_size=32, l2=0.0, early_stopping=False, random_state=7).fit(train.X, train.y_classification)
    np.testing.assert_array_equal(model.coefficients_, manual.coefficients_)
    assert model.intercept_ == manual.intercept_


def test_external_validation_cannot_control_epochs():
    assert "validation" not in " ".join(inspect.signature(select_slp_epochs).parameters)
    assert "validation" not in " ".join(inspect.signature(fit_candidate).parameters)


# ---- executable workflow -----------------------------------------------------------------------

def run(tmp_path, name, inputs=None, config=None, run_id="r1", **kw):
    inputs = inputs or synthetic()
    return train_classifier(name, inputs.variant, tmp_path, run_id, config=config or reduced_config(), inputs=inputs,
                            original_dates=inputs.train.dates, hashes=HASHES, **kw)


@pytest.mark.parametrize("name", M5_MODELS)
def test_workflow_selects_persists_reloads_and_is_resumable(name, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = run(tmp_path, name)
    config, inputs = reduced_config(), synthetic()
    manifest = read_manifest(path)
    assert manifest["status"] == "completed" and manifest["lifecycle"] == "supervised_tuned" and not manifest["test_accessed"]
    assert (manifest["task"], manifest["model"], manifest["implementation"]) == ("classification", name, "scratch")
    assert resume_check(manifest, HASHES, path)["can_skip"]
    for artifact in ("config.json", "model.npz", "model.json", "preprocessor.npz", "preprocessor.json", "search_results.json",
                     "validation_metrics.json", "validation_predictions.csv", "load_verification.json", "reference_result.json"):
        assert (path / artifact).is_file(), artifact
    receipt = json.loads((path / "load_verification.json").read_text())
    assert receipt["passed"] and receipt["labels"]["passed"] and receipt["labels"]["labels_exact"] and receipt["scores"]["passed"]

    search = json.loads((path / "search_results.json").read_text())
    grid = _candidates(config["models"][name])
    assert [c["params"] for c in search["candidates"]] == grid
    valid = [c for c in search["candidates"] if c["error"] is None]
    best = select_candidate([{"id": c["id"], "metrics": c["metrics"]} for c in valid], task="classification")
    assert search["selected"]["id"] == best["id"]
    assert manifest["selected_hyperparameters"] == search["selected"]["params"]

    predictions = pd.read_csv(path / "validation_predictions.csv", float_precision="round_trip")
    kind = "decision" if name == "perceptron" else "probability"
    policy = manifest["output_policy"]
    assert policy["score_kind"] == kind and type(policy["threshold"]) is float
    expected = select_threshold(inputs.validation.y_classification, predictions["raw_score"].to_numpy(), score_kind=kind)
    assert policy["threshold"] == expected["threshold"]
    np.testing.assert_array_equal(predictions["final_prediction"], (predictions["raw_score"] >= policy["threshold"]).astype(int))
    np.testing.assert_array_equal(predictions["target_high_volatility"], inputs.validation.y_classification)   # frozen target, not recomputed
    metrics = json.loads((path / "validation_metrics.json").read_text())
    assert metrics["threshold"] == policy["threshold"] and metrics["selection_split"] == "validation"
    assert classification_metrics(inputs.validation.y_classification, predictions["raw_score"].to_numpy(), threshold=policy["threshold"])["f1"]["value"] == metrics["metrics"]["f1"]["value"]


def test_workflow_slp_records_protocol_and_ignores_validation_labels(tmp_path):
    inputs = synthetic()
    first = json.loads((run(tmp_path, "slp", inputs, run_id="a") / "search_results.json").read_text())
    tampered = dataclasses.replace(inputs.validation, y_classification=1 - inputs.validation.y_classification)
    second = json.loads((run(tmp_path, "slp", TrainValidation("with_spike", inputs.train, tampered), run_id="b") / "search_results.json").read_text())
    for a, b in zip(first["candidates"], second["candidates"]):
        assert a["protocol"] == b["protocol"] and a["protocol"]["selected_epochs"] >= 1
    protocol = first["candidates"][0]["protocol"]
    assert protocol["tail_rows"] == 24 and protocol["purged_rows"] == 5 and protocol["subtrain_rows"] == 160 - 24 - 5
    assert protocol["refit_rows"] == 160 and protocol["refit_seed"] == 42


def test_workflow_reference_is_separate_and_slp_reference_is_sgd(tmp_path):
    result = json.loads((run(tmp_path, "slp") / "reference_result.json").read_text())
    assert result["constructor"] == "SGDClassifier" and result["parameters"]["loss"] == "log_loss" and result["verification_only"] is True
    assert "per-sample" in result["scope_note"]
    tree = json.loads((run(tmp_path, "decision_tree", run_id="t") / "reference_result.json").read_text())
    assert tree["constructor"] == "DecisionTreeClassifier" and tree["max_abs_score_difference"] < 1e-8
    nb = json.loads((run(tmp_path, "gaussian_nb", run_id="n") / "reference_result.json").read_text())
    assert nb["max_abs_score_difference"] < 1e-6
    assert run(tmp_path, "logistic", run_id="l", reference=False) and not (tmp_path / "with_spike/classification/logistic/scratch/l/reference_result.json").exists()


def test_workflow_failure_is_recorded_and_runs_are_immutable(tmp_path):
    config = reduced_config()
    config["models"]["knn"]["grid"] = {"k": [10 ** 6], "weights": ["uniform"]}
    with pytest.raises(ValueError, match="No valid candidates"):
        run(tmp_path, "knn", config=config)
    manifest = read_manifest(tmp_path / "with_spike/classification/knn/scratch/r1")
    assert manifest["status"] == "failed" and manifest["error"]["exception_type"] == "ValueError"
    with pytest.raises(FileExistsError):
        run(tmp_path, "knn", config=config)
    with pytest.raises(ValueError, match="Unsupported"):
        run(tmp_path, "adaboost")


def test_workflow_never_touches_test_or_recomputes_target():
    source = Path(training.__file__).read_text(encoding="utf8")
    for forbidden in ("load_test", "TEST_PATH", "np." + "median", "quantile", "y_" + "regression"):
        assert forbidden not in source, forbidden
    assert "sklearn" not in source


@pytest.mark.parametrize("variant,name", [("with_spike", "decision_tree"), ("non_spike", "slp")])
def test_real_data_smoke_reduced_grid(variant, name, tmp_path):
    config = reduced_config(max_epochs=15)
    model = config["models"][name]
    model["grid"] = {k: v[:1] for k, v in model["grid"].items()}
    path = train_classifier(name, variant, tmp_path, "smoke", config=config, reference=False, hashes=HASHES)
    manifest = read_manifest(path)
    inputs = load_train_validation(variant)
    assert manifest["status"] == "completed" and manifest["input_identity"]["validation"]["rows"] == len(inputs.validation.X)
    predictions = pd.read_csv(path / "validation_predictions.csv", float_precision="round_trip")
    np.testing.assert_array_equal(predictions["target_high_volatility"], inputs.validation.y_classification)
    if name == "slp":
        protocol = json.loads((path / "search_results.json").read_text())["candidates"][0]["protocol"]
        assert protocol["subtrain_rows"] + protocol["purged_rows"] + protocol["tail_rows"] <= len(inputs.train.X)
        assert protocol["refit_rows"] == len(inputs.train.X)


def test_refit_runs_selected_epochs_even_when_train_loss_stalls():
    train = synthetic().train
    config = copy.deepcopy(CONFIG)
    config["neural"]["patience"], config["neural"]["min_delta"] = 1, 0.0
    params = {"batch": "full", "l2": 0.0, "learning_rate": 40.0}   # oscillating loss: patience-1 early stopping would fire in the refit
    result = select_slp_epochs(params, train, train.dates, config)
    config["neural"]["patience"] = 1000
    selected = result["selected_epochs"]
    model, info = fit_candidate("slp", params, train, train.dates, config)
    assert model.early_stopping is False and model.epochs_run_ == info["selected_epochs"]
    config["neural"]["patience"] = 1
    model, info = fit_candidate("slp", params, train, train.dates, config)
    assert info["selected_epochs"] == selected and model.epochs_run_ == selected and model.best_epoch_ == selected


@pytest.mark.parametrize("name", M5_MODELS)
def test_configured_loss_is_called_and_persisted(name, tmp_path, monkeypatch):
    from src.modeling.metrics import loss
    calls = []
    def observed(loss_name, y, **kwargs):
        calls.append(loss_name)
        return loss(loss_name, y, **kwargs)
    monkeypatch.setattr(training, "loss", observed, raising=False)
    path = run(tmp_path, name, reference=False)
    configured = reduced_config()["losses"][name]
    search = json.loads((path / "search_results.json").read_text())
    valid = [c for c in search["candidates"] if c["error"] is None]
    assert len(calls) >= len(valid)
    assert set(calls) == {configured}
    for candidate in valid:
        assert candidate["metrics"]["loss"]["name"] == configured
        assert candidate["metrics"]["loss"]["value"] is not None
    predictions = pd.read_csv(path / "validation_predictions.csv", float_precision="round_trip")
    score = predictions["raw_score"].to_numpy()
    kwargs = {"scores": score} if name == "perceptron" else {"probabilities": score}
    expected = {"name": configured, **loss(configured, synthetic().validation.y_classification, **kwargs)}
    metrics = json.loads((path / "validation_metrics.json").read_text())
    assert metrics["metrics"]["loss"] == expected
    assert metrics["baseline"]["loss"] == expected


@pytest.mark.parametrize("name", M5_MODELS)
def test_reference_deliverables_reproduce_validation(name, tmp_path):
    from src.modeling.persistence import load_npz, load_reference, required_artifacts
    from src.modeling.artifacts import file_sha256
    from src.modeling.classification._common import scaler_from_dict
    from src.modeling.metrics import loss
    path = run(tmp_path, name)
    reference = path.parent.parent / "reference" / path.name
    assert reference != path and reference.is_dir()
    manifest = read_manifest(reference)
    assert manifest["implementation"] == "reference" and manifest["role"] == "finalized_reference"
    assert manifest["status"] == "completed" and not manifest["test_accessed"]
    for artifact in [*required_artifacts(manifest), "manifest.json"]:
        assert (reference / artifact).is_file(), artifact
    assert resume_check(manifest, HASHES, reference)["can_skip"]
    receipt = json.loads((reference / "load_verification.json").read_text())
    assert receipt["passed"] and receipt["scores"]["passed"] and receipt["labels"]["labels_exact"]
    estimator = load_reference(reference / "model.joblib", expected_sha256=file_sha256(reference / "model.joblib"))
    arrays, state = load_npz(reference / "preprocessor.npz")
    X = synthetic().validation.X
    if state["kind"] == "standardizer":
        scaler = scaler_from_dict({k: v for k, v in state.items() if k not in ("kind", "warnings")})
        np.testing.assert_array_equal(arrays["mean"], scaler.mean_)
        np.testing.assert_allclose(scaler.mean_, synthetic().train.X.mean(axis=0))
        X = scaler.transform(X)
    else:
        assert arrays["feature_count"][0] == X.shape[1]
    score = estimator.decision_function(X) if name == "perceptron" else estimator.predict_proba(X)[:, 1]
    stored = pd.read_csv(reference / "validation_predictions.csv", float_precision="round_trip")
    np.testing.assert_allclose(score, stored["raw_score"], rtol=1e-10, atol=1e-12)
    threshold = manifest["output_policy"]["threshold"]
    np.testing.assert_array_equal(score >= threshold, stored["final_prediction"])
    metrics = json.loads((reference / "validation_metrics.json").read_text())
    kwargs = {"scores": score} if name == "perceptron" else {"probabilities": score}
    expected_loss = {"name": CONFIG["losses"][name], **loss(CONFIG["losses"][name], synthetic().validation.y_classification, **kwargs)}
    for key in ("metrics", "baseline"):
        assert metrics[key]["loss"] == expected_loss
        assert {"confusion_matrix", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "roc_curve", "pr_curve"} <= metrics[key].keys()
