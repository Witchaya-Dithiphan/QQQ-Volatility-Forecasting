"""Strict executable configuration schema and canonical identity."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from config import PROJECT_ROOT
from .artifacts import canonical_hash
CONFIG_PATH = PROJECT_ROOT / "configs/modeling.json"
from .config_schema import SCHEMA
INTEGRITY_FIELDS = {"config_id", "config_sha256", "plan_section_sha256"}


def _schema_check(value: Any, schema: dict, path: str = "config") -> None:
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _schema_check(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: invalid type/value")
    kind = schema["type"]
    valid = {"object": type(value) is dict, "array": type(value) is list,
             "integer": type(value) is int, "number": type(value) in (int, float),
             "string": type(value) is str, "boolean": type(value) is bool, "null": value is None}[kind]
    if not valid or ("enum" in schema and value not in schema["enum"]):
        raise ValueError(f"{path}: invalid {kind}")
    if kind in ("integer", "number") and (("minimum" in schema and value < schema["minimum"]) or ("maximum" in schema and value > schema["maximum"])):
        raise ValueError(f"{path}: numerical range violation")
    if kind == "object":
        expected = schema["properties"]
        if set(value) != set(expected):
            raise ValueError(f"{path}: missing {set(expected)-set(value)}; unknown {set(value)-set(expected)}")
        for key, child in expected.items():
            _schema_check(value[key], child, f"{path}.{key}")
    elif kind == "array":
        for item in value:
            _schema_check(item, schema["items"], path + "[]")
    elif kind == "number" and not math.isfinite(value):
        raise ValueError(f"{path}: nonfinite number")


def validate_config(config: dict, *, verify_integrity: bool = True) -> None:
    _schema_check(config, SCHEMA)
    from .contracts import FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET
    if config["plan_revision"] != "modeling-config-2026-10-08-v1":
        raise ValueError("Unsupported plan revision")
    data = config["data"]
    if data["feature_columns"] != list(FEATURE_COLUMNS) or data["regression_target"] != REGRESSION_TARGET or data["classification_target"] != CLASSIFICATION_TARGET:
        raise ValueError("Canonical feature/target contract drift")
    if data["variants"] != ["with_spike", "non_spike"] or data["recompute_features_targets"]:
        raise ValueError("Dataset contract drift")
    components = config["preprocessing"]["pca"]["components"]
    if not components or any(type(k) is not int or not 1 <= k <= len(FEATURE_COLUMNS) for k in components) or len(set(components)) != len(components):
        raise ValueError("Invalid PCA component count")
    if config["test_gate"]["default_allow_test"] or not config["test_gate"]["require_clean_git"]:
        raise ValueError("Test must default-deny and require clean Git")
    if not 0 < config["selection"]["probability_epsilon"] < 0.5:
        raise ValueError("Invalid probability epsilon")
    if config["selection"]["score_only_probability"] or not config["selection"]["retain_raw"]:
        raise ValueError("Invalid output policy")
    if type(config["experiment"]["seed"]) is not int or config["experiment"]["seed"] < 0:
        raise ValueError("Invalid seed")
    for name, model in config["models"].items():
        count = 1
        for key, values in model["grid"].items():
            if not values:
                raise ValueError(f"{name}.{key}: empty grid")
            count *= len(values)
            if key in ("k", "n_estimators", "max_depth", "min_samples_leaf", "degree") and any(type(x) is not int or x <= 0 for x in values):
                raise ValueError(f"{name}.{key}: positive integers required")
            if key in ("alpha", "l2", "C", "learning_rate", "var_smoothing", "l1_ratio") and any(type(x) not in (int, float) or x < 0 or (key != "l2" and x == 0) for x in values):
                raise ValueError(f"{name}.{key}: invalid numerical range")
            if key == "l1_ratio" and any(x > 1 for x in values):
                raise ValueError("Invalid l1 ratio")
            if key == "tuples" and any(len(x) != 5 or x[0] <= 0 or x[1] <= 0 or not 0 < x[2] <= 1 or x[3] < 0 or x[4] < 0 for x in values):
                raise ValueError("Invalid XGBoost tuple")
        if name == "svm": count *= 1 + len(components)
        if name == "random_forest": count += len(components)
        budget = config["experiment"]["polynomial_max_candidates" if name == "polynomial" else "max_candidates"]
        if "final_candidates" in model["parameters"] and model["parameters"]["final_candidates"] != count:
            raise ValueError(f"{name}: final candidate count mismatch")
        if count > budget:
            raise ValueError(f"{name}: candidate budget exceeded")
    if verify_integrity:
        sealed = seal_config(config)
        if any(config[key] != sealed[key] for key in INTEGRITY_FIELDS):
            raise ValueError("Config hash/integrity mismatch")


def seal_config(config: dict) -> dict:
    _schema_check(config, SCHEMA)
    from .config_sync import render_block
    result = copy.deepcopy(config)
    result["plan_section_sha256"] = hashlib.sha256(render_block(result)).hexdigest()
    digest = canonical_hash({k: v for k, v in result.items() if k not in {"config_id", "config_sha256"}})
    result.update(config_id="cfg-" + digest[:12], config_sha256=digest)
    return result


def load_config(path: Path = CONFIG_PATH, *, verify_integrity: bool = True) -> dict:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)
    validate_config(value, verify_integrity=verify_integrity)
    return value
