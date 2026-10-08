"""Deterministic execution inventory from config; no model construction/fitting."""
from __future__ import annotations
import itertools
from .artifacts import canonical_hash
from .configuration import validate_config


def _candidates(model: dict) -> list[dict]:
    grid = model["grid"]
    if "tuples" in grid:
        return [dict(zip(model["parameters"]["tuple_fields"], values, strict=True)) for values in grid["tuples"]]
    fields = sorted(grid)
    return [dict(zip(fields, values, strict=True)) for values in itertools.product(*(grid[field] for field in fields))]


def expected_runs(config: dict, *, include_optional: bool = False) -> list[dict]:
    validate_config(config)
    runs = []
    def add(variant, name, model, implementation, role, parameters=None, required=False):
        row = {"variant": variant, "task": model["task"], "model": name, "implementation": implementation,
               "role": role, "parameters": parameters or {}, "required": required, "pilot_only": role == "pilot_only", "fresh_fit": True,
               "config_sha256": config["config_sha256"]}
        row["run_key"] = "expected-" + canonical_hash(row)[:20]
        runs.append(row)
    for variant in config["experiment"]["variant_order"]:
        for name, model in sorted(config["models"].items()):
            candidates = _candidates(model)
            components = config["preprocessing"]["pca"]["components"]
            if name == "svm":
                candidates = [{**candidate, "pca_components": count} for candidate in candidates for count in [None, *components]]
            if name == "random_forest":
                candidates = [{**candidate, "pca_components": None} for candidate in candidates] + [{"pca_components": count, "use_selected_no_pca_hyperparameters": True} for count in components]
            for implementation in ("scratch", "reference"):
                for parameters in candidates: add(variant, name, model, implementation, "candidate", parameters)
                add(variant, name, model, implementation, "finalized_" + implementation, required=True)
                if name == "stacking":
                    for fold in range(config["stacking"]["folds"]):
                        for base in config["stacking"]["bases"]:
                            add(variant, name, model, implementation, "internal_fold", {"fold": fold, "base": base})
                if name == config["experiment"]["pilot"]["model"]:
                    add(variant, name, model, implementation, "pilot_only", {"train_validation_only": True})
                if include_optional:
                    for seed in config["experiment"]["stability_seeds"]:
                        add(variant, name, model, implementation, "optional_stability", {"seed": seed, "after_mandatory_only": True})
    return runs


def delivery_complete(expected: list[dict], actual: list[dict]) -> dict:
    indexed = {}
    duplicates = set()
    for row in actual:
        key = row["run_key"]
        if key in indexed: duplicates.add(key)
        indexed[key] = row
    missing = []
    for row in expected:
        if not row["required"]: continue
        result = indexed.get(row["run_key"])
        if row["run_key"] in duplicates or result is None or result.get("status") != "completed" or result.get("pilot_only") or any(result.get(key) != row[key] for key in ("variant", "task", "model", "implementation", "role", "config_sha256", "fresh_fit")):
            missing.append(row["run_key"])
    return {"complete": not missing, "missing_or_incomplete": missing}
