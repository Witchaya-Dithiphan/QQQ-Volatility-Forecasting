"""Tables the written report quotes that no other command produces.

Everything here is derived from files that already exist: the saved runs, the
leaderboards, and (only once finalize left its receipt) the stored holdout predictions.
Nothing is fitted and no data split is opened, so the tables can be rebuilt at any time
and a document that quotes them can be checked against them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .core.artifacts import load_model, read_run, run_directory
from .core.bootstrap import REPLICATES, block_indices, difference
from .core.contracts import FEATURE_COLUMNS, VARIANTS
from .core.figures import write_markdown_table
from .core.report import PRIMARY, as_entry, score_entries
from .core.trainer import OUTPUT_ROOT
from .registry import MODELS, model_names

LOCK_NAME = "selection_lock.json"
RECEIPT_NAME = "finalize_receipt.json"
HOLDOUT_PREDICTIONS = "test_predictions.csv"
HOLDOUT_METRICS = "test_metrics.json"
# The taught model each headline is measured against.
CLASSROOM = {"classification": "logistic", "regression": "multiple_linear"}
EXTRACURRICULAR = {"classification": "adaboost", "regression": "elastic_net"}
# Every metric the assignment lists for classification, in its order.
ASSIGNMENT_COLUMNS = ["model", "tn", "fp", "fn", "tp", "accuracy", "precision", "sensitivity", "specificity",
                      "tnr", "f1", "roc_auc", "average_precision"]


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value):
    return None if value in ("", None) else float(value)


def _count(value):
    return None if value in ("", None) else int(float(value))


def _headline(artifact_root: Path) -> dict:
    return json.loads((artifact_root / LOCK_NAME).read_text(encoding="utf-8"))["headline"]


def _holdout_entry(artifact_root: Path, variant: str, name: str) -> dict:
    rows = _read_csv(run_directory(artifact_root, variant, name) / HOLDOUT_PREDICTIONS)
    return {"model": name, "kind": "model",
            "target": np.array([float(row["target"]) for row in rows]),
            "score": np.array([float(row["raw_score"]) for row in rows]),
            "prediction": np.array([float(row["final_prediction"]) for row in rows])}


def _entry(split: str, artifact_root: Path, variant: str, name: str) -> dict:
    if split == "validation":
        return as_entry(read_run(artifact_root, variant, name))
    return _holdout_entry(artifact_root, variant, name)


def _paired(task: str, first: dict, second: dict, replicates: int) -> tuple[dict, dict, dict]:
    """Scores of both entries and the interval of second - first on the same resampled rows."""
    field, _ = PRIMARY[task]
    indices = block_indices(len(first["target"]), replicates=replicates)
    a, b = score_entries(task, [first, second], indices)
    gap = difference(a["point"][field], a["replicates"][field], b["point"][field], b["replicates"][field])
    return a["point"], b["point"], gap


def model_inventory(artifact_root: Path) -> list[dict]:
    rows = []
    for name in model_names():
        runs = {variant: read_run(artifact_root, variant, name) for variant in VARIANTS}
        first = runs[VARIANTS[0]]
        search = json.loads((first.directory / "search_results.json").read_text(encoding="utf-8"))
        model, _ = load_model(first.directory)
        rows.append({
            "model": name, "task": MODELS[name].task, "candidates": len(search),
            "reference": type(model.reference()).__name__,
            "preprocess": first.config["preprocess"],
            **{f"params_{variant}": json.dumps(runs[variant].config["params"], sort_keys=True) for variant in VARIANTS},
            **{f"pca_{variant}": runs[variant].config["pca_components"] for variant in VARIANTS},
        })
    return rows


def headline_vs_classroom(split: str, artifact_root: Path, *, replicates: int = REPLICATES) -> list[dict]:
    """The locked headline against the taught model of the same task, paired on one split."""
    headline = _headline(artifact_root)
    rows = []
    for task, reference in CLASSROOM.items():
        field, higher = PRIMARY[task]
        for variant in VARIANTS:
            name = headline[f"{variant}/{task}"]
            base, best, gap = _paired(task, _entry(split, artifact_root, variant, reference),
                                      _entry(split, artifact_root, variant, name), replicates)
            rows.append({
                "task": task, "variant": variant, "metric": field, "classroom_model": reference,
                "classroom_score": base[field], "headline": name, "headline_score": best[field],
                "delta": gap["point"], "delta_low": gap["low"], "delta_high": gap["high"],
                "improvement_percent": 100 * gap["point"] / base[field] * (1 if higher else -1),
                "interval_excludes_zero": gap["excludes_zero"],
            })
    return rows


def spike_policy_on_holdout(artifact_root: Path, *, replicates: int = REPLICATES) -> list[dict]:
    """The same model trained with and without the spike rows, paired on the holdout."""
    rows = []
    for task in ("classification", "regression"):
        field, higher = PRIMARY[task]
        for name in model_names():
            if MODELS[name].task != task:
                continue
            kept, removed, gap = _paired(task, _holdout_entry(artifact_root, "with_spike", name),
                                         _holdout_entry(artifact_root, "non_spike", name), replicates)
            better = gap["point"] > 0 if higher else gap["point"] < 0
            rows.append({
                "task": task, "model": name, "metric": field, "with_spike": kept[field],
                "non_spike": removed[field], "delta": gap["point"], "delta_low": gap["low"],
                "delta_high": gap["high"], "direction": "non_spike better" if better else "with_spike better",
                "interval_excludes_zero": gap["excludes_zero"],
            })
    return rows


def holdout_segments(artifact_root: Path, task: str) -> list[dict]:
    """Segment metrics of the headline, classroom and extracurricular models of one task."""
    headline = _headline(artifact_root)
    chosen = {task: {CLASSROOM[task], EXTRACURRICULAR[task],
                     *(headline[f"{variant}/{task}"] for variant in VARIANTS)} for task in CLASSROOM}
    rows = []
    for row in _read_csv(artifact_root / "final" / "segmented_metrics.csv"):
        if row["task"] != task or row["model"] not in chosen[task]:
            continue
        entry = {"model": row["model"], "variant": row["variant"], "segment": row["segment"],
                 "rows": _count(row["rows"])}
        if task == "classification":
            entry["positives"] = _count(row["positives"])
            fields = ("average_precision", "roc_auc", "recall", "precision", "f1")
        else:
            fields = ("rmse", "mae", "r2")
        rows.append({**entry, **{name: _number(row.get(name)) for name in fields}})
    return sorted(rows, key=lambda row: (row["model"], row["variant"], row["segment"]))


def headline_summary(artifact_root: Path) -> list[dict]:
    """One row per locked headline: its selection score, its holdout score and the best baseline."""
    rows = []
    for task, (field, higher) in PRIMARY.items():
        for variant in VARIANTS:
            board = _read_csv(artifact_root / "final" / variant / f"leaderboard_{task}.csv")
            chosen = next(row for row in board if row["headline"] == "True")
            rules = [row for row in board if row["kind"] == "baseline"]
            rule = (max if higher else min)(rules, key=lambda row: float(row[field]))
            ranked = sorted((row for row in board if row["kind"] == "model"),
                            key=lambda row: float(row[field]), reverse=higher)
            rows.append({
                "task": task, "variant": variant, "headline": chosen["model"], "metric": field,
                "validation": _number(chosen[f"validation_{field}"]), "test": _number(chosen[field]),
                "test_low": _number(chosen[f"{field}_low"]), "test_high": _number(chosen[f"{field}_high"]),
                "rank_on_test": f"{[row['model'] for row in ranked].index(chosen['model']) + 1} of {len(ranked)}",
                "best_baseline": rule["model"], "baseline_test": _number(rule[field]),
                "beats_baseline": chosen["beats_best_baseline"] == "True",
            })
    return rows


def holdout_classification_metrics(artifact_root: Path, variant: str) -> list[dict]:
    rows = []
    for name in model_names():
        if MODELS[name].task != "classification":
            continue
        metrics = json.loads((run_directory(artifact_root, variant, name) / HOLDOUT_METRICS)
                             .read_text(encoding="utf-8"))
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        rows.append({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                     **{field: metrics[field]["value"] for field in ASSIGNMENT_COLUMNS[5:]}})
    return sorted(rows, key=lambda row: -(row["average_precision"] or 0.0))


def _walk_forward_summary(report_root: Path, variant: str) -> list[dict]:
    table = _read_csv(report_root / variant / "walk_forward.csv")
    rows = []
    for name in sorted({row["model"] for row in table}):
        own = {row["fold"]: row for row in table if row["model"] == name}
        pooled = own["pooled"]
        rows.append({"model": name,
                     **{f"fold_{fold}_ap": _number(own[str(fold)]["average_precision"]) for fold in range(4)},
                     "folds_used": int(float(pooled["folds_used"])), "positives": int(float(pooled["positives"])),
                     "pooled_ap": _number(pooled["average_precision"]),
                     "pooled_low": _number(pooled["average_precision_low"]),
                     "pooled_high": _number(pooled["average_precision_high"])})
    return sorted(rows, key=lambda row: -row["pooled_ap"])


def _importance_summary(report_root: Path, variant: str) -> list[dict]:
    table = _read_csv(report_root / variant / "feature_importance.csv")
    rows = []
    for feature in FEATURE_COLUMNS:
        row = {"feature": feature}
        for task, label in (("classification", "drop_in_ap"), ("regression", "rise_in_rmse")):
            values = np.array([float(item["importance_mean"]) for item in table
                               if item["feature"] == feature and item["task"] == task])
            row[f"{label}_mean"] = float(values.mean())
            row[f"{label}_std_across_models"] = float(values.std(ddof=1))
        rows.append(row)
    return rows


def write_tables(*, artifact_root=None, report_root=None, replicates: int = REPLICATES) -> list[Path]:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    source_root = artifact_root / "report"
    tables = Path(report_root or source_root) / "tables"
    written = []

    def emit(name: str, rows: list[dict], title: str, columns: list[str] | None = None, notes=()) -> None:
        path = tables / f"{name}.md"
        write_markdown_table(rows, path, columns=columns or list(rows[0]), title=title, notes=tuple(notes))
        written.append(path)

    emit("model_inventory", model_inventory(artifact_root), "Model inventory and selected settings")
    emit("validation_headline_vs_classroom", headline_vs_classroom("validation", artifact_root, replicates=replicates),
         "Headline vs classroom model on Validation (paired block bootstrap)")
    for variant in VARIANTS:
        leaderboard = _read_csv(source_root / variant / "leaderboard_classification.csv")
        ranked = sorted(leaderboard, key=lambda row: -float(row["average_precision"]))
        emit(f"classification_metrics_{variant}",
             [{name: (row[name] or None if name in ("model", "loss_name")
                      else _count(row[name]) if name in ("tn", "fp", "fn", "tp") else _number(row[name]))
               for name in ["model", "loss_name", "loss", *ASSIGNMENT_COLUMNS[1:]]} for row in ranked],
             f"Every metric the assignment lists — Validation — {variant}")
        emit(f"walk_forward_{variant}", _walk_forward_summary(source_root, variant),
             f"Walk-forward average precision inside Train — {variant}")
        emit(f"feature_importance_{variant}", _importance_summary(source_root, variant),
             f"Block-permutation importance averaged over models — {variant}")

    if (artifact_root / RECEIPT_NAME).is_file():
        emit("test_headline_summary", headline_summary(artifact_root),
             "Locked headline models: Validation score, one-shot Test score and the best baseline")
        for task in CLASSROOM:
            emit(f"test_segments_{task}", holdout_segments(artifact_root, task),
                 f"Test metrics by diagnostic segment — {task}")
        emit("test_headline_vs_classroom", headline_vs_classroom("test", artifact_root, replicates=replicates),
             "Headline vs classroom model on Test (paired block bootstrap)")
        emit("test_spike_policy", spike_policy_on_holdout(artifact_root, replicates=replicates),
             "Same model with and without spike rows, on Test (paired block bootstrap)")
        for variant in VARIANTS:
            emit(f"test_classification_metrics_{variant}", holdout_classification_metrics(artifact_root, variant),
                 f"Every metric the assignment lists — Test — {variant}", columns=ASSIGNMENT_COLUMNS)
    return written


def main() -> int:
    for path in write_tables():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
