# src/ml/core/report.py
"""Turn the saved validation runs into leaderboards and figures.

Everything here reads artifacts; nothing is fitted and the held-out split is never
opened. Validation numbers are selection scores (each winner was picked on these same
rows), so every table carries block-bootstrap intervals and says which models cannot
be told apart from the leader.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT

from . import figures
from .artifacts import Run, read_runs
from .bootstrap import REPLICATES, block_indices, difference, interval, replicate
from .contracts import FEATURE_COLUMNS
from .data import load
from .metrics import classification_metrics, loss, regression_metrics
from .trainer import OUTPUT_ROOT

FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "modeling"
CLASSIFICATION_FIELDS = ("average_precision", "roc_auc", "f1", "recall", "precision", "specificity", "accuracy")
REGRESSION_FIELDS = ("rmse", "mae", "r2")
INTERVAL_FIELDS = {"classification": ("average_precision", "roc_auc", "f1", "recall"),
                   "regression": ("rmse", "r2")}
# task -> (metric the models were selected on, whether larger is better)
PRIMARY = {"classification": ("average_precision", True), "regression": ("rmse", False)}
PERSISTENCE_FEATURE = "historical_volatility_5d"
SELECTION_NOTE = (
    "ตัวเลขบน Validation เป็น selection score: hyperparameter ของทุกโมเดลถูกเลือกบนแถวชุดนี้ "
    "จึงมองโลกในแง่ดีเกินจริง และโมเดลที่ `tied_with_leader = yes` แยกจากอันดับ 1 ไม่ได้ที่ระดับ 95% "
    "(moving-block bootstrap, block 20 วัน) ห้ามอ้างอันดับภายในกลุ่มนั้น"
)


def classification_scores(y, score, label) -> dict:
    metrics = classification_metrics(y, score, predicted_labels=label)
    return {name: metrics[name]["value"] for name in CLASSIFICATION_FIELDS}


def regression_scores(y, prediction) -> dict:
    metrics = regression_metrics(y, prediction)
    return {name: metrics[name]["value"] for name in REGRESSION_FIELDS}


def baselines(task: str, train, evaluation, threshold: float) -> dict:
    """{name: (score, final prediction)} from rules that need nothing but Train statistics.

    train_mean and majority use the variant's own Train targets. persistence repeats
    the volatility realised over the five days ending today, which is a feature known
    at today's close; for classification it is compared with the Train-fitted Q75.
    """
    rows = len(evaluation.X)
    realised = evaluation.X[:, FEATURE_COLUMNS.index(PERSISTENCE_FEATURE)].copy()
    if task == "regression":
        mean = np.full(rows, train.y_regression.mean())
        return {"baseline_train_mean": (mean, mean), "baseline_persistence": (realised, realised)}
    majority = float(train.y_classification.mean() >= 0.5)
    constant = np.full(rows, majority)
    return {"baseline_majority": (constant, constant),
            "baseline_persistence": (realised, (realised > threshold).astype(np.float64))}


def as_entry(run: Run) -> dict:
    return {"model": run.name, "kind": "model", "target": run.target, "score": run.score,
            "prediction": run.prediction}


def score_entries(task: str, entries: list[dict], indices) -> list[dict]:
    """Add point scores and bootstrap replicates; every entry sees the same resampled rows."""
    scored = []
    for entry in entries:
        if task == "classification":
            arrays, statistic = (entry["target"], entry["score"], entry["prediction"]), classification_scores
        else:
            arrays, statistic = (entry["target"], entry["prediction"]), regression_scores
        scored.append({**entry, "point": statistic(*arrays), "replicates": replicate(statistic, arrays, indices)})
    return scored


def leaderboard(task: str, scored: list[dict]) -> list[dict]:
    field, higher = PRIMARY[task]
    models = [entry for entry in scored if entry["kind"] == "model" and entry["point"][field] is not None]
    if not models:
        raise ValueError("No model rows to rank")
    leader = (max if higher else min)(models, key=lambda entry: entry["point"][field])
    rules = [entry for entry in scored if entry["kind"] == "baseline" and entry["point"][field] is not None]
    best_rule = (max if higher else min)(rules, key=lambda entry: entry["point"][field]) if rules else None
    rows = []
    for entry in scored:
        row = {"model": entry["model"], "kind": entry["kind"]}
        for name, value in entry["point"].items():
            row[name] = value
            if name in INTERVAL_FIELDS[task]:
                summary = interval(value, entry["replicates"][name])
                row[f"{name}_low"], row[f"{name}_high"] = summary["low"], summary["high"]
                row[f"{name}_valid_replicates"] = summary["n_valid"]
        gap = difference(leader["point"][field], leader["replicates"][field],
                         entry["point"][field], entry["replicates"][field])
        row["gap_to_leader"], row["gap_low"], row["gap_high"] = gap["point"], gap["low"], gap["high"]
        # Tied unless the paired interval of the gap clears zero.
        row["tied_with_leader"] = bool(entry is leader or gap["excludes_zero"] is not True)
        # A point comparison only: being tied with the leader says nothing about whether
        # a model is any better than a rule that fits nothing.
        row["best_baseline"] = best_rule["model"] if best_rule else None
        row["beats_best_baseline"] = None
        if best_rule and entry["kind"] == "model" and entry["point"][field] is not None:
            ahead = entry["point"][field] - best_rule["point"][field]
            row["beats_best_baseline"] = bool(ahead > 0 if higher else ahead < 0)
        rows.append(row)
    return rows


def ranked(rows: list[dict], field: str, higher: bool) -> list[dict]:
    """Best first; rows where the metric is undefined go last."""
    defined = [row for row in rows if row[field] is not None]
    undefined = [row for row in rows if row[field] is None]
    return sorted(defined, key=lambda row: row[field], reverse=higher) + undefined


def with_interval(row: dict, name: str) -> str:
    if row.get(name) is None:
        return "n/a"
    if row.get(f"{name}_low") is None:
        return f"{row[name]:.4f} [n/a]"
    return f"{row[name]:.4f} [{row[f'{name}_low']:.4f}, {row[f'{name}_high']:.4f}]"


def _check_current(runs: list[Run], validation) -> None:
    for run in runs:
        if run.config["validation_sha256"] != validation.sha256:
            raise ValueError(f"{run.variant}/{run.name} was scored on a different Validation file; retrain it")
        if not np.array_equal(run.dates, validation.dates.astype("datetime64[D]")):
            raise ValueError(f"{run.variant}/{run.name} predictions do not cover the Validation dates")


def _classification_extras(row: dict, entry: dict, run: Run | None, losses: dict) -> None:
    matrix = classification_metrics(entry["target"], entry["score"],
                                    predicted_labels=entry["prediction"])["confusion_matrix"]
    (tn, fp), (fn, tp) = matrix
    row.update(sensitivity=row["recall"], tnr=row["specificity"], tn=tn, fp=fp, fn=fn, tp=tp)
    if run is None:
        row.update(loss_name=None, loss=None, score_kind=None)
        return
    kind = run.metrics["score_kind"]
    measured = loss(losses[run.name], run.target,
                    scores=run.score if kind == "decision" else None,
                    probabilities=run.score if kind == "probability" else None)
    row.update(loss_name=losses[run.name], loss=measured["value"], score_kind=kind)


def build_report(variant: str, *, artifact_root=None, report_root=None, figure_root=None,
                 replicates: int = REPLICATES, require_all: bool = True) -> dict:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    report_root = Path(report_root or artifact_root / "report") / variant
    figure_root = Path(figure_root or FIGURE_ROOT) / variant
    data = load(variant)
    validation = data.validation
    indices = block_indices(len(validation.X), replicates=replicates)
    losses = json.loads((PROJECT_ROOT / "configs" / "modeling.json").read_text(encoding="utf-8"))["losses"]
    written, tables = [], {}

    for task in ("regression", "classification"):
        runs = read_runs(artifact_root, variant, task=task, require_all=require_all)
        if not runs:
            continue
        _check_current(runs, validation)
        target = (validation.y_regression if task == "regression" else validation.y_classification).astype(np.float64)
        entries = [as_entry(run) for run in runs]
        entries += [{"model": name, "kind": "baseline", "target": target, "score": score, "prediction": prediction}
                    for name, (score, prediction) in baselines(task, data.train, validation, data.threshold).items()]
        rows = leaderboard(task, score_entries(task, entries, indices))
        by_name = {run.name: run for run in runs}
        field, higher = PRIMARY[task]
        for row, entry in zip(rows, entries):
            row["variant"] = variant
            if task == "classification":
                _classification_extras(row, entry, by_name.get(row["model"]), losses)
            else:
                row["mse"] = None if row["rmse"] is None else row["rmse"] ** 2
            for name in INTERVAL_FIELDS[task]:
                row[f"{name}_ci"] = with_interval(row, name)
        tables[task] = rows
        csv_path, md_path = report_root / f"leaderboard_{task}.csv", report_root / f"leaderboard_{task}.md"
        figures.write_leaderboard(rows, csv_path, sort_by=field, descending=higher)
        columns = (["model", "kind", "average_precision_ci", "roc_auc_ci", "f1_ci", "recall_ci", "precision",
                    "specificity", "accuracy", "loss_name", "loss", "tied_with_leader", "beats_best_baseline"]
                   if task == "classification"
                   else ["model", "kind", "rmse_ci", "mae", "r2_ci", "tied_with_leader", "beats_best_baseline"])
        tied = sum(1 for row in rows if row["kind"] == "model" and row["tied_with_leader"])
        figures.write_markdown_table(
            ranked(rows, field, higher), md_path, columns=columns,
            title=f"Validation leaderboard — {task} — {variant}",
            notes=(SELECTION_NOTE, f"{tied} จาก {len(runs)} โมเดลอยู่ในกลุ่มที่แยกจากอันดับ 1 ไม่ได้"),
        )
        written += [csv_path, md_path]

        if task == "classification":
            defined = [run for run in runs if run.metrics["roc_curve"]["value"] is not None]
            figures.plot_roc_curves(
                {run.name: (run.metrics["roc_curve"]["value"]["fpr"], run.metrics["roc_curve"]["value"]["tpr"],
                            run.metrics["roc_auc"]["value"]) for run in defined},
                figure_root / "roc_overlay.png", title=f"ROC on Validation — {variant}")
            figures.plot_pr_curves(
                {run.name: (run.metrics["pr_curve"]["value"]["recall"], run.metrics["pr_curve"]["value"]["precision"],
                            run.metrics["average_precision"]["value"]) for run in defined},
                figure_root / "pr_overlay.png", prevalence=float(target.mean()),
                title=f"Precision-recall on Validation — {variant}")
            figures.plot_confusion_grid({run.name: run.metrics["confusion_matrix"] for run in runs},
                                        figure_root / "confusion_matrices.png",
                                        title=f"Confusion matrices on Validation — {variant}")
            written += [figure_root / name for name in ("roc_overlay.png", "pr_overlay.png", "confusion_matrices.png")]
        else:
            figures.plot_predicted_vs_actual({run.name: (run.target, run.prediction) for run in runs},
                                             figure_root / "regression_predicted_vs_actual.png",
                                             title=f"Predicted vs actual on Validation — {variant}")
            figures.plot_residuals({run.name: (run.dates, run.target - run.prediction) for run in runs},
                                   figure_root / "regression_residuals.png",
                                   title=f"Residuals on Validation — {variant}")
            written += [figure_root / "regression_predicted_vs_actual.png", figure_root / "regression_residuals.png"]

    everything = read_runs(artifact_root, variant, require_all=require_all)
    curves = {run.name: run.history for run in everything if run.history and len(run.history["loss"]) > 1}
    if curves:
        figures.plot_performance_grid(curves, figure_root / "performance_curves.png",
                                      title=f"Training loss per iteration — {variant}")
        written.append(figure_root / "performance_curves.png")

    clusters = [run for run in everything if run.task == "clustering"]
    if clusters:
        rows = [{"model": run.name, "variant": variant, "k": run.config["params"]["k"],
                 "silhouette": run.metrics["silhouette"]["value"], "inertia": run.metrics["inertia"]["value"],
                 "cluster_sizes": json.dumps(run.metrics["cluster_sizes"], sort_keys=True)} for run in clusters]
        tables["clustering"] = rows
        csv_path, md_path = report_root / "leaderboard_clustering.csv", report_root / "leaderboard_clustering.md"
        figures.write_leaderboard(rows, csv_path, sort_by="silhouette", descending=True)
        figures.write_markdown_table(rows, md_path, columns=list(rows[0]),
                                     title=f"Validation clustering metrics — {variant}",
                                     notes=("silhouette/inertia คำนวณบน Validation ไม่มีช่วงความเชื่อมั่น",))
        written += [csv_path, md_path]
    return {"written": written, "tables": tables}
