# src/ml/finalize.py
"""One-shot evaluation of the locked models on the held-out split.

This module never opens the held-out files itself: `run.py finalize` loads them and
hands the arrays in. That keeps the static guard in tests/test_audit_guards.py true
without adding an exception for this file.
"""
from __future__ import annotations

import csv
import json
import subprocess
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT

from .core import figures
from .core.artifacts import HASHED_FILES, load_model, read_run, run_directory, sha256_file
from .core.bootstrap import REPLICATES, block_indices
from .core.contracts import VARIANTS
from .core.data import load
from .core.metrics import classification_metrics, regression_metrics
from .core.report import (
    INTERVAL_FIELDS, PRIMARY, baselines, leaderboard, ranked, score_entries, with_interval,
)
from .core.trainer import OUTPUT_ROOT, evaluate, write_json
from .registry import MODELS, model_names

LOCK_NAME = "selection_lock.json"
RECEIPT_NAME = "finalize_receipt.json"
SEGMENTS = ("spike_affected", "non_spike")
HOLDOUT_NOTE = (
    "ประเมินบน Test ครั้งเดียวด้วยโมเดลที่เทรนจาก Train และเลือกบน Validation (ไม่ refit) "
    "โมเดลเด่นของแต่ละ task ถูกระบุไว้ใน selection_lock.json ก่อนเปิด Test "
    "แถวอื่นในตารางเป็นข้อมูลประกอบ ห้ามเลือกโมเดลจากตารางนี้"
)


def _selection_value(run, task: str, field: str):
    metrics = run.metrics["clipped_metrics"] if task == "regression" else run.metrics
    return metrics[field]["value"]


def build_lock(artifact_root: Path) -> dict:
    artifact_root = Path(artifact_root)
    runs, headline = {}, {}
    for variant in VARIANTS:
        for name in model_names():
            directory = run_directory(artifact_root, variant, name)
            verification = json.loads((directory / "load_verification.json").read_text(encoding="utf-8"))
            if verification["passed"] is not True:
                raise ValueError(f"{variant}/{name}: reload verification did not pass")
            runs[f"{variant}/{name}"] = {file: sha256_file(directory / file) for file in HASHED_FILES}
        for task, (field, higher) in PRIMARY.items():
            candidates = [read_run(artifact_root, variant, name) for name in model_names()
                          if MODELS[name].task == task]
            best = (max if higher else min)(candidates, key=lambda run: _selection_value(run, task, field))
            headline[f"{variant}/{task}"] = best.name
    return {"runs": runs, "run_count": len(runs), "headline": headline}


def write_lock(artifact_root: Path) -> Path:
    path = Path(artifact_root) / LOCK_NAME
    if (Path(artifact_root) / RECEIPT_NAME).is_file():
        # Re-locking after the read would let a changed artifact pass as pre-registered.
        raise FileExistsError(f"The held-out split was already evaluated ({RECEIPT_NAME} exists); "
                              "the selection lock can no longer be rewritten")
    write_json(path, build_lock(artifact_root))
    return path


def verify_lock(artifact_root: Path) -> dict:
    path = Path(artifact_root) / LOCK_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No {LOCK_NAME}; run `finalize --dry-run` and commit it first")
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = build_lock(artifact_root)["runs"]
    changed = sorted(key for key in set(stored["runs"]) | set(current) if stored["runs"].get(key) != current.get(key))
    if changed:
        raise ValueError(f"Artifacts changed since the selection lock: {changed}")
    return stored


def check_receipt(artifact_root: Path, rerun_reason: str | None) -> list[dict]:
    """Earlier receipts, or FileExistsError when a second run was not explicitly justified."""
    path = Path(artifact_root) / RECEIPT_NAME
    if not path.is_file():
        return []
    if not rerun_reason:
        raise FileExistsError(
            f"The held-out split was already evaluated ({RECEIPT_NAME} exists). "
            "A second run needs --rerun-reason and must be disclosed in the report."
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    return [*stored.pop("previous_receipts", []), stored]


def segment_metrics(task: str, target, score, prediction, segments) -> dict:
    result = {}
    for segment in SEGMENTS:
        mask = segments == segment
        entry = {"rows": int(mask.sum())}
        if mask.any() and task == "classification":
            metrics = classification_metrics(target[mask], score[mask], predicted_labels=prediction[mask])
            entry["positives"] = int(target[mask].sum())
            entry.update({name: value for name, value in metrics.items() if name not in ("roc_curve", "pr_curve")})
        elif mask.any():
            entry.update(regression_metrics(target[mask], prediction[mask]))
        result[segment] = entry
    return result


def _write_predictions(path: Path, rows: list) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Date", "target", "raw_score", "final_prediction", "diagnostic_segment"])
        writer.writerows(rows)


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def evaluate_holdout(holdout, segment_dates, segments, *, artifact_root=None, replicates: int = REPLICATES,
                     command: str = "", rerun_reason: str | None = None) -> dict:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    lock = verify_lock(artifact_root)
    previous = check_receipt(artifact_root, rerun_reason)
    segments = np.asarray(segments)
    if (len(segments) != len(holdout.dates) or not np.array_equal(np.asarray(segment_dates), holdout.dates)
            or not np.isin(segments, SEGMENTS).all()):
        raise ValueError("Segment flags do not line up with the held-out rows")

    indices = block_indices(len(holdout.X), replicates=replicates)
    # Nothing is written until every model is scored and every table is built: the read
    # cannot be repeated, so a failure must not leave half the outputs with no receipt.
    written, segment_rows, pending = [], [], []
    for variant in VARIANTS:
        data = load(variant)
        entries = {"regression": [], "classification": []}
        for name in model_names():
            task = MODELS[name].task
            directory = run_directory(artifact_root, variant, name)
            model, transform = load_model(directory)
            metrics, prediction, score = evaluate(model, transform(holdout.X), holdout.y_regression,
                                                  holdout.y_classification, task)
            target = (holdout.y_regression if task == "regression" else holdout.y_classification).astype(np.float64)
            prediction = np.asarray(prediction, dtype=np.float64)
            score = np.asarray(score, dtype=np.float64)
            if task != "clustering":
                by_segment = segment_metrics(task, target, score, prediction, segments)
                metrics = {**metrics, "segments": by_segment}
                entries[task].append({"model": name, "kind": "model", "target": target, "score": score,
                                      "prediction": prediction})
                for segment, values in by_segment.items():
                    segment_rows.append({
                        "variant": variant, "model": name, "task": task, "segment": segment,
                        **{key: (value["value"] if isinstance(value, dict) else value)
                           for key, value in values.items() if key != "confusion_matrix"},
                    })
            # write_json refuses NaN and NumPy scalars; find that out now, not after
            # some folders are already on disk.
            json.dumps(metrics, allow_nan=False)
            pending.append(partial(write_json, directory / "test_metrics.json", metrics))
            pending.append(partial(
                _write_predictions, directory / "test_predictions.csv",
                [[str(date)[:10], *rest] for date, *rest in zip(holdout.dates, target, score, prediction, segments)],
            ))
            written += [directory / "test_metrics.json", directory / "test_predictions.csv"]

        for task, task_entries in entries.items():
            field, higher = PRIMARY[task]
            target = (holdout.y_regression if task == "regression" else holdout.y_classification).astype(np.float64)
            rules = baselines(task, data.train, holdout, data.threshold)
            task_entries = task_entries + [
                {"model": name, "kind": "baseline", "target": target, "score": score, "prediction": prediction}
                for name, (score, prediction) in rules.items()
            ]
            rows = leaderboard(task, score_entries(task, task_entries, indices))
            for row in rows:
                row["variant"] = variant
                row["headline"] = row["model"] == lock["headline"][f"{variant}/{task}"]
                row[f"validation_{field}"] = (
                    _selection_value(read_run(artifact_root, variant, row["model"]), task, field)
                    if row["kind"] == "model" else None
                )
                for name in INTERVAL_FIELDS[task]:
                    row[f"{name}_ci"] = with_interval(row, name)
            csv_path = artifact_root / "final" / variant / f"leaderboard_{task}.csv"
            md_path = csv_path.with_suffix(".md")
            pending.append(partial(figures.write_leaderboard, rows, csv_path, sort_by=field, descending=higher))
            columns = ["model", "kind", "headline", f"validation_{field}"] + [
                f"{name}_ci" for name in INTERVAL_FIELDS[task]
            ] + ["tied_with_leader", "beats_best_baseline"]
            pending.append(partial(figures.write_markdown_table, ranked(rows, field, higher), md_path,
                                   columns=columns, title=f"Test leaderboard — {task} — {variant}",
                                   notes=(HOLDOUT_NOTE,)))
            written += [csv_path, md_path]

    segment_path = artifact_root / "final" / "segmented_metrics.csv"
    pending.append(partial(figures.write_leaderboard, segment_rows, segment_path, sort_by="model"))
    written.append(segment_path)

    receipt = {
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "holdout_sha256": holdout.sha256,
        "holdout_rows": int(len(holdout.X)),
        "segment_rows": {segment: int((segments == segment).sum()) for segment in SEGMENTS},
        "lock_sha256": sha256_file(artifact_root / LOCK_NAME),
        "headline": lock["headline"],
        "run_count": lock["run_count"],
        "replicates": replicates,
        "rerun_reason": rerun_reason,
        "previous_receipts": previous,
    }
    json.dumps(receipt, allow_nan=False)
    for write in pending:
        write()
    write_json(artifact_root / RECEIPT_NAME, receipt)
    written.append(artifact_root / RECEIPT_NAME)
    return {"written": written, "receipt": receipt}
