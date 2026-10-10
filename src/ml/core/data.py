"""Read-only accepted datasets, with the Test split denied unless explicitly asked for.

Keeps the SHA-256 and label-threshold checks that catch silent data corruption, but
not the old cross-report identity chain: it re-read the full original train set on
every non_spike load to prove subsetting, which belongs in a test rather than in the
hot path of 38 training runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXPERIMENT_DATASET_REPORT_PATH, SPIKE_INPUT_CONTRACT_REPORT_PATH

from .contracts import (
    CLASSIFICATION_TARGET,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    REGRESSION_TARGET,
    TEST_PATH,
    VALIDATION_PATH,
    VARIANT_TRAIN_PATHS,
)


@dataclass(frozen=True)
class Split:
    name: str
    dates: np.ndarray
    X: np.ndarray
    y_regression: np.ndarray
    y_classification: np.ndarray
    sha256: str


@dataclass(frozen=True)
class Dataset:
    variant: str
    train: Split
    validation: Split
    threshold: float
    feature_names: list[str]


def _readonly(values, dtype=None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _accepted_threshold() -> float:
    contract = json.loads(SPIKE_INPUT_CONTRACT_REPORT_PATH.read_text(encoding="utf-8"))
    return float(contract["classification_contract"]["threshold"])


def _expected_sha256() -> dict[str, str]:
    report = json.loads(EXPERIMENT_DATASET_REPORT_PATH.read_text(encoding="utf-8"))
    return {name: entry["sha256"].lower() for name, entry in report["artifacts"].items()}


def _read_split(path: Path, name: str, *, threshold: float, expected_sha256: str | None) -> Split:
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Accepted input changed on disk: {path}")

    frame = pd.read_csv(path, float_precision="round_trip")
    required = (DATE_COLUMN, *FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing accepted columns in {path}: {missing}")
    if frame.isna().any().any():
        raise ValueError(f"Null values in {path}")

    dates = pd.to_datetime(frame[DATE_COLUMN], format="%Y-%m-%d", errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"Dates must be unique and chronological in {path}")

    targets = frame[REGRESSION_TARGET].to_numpy(dtype=np.float64)
    labels = frame[CLASSIFICATION_TARGET].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Classification labels must be binary in {path}")
    if not np.array_equal(labels, (targets > threshold).astype(np.int64)):
        raise ValueError(f"Labels disagree with the accepted Q75 threshold in {path}")

    return Split(
        name=name,
        dates=_readonly(dates.to_numpy()),
        X=_readonly(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)),
        y_regression=_readonly(targets),
        y_classification=_readonly(labels),
        sha256=digest,
    )


def load(variant: str = "with_spike") -> Dataset:
    if variant not in VARIANT_TRAIN_PATHS:
        raise ValueError(
            f"Unknown dataset variant: {variant!r}; expected one of {sorted(VARIANT_TRAIN_PATHS)}"
        )
    threshold = _accepted_threshold()
    digests = _expected_sha256()
    train_key = "with_spikes_train" if variant == "with_spike" else "non_spike_train"

    train = _read_split(
        VARIANT_TRAIN_PATHS[variant], "train", threshold=threshold, expected_sha256=digests.get(train_key)
    )
    validation = _read_split(
        VALIDATION_PATH, "validation", threshold=threshold, expected_sha256=digests.get("validation_labeled")
    )
    if train.dates[-1] >= validation.dates[0]:
        raise ValueError("Train and Validation overlap in time")

    return Dataset(variant, train, validation, threshold, list(FEATURE_COLUMNS))


def load_test(*, allow_test: bool = False) -> Split:
    """Default-deny so the Test split cannot leak into model selection."""
    if allow_test is not True:
        raise PermissionError(
            "Test split is only available during finalize; "
            "pass allow_test=True from `src.ml.run finalize --allow-test`"
        )
    digests = _expected_sha256()
    return _read_split(
        TEST_PATH, "test", threshold=_accepted_threshold(), expected_sha256=digests.get("test_labeled")
    )
