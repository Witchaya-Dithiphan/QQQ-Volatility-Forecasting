"""Read-only accepted Train/Validation inputs with a default-deny Test boundary."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from config import EXPERIMENT_DATASET_REPORT_PATH, SPIKE_INPUT_CONTRACT_REPORT_PATH
from .artifacts import file_sha256
from .contracts import FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET, VARIANT_TRAIN_PATHS, VALIDATION_PATH, TEST_PATH

REQUIRED_COLUMNS = ("Date", "Close", "Volume", "Open", "High", "Low", *FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET)


@dataclass(frozen=True)
class DatasetSplit:
    dates: np.ndarray
    X: np.ndarray
    y_regression: np.ndarray
    y_classification: np.ndarray
    sha256: str
    source: str


@dataclass(frozen=True)
class TrainValidation:
    variant: str
    train: DatasetSplit
    validation: DatasetSplit


def _readonly(values, dtype=None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def split_from_frame(frame: pd.DataFrame, *, threshold: float, sha256: str = "", source: str = "synthetic") -> DatasetSplit:
    if frame.empty or frame.columns.has_duplicates or tuple(frame.columns[:len(REQUIRED_COLUMNS)]) != REQUIRED_COLUMNS:
        raise ValueError("Missing, duplicate or reordered accepted columns")
    if frame.isna().any().any():
        raise ValueError("Null dataset values")
    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be valid, unique and chronological")
    numeric = frame[list(REQUIRED_COLUMNS[1:])]
    if any(not pd.api.types.is_numeric_dtype(numeric[col]) for col in numeric):
        raise ValueError("Non-numeric accepted column")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite dataset values")
    labels = frame[CLASSIFICATION_TARGET].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Classification labels must be binary")
    targets = frame[REGRESSION_TARGET].to_numpy(dtype=np.float64)
    if not np.isfinite(threshold) or (targets < 0).any() or not np.array_equal(labels, (targets > threshold).astype(int)):
        raise ValueError("Original Train Q75 label contract drift")
    return DatasetSplit(_readonly(dates.to_numpy()), _readonly(frame[list(FEATURE_COLUMNS)], np.float64),
                        _readonly(targets, np.float64), _readonly(labels, np.int64), sha256, source)


def _load_reports() -> tuple[dict, dict]:
    experiment = json.loads(EXPERIMENT_DATASET_REPORT_PATH.read_text(encoding="utf-8"))
    raw = SPIKE_INPUT_CONTRACT_REPORT_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != experiment["artifacts"]["spike_input_contract"]["sha256"].lower():
        raise ValueError("Accepted contract report checksum drift")
    contract = json.loads(raw)
    classification = contract["classification_contract"]
    if contract["required_columns"] != list(REQUIRED_COLUMNS) or not classification["labels_valid"] or classification["threshold_source"] != "original_train_only":
        raise ValueError("Accepted report identity drift")
    if experiment["classification_contract"]["threshold"] != classification["threshold"] or not experiment["classification_contract"]["labels_consistent"]:
        raise ValueError("Q75 report identity drift")
    return experiment, contract


def _read_verified(path: Path, identity: dict, *, threshold: float) -> DatasetSplit:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != identity["sha256"].lower():
        raise ValueError(f"Input checksum drift: {path}")
    header = next(csv.reader(io.StringIO(raw.decode("utf-8"))))
    if len(header) != len(set(header)) or tuple(header) != REQUIRED_COLUMNS:
        raise ValueError("Accepted CSV header drift")
    frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    split = split_from_frame(frame, threshold=threshold, sha256=digest, source=str(path))
    if "rows" in identity and len(split.X) != identity["rows"]:
        raise ValueError("Input row count drift")
    if "date_range" in identity:
        actual = {"start": str(split.dates[0])[:10], "end": str(split.dates[-1])[:10]}
        if actual != identity["date_range"]:
            raise ValueError("Input date range drift")
    if "class_counts" in identity:
        actual = {str(k): int((split.y_classification == k).sum()) for k in [0, 1]}
        if actual != identity["class_counts"]:
            raise ValueError("Input class count drift")
    return split


def load_train_validation(variant: str = "with_spike") -> TrainValidation:
    if variant not in VARIANT_TRAIN_PATHS:
        raise ValueError(f"Unknown dataset variant: {variant}")
    experiment, contract = _load_reports()
    threshold = contract["classification_contract"]["threshold"]
    key = "with_spikes_train" if variant == "with_spike" else "non_spike_train"
    identity = {**experiment["datasets"][key], **experiment["artifacts"][key]}
    train = _read_verified(VARIANT_TRAIN_PATHS[variant], identity, threshold=threshold)
    validation = _read_verified(VALIDATION_PATH, contract["inputs"]["validation"], threshold=threshold)
    if validation.sha256 != experiment["artifacts"]["validation_labeled"]["sha256"].lower():
        raise ValueError("Validation report identity drift")
    if train.dates[-1] >= validation.dates[0]:
        raise ValueError("Train/Validation overlap")
    original_hash = contract["inputs"]["train"]["sha256"].lower()
    if experiment["artifacts"]["with_spikes_train"]["sha256"].lower() != original_hash:
        raise ValueError("With-Spike must equal Original Train")
    if variant == "non_spike":
        original = _read_verified(VARIANT_TRAIN_PATHS["with_spike"], contract["inputs"]["train"], threshold=threshold)
        positions = np.searchsorted(original.dates, train.dates)
        if (positions >= len(original.dates)).any() or not np.array_equal(original.dates[positions], train.dates):
            raise ValueError("Non-Spike dates not in Original Train")
        for field in ("X", "y_regression", "y_classification"):
            if not np.array_equal(getattr(original, field)[positions], getattr(train, field)):
                raise ValueError("Non-Spike accepted values changed")
    return TrainValidation(variant, train, validation)


def load_test(*, allow_test: bool = False, authorization=None) -> DatasetSplit:
    if allow_test is not True or authorization is None:
        raise PermissionError("Test requires explicit finalize-test authorization")
    from .persistence import FinalizeTestAuthorization
    if not isinstance(authorization, FinalizeTestAuthorization):
        raise PermissionError("Invalid finalize-test authorization")
    authorization.validate()
    experiment, contract = _load_reports()
    split = _read_verified(TEST_PATH, contract["inputs"]["test"], threshold=contract["classification_contract"]["threshold"])
    if split.sha256 != experiment["artifacts"]["test_labeled"]["sha256"].lower():
        raise ValueError("Test report identity drift")
    authorization.record_access(split.sha256)
    return split
