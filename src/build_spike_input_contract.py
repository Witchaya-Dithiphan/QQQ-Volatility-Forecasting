"""Validate and persist the Phase 2 baseline input contract.

The contract reads Phase 1 labeled splits and reports without modifying them.
It records portable paths and direct SHA-256 values so Phase 2 does not depend
on machine-specific paths embedded in older saved reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from config import (
    CLASSIFICATION_THRESHOLD_REPORT_PATH,
    CLEAN_DATA_PATH,
    DATA_SPLIT_REPORT_PATH,
    FEATURE_DATA_PATH,
    PRIMARY_SPIKE_IQR_MULTIPLIER,
    RAW_DATA_MANIFEST_PATH,
    RAW_DATA_PATH,
    REGRESSION_TARGET_DATA_PATH,
    SPIKE_COMPARISON_RULE,
    SPIKE_EXPERIMENT_ROOT,
    SPIKE_FEATURE_FORWARD_REACH,
    SPIKE_INPUT_CONTRACT_REPORT_PATH,
    SPIKE_RSI_POLICY,
    SPIKE_TARGET_BACKWARD_REACH,
    TEST_DATA_PATH,
    TEST_LABELED_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
)
from src.build_features import FEATURE_COLUMNS
from src.build_targets import (
    CLASSIFICATION_TARGET,
    HIGH_VOLATILITY_QUANTILE,
    QUANTILE_METHOD,
    TARGET_COLUMN,
    calculate_high_volatility_threshold,
)
from src.report_paths import report_relative_path
from src.split_data import SPLIT_GAP

CONTRACT_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")
BASE_COLUMNS = ("Date", "Close", "Volume", "Open", "High", "Low")
REQUIRED_COLUMNS = (
    *BASE_COLUMNS,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    CLASSIFICATION_TARGET,
)
MODEL_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
NUMERIC_COLUMNS = (
    "Close",
    "Volume",
    "Open",
    "High",
    "Low",
    *MODEL_COLUMNS,
    CLASSIFICATION_TARGET,
)

PROTECTED_BASELINE_PATHS = (
    RAW_DATA_PATH,
    RAW_DATA_MANIFEST_PATH,
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    REGRESSION_TARGET_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
    TEST_LABELED_DATA_PATH,
    DATA_SPLIT_REPORT_PATH,
    CLASSIFICATION_THRESHOLD_REPORT_PATH,
)


@dataclass(frozen=True, slots=True)
class SplitAuditExpectation:
    """Expected row and date bounds used only as drift assertions."""

    rows: int
    start_date: str
    end_date: str


@dataclass(frozen=True, slots=True)
class AuditExpectations:
    """Snapshot-specific regression checks, never calculation inputs."""

    splits: Mapping[str, SplitAuditExpectation]
    gap_dates: Mapping[str, tuple[str, ...]]
    classification_q75: float


DEFAULT_AUDIT_EXPECTATIONS = AuditExpectations(
    splits={
        "train": SplitAuditExpectation(1733, "2016-09-29", "2023-08-18"),
        "validation": SplitAuditExpectation(371, "2023-08-28", "2025-02-19"),
        "test": SplitAuditExpectation(373, "2025-02-27", "2026-08-21"),
    },
    gap_dates={
        "gap1": (
            "2023-08-21",
            "2023-08-22",
            "2023-08-23",
            "2023-08-24",
            "2023-08-25",
        ),
        "gap2": (
            "2025-02-20",
            "2025-02-21",
            "2025-02-24",
            "2025-02-25",
            "2025-02-26",
        ),
    },
    classification_q75=0.2530580184684854,
)


@dataclass(frozen=True, slots=True)
class SpikeInputContractPaths:
    """All protected inputs, isolated evidence, and the generated report."""

    train_labeled: Path
    validation_labeled: Path
    test_labeled: Path
    split_report: Path
    classification_report: Path
    reproduced_root: Path
    output_report: Path
    experiment_root: Path = SPIKE_EXPERIMENT_ROOT

    @classmethod
    def defaults(cls, reproduced_root: str | Path) -> SpikeInputContractPaths:
        """Build the default project-root input and output path contract."""
        return cls(
            train_labeled=TRAIN_LABELED_DATA_PATH,
            validation_labeled=VALIDATION_LABELED_DATA_PATH,
            test_labeled=TEST_LABELED_DATA_PATH,
            split_report=DATA_SPLIT_REPORT_PATH,
            classification_report=CLASSIFICATION_THRESHOLD_REPORT_PATH,
            reproduced_root=Path(reproduced_root),
            output_report=SPIKE_INPUT_CONTRACT_REPORT_PATH,
        )

    def labeled_inputs(self) -> dict[str, Path]:
        """Return labeled inputs keyed by split name."""
        return {
            "train": self.train_labeled,
            "validation": self.validation_labeled,
            "test": self.test_labeled,
        }

    def protected_inputs(self) -> dict[str, Path]:
        """Return every file whose checksum must remain unchanged."""
        return {
            **self.labeled_inputs(),
            "data_split_report": self.split_report,
            "classification_threshold_report": self.classification_report,
        }


def _sha256(path: str | Path) -> str:
    """Return an uppercase SHA-256 digest without modifying the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Return whether paths resolve or alias the same existing file."""
    first_path = Path(first)
    second_path = Path(second)
    if first_path.resolve(strict=False) == second_path.resolve(strict=False):
        return True
    if first_path.exists() and second_path.exists():
        try:
            return os.path.samefile(first_path, second_path)
        except OSError:
            return False
    return False


def _strict_json_load(path: str | Path) -> dict[str, Any]:
    """Load one JSON object while rejecting nonstandard numeric constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonstandard JSON constant {value} in {path}")

    with Path(path).open("r", encoding="utf-8") as source_file:
        payload = json.load(source_file, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def load_labeled_split(path: str | Path, split_name: str) -> pd.DataFrame:
    """Load one labeled CSV and strictly parse its Date column."""
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read {split_name} labeled CSV: {exc}") from exc
    if "Date" not in frame.columns:
        raise ValueError(f"{split_name} is missing required column: Date")
    try:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").astype(
            "datetime64[ns]"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{split_name} Date could not be parsed strictly: {exc}") from exc
    return frame


def _format_date(value: Any) -> str:
    """Format a validated date scalar for strict JSON."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _validate_one_split(
    frame: pd.DataFrame,
    split_name: str,
    expectation: SplitAuditExpectation,
) -> dict[str, Any]:
    """Validate schema, types, chronology, finiteness, labels, and drift."""
    actual_columns = tuple(frame.columns)
    if actual_columns != REQUIRED_COLUMNS:
        raise ValueError(
            f"{split_name} columns must exactly match the Phase 2 contract; "
            f"expected {list(REQUIRED_COLUMNS)}, got {list(actual_columns)}"
        )
    if not is_datetime64_any_dtype(frame["Date"]):
        raise ValueError(f"{split_name} Date must have datetime64 dtype")
    if frame["Date"].isna().any():
        raise ValueError(f"{split_name} Date must not contain NaT")
    if frame["Date"].duplicated().any():
        raise ValueError(f"{split_name} Date must be unique")
    if not frame["Date"].is_monotonic_increasing:
        raise ValueError(f"{split_name} Date must be sorted oldest to newest")

    nonnumeric = [
        column for column in NUMERIC_COLUMNS if not is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        raise ValueError(
            f"{split_name} columns must have numeric dtype: " + ", ".join(nonnumeric)
        )
    numeric_values = frame[list(NUMERIC_COLUMNS)].to_numpy(
        dtype=float, na_value=np.nan
    )
    if np.isnan(numeric_values).any():
        raise ValueError(f"{split_name} numeric columns must not contain NaN")
    if np.isinf(numeric_values).any():
        raise ValueError(f"{split_name} numeric columns must not contain Infinity")

    invalid_labels = sorted(
        set(frame[CLASSIFICATION_TARGET].astype(float).unique()) - {0.0, 1.0}
    )
    if invalid_labels:
        raise ValueError(
            f"{split_name} labels must contain only 0 and 1: {invalid_labels}"
        )

    start_date = _format_date(frame["Date"].iloc[0]) if not frame.empty else None
    end_date = _format_date(frame["Date"].iloc[-1]) if not frame.empty else None
    if len(frame) != expectation.rows:
        raise ValueError(
            f"{split_name} row-count audit mismatch: "
            f"expected {expectation.rows}, got {len(frame)}"
        )
    if (start_date, end_date) != (expectation.start_date, expectation.end_date):
        raise ValueError(
            f"{split_name} date-range audit mismatch: expected "
            f"{expectation.start_date}..{expectation.end_date}, got "
            f"{start_date}..{end_date}"
        )

    return {
        "rows": len(frame),
        "date_range": {"start": start_date, "end": end_date},
        "schema_valid": True,
        "numeric_dtypes_valid": True,
        "finite_numeric_values": True,
        "date_unique": True,
        "date_sorted": True,
        "labels_binary": True,
    }


def _validate_split_order(frames: Mapping[str, pd.DataFrame]) -> None:
    """Require non-overlapping chronological Train, Validation, and Test."""
    for earlier, later in (("train", "validation"), ("validation", "test")):
        if not frames[earlier]["Date"].max() < frames[later]["Date"].min():
            raise ValueError(f"{earlier} and {later} overlap or are out of order")


def _validate_gap_contract(
    split_report: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    expectations: AuditExpectations,
) -> dict[str, Any]:
    """Validate gap sizes, positions, dates, and split summary semantics."""
    if split_report.get("gap_size") != SPLIT_GAP:
        raise ValueError(
            f"Split gap mismatch: expected {SPLIT_GAP}, "
            f"got {split_report.get('gap_size')}"
        )
    if split_report.get("total_gap_rows") != 2 * SPLIT_GAP:
        raise ValueError("Split report total_gap_rows is inconsistent")

    expected_rows = {name: len(frame) for name, frame in frames.items()}
    if split_report.get("split_rows") != expected_rows:
        raise ValueError("Split report row counts do not match labeled CSV inputs")
    expected_ranges = {
        name: {
            "start": _format_date(frame["Date"].iloc[0]),
            "end": _format_date(frame["Date"].iloc[-1]),
        }
        for name, frame in frames.items()
    }
    if split_report.get("date_ranges") != expected_ranges:
        raise ValueError("Split report date ranges do not match labeled CSV inputs")

    positions = split_report.get("positions")
    if not isinstance(positions, dict):
        raise TypeError("Split report positions must be an object")
    for earlier, gap_name, later in (
        ("train", "gap1", "validation"),
        ("validation", "gap2", "test"),
    ):
        try:
            earlier_end = int(positions[earlier]["end"])
            gap_start = int(positions[gap_name]["start"])
            gap_end = int(positions[gap_name]["end"])
            later_start = int(positions[later]["start"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Split report positions are incomplete") from exc
        if gap_start != earlier_end + 1 or later_start != gap_end + 1:
            raise ValueError(f"{gap_name} positions are not contiguous with splits")
        if gap_end - gap_start + 1 != SPLIT_GAP:
            raise ValueError(f"{gap_name} does not contain {SPLIT_GAP} positions")

    gap_dates = split_report.get("gap_dates")
    if not isinstance(gap_dates, dict):
        raise TypeError("Split report gap_dates must be an object")
    normalized_gap_dates: dict[str, list[str]] = {}
    for gap_name, expected_dates in expectations.gap_dates.items():
        actual_dates = gap_dates.get(gap_name)
        if not isinstance(actual_dates, list):
            raise TypeError(f"{gap_name} dates must be a list")
        try:
            parsed = pd.to_datetime(actual_dates, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{gap_name} contains invalid dates") from exc
        if len(actual_dates) != SPLIT_GAP or not parsed.is_monotonic_increasing:
            raise ValueError(f"{gap_name} dates do not satisfy the gap contract")
        if tuple(actual_dates) != tuple(expected_dates):
            raise ValueError(
                f"{gap_name} date audit mismatch: expected {list(expected_dates)}, "
                f"got {actual_dates}"
            )
        normalized_gap_dates[gap_name] = list(actual_dates)

    if not split_report.get("source_input_unchanged", False):
        raise ValueError("Split report does not confirm its source remained unchanged")
    return {
        "ordering_valid": True,
        "non_overlapping": True,
        "gap_size": SPLIT_GAP,
        "total_gap_rows": 2 * SPLIT_GAP,
        "gap_dates": normalized_gap_dates,
        "positions_valid": True,
    }


def _validate_classification_contract(
    frames: Mapping[str, pd.DataFrame],
    classification_report: Mapping[str, Any],
    expectations: AuditExpectations,
) -> dict[str, Any]:
    """Recompute Train Q75 and verify strict labels across every split."""
    threshold = calculate_high_volatility_threshold(
        frames["train"].drop(columns=[CLASSIFICATION_TARGET])
    )
    if not np.isclose(
        threshold,
        expectations.classification_q75,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Original Train Q75 audit mismatch: expected "
            f"{expectations.classification_q75}, got {threshold}"
        )

    mismatch_counts: dict[str, int] = {}
    equality_counts: dict[str, int] = {}
    equality_positive_counts: dict[str, int] = {}
    for split_name, frame in frames.items():
        expected_labels = frame[TARGET_COLUMN].gt(threshold).astype("int8")
        actual_labels = frame[CLASSIFICATION_TARGET].astype("int8")
        mismatch_count = int(actual_labels.ne(expected_labels).sum())
        if mismatch_count:
            raise ValueError(
                f"{split_name} contains {mismatch_count} labels inconsistent "
                "with strict Original Train Q75"
            )
        equal_mask = frame[TARGET_COLUMN].eq(threshold)
        equality_positive = int(actual_labels.loc[equal_mask].eq(1).sum())
        mismatch_counts[split_name] = mismatch_count
        equality_counts[split_name] = int(equal_mask.sum())
        equality_positive_counts[split_name] = equality_positive

    expected_report_fields = {
        "quantile": HIGH_VOLATILITY_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "threshold_source": "train_only",
        "regression_target": TARGET_COLUMN,
        "classification_target": CLASSIFICATION_TARGET,
        "comparison_rule": f"{TARGET_COLUMN} > threshold",
    }
    for field, expected in expected_report_fields.items():
        if classification_report.get(field) != expected:
            raise ValueError(f"Classification report field {field} is inconsistent")
    report_threshold = classification_report.get("threshold_decimal")
    if not isinstance(report_threshold, (int, float)) or not np.isclose(
        float(report_threshold), threshold, rtol=1e-12, atol=1e-15
    ):
        raise ValueError("Classification report threshold differs from recomputed Q75")

    for split_name, frame in frames.items():
        saved_stats = classification_report.get(split_name)
        if not isinstance(saved_stats, dict):
            raise TypeError(f"Classification report is missing {split_name} stats")
        high_count = int(frame[CLASSIFICATION_TARGET].eq(1).sum())
        normal_count = int(frame[CLASSIFICATION_TARGET].eq(0).sum())
        if (
            saved_stats.get("rows") != len(frame)
            or saved_stats.get("high_count") != high_count
            or saved_stats.get("normal_count") != normal_count
            or saved_stats.get("start_date") != _format_date(frame["Date"].iloc[0])
            or saved_stats.get("end_date") != _format_date(frame["Date"].iloc[-1])
        ):
            raise ValueError(
                f"Classification report {split_name} stats differ from labeled CSV"
            )

    return {
        "threshold_source": "original_train_only",
        "quantile": HIGH_VOLATILITY_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "threshold": threshold,
        "comparison_rule": f"{TARGET_COLUMN} > threshold",
        "comparison_operator": ">",
        "label_mismatch_counts": mismatch_counts,
        "equality_boundary_counts": equality_counts,
        "equality_boundary_positive_counts": equality_positive_counts,
        "labels_valid": True,
    }


def _is_absolute_serialized_path(value: str) -> bool:
    """Recognize absolute Windows or POSIX paths independent of host OS."""
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _path_fields(report: Mapping[str, Any]) -> dict[str, str]:
    """Collect serialized input/output paths from a baseline report."""
    fields: dict[str, str] = {}
    input_path = report.get("input_path")
    if isinstance(input_path, str):
        fields["input_path"] = input_path
    for group_name in ("input_paths", "output_paths"):
        group = report.get(group_name)
        if isinstance(group, dict):
            for name, value in group.items():
                if isinstance(value, str):
                    fields[f"{group_name}.{name}"] = value
    return fields


def _provenance_status(
    report: Mapping[str, Any],
    *,
    semantic_fields_verified: Sequence[str],
) -> dict[str, Any]:
    """Describe path portability separately from validated semantic fields."""
    path_fields = _path_fields(report)
    absolute_fields = sorted(
        name for name, value in path_fields.items() if _is_absolute_serialized_path(value)
    )
    paths_relative_to = report.get("paths_relative_to")
    portable = paths_relative_to == "report_directory" and not absolute_fields
    return {
        "portable": portable,
        "paths_relative_to": paths_relative_to,
        "absolute_path_fields": absolute_fields,
        "discrepancy_type": None if portable else "artifact_provenance_discrepancy",
        "calculation_bug_inferred": False,
        "semantic_fields_verified": list(semantic_fields_verified),
    }


def validate_input_contract(
    frames: Mapping[str, pd.DataFrame],
    split_report: Mapping[str, Any],
    classification_report: Mapping[str, Any],
    *,
    expectations: AuditExpectations = DEFAULT_AUDIT_EXPECTATIONS,
) -> dict[str, Any]:
    """Validate Phase 2 labeled inputs without reading or writing files.

    Args:
        frames: Train, Validation, and Test labeled DataFrames with parsed dates.
        split_report: Phase 1 data-split report payload.
        classification_report: Phase 1 classification-threshold report payload.
        expectations: Snapshot-specific regression assertions.

    Returns:
        JSON-ready semantic contract sections.

    Raises:
        ValueError: If any schema, value, split, gap, label, or audit check fails.
    """
    if set(frames) != set(SPLIT_NAMES):
        raise ValueError(f"frames must contain exactly {list(SPLIT_NAMES)}")
    if set(expectations.splits) != set(SPLIT_NAMES):
        raise ValueError("Audit expectations must cover every split")

    input_stats = {
        split_name: _validate_one_split(
            frames[split_name], split_name, expectations.splits[split_name]
        )
        for split_name in SPLIT_NAMES
    }
    _validate_split_order(frames)
    split_contract = _validate_gap_contract(split_report, frames, expectations)
    classification_contract = _validate_classification_contract(
        frames, classification_report, expectations
    )
    return {
        "input_stats": input_stats,
        "split_contract": split_contract,
        "classification_contract": classification_contract,
        "saved_report_provenance": {
            "data_split_report": _provenance_status(
                split_report,
                semantic_fields_verified=(
                    "split_rows",
                    "date_ranges",
                    "positions",
                    "gap_size",
                    "gap_dates",
                    "source_input_unchanged",
                ),
            ),
            "classification_threshold_report": _provenance_status(
                classification_report,
                semantic_fields_verified=(
                    "quantile",
                    "quantile_method",
                    "threshold_source",
                    "threshold_decimal",
                    "comparison_rule",
                    "split class counts",
                    "split date ranges",
                ),
            ),
        },
    }


def _resolve_report_links(report_path: Path, report: Mapping[str, Any]) -> bool:
    """Return whether every serialized relative path resolves to an existing file."""
    if report.get("paths_relative_to") != "report_directory":
        return False
    for value in _path_fields(report).values():
        if _is_absolute_serialized_path(value):
            return False
        resolved = (report_path.parent / value).resolve(strict=False)
        if not resolved.is_file():
            return False
    return True


def _collect_reproduction_evidence(
    reproduced_root: Path,
    authoritative_hashes: Mapping[str, str],
    output_report: Path,
) -> dict[str, Any]:
    """Verify fresh portable reports and byte-identical reproduced labeled CSVs."""
    split_report_path = reproduced_root / "outputs" / "reports" / "data_split_report.json"
    classification_report_path = (
        reproduced_root / "outputs" / "reports" / "classification_threshold.json"
    )
    fresh_reports = {
        "data_split_report": (split_report_path, _strict_json_load(split_report_path)),
        "classification_threshold_report": (
            classification_report_path,
            _strict_json_load(classification_report_path),
        ),
    }
    report_status = {
        name: {
            "paths_relative_to": report.get("paths_relative_to"),
            "portable_links_resolve": _resolve_report_links(path, report),
        }
        for name, (path, report) in fresh_reports.items()
    }
    if not all(
        status["paths_relative_to"] == "report_directory"
        and status["portable_links_resolve"]
        for status in report_status.values()
    ):
        raise ValueError("Fresh Phase 1 reports do not satisfy portable-path contract")

    reproduced_hashes: dict[str, str] = {}
    hash_matches: dict[str, bool] = {}
    for split_name in SPLIT_NAMES:
        path = reproduced_root / "data" / "processed" / f"{split_name}_labeled.csv"
        reproduced_hashes[split_name] = _sha256(path)
        hash_matches[split_name] = (
            reproduced_hashes[split_name] == authoritative_hashes[split_name]
        )
    if not all(hash_matches.values()):
        changed = [name for name, matches in hash_matches.items() if not matches]
        raise ValueError(
            "Reproduced labeled CSV hashes differ from authoritative inputs: "
            + ", ".join(changed)
        )
    return {
        "output_root": report_relative_path(reproduced_root, output_report),
        "reports": report_status,
        "reproduced_labeled_sha256": reproduced_hashes,
        "labeled_hash_matches": hash_matches,
        "all_labeled_csvs_byte_identical": True,
    }


def _preflight(
    paths: SpikeInputContractPaths,
    *,
    overwrite_generated: bool,
) -> None:
    """Validate protected inputs and the sole generated output before writes."""
    for name, path in paths.protected_inputs().items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected input {name}: {path}")
    if not paths.reproduced_root.is_dir():
        raise FileNotFoundError(
            f"Missing isolated Phase 1 reproduction root: {paths.reproduced_root}"
        )

    output = paths.output_report
    protected = (*paths.protected_inputs().values(), *PROTECTED_BASELINE_PATHS)
    for source in protected:
        if _same_path(output, source):
            raise ValueError(f"Contract output aliases protected baseline file: {source}")
    if output.exists() or output.is_symlink():
        if output.is_dir():
            raise ValueError(f"Contract output path is a directory: {output}")
        if not overwrite_generated:
            raise FileExistsError(
                "Spike input contract already exists; pass --overwrite-generated "
                f"to replace only this generated report: {output}"
            )
        if output.is_symlink() or output.stat().st_nlink > 1:
            raise ValueError(f"Refusing to overwrite aliased contract report: {output}")


def _write_and_verify_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write deterministic strict JSON and verify its round trip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(report), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(serialized)
            temporary_name = temporary_file.name
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    saved = _strict_json_load(output_path)
    if saved != dict(report):
        raise RuntimeError("Saved spike input contract does not match generated report")
    if output_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("Saved spike input contract serialization is not deterministic")


def _audit_expectations_report(expectations: AuditExpectations) -> dict[str, Any]:
    """Serialize snapshot checks with an explicit non-formula role."""
    return {
        "role": "regression_assertions_not_calculation_inputs",
        "splits": {
            name: {
                "rows": value.rows,
                "start_date": value.start_date,
                "end_date": value.end_date,
            }
            for name, value in expectations.splits.items()
        },
        "gap_dates": {
            name: list(values) for name, values in expectations.gap_dates.items()
        },
        "classification_q75": expectations.classification_q75,
    }


def run_spike_input_contract_pipeline(
    paths: SpikeInputContractPaths,
    *,
    overwrite_generated: bool = False,
    expectations: AuditExpectations = DEFAULT_AUDIT_EXPECTATIONS,
) -> dict[str, Any]:
    """Validate Phase 2 inputs and save one portable provenance contract.

    Args:
        paths: Protected labeled inputs, Phase 1 reports, reproduction root,
            and the sole generated report destination.
        overwrite_generated: Permit replacing only ``paths.output_report``.
        expectations: Snapshot-specific regression assertions.

    Returns:
        The exact JSON-ready report written to disk.

    Raises:
        FileExistsError: If the generated report exists without permission.
        ValueError: If any input, path, semantic, or provenance check fails.
        RuntimeError: If a protected input changes or report verification fails.
    """
    _preflight(paths, overwrite_generated=overwrite_generated)
    checksums_before = {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    }

    frames = {
        name: load_labeled_split(path, name)
        for name, path in paths.labeled_inputs().items()
    }
    split_report = _strict_json_load(paths.split_report)
    classification_report = _strict_json_load(paths.classification_report)
    validated = validate_input_contract(
        frames,
        split_report,
        classification_report,
        expectations=expectations,
    )
    authoritative_hashes = {
        name: checksums_before[name] for name in SPLIT_NAMES
    }
    reproduction = _collect_reproduction_evidence(
        paths.reproduced_root,
        authoritative_hashes,
        paths.output_report,
    )

    checksums_after_validation = {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    }
    if checksums_after_validation != checksums_before:
        raise RuntimeError("Protected Phase 1 inputs changed during validation")

    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "paths_relative_to": "report_directory",
        "experiment_root": report_relative_path(
            paths.experiment_root, paths.output_report
        ),
        "frozen_rules": {
            "detection_variable": "abs(return_1d)",
            "primary_iqr_multiplier": PRIMARY_SPIKE_IQR_MULTIPLIER,
            "comparison_rule_identifier": SPIKE_COMPARISON_RULE,
            "comparison_formula": "abs(return_1d) > Q3 + 3 * IQR",
            "target_backward_reach": SPIKE_TARGET_BACKWARD_REACH,
            "feature_forward_reach": SPIKE_FEATURE_FORWARD_REACH,
            "affected_window": "inclusive [s-5, s+19]",
            "rsi_policy": SPIKE_RSI_POLICY,
            "threshold_source": "original_train_only",
        },
        "required_columns": list(REQUIRED_COLUMNS),
        "model_columns": list(MODEL_COLUMNS),
        "inputs": {
            name: {
                "path": report_relative_path(path, paths.output_report),
                "sha256": authoritative_hashes[name],
                **validated["input_stats"][name],
            }
            for name, path in paths.labeled_inputs().items()
        },
        "split_contract": validated["split_contract"],
        "classification_contract": validated["classification_contract"],
        "saved_report_provenance": {
            "reports": {
                "data_split_report": {
                    "path": report_relative_path(
                        paths.split_report, paths.output_report
                    ),
                    **validated["saved_report_provenance"]["data_split_report"],
                },
                "classification_threshold_report": {
                    "path": report_relative_path(
                        paths.classification_report, paths.output_report
                    ),
                    **validated["saved_report_provenance"][
                        "classification_threshold_report"
                    ],
                },
            },
            "interpretation": (
                "Non-portable saved paths are an artifact provenance discrepancy; "
                "they do not by themselves establish a calculation-code defect."
            ),
            "isolated_phase1_reproduction": reproduction,
        },
        "source_checksums": {
            name: {
                "before": checksums_before[name],
                "after": checksums_after_validation[name],
                "unchanged": True,
            }
            for name in checksums_before
        },
        "expected_audit_checks": _audit_expectations_report(expectations),
    }
    _write_and_verify_report(report, paths.output_report)

    checksums_after_write = {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    }
    if checksums_after_write != checksums_before:
        raise RuntimeError("Protected Phase 1 inputs changed while writing contract")
    return report


def _build_parser() -> argparse.ArgumentParser:
    """Create the M1 contract command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 labeled inputs and save the Phase 2 M1 contract."
    )
    parser.add_argument(
        "--reproduced-root",
        type=Path,
        required=True,
        help="Fresh isolated Phase 1 output root used as portability evidence.",
    )
    parser.add_argument("--train", type=Path, default=TRAIN_LABELED_DATA_PATH)
    parser.add_argument(
        "--validation", type=Path, default=VALIDATION_LABELED_DATA_PATH
    )
    parser.add_argument("--test", type=Path, default=TEST_LABELED_DATA_PATH)
    parser.add_argument("--split-report", type=Path, default=DATA_SPLIT_REPORT_PATH)
    parser.add_argument(
        "--classification-report",
        type=Path,
        default=CLASSIFICATION_THRESHOLD_REPORT_PATH,
    )
    parser.add_argument(
        "--output", type=Path, default=SPIKE_INPUT_CONTRACT_REPORT_PATH
    )
    parser.add_argument(
        "--overwrite-generated",
        action="store_true",
        help="Replace only the declared generated spike input contract report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the M1 CLI and return a process exit code."""
    arguments = _build_parser().parse_args(argv)
    paths = SpikeInputContractPaths(
        train_labeled=arguments.train,
        validation_labeled=arguments.validation,
        test_labeled=arguments.test,
        split_report=arguments.split_report,
        classification_report=arguments.classification_report,
        reproduced_root=arguments.reproduced_root,
        output_report=arguments.output,
    )
    try:
        report = run_spike_input_contract_pipeline(
            paths, overwrite_generated=arguments.overwrite_generated
        )
    except Exception as exc:  # noqa: BLE001 - CLI converts validation failures to exit 1.
        print(f"Spike input contract failed: {exc}", file=sys.stderr)
        return 1
    print(f"Spike input contract saved: {paths.output_report}")
    print(
        "Authoritative labeled SHA-256: "
        + ", ".join(
            f"{name}={report['inputs'][name]['sha256']}" for name in SPLIT_NAMES
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
