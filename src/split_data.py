"""Modeling-row preparation and leakage-aware chronological splitting."""

import hashlib
import json
from numbers import Real
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import (
    CLEAN_DATA_PATH,
    DATA_SPLIT_REPORT_PATH,
    FEATURE_DATA_PATH,
    RAW_DATA_PATH,
    REGRESSION_TARGET_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from src.build_features import FEATURE_COLUMNS
from src.build_targets import TARGET_COLUMN
from src.load_data import load_qqq_data
from src.report_paths import report_relative_path


REGRESSION_TARGET = TARGET_COLUMN
MODEL_COLUMNS = [*FEATURE_COLUMNS, REGRESSION_TARGET]
SPLIT_GAP = 5
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
FORBIDDEN_PRICE_COLUMNS = {"Adj Close", "adj_close", "Adjusted Close"}
PROTECTED_SOURCE_PATHS = (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    REGRESSION_TARGET_DATA_PATH,
)


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    return Path(first).resolve(strict=False) == Path(second).resolve(strict=False)


def _format_date(value: Any) -> str | None:
    """Format a valid date-like scalar for strict JSON."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _sha256(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file without modifying it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_schema_and_dates(df: pd.DataFrame) -> None:
    """Validate required schema and chronological Date invariants."""
    forbidden = sorted(FORBIDDEN_PRICE_COLUMNS.intersection(df.columns))
    if forbidden:
        raise ValueError("Forbidden adjusted-close columns found: " + ", ".join(forbidden))

    required_columns = ["Date", "Close", *MODEL_COLUMNS]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if not is_datetime64_any_dtype(df["Date"]):
        raise ValueError("Date must have a pandas datetime dtype")
    if df["Date"].isna().any():
        raise ValueError("Date contains NaT values")
    if df["Date"].duplicated().any():
        raise ValueError("Date contains duplicate values")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Date must be sorted from oldest to newest")

    numeric_columns = ["Close", *MODEL_COLUMNS]
    nonnumeric = [
        column for column in numeric_columns if not is_numeric_dtype(df[column])
    ]
    if nonnumeric:
        raise ValueError("Columns must have numeric dtype: " + ", ".join(nonnumeric))


def prepare_modeling_data(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = REGRESSION_TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove rows unusable for modeling and report NaN/Infinity diagnostics.

    Only the selected feature and target columns participate in filtering.
    Infinity is counted, converted to NaN in a deep copy, and then removed via
    ``dropna(subset=...)``. The input DataFrame is never modified.

    Args:
        df: Chronological target DataFrame with a datetime Date column.
        feature_columns: Features used by the model. Defaults to the project's
            eight feature columns.
        target_column: Regression target used by the model.

    Returns:
        The modeling-ready DataFrame and filtering diagnostics.
    """
    selected_features = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    selected_model_columns = [*selected_features, target_column]
    if selected_features == FEATURE_COLUMNS and target_column == REGRESSION_TARGET:
        _validate_schema_and_dates(df)
    else:
        forbidden = sorted(FORBIDDEN_PRICE_COLUMNS.intersection(df.columns))
        if forbidden:
            raise ValueError(
                "Forbidden adjusted-close columns found: " + ", ".join(forbidden)
            )
        required = ["Date", "Close", *selected_model_columns]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        if not is_datetime64_any_dtype(df["Date"]):
            raise ValueError("Date must have a pandas datetime dtype")
        if df["Date"].isna().any() or df["Date"].duplicated().any():
            raise ValueError("Date contains missing or duplicate values")
        if not df["Date"].is_monotonic_increasing:
            raise ValueError("Date must be sorted from oldest to newest")
        nonnumeric = [
            column
            for column in ["Close", *selected_model_columns]
            if not is_numeric_dtype(df[column])
        ]
        if nonnumeric:
            raise ValueError("Columns must have numeric dtype: " + ", ".join(nonnumeric))

    nan_counts = {
        column: int(df[column].isna().sum()) for column in selected_model_columns
    }
    positive_infinity_counts = {
        column: int(np.isposinf(df[column].to_numpy(dtype=float, na_value=np.nan)).sum())
        for column in selected_model_columns
    }
    negative_infinity_counts = {
        column: int(np.isneginf(df[column].to_numpy(dtype=float, na_value=np.nan)).sum())
        for column in selected_model_columns
    }
    nan_row_mask = df[selected_model_columns].isna().any(axis=1)
    infinity_row_mask = pd.Series(False, index=df.index)
    for column in selected_model_columns:
        infinity_row_mask |= np.isinf(
            df[column].to_numpy(dtype=float, na_value=np.nan)
        )
    invalid_row_mask = nan_row_mask | infinity_row_mask

    model_df = df.copy(deep=True)
    model_df[selected_model_columns] = model_df[selected_model_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    model_df = model_df.dropna(subset=selected_model_columns).copy()
    model_df = model_df.reset_index(drop=True)

    report: dict[str, Any] = {
        "rows_before_filtering": int(len(df)),
        "rows_after_filtering": int(len(model_df)),
        "rows_removed": int(len(df) - len(model_df)),
        "nan_counts_before_filtering": nan_counts,
        "positive_infinity_counts_before_filtering": positive_infinity_counts,
        "negative_infinity_counts_before_filtering": negative_infinity_counts,
        "rows_with_nan": int(nan_row_mask.sum()),
        "rows_with_infinity": int(infinity_row_mask.sum()),
        "rows_with_invalid_model_values": int(invalid_row_mask.sum()),
        "date_min_before_filtering": _format_date(df["Date"].min()),
        "date_max_before_filtering": _format_date(df["Date"].max()),
        "date_min_after_filtering": (
            _format_date(model_df["Date"].min()) if not model_df.empty else None
        ),
        "date_max_after_filtering": (
            _format_date(model_df["Date"].max()) if not model_df.empty else None
        ),
    }
    return model_df, report


def _validate_split_parameters(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    gap: int,
) -> None:
    """Validate ratios and gap without accepting booleans as numbers."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(
        isinstance(ratio, bool)
        or not isinstance(ratio, Real)
        or not 0 < ratio < 1
        for ratio in ratios
    ):
        raise ValueError("Split ratios must be numeric values between 0 and 1")
    if not np.isclose(sum(ratios), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Split ratios must sum to 1")
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
        raise ValueError("gap must be a non-negative integer")


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    """Return JSON-ready minimum and maximum dates for a frame."""
    if df.empty:
        return {"start": None, "end": None}
    return {
        "start": _format_date(df["Date"].iloc[0]),
        "end": _format_date(df["Date"].iloc[-1]),
    }


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    gap: int = SPLIT_GAP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split modeling-ready data chronologically with two purging gaps.

    Ratios are applied only after reserving two gaps. Returned split indices
    retain their positions in ``df`` so leakage boundaries remain auditable.
    """
    _validate_split_parameters(train_ratio, validation_ratio, test_ratio, gap)
    _validate_schema_and_dates(df)

    values = df[MODEL_COLUMNS].to_numpy(dtype=float, na_value=np.nan)
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("Modeling data must not contain NaN or Infinity")

    total_gap_rows = 2 * int(gap)
    usable_rows = len(df) - total_gap_rows
    n_train = int(np.floor(usable_rows * train_ratio))
    n_validation = int(np.floor(usable_rows * validation_ratio))
    n_test = int(usable_rows - n_train - n_validation)
    if usable_rows <= 0 or min(n_train, n_validation, n_test) < 1:
        raise ValueError("Insufficient rows for three non-empty splits and two gaps")

    train_start, train_end = 0, n_train
    gap1_start, gap1_end = train_end, train_end + gap
    validation_start = gap1_end
    validation_end = validation_start + n_validation
    gap2_start, gap2_end = validation_end, validation_end + gap
    test_start, test_end = gap2_end, len(df)

    train = df.iloc[train_start:train_end].copy()
    gap1 = df.iloc[gap1_start:gap1_end]
    validation = df.iloc[validation_start:validation_end].copy()
    gap2 = df.iloc[gap2_start:gap2_end]
    test = df.iloc[test_start:test_end].copy()

    split_report: dict[str, Any] = {
        "split_ratios": {
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": float(test_ratio),
        },
        "gap_size": int(gap),
        "total_gap_rows": int(total_gap_rows),
        "usable_rows": int(usable_rows),
        "split_rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "positions": {
            "train": {"start": train_start, "end": train_end - 1},
            "gap1": {
                "start": gap1_start if gap else None,
                "end": gap1_end - 1 if gap else None,
            },
            "validation": {"start": validation_start, "end": validation_end - 1},
            "gap2": {
                "start": gap2_start if gap else None,
                "end": gap2_end - 1 if gap else None,
            },
            "test": {"start": test_start, "end": test_end - 1},
        },
        "date_ranges": {
            "train": _date_range(train),
            "validation": _date_range(validation),
            "test": _date_range(test),
        },
        "gap_dates": {
            "gap1": [_format_date(date) for date in gap1["Date"]],
            "gap2": [_format_date(date) for date in gap2["Date"]],
        },
    }
    return train, validation, test, split_report


def _ensure_safe_outputs(input_path: str | Path, outputs: list[str | Path]) -> None:
    """Ensure outputs are distinct and do not overwrite source datasets."""
    all_paths = [Path(path) for path in outputs]
    for output in all_paths:
        if _same_path(input_path, output):
            raise ValueError("Input and output paths must refer to different files")
        for protected in PROTECTED_SOURCE_PATHS:
            if _same_path(output, protected):
                raise ValueError(f"Refusing to overwrite source data file: {output}")
    resolved = [path.resolve(strict=False) for path in all_paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("All output paths must refer to different files")


def _save_split(df: pd.DataFrame, path: str | Path) -> None:
    """Save one split without its DataFrame index."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, date_format="%Y-%m-%d")


def _verify_saved_split(expected: pd.DataFrame, path: str | Path) -> None:
    """Reload one saved split and compare it with the returned frame."""
    output_path = Path(path)
    saved = pd.read_csv(output_path)
    saved["Date"] = pd.to_datetime(saved["Date"], errors="raise").astype(
        "datetime64[ns]"
    )
    pd.testing.assert_frame_equal(
        saved.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def run_split_pipeline(
    input_path: str | Path = REGRESSION_TARGET_DATA_PATH,
    train_output_path: str | Path = TRAIN_DATA_PATH,
    validation_output_path: str | Path = VALIDATION_DATA_PATH,
    test_output_path: str | Path = TEST_DATA_PATH,
    report_output_path: str | Path = DATA_SPLIT_REPORT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare, chronologically split, save, verify, and report modeling data."""
    output_paths = [
        train_output_path,
        validation_output_path,
        test_output_path,
        report_output_path,
    ]
    _ensure_safe_outputs(input_path, output_paths)
    checksum_before = _sha256(input_path)

    source_df = load_qqq_data(input_path)
    try:
        source_df["Date"] = pd.to_datetime(
            source_df["Date"], errors="raise"
        ).astype("datetime64[ns]")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Could not parse modeling-data Date column: {exc}") from exc

    model_df, preparation_report = prepare_modeling_data(source_df)
    train, validation, test, split_report = chronological_split(model_df)

    for frame, path in (
        (train, train_output_path),
        (validation, validation_output_path),
        (test, test_output_path),
    ):
        _save_split(frame, path)
        _verify_saved_split(frame, path)

    checksum_after = _sha256(input_path)
    source_unchanged = checksum_before == checksum_after
    if not source_unchanged:
        raise RuntimeError("Source input checksum changed during split pipeline")

    report: dict[str, Any] = {
        "paths_relative_to": "report_directory",
        "input_path": report_relative_path(input_path, report_output_path),
        "output_paths": {
            "train": report_relative_path(train_output_path, report_output_path),
            "validation": report_relative_path(
                validation_output_path, report_output_path
            ),
            "test": report_relative_path(test_output_path, report_output_path),
            "report": report_relative_path(report_output_path, report_output_path),
        },
        "feature_columns": FEATURE_COLUMNS.copy(),
        "regression_target": REGRESSION_TARGET,
        **preparation_report,
        **split_report,
        "source_checksum_before": checksum_before,
        "source_checksum_after": checksum_after,
        "source_input_unchanged": source_unchanged,
    }
    report_path = Path(report_output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, allow_nan=False)
        report_file.write("\n")
    return train, validation, test


def main() -> None:
    """Run the default split pipeline and print its saved report."""
    run_split_pipeline()
    print(DATA_SPLIT_REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
