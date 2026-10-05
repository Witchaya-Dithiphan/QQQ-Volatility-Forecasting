"""Regression and classification targets for future QQQ volatility."""

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
    CLASSIFICATION_THRESHOLD_REPORT_PATH,
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    RAW_DATA_MANIFEST_PATH,
    RAW_DATA_PATH,
    REGRESSION_TARGET_DATA_PATH,
    REGRESSION_TARGET_REPORT_PATH,
    TEST_DATA_PATH,
    TEST_LABELED_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
)
from src.build_features import FEATURE_COLUMNS
from src.load_data import load_qqq_data
from src.report_paths import report_relative_path


TARGET_COLUMN = "target_volatility_5d"
FORWARD_HORIZON = 5
DEFAULT_ANNUALIZATION_FACTOR = 252
RETURN_RTOL = 1e-10
RETURN_ATOL = 1e-12
TARGET_REQUIRED_COLUMNS = ["Date", "Close", *FEATURE_COLUMNS]
FORBIDDEN_PRICE_COLUMNS = {"Adj Close", "adj_close", "Adjusted Close"}
CLASSIFICATION_TARGET = "target_high_volatility"
HIGH_VOLATILITY_QUANTILE = 0.75
QUANTILE_METHOD = "linear"
CLASSIFICATION_REQUIRED_COLUMNS = ["Date", "Close", *FEATURE_COLUMNS, TARGET_COLUMN]


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    return Path(first).resolve(strict=False) == Path(second).resolve(strict=False)


def _ensure_safe_target_output(path: str | Path) -> Path:
    """Reject destinations that would overwrite any source dataset."""
    output_path = Path(path)
    for protected_path in (
        RAW_DATA_PATH,
        RAW_DATA_MANIFEST_PATH,
        CLEAN_DATA_PATH,
        FEATURE_DATA_PATH,
    ):
        if _same_path(output_path, protected_path):
            raise ValueError(f"Refusing to overwrite source data file: {output_path}")
    return output_path


def _validate_annualization_factor(annualization_factor: int) -> None:
    """Validate the annualization factor used by the regression target."""
    if (
        isinstance(annualization_factor, bool)
        or not isinstance(annualization_factor, (int, np.integer))
        or annualization_factor <= 0
    ):
        raise ValueError("annualization_factor must be a positive integer")


def validate_regression_target_input(df: pd.DataFrame) -> None:
    """Validate feature data used to build the regression target.

    The existing one-day return must match ``Close.pct_change()`` within
    floating-point tolerances. Natural first-row NaN values are supported.

    Args:
        df: Feature DataFrame to validate without modification.

    Raises:
        ValueError: If schema, date order, types, finite values, or return
            consistency are invalid.
    """
    forbidden_columns = sorted(FORBIDDEN_PRICE_COLUMNS.intersection(df.columns))
    if forbidden_columns:
        raise ValueError(
            "Forbidden adjusted-close columns found: "
            + ", ".join(forbidden_columns)
        )

    missing_columns = [
        column for column in TARGET_REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    if not is_datetime64_any_dtype(df["Date"]):
        raise ValueError("Date must have a pandas datetime dtype")
    if df["Date"].isna().any():
        raise ValueError("Date contains missing values")
    if df["Date"].duplicated().any():
        raise ValueError("Date contains duplicate values")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Date must be sorted from oldest to newest")

    nonnumeric_columns = [
        column
        for column in ("Close", "return_1d")
        if not is_numeric_dtype(df[column])
    ]
    if nonnumeric_columns:
        raise ValueError(
            "Columns must have numeric dtype: " + ", ".join(nonnumeric_columns)
        )
    if df["Close"].isna().any():
        raise ValueError("Close contains missing values")
    if (df["Close"] <= 0).any():
        raise ValueError("Close must contain values greater than zero")

    calculation_values = df[["Close", "return_1d"]].to_numpy(
        dtype=float, na_value=np.nan
    )
    if np.isinf(calculation_values).any():
        raise ValueError("Close and return_1d must not contain Infinity")

    expected_returns = df["Close"].pct_change(periods=1, fill_method=None)
    actual_values = df["return_1d"].to_numpy(dtype=float, na_value=np.nan)
    expected_values = expected_returns.to_numpy(dtype=float, na_value=np.nan)
    if not np.allclose(
        actual_values,
        expected_values,
        rtol=RETURN_RTOL,
        atol=RETURN_ATOL,
        equal_nan=True,
    ):
        mismatch_mask = ~np.isclose(
            actual_values,
            expected_values,
            rtol=RETURN_RTOL,
            atol=RETURN_ATOL,
            equal_nan=True,
        )
        mismatch_positions = np.flatnonzero(mismatch_mask)
        preview = ", ".join(str(int(position)) for position in mismatch_positions[:5])
        raise ValueError(
            "return_1d is inconsistent with Close at row positions: " + preview
        )


def add_regression_target(
    df: pd.DataFrame,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.DataFrame:
    """Add fixed five-row forward annualized realized volatility.

    For row ``t``, the target is the sample standard deviation of
    ``return_1d`` at rows ``t+1`` through ``t+5``, multiplied by the square
    root of ``annualization_factor``. The final five targets remain NaN.

    Args:
        df: Valid feature DataFrame in ascending date order.
        annualization_factor: Positive integer trading periods per year.

    Returns:
        A copy containing all original columns followed by
        ``target_volatility_5d``.
    """
    _validate_annualization_factor(annualization_factor)
    validate_regression_target_input(df)
    result = df.copy(deep=True)

    future_return_1d = result["return_1d"].shift(-1)
    result[TARGET_COLUMN] = (
        future_return_1d
        .rolling(window=FORWARD_HORIZON, min_periods=FORWARD_HORIZON)
        .std(ddof=1)
        .shift(-(FORWARD_HORIZON - 1))
        * np.sqrt(annualization_factor)
    )
    return result


def save_regression_target_data(
    df: pd.DataFrame,
    output_path: str | Path = REGRESSION_TARGET_DATA_PATH,
) -> None:
    """Save regression-target data without overwriting source datasets."""
    path = _ensure_safe_target_output(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")


def _parse_saved_dates(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Parse a saved Date column strictly for round-trip verification."""
    verified = df.copy(deep=True)
    try:
        verified["Date"] = pd.to_datetime(
            verified["Date"], errors="raise"
        ).astype("datetime64[ns]")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Could not parse saved Date column in {path}: {exc}") from exc
    return verified


def _verify_saved_target_data(
    expected: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Reload and verify row order, source columns, and trailing targets."""
    path = Path(output_path)
    saved = _parse_saved_dates(pd.read_csv(path), path)
    if len(saved) != len(expected):
        raise ValueError("Saved target data row count does not match input")
    if not saved["Date"].equals(expected["Date"]):
        raise ValueError("Saved target data changed Date values or order")

    source_columns = [column for column in expected.columns if column != TARGET_COLUMN]
    try:
        pd.testing.assert_frame_equal(
            saved[source_columns],
            expected[source_columns],
            check_dtype=False,
            check_exact=False,
            rtol=RETURN_RTOL,
            atol=RETURN_ATOL,
        )
    except AssertionError as exc:
        raise ValueError("Saved target data changed existing feature values") from exc

    if not saved[TARGET_COLUMN].tail(FORWARD_HORIZON).isna().all():
        raise ValueError("Saved target data must end with five NaN target values")


def _file_sha256(path: str | Path) -> str:
    """Return the uppercase SHA-256 checksum of one artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _create_regression_target_report(
    saved: pd.DataFrame,
    *,
    input_rows: int,
    annualization_factor: int,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    input_checksum_before: str,
    input_checksum_after: str,
) -> dict[str, Any]:
    """Describe the persisted target without fitting a model or threshold."""
    target = saved[TARGET_COLUMN]
    valid = target.dropna()
    boundary_nan_count = min(FORWARD_HORIZON, len(saved))
    quantiles = {
        f"p{int(q * 100):02d}": float(valid.quantile(q))
        for q in (0.05, 0.25, 0.50, 0.75, 0.95)
    }
    statistics: dict[str, Any] = {
        "min": float(valid.min()) if not valid.empty else None,
        "max": float(valid.max()) if not valid.empty else None,
        "mean": float(valid.mean()) if not valid.empty else None,
        "median": float(valid.median()) if not valid.empty else None,
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else None,
        "quantiles": quantiles if not valid.empty else {},
    }
    return {
        "target_column": TARGET_COLUMN,
        "horizon_trading_days": FORWARD_HORIZON,
        "annualization_factor": annualization_factor,
        "formula": "sample_std(return_1d at t+1 through t+5, ddof=1) * sqrt(annualization_factor)",
        "future_returns_description": (
            "The target at trading row t uses return_1d from t+1 through t+5."
        ),
        "unit": "annualized decimal volatility (0.20 = 20%)",
        "input_rows": input_rows,
        "output_rows": len(saved),
        "valid_target_count": int(target.notna().sum()),
        "target_nan_count": int(target.isna().sum()),
        "boundary_nan_count": boundary_nan_count,
        "other_nan_count": int(target.isna().sum()) - boundary_nan_count,
        "nan_reason": (
            "The final five trading rows lack a complete t+1 through t+5 "
            "return window."
        ),
        "statistics": statistics,
        "paths_relative_to": "report_directory",
        "input_path": report_relative_path(input_path, report_path),
        "output_path": report_relative_path(output_path, report_path),
        "report_path": report_relative_path(report_path, report_path),
        "input_checksum_before": input_checksum_before,
        "input_checksum_after": input_checksum_after,
        "source_input_unchanged": input_checksum_before == input_checksum_after,
        "output_checksum": _file_sha256(output_path),
    }


def save_regression_target_report(
    report: dict[str, Any],
    report_path: str | Path = REGRESSION_TARGET_REPORT_PATH,
) -> None:
    """Save a strict JSON report without overwriting configured source files."""
    path = _ensure_safe_target_output(report_path)
    if _same_path(path, REGRESSION_TARGET_DATA_PATH):
        raise ValueError(f"Refusing to overwrite regression target data file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, allow_nan=False)
        report_file.write("\n")


def run_regression_target_pipeline(
    input_path: str | Path = FEATURE_DATA_PATH,
    output_path: str | Path = REGRESSION_TARGET_DATA_PATH,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    report_path: str | Path = REGRESSION_TARGET_REPORT_PATH,
) -> pd.DataFrame:
    """Save and verify regression target data, then report its provenance."""
    if _same_path(input_path, output_path):
        raise ValueError("Input and output paths must refer to different files")
    if _same_path(input_path, report_path) or _same_path(output_path, report_path):
        raise ValueError("Regression report path must differ from input and output")
    _ensure_safe_target_output(output_path)
    _ensure_safe_target_output(report_path)
    if _same_path(report_path, REGRESSION_TARGET_DATA_PATH):
        raise ValueError("Regression report must not overwrite target data")

    input_checksum_before = _file_sha256(input_path)

    feature_df = load_qqq_data(input_path)
    try:
        parsed_dates = pd.to_datetime(feature_df["Date"], errors="raise")
        if isinstance(parsed_dates.dtype, pd.DatetimeTZDtype):
            parsed_dates = parsed_dates.dt.tz_localize(None)
        feature_df["Date"] = parsed_dates.astype("datetime64[ns]")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Could not parse feature-data Date column: {exc}") from exc

    result = add_regression_target(feature_df, annualization_factor)
    save_regression_target_data(result, output_path)
    _verify_saved_target_data(result, output_path)
    input_checksum_after = _file_sha256(input_path)
    if input_checksum_after != input_checksum_before:
        raise RuntimeError("Feature input checksum changed during target pipeline")
    saved = pd.read_csv(output_path)
    report = _create_regression_target_report(
        saved,
        input_rows=len(feature_df),
        annualization_factor=annualization_factor,
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        input_checksum_before=input_checksum_before,
        input_checksum_after=input_checksum_after,
    )
    save_regression_target_report(report, report_path)
    return result


def _validate_regression_target_values(df: pd.DataFrame) -> None:
    """Validate regression-target values used for classification."""
    if df.empty:
        raise ValueError("Classification input DataFrame must not be empty")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {TARGET_COLUMN}")
    if CLASSIFICATION_TARGET in df.columns:
        raise ValueError(f"{CLASSIFICATION_TARGET} already exists")
    if not is_numeric_dtype(df[TARGET_COLUMN]):
        raise ValueError(f"{TARGET_COLUMN} must have numeric dtype")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} must not contain NaN")
    target_values = df[TARGET_COLUMN].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(target_values).any():
        raise ValueError(f"{TARGET_COLUMN} must not contain Infinity")


def _validate_classification_frame(df: pd.DataFrame, split_name: str) -> None:
    """Validate one chronological split before classification labeling."""
    _validate_regression_target_values(df)
    forbidden = sorted(FORBIDDEN_PRICE_COLUMNS.intersection(df.columns))
    if forbidden:
        raise ValueError(
            f"{split_name} contains forbidden adjusted-close columns: "
            + ", ".join(forbidden)
        )
    missing = [
        column for column in CLASSIFICATION_REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{split_name} is missing required columns: " + ", ".join(missing)
        )
    if not is_datetime64_any_dtype(df["Date"]):
        raise ValueError(f"{split_name} Date must have a pandas datetime dtype")
    if df["Date"].isna().any():
        raise ValueError(f"{split_name} Date contains NaT values")
    if df["Date"].duplicated().any():
        raise ValueError(f"{split_name} Date contains duplicate values")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError(f"{split_name} Date must be sorted oldest to newest")


def validate_classification_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Validate schema, chronology, and non-overlap across all three splits."""
    frames = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }
    for split_name, frame in frames.items():
        _validate_classification_frame(frame, split_name)

    if not train_df["Date"].max() < validation_df["Date"].min():
        raise ValueError("Train and validation dates overlap or are out of order")
    if not validation_df["Date"].max() < test_df["Date"].min():
        raise ValueError("Validation and test dates overlap or are out of order")

    date_sets = {name: set(frame["Date"]) for name, frame in frames.items()}
    if (
        date_sets["train"] & date_sets["validation"]
        or date_sets["train"] & date_sets["test"]
        or date_sets["validation"] & date_sets["test"]
    ):
        raise ValueError("Split dates must not overlap")


def calculate_high_volatility_threshold(train_df: pd.DataFrame) -> float:
    """Calculate the train-only 75th percentile of regression volatility."""
    _validate_regression_target_values(train_df)
    threshold = train_df[TARGET_COLUMN].quantile(
        q=HIGH_VOLATILITY_QUANTILE,
        interpolation=QUANTILE_METHOD,
    )
    return float(threshold)


def _validate_threshold(threshold: float) -> None:
    """Reject nonnumeric, boolean, NaN, and infinite thresholds."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
    ):
        raise ValueError("threshold must be a finite numeric value")


def add_classification_target(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Add a binary high-volatility label using a supplied fixed threshold."""
    _validate_threshold(threshold)
    _validate_regression_target_values(df)
    result = df.copy(deep=True)
    result[CLASSIFICATION_TARGET] = (
        result[TARGET_COLUMN] > threshold
    ).astype("int8")
    return result


def _classification_sha256(path: str | Path) -> str:
    """Return the uppercase SHA-256 checksum of a classification source file."""
    return _file_sha256(path)


def _load_classification_split(path: str | Path, split_name: str) -> pd.DataFrame:
    """Load one split and strictly deserialize its Date column."""
    frame = load_qqq_data(path)
    try:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").astype(
            "datetime64[ns]"
        )
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Could not parse {split_name} Date column: {exc}") from exc
    return frame


def _ensure_classification_output_paths(
    source_paths: list[str | Path],
    output_paths: list[str | Path],
) -> None:
    """Ensure labeled outputs and report cannot overwrite any source file."""
    sources = [Path(path).resolve(strict=False) for path in source_paths]
    outputs = [Path(path).resolve(strict=False) for path in output_paths]
    if len(set(outputs)) != len(outputs):
        raise ValueError("All classification output paths must be different")
    for output in outputs:
        if output in sources:
            raise ValueError("Classification output must not overwrite a source split file")
        for protected in (
            RAW_DATA_PATH,
            CLEAN_DATA_PATH,
            FEATURE_DATA_PATH,
            REGRESSION_TARGET_DATA_PATH,
        ):
            if _same_path(output, protected):
                raise ValueError(f"Refusing to overwrite source data file: {output}")


def _save_and_verify_classification_data(
    expected: pd.DataFrame,
    output_path: str | Path,
    threshold: float,
) -> None:
    """Save one labeled split and verify its complete round trip."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected.to_csv(path, index=False, date_format="%Y-%m-%d")
    saved = _parse_saved_dates(pd.read_csv(path), path)
    try:
        pd.testing.assert_frame_equal(
            saved,
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=RETURN_RTOL,
            atol=RETURN_ATOL,
        )
    except AssertionError as exc:
        raise ValueError(f"Saved classification output does not match: {path}") from exc
    expected_labels = (saved[TARGET_COLUMN] > threshold).astype("int8")
    if not saved[CLASSIFICATION_TARGET].astype("int8").equals(expected_labels):
        raise ValueError(f"Saved classification labels are incorrect: {path}")


def _classification_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Create row, class-distribution, and date-range report values."""
    high_count = int(df[CLASSIFICATION_TARGET].eq(1).sum())
    normal_count = int(df[CLASSIFICATION_TARGET].eq(0).sum())
    rows = int(len(df))
    return {
        "rows": rows,
        "normal_count": normal_count,
        "high_count": high_count,
        "normal_ratio": float(normal_count / rows),
        "high_ratio": float(high_count / rows),
        "start_date": df["Date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
    }


def run_classification_target_pipeline(
    train_input_path: str | Path = TRAIN_DATA_PATH,
    validation_input_path: str | Path = VALIDATION_DATA_PATH,
    test_input_path: str | Path = TEST_DATA_PATH,
    train_output_path: str | Path = TRAIN_LABELED_DATA_PATH,
    validation_output_path: str | Path = VALIDATION_LABELED_DATA_PATH,
    test_output_path: str | Path = TEST_LABELED_DATA_PATH,
    threshold_report_path: str | Path = CLASSIFICATION_THRESHOLD_REPORT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Fit a train-only Q75 threshold and label all splits consistently."""
    source_paths = [train_input_path, validation_input_path, test_input_path]
    output_paths = [
        train_output_path,
        validation_output_path,
        test_output_path,
        threshold_report_path,
    ]
    _ensure_classification_output_paths(source_paths, output_paths)
    split_names = ("train", "validation", "test")
    checksums_before = {
        name: _classification_sha256(path)
        for name, path in zip(split_names, source_paths, strict=True)
    }

    train_df = _load_classification_split(train_input_path, "train")
    validation_df = _load_classification_split(validation_input_path, "validation")
    test_df = _load_classification_split(test_input_path, "test")
    validate_classification_splits(train_df, validation_df, test_df)

    threshold = calculate_high_volatility_threshold(train_df)
    train_labeled = add_classification_target(train_df, threshold)
    validation_labeled = add_classification_target(validation_df, threshold)
    test_labeled = add_classification_target(test_df, threshold)

    for labeled, output_path in (
        (train_labeled, train_output_path),
        (validation_labeled, validation_output_path),
        (test_labeled, test_output_path),
    ):
        _save_and_verify_classification_data(labeled, output_path, threshold)

    checksums_after = {
        name: _classification_sha256(path)
        for name, path in zip(split_names, source_paths, strict=True)
    }
    checksum_report = {
        name: {
            "before": checksums_before[name],
            "after": checksums_after[name],
            "unchanged": checksums_before[name] == checksums_after[name],
        }
        for name in split_names
    }
    changed_sources = [
        name for name, values in checksum_report.items() if not values["unchanged"]
    ]
    if changed_sources:
        raise RuntimeError(
            "Classification pipeline changed source files: "
            + ", ".join(changed_sources)
        )

    report: dict[str, Any] = {
        "quantile": HIGH_VOLATILITY_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "threshold_source": "train_only",
        "regression_target": TARGET_COLUMN,
        "classification_target": CLASSIFICATION_TARGET,
        "threshold_decimal": threshold,
        "threshold_percent": threshold * 100,
        "comparison_rule": f"{TARGET_COLUMN} > threshold",
        "train": _classification_stats(train_labeled),
        "validation": _classification_stats(validation_labeled),
        "test": _classification_stats(test_labeled),
        "input_paths": {
            name: report_relative_path(path, threshold_report_path)
            for name, path in zip(split_names, source_paths, strict=True)
        },
        "output_paths": {
            "train": report_relative_path(train_output_path, threshold_report_path),
            "validation": report_relative_path(
                validation_output_path, threshold_report_path
            ),
            "test": report_relative_path(test_output_path, threshold_report_path),
            "report": report_relative_path(
                threshold_report_path, threshold_report_path
            ),
        },
        "paths_relative_to": "report_directory",
        "source_checksums": checksum_report,
    }
    report_path = Path(threshold_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, allow_nan=False)
        report_file.write("\n")
    return train_labeled, validation_labeled, test_labeled, threshold


def main() -> None:
    """Run the default regression-target pipeline and print a summary."""
    result = run_regression_target_pipeline()
    summary: dict[str, Any] = {
        "rows": int(len(result)),
        "date_min": result["Date"].min().strftime("%Y-%m-%d"),
        "date_max": result["Date"].max().strftime("%Y-%m-%d"),
        "target_column": TARGET_COLUMN,
        "valid_targets": int(result[TARGET_COLUMN].notna().sum()),
        "nan_targets": int(result[TARGET_COLUMN].isna().sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
