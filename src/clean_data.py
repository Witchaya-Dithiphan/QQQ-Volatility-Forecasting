"""Cleaning pipeline for daily QQQ OHLCV data."""

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Support direct execution with ``python src/clean_data.py``. In normal package
# imports and ``python -m src.clean_data``, pytest or Python already exposes the
# project root and this branch does nothing.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import CLEAN_DATA_PATH, CLEANING_REPORT_PATH, RAW_DATA_PATH
from src.load_data import REQUIRED_COLUMNS, load_qqq_data, validate_required_columns


NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def _format_date(value: pd.Timestamp | None) -> str | None:
    """Format a report date as ISO calendar date."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_bounds(values: pd.Series) -> tuple[str | None, str | None]:
    """Return serializable minimum and maximum valid dates."""
    valid_values = values.dropna()
    if valid_values.empty:
        return None, None
    return _format_date(valid_values.min()), _format_date(valid_values.max())


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    return Path(first).resolve(strict=False) == Path(second).resolve(strict=False)


def _json_default(value: Any) -> Any:
    """Convert pandas and NumPy scalar values into JSON-compatible values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _format_date(pd.Timestamp(value))
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _remove_duplicate_dates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove equivalent duplicate dates and reject conflicting OHLCV rows."""
    duplicate_date_mask = df.duplicated(subset=["Date"], keep=False)
    if not duplicate_date_mask.any():
        return df, 0

    duplicate_rows = df.loc[duplicate_date_mask, REQUIRED_COLUMNS]
    conflicting_dates = [
        date
        for date, group in duplicate_rows.groupby("Date", sort=True)
        if len(group.drop_duplicates(subset=REQUIRED_COLUMNS)) > 1
    ]
    if conflicting_dates:
        formatted_dates = ", ".join(
            date.strftime("%Y-%m-%d") for date in conflicting_dates
        )
        raise ValueError(f"Conflicting OHLCV data for duplicate dates: {formatted_dates}")

    rows_before = len(df)
    deduplicated = df.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first")
    return deduplicated, int(rows_before - len(deduplicated))


def clean_qqq_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean daily QQQ OHLCV data and return it with a quality report.

    The input DataFrame is never modified. Invalid dates, missing required
    values, nonpositive prices, impossible OHLC relationships, and negative
    volume rows are removed. Equivalent duplicates are collapsed, while
    conflicting records for the same date raise ``ValueError``.

    Args:
        df: DataFrame produced by :func:`src.load_data.load_qqq_data`.

    Returns:
        A tuple containing the cleaned DataFrame and a JSON-serializable report.

    Raises:
        ValueError: If required columns are missing or a date has conflicting
            OHLCV records.
    """
    validate_required_columns(df)
    cleaned_df = df.copy(deep=True)
    rows_before = len(cleaned_df)

    missing_values_before = {
        column: int(cleaned_df[column].isna().sum())
        for column in REQUIRED_COLUMNS
    }

    parsed_dates = pd.to_datetime(cleaned_df["Date"], errors="coerce")
    if isinstance(parsed_dates.dtype, pd.DatetimeTZDtype):
        parsed_dates = parsed_dates.dt.tz_localize(None)
    parsed_dates = parsed_dates.astype("datetime64[ns]")
    date_min_before, date_max_before = _date_bounds(parsed_dates)
    cleaned_df["Date"] = parsed_dates

    invalid_date_mask = cleaned_df["Date"].isna()
    invalid_date_rows = int(invalid_date_mask.sum())
    cleaned_df = cleaned_df.loc[~invalid_date_mask].copy()

    numeric_values_coerced: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        values_before = cleaned_df[column]
        converted_values = pd.to_numeric(values_before, errors="coerce")
        numeric_values_coerced[column] = int(
            (values_before.notna() & converted_values.isna()).sum()
        )
        cleaned_df[column] = converted_values

    exact_duplicate_rows = int(cleaned_df.duplicated(keep="first").sum())
    cleaned_df = cleaned_df.drop_duplicates(keep="first")
    cleaned_df, duplicate_date_rows = _remove_duplicate_dates(cleaned_df)

    missing_required_mask = cleaned_df[REQUIRED_COLUMNS].isna().any(axis=1)
    rows_removed_for_missing = int(missing_required_mask.sum())
    cleaned_df = cleaned_df.loc[~missing_required_mask].copy()

    nonpositive_price_mask = cleaned_df[PRICE_COLUMNS].le(0).any(axis=1)
    nonpositive_price_rows = int(nonpositive_price_mask.sum())
    cleaned_df = cleaned_df.loc[~nonpositive_price_mask].copy()

    high_below_low_mask = cleaned_df["High"] < cleaned_df["Low"]
    open_outside_range_mask = (
        (cleaned_df["Open"] < cleaned_df["Low"])
        | (cleaned_df["Open"] > cleaned_df["High"])
    )
    high_below_low_rows = int(high_below_low_mask.sum())
    open_outside_range_rows = int(open_outside_range_mask.sum())
    invalid_ohlc_mask = high_below_low_mask | open_outside_range_mask
    cleaned_df = cleaned_df.loc[~invalid_ohlc_mask].copy()

    negative_volume_mask = cleaned_df["Volume"] < 0
    negative_volume_rows = int(negative_volume_mask.sum())
    cleaned_df = cleaned_df.loc[~negative_volume_mask].copy()

    cleaned_df = cleaned_df.sort_values("Date").reset_index(drop=True)
    zero_volume_rows_preserved = int(cleaned_df["Volume"].eq(0).sum())
    missing_values_after = {
        column: int(cleaned_df[column].isna().sum())
        for column in REQUIRED_COLUMNS
    }
    date_min_after, date_max_after = _date_bounds(cleaned_df["Date"])

    report: dict[str, Any] = {
        "rows_before": int(rows_before),
        "rows_after": int(len(cleaned_df)),
        "rows_removed": int(rows_before - len(cleaned_df)),
        "date_min_before": date_min_before,
        "date_max_before": date_max_before,
        "date_min_after": date_min_after,
        "date_max_after": date_max_after,
        "invalid_date_rows": invalid_date_rows,
        "numeric_values_coerced": numeric_values_coerced,
        "exact_duplicate_rows": exact_duplicate_rows,
        "duplicate_date_rows": duplicate_date_rows,
        "missing_values_before": missing_values_before,
        "missing_values_after": missing_values_after,
        "rows_removed_for_missing_required_values": rows_removed_for_missing,
        "nonpositive_price_rows": nonpositive_price_rows,
        "high_below_low_rows": high_below_low_rows,
        "open_outside_range_rows": open_outside_range_rows,
        "negative_volume_rows": negative_volume_rows,
        "zero_volume_rows_preserved": zero_volume_rows_preserved,
    }
    return cleaned_df, report


def save_clean_data(
    df: pd.DataFrame,
    output_path: str | Path = CLEAN_DATA_PATH,
) -> None:
    """Save cleaned data as CSV without allowing raw QQQ data overwrite.

    Parent directories are created when needed. Dates are written in
    ``YYYY-MM-DD`` format and the DataFrame index is omitted.

    Args:
        df: Cleaned QQQ DataFrame to save.
        output_path: Destination CSV path.

    Raises:
        ValueError: If the destination is the project's raw QQQ data file.
    """
    path = Path(output_path)
    if _same_path(path, RAW_DATA_PATH):
        raise ValueError(f"Refusing to overwrite raw data file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")


def save_cleaning_report(
    report: dict[str, Any],
    output_path: str | Path = CLEANING_REPORT_PATH,
) -> None:
    """Save a cleaning report as human-readable JSON.

    Args:
        report: Cleaning metrics to serialize.
        output_path: Destination JSON path.
    """
    path = Path(output_path)
    if _same_path(path, RAW_DATA_PATH):
        raise ValueError(f"Refusing to overwrite raw data file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(
            report,
            report_file,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        report_file.write("\n")


def run_cleaning_pipeline(
    input_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = CLEAN_DATA_PATH,
    report_path: str | Path = CLEANING_REPORT_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, clean, and persist daily QQQ data and its cleaning report.

    Args:
        input_path: Source raw CSV path.
        output_path: Destination cleaned CSV path.
        report_path: Destination cleaning report path.

    Returns:
        The cleaned DataFrame and its report.

    Raises:
        ValueError: If input and output resolve to the same file.
    """
    if _same_path(input_path, output_path):
        raise ValueError("Input and output paths must refer to different files")
    if _same_path(input_path, report_path):
        raise ValueError("Input and report paths must refer to different files")
    if _same_path(output_path, report_path):
        raise ValueError("Output and report paths must refer to different files")

    raw_df = load_qqq_data(input_path)
    cleaned_df, report = clean_qqq_data(raw_df)
    save_clean_data(cleaned_df, output_path)
    save_cleaning_report(report, report_path)
    return cleaned_df, report


def main() -> None:
    """Run the default cleaning pipeline and print its resulting report."""
    _, report = run_cleaning_pipeline()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
