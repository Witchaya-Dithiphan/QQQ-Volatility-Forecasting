"""Utilities for loading raw daily QQQ market data."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
) -> None:
    """Validate that ``df`` contains every required column.

    Args:
        df: DataFrame whose column names will be validated.
        required_columns: Column names to require. When omitted, the QQQ daily
            columns in :data:`REQUIRED_COLUMNS` are used.

    Raises:
        ValueError: If one or more required columns are missing. The error
            message lists the missing columns.

    This function only inspects column names and does not modify ``df``.
    """
    columns_to_check = (
        REQUIRED_COLUMNS if required_columns is None else list(required_columns)
    )
    missing_columns = [
        column for column in columns_to_check if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )


def load_qqq_data(
    file_path: str | Path = "data/raw/qqq_daily.csv",
) -> pd.DataFrame:
    """Load raw daily QQQ data from a CSV file without transforming it.

    Args:
        file_path: Relative or absolute path to the source CSV file.

    Returns:
        The DataFrame returned by :func:`pandas.read_csv`, with its values,
        columns, and row order unchanged.

    Raises:
        FileNotFoundError: If ``file_path`` does not point to an existing file.
        ValueError: If the CSV is empty or required columns are missing.
        pandas.errors.ParserError: If pandas cannot parse malformed CSV data.
        OSError: If the file exists but cannot be read.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"QQQ data file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"QQQ data file is empty: {path}") from exc

    if df.empty:
        raise ValueError(f"QQQ data file contains no data rows: {path}")

    validate_required_columns(df)
    return df
