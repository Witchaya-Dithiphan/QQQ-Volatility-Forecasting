"""Causal feature engineering for cleaned daily QQQ data."""

import json
from collections.abc import Mapping
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
    FEATURE_DATA_PATH,
    FEATURE_REPORT_PATH,
    RAW_DATA_PATH,
)
from src.load_data import REQUIRED_COLUMNS, load_qqq_data, validate_required_columns


NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "historical_volatility_5d",
    "historical_volatility_20d",
    "intraday_range",
    "sma_ratio_5_20",
    "rsi_14",
    "volume_zscore_20",
]
ANNUALIZATION_FACTOR = 252
RSI_PERIOD = 14


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Return whether two paths resolve to the same location."""
    return Path(first).resolve(strict=False) == Path(second).resolve(strict=False)


def _ensure_safe_feature_output(path: str | Path) -> Path:
    """Reject destinations that would overwrite source datasets."""
    output_path = Path(path)
    for protected_path in (RAW_DATA_PATH, CLEAN_DATA_PATH):
        if _same_path(output_path, protected_path):
            raise ValueError(f"Refusing to overwrite source data file: {output_path}")
    return output_path


def _format_date(value: Any) -> str | None:
    """Convert a valid date-like scalar to YYYY-MM-DD or return None."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _finite_stat(value: Any) -> int | float | None:
    """Convert a numeric scalar to a strict-JSON-compatible Python value."""
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _json_default(value: Any) -> Any:
    """Convert supported pandas and NumPy values for strict JSON output."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _finite_stat(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _format_date(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def validate_feature_input(df: pd.DataFrame) -> None:
    """Validate cleaned QQQ data without modifying or repairing it.

    Args:
        df: Candidate cleaned DataFrame.

    Raises:
        ValueError: If schema, types, order, uniqueness, missing values, or
            cleaned OHLCV constraints are invalid.
    """
    validate_required_columns(df)

    if not is_datetime64_any_dtype(df["Date"]):
        raise ValueError("Date must have a pandas datetime dtype")

    nonnumeric_columns = [
        column for column in NUMERIC_COLUMNS if not is_numeric_dtype(df[column])
    ]
    if nonnumeric_columns:
        raise ValueError(
            "Numeric columns have non-numeric dtype: "
            + ", ".join(nonnumeric_columns)
        )

    missing_counts = df[REQUIRED_COLUMNS].isna().sum()
    columns_with_missing = [
        column for column in REQUIRED_COLUMNS if missing_counts[column] > 0
    ]
    if columns_with_missing:
        raise ValueError(
            "Missing values in required columns: "
            + ", ".join(columns_with_missing)
        )

    if df["Date"].duplicated().any():
        raise ValueError("Date contains duplicate values")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Date must be sorted from oldest to newest")
    if df[PRICE_COLUMNS].le(0).any(axis=None):
        raise ValueError("Price columns must contain values greater than zero")
    if (df["High"] < df["Low"]).any():
        raise ValueError("High must be greater than or equal to Low")
    if ((df["Open"] < df["Low"]) | (df["Open"] > df["High"])).any():
        raise ValueError("Open must be between Low and High")
    if (df["Volume"] < 0).any():
        raise ValueError("Volume must be greater than or equal to zero")


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal one-day and five-day returns based only on ``Close``."""
    result = df.copy(deep=True)
    result["return_1d"] = result["Close"].pct_change(
        periods=1, fill_method=None
    )
    result["return_5d"] = result["Close"].pct_change(
        periods=5, fill_method=None
    )
    return result


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5-day and 20-day annualized historical volatility features."""
    result = df.copy(deep=True)
    daily_returns = result["Close"].pct_change(periods=1, fill_method=None)
    result["historical_volatility_5d"] = (
        daily_returns.rolling(window=5, min_periods=5).std(ddof=1)
        * np.sqrt(ANNUALIZATION_FACTOR)
    )
    result["historical_volatility_20d"] = (
        daily_returns.rolling(window=20, min_periods=20).std(ddof=1)
        * np.sqrt(ANNUALIZATION_FACTOR)
    )
    return result


def add_price_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add intraday range and 5-to-20-day simple moving-average ratio."""
    result = df.copy(deep=True)
    result["intraday_range"] = (
        result["High"] - result["Low"]
    ) / result["Close"]
    sma_5 = result["Close"].rolling(window=5, min_periods=5).mean()
    sma_20 = result["Close"].rolling(window=20, min_periods=20).mean()
    result["sma_ratio_5_20"] = sma_5 / sma_20
    return result


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Return Wilder's average seeded by the first period's simple mean."""
    averages = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) <= period:
        return averages

    averages.iloc[period] = float(values.iloc[1 : period + 1].mean())
    for position in range(period + 1, len(values)):
        averages.iloc[position] = (
            averages.iloc[position - 1] * (period - 1)
            + float(values.iloc[position])
        ) / period
    return averages


def add_rsi_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add exact 14-period Wilder RSI based only on historical ``Close``."""
    result = df.copy(deep=True)
    delta = result["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_average(gain, RSI_PERIOD)
    avg_loss = _wilder_average(loss, RSI_PERIOD)

    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    result["rsi_14"] = rsi.clip(lower=0, upper=100)
    return result


def add_volume_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal 20-day volume z-score, returning NaN for zero variance."""
    result = df.copy(deep=True)
    volume_mean_20 = result["Volume"].rolling(window=20, min_periods=20).mean()
    volume_std_20 = (
        result["Volume"].rolling(window=20, min_periods=20).std(ddof=1)
    )
    safe_std = volume_std_20.mask(volume_std_20 == 0)
    result["volume_zscore_20"] = (
        result["Volume"] - volume_mean_20
    ) / safe_std
    return result


def _build_all_features_with_diagnostics(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build all features and return infinity counts captured before cleanup."""
    validate_feature_input(df)
    original_columns = list(df.columns)
    featured_df = add_return_features(df)
    featured_df = add_volatility_features(featured_df)
    featured_df = add_price_trend_features(featured_df)
    featured_df = add_rsi_feature(featured_df)
    featured_df = add_volume_feature(featured_df)
    featured_df = featured_df[original_columns + FEATURE_COLUMNS]

    infinite_counts = {
        column: int(np.isinf(featured_df[column]).sum())
        for column in FEATURE_COLUMNS
    }
    featured_df[FEATURE_COLUMNS] = featured_df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    return featured_df, infinite_counts


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all eight causal features without modifying the input DataFrame."""
    featured_df, _ = _build_all_features_with_diagnostics(df)
    return featured_df


def create_feature_report(
    df: pd.DataFrame,
    infinite_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Create a strict-JSON-compatible summary of engineered features.

    Args:
        df: DataFrame containing every feature in :data:`FEATURE_COLUMNS`.
        infinite_counts: Infinity counts captured before they were replaced by
            NaN. Omitted counts default to zero.

    Returns:
        Feature metadata, missing counts, first valid dates, and extrema.
    """
    missing_features = [
        column for column in FEATURE_COLUMNS if column not in df.columns
    ]
    if missing_features:
        raise ValueError("Missing feature columns: " + ", ".join(missing_features))

    infinity_report = {
        column: int((infinite_counts or {}).get(column, 0))
        for column in FEATURE_COLUMNS
    }
    first_valid_date: dict[str, str | None] = {}
    for column in FEATURE_COLUMNS:
        valid_mask = df[column].notna()
        first_valid_date[column] = (
            _format_date(df.loc[valid_mask, "Date"].iloc[0])
            if valid_mask.any()
            else None
        )

    return {
        "rows": int(len(df)),
        "date_min": _format_date(df["Date"].min()) if not df.empty else None,
        "date_max": _format_date(df["Date"].max()) if not df.empty else None,
        "price_column": "Close",
        "annualization_factor": ANNUALIZATION_FACTOR,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "nan_counts": {
            column: int(df[column].isna().sum()) for column in FEATURE_COLUMNS
        },
        "infinite_values_replaced": infinity_report,
        "first_valid_date": first_valid_date,
        "min_values": {
            column: _finite_stat(df[column].min(skipna=True))
            for column in FEATURE_COLUMNS
        },
        "max_values": {
            column: _finite_stat(df[column].max(skipna=True))
            for column in FEATURE_COLUMNS
        },
    }


def save_feature_data(
    df: pd.DataFrame,
    output_path: str | Path = FEATURE_DATA_PATH,
) -> None:
    """Save feature data without overwriting raw or cleaned source data."""
    path = _ensure_safe_feature_output(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")


def save_feature_report(
    report: Mapping[str, Any],
    output_path: str | Path = FEATURE_REPORT_PATH,
) -> None:
    """Save a feature report as strict JSON with no nonstandard NaN values."""
    path = _ensure_safe_feature_output(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(
            dict(report),
            report_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        report_file.write("\n")


def run_feature_pipeline(
    input_path: str | Path = CLEAN_DATA_PATH,
    output_path: str | Path = FEATURE_DATA_PATH,
    report_path: str | Path = FEATURE_REPORT_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load cleaned CSV data, build features, and save data and diagnostics."""
    if _same_path(input_path, output_path):
        raise ValueError("Input and output paths must refer to different files")
    if _same_path(input_path, report_path):
        raise ValueError("Input and report paths must refer to different files")
    if _same_path(output_path, report_path):
        raise ValueError("Output and report paths must refer to different files")

    clean_df = load_qqq_data(input_path)
    try:
        parsed_dates = pd.to_datetime(clean_df["Date"], errors="raise")
        if isinstance(parsed_dates.dtype, pd.DatetimeTZDtype):
            parsed_dates = parsed_dates.dt.tz_localize(None)
        clean_df["Date"] = parsed_dates.astype("datetime64[ns]")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Could not parse clean-data Date column: {exc}") from exc

    featured_df, infinite_counts = _build_all_features_with_diagnostics(clean_df)
    report = create_feature_report(featured_df, infinite_counts)
    save_feature_data(featured_df, output_path)
    save_feature_report(report, report_path)
    return featured_df, report


def main() -> None:
    """Run the default feature pipeline and print its report."""
    _, report = run_feature_pipeline()
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
