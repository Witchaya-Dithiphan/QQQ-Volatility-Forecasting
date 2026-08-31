"""Unit tests for raw QQQ data loading and schema validation."""

from pathlib import Path

import pandas as pd
import pytest

from src.load_data import REQUIRED_COLUMNS, load_qqq_data, validate_required_columns


def _valid_data() -> pd.DataFrame:
    """Return a small valid QQQ-like DataFrame for isolated tests."""
    return pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [400.0, 402.0],
            "High": [405.0, 406.0],
            "Low": [399.0, 401.0],
            "Close": [404.0, 403.0],
            "Volume": [10_000_000, 11_000_000],
        }
    )


def test_load_valid_csv_successfully(tmp_path: Path) -> None:
    """A valid CSV is returned without changing its contents or layout."""
    expected = _valid_data()
    csv_path = tmp_path / "valid.csv"
    expected.to_csv(csv_path, index=False)

    actual = load_qqq_data(csv_path)

    pd.testing.assert_frame_equal(actual, expected)


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """A missing source file raises FileNotFoundError containing its path."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        load_qqq_data(missing_path)


def test_load_empty_csv_raises_value_error(tmp_path: Path) -> None:
    """A zero-byte CSV raises a clear ValueError."""
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_qqq_data(empty_path)


def test_load_csv_missing_one_required_column(tmp_path: Path) -> None:
    """A schema missing one required column reports that column."""
    csv_path = tmp_path / "missing_volume.csv"
    _valid_data().drop(columns=["Volume"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=r"Missing required columns: Volume"):
        load_qqq_data(csv_path)


def test_load_csv_missing_multiple_required_columns(tmp_path: Path) -> None:
    """A schema missing several required columns reports all of them."""
    csv_path = tmp_path / "missing_columns.csv"
    _valid_data().drop(columns=["Open", "Low", "Volume"]).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError) as exc_info:
        load_qqq_data(csv_path)

    error_message = str(exc_info.value)
    assert "Open" in error_message
    assert "Low" in error_message
    assert "Volume" in error_message


def test_validate_required_columns_does_not_modify_dataframe() -> None:
    """Schema validation leaves the supplied DataFrame untouched."""
    original = _valid_data()
    before_validation = original.copy(deep=True)

    validate_required_columns(original, REQUIRED_COLUMNS)

    pd.testing.assert_frame_equal(original, before_validation)
