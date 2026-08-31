"""Unit tests for the QQQ cleaning pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean_data import (
    clean_qqq_data,
    run_cleaning_pipeline,
    save_clean_data,
    save_cleaning_report,
)


def _valid_data() -> pd.DataFrame:
    """Return three valid, reverse-chronological QQQ rows."""
    return pd.DataFrame(
        {
            "Date": ["2024-01-04", "2024-01-03", "2024-01-02"],
            "Open": [405.0, 402.0, 400.0],
            "High": [408.0, 406.0, 405.0],
            "Low": [403.0, 401.0, 399.0],
            "Close": [407.0, 403.0, 404.0],
            "Volume": [12_000_000, 11_000_000, 10_000_000],
        }
    )


def test_valid_dataframe_keeps_all_rows() -> None:
    cleaned, report = clean_qqq_data(_valid_data())

    assert len(cleaned) == 3
    assert report["rows_removed"] == 0


def test_cleaning_does_not_modify_input_dataframe() -> None:
    original = _valid_data()
    snapshot = original.copy(deep=True)

    clean_qqq_data(original)

    pd.testing.assert_frame_equal(original, snapshot)


def test_dates_are_datetime_and_sorted_oldest_first() -> None:
    cleaned, _ = clean_qqq_data(_valid_data())

    assert cleaned["Date"].dtype == "datetime64[ns]"
    assert cleaned["Date"].is_monotonic_increasing
    assert list(cleaned.index) == [0, 1, 2]


def test_invalid_date_is_removed_and_reported() -> None:
    data = _valid_data()
    data.loc[1, "Date"] = "not-a-date"

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 2
    assert report["invalid_date_rows"] == 1


def test_exact_duplicate_is_removed() -> None:
    data = pd.concat([_valid_data(), _valid_data().iloc[[0]]], ignore_index=True)

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 3
    assert report["exact_duplicate_rows"] == 1


def test_equivalent_duplicate_date_keeps_one_row() -> None:
    data = _valid_data()
    data["Source"] = ["A", "A", "A"]
    duplicate = data.iloc[[0]].copy()
    duplicate["Source"] = "B"
    data = pd.concat([data, duplicate], ignore_index=True)

    cleaned, report = clean_qqq_data(data)

    assert cleaned["Date"].nunique() == 3
    assert report["duplicate_date_rows"] == 1
    assert "Source" in cleaned.columns


def test_conflicting_duplicate_date_raises_value_error() -> None:
    data = _valid_data()
    conflict = data.iloc[[0]].copy()
    conflict["Open"] = 406.0
    data = pd.concat([data, conflict], ignore_index=True)

    with pytest.raises(ValueError, match=r"2024-01-04"):
        clean_qqq_data(data)


def test_missing_required_value_is_removed() -> None:
    data = _valid_data()
    data.loc[1, "Close"] = np.nan

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 2
    assert report["rows_removed_for_missing_required_values"] == 1
    assert report["missing_values_after"]["Close"] == 0


def test_invalid_numeric_value_is_coerced_reported_and_removed() -> None:
    data = _valid_data()
    data["Volume"] = data["Volume"].astype(object)
    data.loc[1, "Volume"] = "not-a-number"

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 2
    assert report["numeric_values_coerced"]["Volume"] == 1
    assert report["rows_removed_for_missing_required_values"] == 1


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_nonpositive_price_is_removed(price: float) -> None:
    data = _valid_data()
    data.loc[0, "Close"] = price

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 2
    assert report["nonpositive_price_rows"] == 1


def test_high_below_low_is_removed() -> None:
    data = _valid_data()
    data.loc[0, ["High", "Low"]] = [400.0, 403.0]

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 2
    assert report["high_below_low_rows"] == 1


@pytest.mark.parametrize("open_price", [398.0, 409.0])
def test_open_outside_low_high_range_is_removed(open_price: float) -> None:
    data = _valid_data()
    data.loc[0, "Open"] = open_price

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 2
    assert report["open_outside_range_rows"] == 1


def test_negative_volume_is_removed() -> None:
    data = _valid_data()
    data.loc[0, "Volume"] = -1

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 2
    assert report["negative_volume_rows"] == 1


def test_zero_volume_is_preserved_and_reported() -> None:
    data = _valid_data()
    data.loc[0, "Volume"] = 0

    cleaned, report = clean_qqq_data(data)

    assert len(cleaned) == 3
    assert cleaned["Volume"].eq(0).sum() == 1
    assert report["zero_volume_rows_preserved"] == 1


def test_valid_extreme_market_move_is_preserved() -> None:
    data = _valid_data()
    extreme = pd.DataFrame(
        {
            "Date": ["2024-01-05"],
            "Open": [800.0],
            "High": [900.0],
            "Low": [700.0],
            "Close": [850.0],
            "Volume": [100_000_000],
        }
    )
    data = pd.concat([data, extreme], ignore_index=True)

    cleaned, _ = clean_qqq_data(data)

    assert pd.Timestamp("2024-01-05") in cleaned["Date"].values


def test_result_has_no_duplicate_dates() -> None:
    cleaned, _ = clean_qqq_data(_valid_data())

    assert not cleaned["Date"].duplicated().any()


def test_result_has_no_missing_required_values() -> None:
    data = _valid_data()
    data.loc[0, "Volume"] = None

    cleaned, _ = clean_qqq_data(data)

    assert not cleaned[["Date", "Open", "High", "Low", "Close", "Volume"]].isna().any().any()


def test_pipeline_rejects_same_input_and_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "qqq.csv"
    _valid_data().to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="different files"):
        run_cleaning_pipeline(input_path, input_path, tmp_path / "report.json")


def test_save_clean_data_rejects_project_raw_path() -> None:
    with pytest.raises(ValueError, match="raw data"):
        save_clean_data(_valid_data(), "data/raw/qqq_daily.csv")


def test_clean_csv_and_report_are_saved_correctly(tmp_path: Path) -> None:
    cleaned, report = clean_qqq_data(_valid_data())
    csv_path = tmp_path / "nested" / "qqq_clean.csv"
    report_path = tmp_path / "reports" / "cleaning_report.json"

    save_clean_data(cleaned, csv_path)
    serializable_report = {**report, "numpy_integer": np.int64(7)}
    save_cleaning_report(serializable_report, report_path)

    saved_data = pd.read_csv(csv_path)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert list(saved_data.columns) == list(cleaned.columns)
    assert saved_data.loc[0, "Date"] == "2024-01-02"
    assert saved_report["rows_after"] == 3
    assert saved_report["numpy_integer"] == 7


def test_pipeline_loads_cleans_and_saves_with_tmp_files(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "interim" / "clean.csv"
    report_path = tmp_path / "reports" / "cleaning.json"
    _valid_data().to_csv(input_path, index=False)

    cleaned, report = run_cleaning_pipeline(input_path, output_path, report_path)

    assert len(cleaned) == 3
    assert report["rows_after"] == 3
    assert output_path.is_file()
    assert report_path.is_file()
