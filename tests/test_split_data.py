"""Tests for modeling-data preparation and chronological splitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_features import FEATURE_COLUMNS
from src.build_targets import TARGET_COLUMN
from src.split_data import (
    MODEL_COLUMNS,
    chronological_split,
    prepare_modeling_data,
    run_split_pipeline,
)


def _model_data(rows: int = 100) -> pd.DataFrame:
    """Return complete chronological data suitable for direct splitting."""
    positions = np.arange(rows, dtype=float)
    close = 100 + positions
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Open": close,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 1_000_000 + positions,
            "unused": positions,
        }
    )
    for number, column in enumerate(FEATURE_COLUMNS, start=1):
        data[column] = positions / 1000 + number
    data[TARGET_COLUMN] = positions / 100 + 0.5
    return data


def test_prepare_drops_nan_only_from_model_columns() -> None:
    data = _model_data()
    data.loc[5, "rsi_14"] = np.nan
    prepared, report = prepare_modeling_data(data)
    assert len(prepared) == 99
    assert report["nan_counts_before_filtering"]["rsi_14"] == 1
    assert report["rows_removed"] == 1


def test_prepare_does_not_drop_for_unused_column_nan() -> None:
    data = _model_data()
    data.loc[5, "unused"] = np.nan
    prepared, _ = prepare_modeling_data(data)
    assert len(prepared) == 100
    assert pd.isna(prepared.loc[5, "unused"])


def test_prepare_removes_positive_and_negative_infinity() -> None:
    data = _model_data()
    data.loc[4, "return_1d"] = np.inf
    data.loc[8, "rsi_14"] = -np.inf
    prepared, report = prepare_modeling_data(data)
    assert len(prepared) == 98
    assert report["positive_infinity_counts_before_filtering"]["return_1d"] == 1
    assert report["negative_infinity_counts_before_filtering"]["rsi_14"] == 1
    assert report["rows_with_infinity"] == 2


def test_prepare_does_not_impute_or_change_retained_values() -> None:
    data = _model_data()
    data.loc[5, "return_5d"] = np.nan
    prepared, _ = prepare_modeling_data(data)
    expected = data.drop(index=5).reset_index(drop=True)
    pd.testing.assert_frame_equal(prepared, expected)


def test_prepare_does_not_mutate_input() -> None:
    data = _model_data()
    data.loc[5, "return_5d"] = np.inf
    snapshot = data.copy(deep=True)
    prepare_modeling_data(data)
    pd.testing.assert_frame_equal(data, snapshot)


def test_overlapping_nan_and_infinity_rows_are_counted_once_as_removed() -> None:
    data = _model_data()
    data.loc[5, "return_1d"] = np.nan
    data.loc[5, "return_5d"] = np.inf
    _, report = prepare_modeling_data(data)
    assert report["rows_with_nan"] == 1
    assert report["rows_with_infinity"] == 1
    assert report["rows_with_invalid_model_values"] == 1
    assert report["rows_removed"] == 1


def test_missing_required_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_modeling_data(_model_data().drop(columns=["Close"]))


def test_adjusted_close_is_rejected() -> None:
    data = _model_data()
    data["Adj Close"] = data["Close"]
    with pytest.raises(ValueError, match="adjusted-close"):
        prepare_modeling_data(data)


def test_non_datetime_date_is_rejected_by_pure_function() -> None:
    data = _model_data()
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="datetime"):
        prepare_modeling_data(data)


def test_duplicate_dates_are_rejected() -> None:
    data = _model_data()
    data.loc[1, "Date"] = data.loc[0, "Date"]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_modeling_data(data)


def test_unsorted_dates_are_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        prepare_modeling_data(_model_data().iloc[::-1].reset_index(drop=True))


def test_chronological_split_never_shuffles_rows() -> None:
    data = _model_data()
    train, validation, test, _ = chronological_split(data)
    for split in (train, validation, test):
        assert split["Date"].is_monotonic_increasing
        assert split.index.is_monotonic_increasing


def test_split_ratios_apply_after_reserving_gaps() -> None:
    data = _model_data(rows=100)
    train, validation, test, report = chronological_split(data)
    usable = 100 - 10
    assert len(train) == int(np.floor(usable * 0.70))
    assert len(validation) == int(np.floor(usable * 0.15))
    assert len(test) == usable - len(train) - len(validation)
    assert report["usable_rows"] == usable


def test_five_rows_are_excluded_at_each_boundary() -> None:
    data = _model_data()
    train, validation, test, report = chronological_split(data)
    positions = report["positions"]
    assert positions["gap1"]["end"] - positions["gap1"]["start"] + 1 == 5
    assert positions["gap2"]["end"] - positions["gap2"]["start"] + 1 == 5
    assert validation.index.min() > train.index.max() + 5
    assert test.index.min() > validation.index.max() + 5


def test_gap_rows_do_not_appear_in_any_split() -> None:
    data = _model_data()
    train, validation, test, report = chronological_split(data)
    split_indices = set(train.index) | set(validation.index) | set(test.index)
    for gap_name in ("gap1", "gap2"):
        gap = report["positions"][gap_name]
        assert split_indices.isdisjoint(range(gap["start"], gap["end"] + 1))


def test_splits_do_not_overlap_and_assign_every_non_gap_row_once() -> None:
    data = _model_data()
    train, validation, test, report = chronological_split(data)
    split_indices = [*train.index, *validation.index, *test.index]
    assert len(split_indices) == len(set(split_indices))
    assert len(split_indices) + report["total_gap_rows"] == len(data)


def test_each_split_has_no_modeling_nan_or_infinity() -> None:
    for split in chronological_split(_model_data())[:3]:
        values = split[MODEL_COLUMNS].to_numpy()
        assert not np.isnan(values).any()
        assert not np.isinf(values).any()


@pytest.mark.parametrize(
    "ratios",
    [
        (0.0, 0.5, 0.5),
        (0.7, 0.15, 0.10),
        (-0.1, 0.5, 0.6),
        (True, 0.0, 0.0),
    ],
)
def test_invalid_ratios_are_rejected(ratios: tuple[object, object, object]) -> None:
    with pytest.raises(ValueError, match="ratios"):
        chronological_split(
            _model_data(),
            train_ratio=ratios[0],  # type: ignore[arg-type]
            validation_ratio=ratios[1],  # type: ignore[arg-type]
            test_ratio=ratios[2],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("gap", [-1, 1.5, True])
def test_invalid_gap_is_rejected(gap: object) -> None:
    with pytest.raises(ValueError, match="gap"):
        chronological_split(_model_data(), gap=gap)  # type: ignore[arg-type]


def test_insufficient_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="Insufficient"):
        chronological_split(_model_data(rows=12), gap=5)


def test_pipeline_does_not_overwrite_input_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    _model_data().to_csv(input_path, index=False, date_format="%Y-%m-%d")
    with pytest.raises(ValueError, match="different files"):
        run_split_pipeline(
            input_path,
            input_path,
            tmp_path / "validation.csv",
            tmp_path / "test.csv",
            tmp_path / "report.json",
        )


def test_pipeline_saves_matching_csvs_and_strict_report(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    train_path = tmp_path / "processed" / "train.csv"
    validation_path = tmp_path / "processed" / "validation.csv"
    test_path = tmp_path / "processed" / "test.csv"
    report_path = tmp_path / "reports" / "split.json"
    _model_data().to_csv(input_path, index=False, date_format="%Y-%m-%d")

    train, validation, test = run_split_pipeline(
        input_path, train_path, validation_path, test_path, report_path
    )

    for expected, path in (
        (train, train_path),
        (validation, validation_path),
        (test, test_path),
    ):
        saved = pd.read_csv(path)
        assert len(saved) == len(expected)
        assert list(saved.columns) == list(expected.columns)

    report_text = report_path.read_text(encoding="utf-8")
    assert "NaN" not in report_text
    assert "Infinity" not in report_text
    report = json.loads(
        report_text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert report["source_input_unchanged"] is True
    assert report["paths_relative_to"] == "report_directory"
    assert report["input_path"] == "../input.csv"
    assert report["output_paths"] == {
        "train": "../processed/train.csv",
        "validation": "../processed/validation.csv",
        "test": "../processed/test.csv",
        "report": "split.json",
    }
