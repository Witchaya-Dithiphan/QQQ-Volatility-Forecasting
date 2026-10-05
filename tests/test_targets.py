"""Tests for the fixed five-row forward regression target."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_features import build_all_features
from src.build_targets import (
    FORWARD_HORIZON,
    TARGET_COLUMN,
    add_regression_target,
    run_regression_target_pipeline,
    save_regression_target_data,
    validate_regression_target_input,
)


def _feature_data(rows: int = 30) -> pd.DataFrame:
    """Return deterministic valid feature data with varying daily returns."""
    positions = np.arange(rows, dtype=float)
    close = pd.Series(100 * np.exp(0.002 * positions + 0.015 * np.sin(positions)))
    clean = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": 1_000_000 + np.arange(rows) * 10_000,
        }
    )
    return build_all_features(clean)


def _direct_expected_target(
    data: pd.DataFrame,
    position: int,
    annualization_factor: int = 252,
) -> float:
    """Calculate a target directly without pandas rolling operations."""
    future_returns = (
        data["return_1d"].iloc[position + 1 : position + FORWARD_HORIZON + 1].to_numpy()
    )
    return float(np.std(future_returns, ddof=1) * np.sqrt(annualization_factor))


def test_regression_target_uses_next_five_returns() -> None:
    data = _feature_data()
    result = add_regression_target(data)
    assert result[TARGET_COLUMN].iloc[3] == pytest.approx(
        _direct_expected_target(data, 3)
    )


def test_regression_target_excludes_current_return() -> None:
    data = _feature_data()
    position = 7
    baseline = add_regression_target(data)[TARGET_COLUMN].iloc[position]

    modified = data.copy(deep=True)
    modified.loc[modified.index[position] :, "Close"] *= 10
    modified["return_1d"] = modified["Close"].pct_change(fill_method=None)
    assert modified["return_1d"].iloc[position] != data["return_1d"].iloc[position]
    pd.testing.assert_series_equal(
        modified["return_1d"].iloc[position + 1 :],
        data["return_1d"].iloc[position + 1 :],
    )

    changed = add_regression_target(modified)[TARGET_COLUMN].iloc[position]
    assert changed == pytest.approx(baseline)


def test_regression_target_is_annualized() -> None:
    data = _feature_data()
    result = add_regression_target(data, annualization_factor=252)
    expected_unannualized = np.std(data["return_1d"].iloc[1:6], ddof=1)
    assert result[TARGET_COLUMN].iloc[0] == pytest.approx(
        expected_unannualized * np.sqrt(252)
    )


def test_regression_target_uses_ddof_one() -> None:
    data = _feature_data()
    result = add_regression_target(data)
    expected = np.std(data["return_1d"].iloc[5:10], ddof=1) * np.sqrt(252)
    population_value = np.std(data["return_1d"].iloc[5:10], ddof=0) * np.sqrt(252)
    assert result[TARGET_COLUMN].iloc[4] == pytest.approx(expected)
    assert result[TARGET_COLUMN].iloc[4] != pytest.approx(population_value)


def test_last_five_rows_have_nan_target() -> None:
    result = add_regression_target(_feature_data())
    assert result[TARGET_COLUMN].tail(FORWARD_HORIZON).isna().all()
    assert result[TARGET_COLUMN].iloc[:-FORWARD_HORIZON].notna().all()


def test_target_creation_preserves_row_count_and_order() -> None:
    data = _feature_data()
    result = add_regression_target(data)
    assert len(result) == len(data)
    pd.testing.assert_series_equal(result["Date"], data["Date"])


def test_target_creation_does_not_mutate_input() -> None:
    data = _feature_data()
    snapshot = data.copy(deep=True)
    add_regression_target(data)
    pd.testing.assert_frame_equal(data, snapshot)


def test_existing_columns_are_unchanged_and_target_is_appended() -> None:
    data = _feature_data()
    result = add_regression_target(data)
    assert list(result.columns) == list(data.columns) + [TARGET_COLUMN]
    pd.testing.assert_frame_equal(result[data.columns], data)


def test_missing_required_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="Missing required columns: rsi_14"):
        add_regression_target(_feature_data().drop(columns=["rsi_14"]))


@pytest.mark.parametrize("column", ["Adj Close", "adj_close", "Adjusted Close"])
def test_adjusted_close_columns_are_rejected(column: str) -> None:
    data = _feature_data()
    data[column] = data["Close"]
    with pytest.raises(ValueError, match="Forbidden adjusted-close"):
        add_regression_target(data)


def test_duplicate_dates_are_rejected() -> None:
    data = _feature_data()
    data.loc[1, "Date"] = data.loc[0, "Date"]
    with pytest.raises(ValueError, match="duplicate"):
        add_regression_target(data)


def test_unsorted_dates_are_rejected() -> None:
    data = _feature_data().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        add_regression_target(data)


@pytest.mark.parametrize("factor", [0, -252, 1.5, True])
def test_invalid_annualization_factor_is_rejected(factor: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        add_regression_target(_feature_data(), annualization_factor=factor)  # type: ignore[arg-type]


def test_return_consistency_uses_tolerance_and_equal_nan() -> None:
    data = _feature_data()
    data.loc[10, "return_1d"] += 1e-13
    validate_regression_target_input(data)


def test_inconsistent_return_is_rejected() -> None:
    data = _feature_data()
    data.loc[10, "return_1d"] += 0.01
    with pytest.raises(ValueError, match="inconsistent with Close"):
        add_regression_target(data)


def test_infinity_in_calculation_columns_is_rejected() -> None:
    data = _feature_data()
    data.loc[10, "return_1d"] = np.inf
    with pytest.raises(ValueError, match="Infinity"):
        add_regression_target(data)


def test_non_datetime_date_is_rejected_by_calculation_function() -> None:
    data = _feature_data()
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="datetime"):
        add_regression_target(data)


def test_public_function_has_no_horizon_parameter() -> None:
    assert "horizon" not in inspect.signature(add_regression_target).parameters


@pytest.mark.parametrize(
    "protected_path",
    [
        "data/raw/qqq_daily.csv",
        "data/interim/qqq_clean.csv",
        "data/interim/qqq_features.csv",
    ],
)
def test_save_does_not_overwrite_source_files(protected_path: str) -> None:
    with pytest.raises(ValueError, match="source data"):
        save_regression_target_data(
            add_regression_target(_feature_data()), protected_path
        )


def test_pipeline_parses_dates_saves_and_verifies_output(tmp_path: Path) -> None:
    input_path = tmp_path / "features.csv"
    output_path = tmp_path / "targets" / "regression.csv"
    report_path = tmp_path / "reports" / "regression_target_report.json"
    _feature_data().to_csv(input_path, index=False, date_format="%Y-%m-%d")

    result = run_regression_target_pipeline(
        input_path, output_path, report_path=report_path
    )
    saved = pd.read_csv(output_path)

    assert str(result["Date"].dtype) == "datetime64[ns]"
    assert len(saved) == len(result)
    assert saved[TARGET_COLUMN].tail(FORWARD_HORIZON).isna().all()
    assert "target_high_volatility" not in saved.columns
    assert report_path.is_file()


def test_regression_target_report_matches_saved_csv_and_is_repeatable(
    tmp_path: Path,
) -> None:
    """Report counts, statistics, and checksums describe persisted bytes."""
    input_path = tmp_path / "features.csv"
    output_path = tmp_path / "regression.csv"
    report_path = tmp_path / "regression_target_report.json"
    _feature_data().to_csv(input_path, index=False, date_format="%Y-%m-%d")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest().upper()

    run_regression_target_pipeline(input_path, output_path, report_path=report_path)
    saved = pd.read_csv(output_path)
    report_bytes = report_path.read_bytes()
    report = json.loads(
        report_bytes,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    valid = saved[TARGET_COLUMN].dropna()

    assert report["target_column"] == TARGET_COLUMN
    assert report["horizon_trading_days"] == FORWARD_HORIZON
    assert report["annualization_factor"] == 252
    assert "t+1 through t+5" in report["future_returns_description"]
    assert "ddof=1" in report["formula"]
    assert report["unit"] == "annualized decimal volatility (0.20 = 20%)"
    assert report["input_rows"] == report["output_rows"] == len(saved) == 30
    assert report["valid_target_count"] == len(valid) == 25
    assert report["target_nan_count"] == report["boundary_nan_count"] == 5
    assert report["other_nan_count"] == 0
    assert saved[TARGET_COLUMN].tail(5).isna().all()
    assert saved[TARGET_COLUMN].iloc[:-5].notna().all()
    for key, expected in {
        "min": valid.min(),
        "max": valid.max(),
        "mean": valid.mean(),
        "median": valid.median(),
        "std": valid.std(ddof=1),
    }.items():
        assert report["statistics"][key] == pytest.approx(expected)
    for label, quantile in {
        "p05": 0.05,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p95": 0.95,
    }.items():
        assert report["statistics"]["quantiles"][label] == pytest.approx(
            valid.quantile(quantile)
        )
    assert report["paths_relative_to"] == "report_directory"
    assert report["input_path"] == "features.csv"
    assert report["output_path"] == "regression.csv"
    assert report["report_path"] == "regression_target_report.json"
    assert (
        report["input_checksum_before"] == report["input_checksum_after"] == input_hash
    )
    assert report["source_input_unchanged"] is True
    assert (
        report["output_checksum"]
        == hashlib.sha256(output_path.read_bytes()).hexdigest().upper()
    )
    assert hashlib.sha256(input_path.read_bytes()).hexdigest().upper() == input_hash

    run_regression_target_pipeline(input_path, output_path, report_path=report_path)
    assert report_path.read_bytes() == report_bytes


@pytest.mark.parametrize("collision", ["input", "output"])
def test_regression_report_refuses_source_or_target_path(
    tmp_path: Path,
    collision: str,
) -> None:
    """A report cannot overwrite its source CSV or target CSV."""
    input_path = tmp_path / "features.csv"
    output_path = tmp_path / "regression.csv"
    _feature_data().to_csv(input_path, index=False)
    source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    report_path = input_path if collision == "input" else output_path
    with pytest.raises(ValueError, match="report path must differ"):
        run_regression_target_pipeline(input_path, output_path, report_path=report_path)
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == source_hash
    assert not output_path.exists()
