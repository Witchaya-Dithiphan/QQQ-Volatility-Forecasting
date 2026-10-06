"""Unit and read-only audit tests for the canonical direct-spike detector."""

from __future__ import annotations

import builtins
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from config import (
    PRIMARY_SPIKE_IQR_MULTIPLIER,
    SPIKE_INPUT_CONTRACT_REPORT_PATH,
    TRAIN_LABELED_DATA_PATH,
)
from src.detect_spikes import (
    apply_spike_detector,
    fit_primary_spike_detector,
)


def _data(*returns: object, index: pd.Index | None = None) -> pd.DataFrame:
    """Build a minimal detector input without relying on production data."""
    selected_index = pd.RangeIndex(len(returns)) if index is None else index
    return pd.DataFrame(
        {
            "return_1d": pd.Series(list(returns), index=selected_index),
            "unrelated": pd.Series(range(len(returns)), index=selected_index),
        },
        index=selected_index,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_fit_returns_expected_immutable_metadata_from_original_train() -> None:
    # For |returns| = [0, 1, 2, 3], linear Q1=.75 and Q3=2.25.
    train = _data(0.0, -1.0, 2.0, -3.0)

    fitted = fit_primary_spike_detector(train)

    assert fitted.source_split == "original_train"
    assert fitted.return_column == "return_1d"
    assert fitted.quantile_method == "linear"
    assert fitted.q1 == pytest.approx(0.75)
    assert fitted.q3 == pytest.approx(2.25)
    assert fitted.iqr == pytest.approx(1.5)
    assert fitted.multiplier == PRIMARY_SPIKE_IQR_MULTIPLIER == 3.0
    assert fitted.threshold == pytest.approx(6.75)
    assert fitted.comparison_rule == "abs(return_1d) > threshold"

    with pytest.raises(FrozenInstanceError):
        fitted.threshold = 0.0  # type: ignore[misc]


def test_apply_uses_strict_greater_than_and_preserves_index() -> None:
    fitted = fit_primary_spike_detector(_data(0.0, 1.0, 2.0, 3.0))
    above = np.nextafter(fitted.threshold, np.inf)
    index = pd.Index(["equal+", "equal-", "above+", "above-"], name="case")
    candidate = _data(
        fitted.threshold,
        -fitted.threshold,
        above,
        -above,
        index=index,
    )

    result = apply_spike_detector(candidate, fitted)

    expected = pd.Series(
        [False, False, True, True], index=index, dtype=bool, name="is_spike"
    )
    pd.testing.assert_series_equal(result, expected)


def test_positive_and_negative_returns_are_symmetric() -> None:
    fitted = fit_primary_spike_detector(_data(0.0, 0.1, 0.2, 0.3))
    magnitude = np.nextafter(fitted.threshold, np.inf)

    result = apply_spike_detector(_data(magnitude, -magnitude), fitted)

    assert result.tolist() == [True, True]


@pytest.mark.parametrize(
    "mutated_returns",
    [
        pytest.param([1000.0, -2000.0], id="validation"),
        pytest.param([3000.0, -4000.0], id="test"),
    ],
)
def test_non_train_mutation_cannot_change_train_fit(
    mutated_returns: list[float],
) -> None:
    train = _data(0.0, 0.01, -0.02, 0.03, -0.04)
    diagnostic_split = _data(0.001, 0.002)
    fitted = fit_primary_spike_detector(train)
    original_threshold = fitted.threshold

    diagnostic_split.loc[:, "return_1d"] = mutated_returns
    flags = apply_spike_detector(diagnostic_split, fitted)

    assert flags.all()
    assert fitted.threshold == original_threshold
    assert fitted == fit_primary_spike_detector(train)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (pd.DataFrame({"other": [0.1]}), "missing required column"),
        (pd.DataFrame({"return_1d": pd.Series(dtype=float)}), "must not be empty"),
        (_data("not-numeric"), "numeric dtype"),
        (_data(0.0, np.nan), "finite values"),
        (_data(0.0, np.inf), "finite values"),
        (_data(0.0, -np.inf), "finite values"),
    ],
)
def test_fit_rejects_invalid_data(data: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_primary_spike_detector(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (pd.DataFrame({"other": [0.1]}), "missing required column"),
        (pd.DataFrame({"return_1d": pd.Series(dtype=float)}), "must not be empty"),
        (_data("not-numeric"), "numeric dtype"),
        (_data(0.0, np.nan), "finite values"),
        (_data(0.0, np.inf), "finite values"),
        (_data(0.0, -np.inf), "finite values"),
    ],
)
def test_apply_rejects_invalid_data(data: pd.DataFrame, message: str) -> None:
    fitted = fit_primary_spike_detector(_data(0.0, 0.1, 0.2, 0.3))

    with pytest.raises(ValueError, match=message):
        apply_spike_detector(data, fitted)


@pytest.mark.parametrize("value", [True, False, "3", None, np.nan, np.inf, -np.inf, 0, -1])
def test_fit_rejects_invalid_multiplier(value: object) -> None:
    with pytest.raises(ValueError, match="finite positive number"):
        fit_primary_spike_detector(_data(0.0, 0.1), multiplier=value)  # type: ignore[arg-type]


def test_fit_rejects_non_dataframe_input() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        fit_primary_spike_detector(pd.Series([0.0, 0.1]))  # type: ignore[arg-type]


def test_apply_rejects_non_dataframe_and_non_fitted_inputs() -> None:
    fitted = fit_primary_spike_detector(_data(0.0, 0.1, 0.2, 0.3))
    with pytest.raises(TypeError, match="pandas DataFrame"):
        apply_spike_detector(pd.Series([0.0]), fitted)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ExtremeIQRThreshold"):
        apply_spike_detector(_data(0.0), object())  # type: ignore[arg-type]


def test_fit_rejects_nonfinite_derived_threshold() -> None:
    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="non-finite detector metadata"):
        fit_primary_spike_detector(_data(0.0, maximum))


def test_fit_and_apply_do_not_mutate_inputs_or_metadata() -> None:
    train = _data(0.0, -0.1, 0.2, -0.3)
    candidate = _data(0.0, 10.0, -10.0)
    original_train = train.copy(deep=True)
    original_candidate = candidate.copy(deep=True)

    fitted = fit_primary_spike_detector(train)
    metadata_before = fitted
    apply_spike_detector(candidate, fitted)

    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(candidate, original_candidate)
    assert fitted == metadata_before


def test_fit_and_apply_are_deterministic() -> None:
    train = _data(0.0, 0.01, -0.02, 0.03, -0.04, 0.05)
    candidate = _data(-0.5, 0.0, 0.5)

    first_fit = fit_primary_spike_detector(train)
    second_fit = fit_primary_spike_detector(train)
    first_flags = apply_spike_detector(candidate, first_fit)
    second_flags = apply_spike_detector(candidate, second_fit)

    assert first_fit == second_fit
    pd.testing.assert_series_equal(first_flags, second_flags)


def test_pure_api_does_not_open_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_open(*args: object, **kwargs: object) -> Any:
        raise AssertionError("pure detector attempted filesystem access")

    monkeypatch.setattr(builtins, "open", fail_open)
    fitted = fit_primary_spike_detector(_data(0.0, 0.1, 0.2, 0.3))

    assert apply_spike_detector(_data(-10.0, 10.0), fitted).all()


def test_primary_train_audit_matches_m1_verified_regression_checks() -> None:
    if not TRAIN_LABELED_DATA_PATH.is_file():
        pytest.skip("Ignored production Train CSV is unavailable")
    if not SPIKE_INPUT_CONTRACT_REPORT_PATH.is_file():
        pytest.skip("Generated M1 input contract is unavailable")

    contract = json.loads(SPIKE_INPUT_CONTRACT_REPORT_PATH.read_text(encoding="utf-8"))
    expected_hash = contract["inputs"]["train"]["sha256"]
    assert _sha256(TRAIN_LABELED_DATA_PATH) == expected_hash
    train = pd.read_csv(TRAIN_LABELED_DATA_PATH)

    fitted = fit_primary_spike_detector(train)
    flags = apply_spike_detector(train, fitted)

    assert fitted.q1 == pytest.approx(0.0029797377830751)
    assert fitted.q3 == pytest.approx(0.0138707144726510)
    assert fitted.iqr == pytest.approx(0.0108909766895759)
    assert fitted.threshold == pytest.approx(0.0465436445413787)
    assert int(flags.sum()) == 19
