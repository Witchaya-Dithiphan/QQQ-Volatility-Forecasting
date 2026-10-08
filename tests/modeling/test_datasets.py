"""Read-only accepted loaders, input drift, explicit features and Test denial."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from config import PROJECT_ROOT
from src.modeling import datasets as ds
from src.modeling.contracts import FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET, VARIANT_TRAIN_PATHS
from src.modeling.artifacts import file_sha256


def frame():
    result = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    for col in ["Close", "Volume", "Open", "High", "Low", *FEATURE_COLUMNS]: result[col] = [1., 2., 3.]
    result[REGRESSION_TARGET] = [0.1, 0.2, 0.3]
    result[CLASSIFICATION_TARGET] = [0, 0, 1]
    return result


def test_explicit_features_no_leakage_or_mutation():
    data = frame()
    data["spike_flag"] = 99
    data["auxiliary"] = 1000
    before = data.copy(deep=True)
    split = ds.split_from_frame(data, threshold=0.25)
    assert split.X.shape == (3, 8)
    np.testing.assert_array_equal(split.X, data[list(FEATURE_COLUMNS)])
    pd.testing.assert_frame_equal(data, before)
    assert not split.X.flags.writeable
    assert not split.y_regression.flags.writeable


@pytest.mark.parametrize("drift", ["missing", "null", "infinite", "duplicate_date", "unsorted", "invalid_date", "label", "threshold", "duplicate_column", "column_order"])
def test_input_validation(drift):
    data = frame()
    if drift == "missing": data = data.drop(columns=FEATURE_COLUMNS[0])
    if drift == "null": data.loc[1, FEATURE_COLUMNS[0]] = np.nan
    if drift == "infinite": data.loc[1, FEATURE_COLUMNS[0]] = np.inf
    if drift == "duplicate_date": data.loc[1, "Date"] = data.loc[0, "Date"]
    if drift == "unsorted": data = data.iloc[::-1]
    if drift == "invalid_date": data.loc[1, "Date"] = "bad"
    if drift == "label": data.loc[1, CLASSIFICATION_TARGET] = 2
    if drift == "threshold": data.loc[1, CLASSIFICATION_TARGET] = 1
    if drift == "duplicate_column": data.columns = [*data.columns[:-1], REGRESSION_TARGET]
    if drift == "column_order": data = data[[*data.columns[1:], "Date"]]
    with pytest.raises(ValueError): ds.split_from_frame(data, threshold=0.25)


def test_test_gate_rejects_before_any_io(monkeypatch):
    monkeypatch.setattr(ds, "_load_reports", lambda: pytest.fail("Report IO before denial"))
    with pytest.raises(PermissionError): ds.load_test()
    with pytest.raises(PermissionError): ds.load_test(allow_test=True)


@pytest.mark.parametrize("variant,rows", [("with_spike", 1733), ("non_spike", 1520)])
def test_real_variants_read_only_and_never_touch_test(monkeypatch, variant, rows):
    if not VARIANT_TRAIN_PATHS[variant].exists(): pytest.skip("Accepted CSVs unavailable")
    before = file_sha256(VARIANT_TRAIN_PATHS[variant])
    original = Path.read_bytes
    def guard(path):
        assert "test" not in path.name.lower(), "Test input accessed"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guard)
    pair = ds.load_train_validation(variant)
    assert len(pair.train.X) == rows and len(pair.validation.X) == 371
    assert pair.train.X.shape[1] == 8
    assert pair.train.dates[-1] < pair.validation.dates[0]
    assert file_sha256(VARIANT_TRAIN_PATHS[variant]) == before


def test_unknown_variant_rejected():
    with pytest.raises(ValueError): ds.load_train_validation("other")


def test_checksum_drift_rejected_before_parse(tmp_path):
    path = tmp_path / "input.csv"
    frame().to_csv(path, index=False)
    with pytest.raises(ValueError, match="checksum"):
        ds._read_verified(path, {"sha256": "0" * 64}, threshold=0.25)
