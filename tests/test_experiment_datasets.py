"""Tests for the Phase 2 M1 baseline input contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_features import FEATURE_COLUMNS
from src.build_spike_input_contract import (
    MODEL_COLUMNS,
    REQUIRED_COLUMNS,
    AuditExpectations,
    SpikeInputContractPaths,
    SplitAuditExpectation,
    load_labeled_split,
    main,
    run_spike_input_contract_pipeline,
    validate_input_contract,
)
from src.build_targets import CLASSIFICATION_TARGET, TARGET_COLUMN


@dataclass
class ContractCase:
    """Synthetic contract inputs and their matching audit metadata."""

    frames: dict[str, pd.DataFrame]
    split_report: dict[str, object]
    classification_report: dict[str, object]
    expectations: AuditExpectations


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _frame(start: str, targets: list[float]) -> pd.DataFrame:
    rows = len(targets)
    sequence = np.arange(rows, dtype=float)
    data: dict[str, object] = {
        "Date": pd.bdate_range(start=start, periods=rows),
        "Close": 100.0 + sequence,
        "Volume": 1_000_000.0 + sequence,
        "Open": 99.5 + sequence,
        "High": 101.0 + sequence,
        "Low": 99.0 + sequence,
    }
    for index, column in enumerate(FEATURE_COLUMNS, start=1):
        data[column] = sequence / 100.0 + index
    data[TARGET_COLUMN] = np.asarray(targets, dtype=float)
    data[CLASSIFICATION_TARGET] = np.zeros(rows, dtype="int8")
    return pd.DataFrame(data, columns=REQUIRED_COLUMNS)


def _classification_report(
    frames: dict[str, pd.DataFrame], threshold: float
) -> dict[str, object]:
    report: dict[str, object] = {
        "quantile": 0.75,
        "quantile_method": "linear",
        "threshold_source": "train_only",
        "regression_target": TARGET_COLUMN,
        "classification_target": CLASSIFICATION_TARGET,
        "threshold_decimal": threshold,
        "comparison_rule": f"{TARGET_COLUMN} > threshold",
        "input_paths": {
            name: f"C:\\old-machine\\data\\{name}.csv" for name in frames
        },
        "output_paths": {
            name: f"C:\\old-machine\\data\\{name}_labeled.csv" for name in frames
        },
    }
    for name, frame in frames.items():
        labels = frame[CLASSIFICATION_TARGET]
        report[name] = {
            "rows": len(frame),
            "normal_count": int(labels.eq(0).sum()),
            "high_count": int(labels.eq(1).sum()),
            "start_date": frame["Date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": frame["Date"].iloc[-1].strftime("%Y-%m-%d"),
        }
    return report


def _build_case() -> ContractCase:
    frames = {
        "train": _frame("2020-01-01", [1.0, 2.0, 3.0, 4.0, 4.0]),
        "validation": _frame("2020-01-15", [3.0, 4.0, 4.1, 5.0]),
        "test": _frame("2020-01-28", [2.0, 4.0, 4.0, 8.0]),
    }
    threshold = float(frames["train"][TARGET_COLUMN].quantile(0.75))
    for frame in frames.values():
        frame[CLASSIFICATION_TARGET] = (
            frame[TARGET_COLUMN].gt(threshold).astype("int8")
        )
    split_report: dict[str, object] = {
        "input_path": "C:\\old-machine\\data\\regression_target.csv",
        "output_paths": {
            name: f"C:\\old-machine\\data\\{name}.csv" for name in frames
        },
        "gap_size": 5,
        "total_gap_rows": 10,
        "split_rows": {name: len(frame) for name, frame in frames.items()},
        "positions": {
            "train": {"start": 0, "end": 4},
            "gap1": {"start": 5, "end": 9},
            "validation": {"start": 10, "end": 13},
            "gap2": {"start": 14, "end": 18},
            "test": {"start": 19, "end": 22},
        },
        "date_ranges": {
            name: {
                "start": frame["Date"].iloc[0].strftime("%Y-%m-%d"),
                "end": frame["Date"].iloc[-1].strftime("%Y-%m-%d"),
            }
            for name, frame in frames.items()
        },
        "gap_dates": {
            "gap1": [
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-13",
                "2020-01-14",
            ],
            "gap2": [
                "2020-01-21",
                "2020-01-22",
                "2020-01-23",
                "2020-01-24",
                "2020-01-27",
            ],
        },
        "source_input_unchanged": True,
    }
    expectations = AuditExpectations(
        splits={
            name: SplitAuditExpectation(
                rows=len(frame),
                start_date=frame["Date"].iloc[0].strftime("%Y-%m-%d"),
                end_date=frame["Date"].iloc[-1].strftime("%Y-%m-%d"),
            )
            for name, frame in frames.items()
        },
        gap_dates={
            name: tuple(dates)
            for name, dates in split_report["gap_dates"].items()  # type: ignore[union-attr]
        },
        classification_q75=threshold,
    )
    return ContractCase(
        frames=frames,
        split_report=split_report,
        classification_report=_classification_report(frames, threshold),
        expectations=expectations,
    )


@pytest.fixture
def contract_case() -> ContractCase:
    return _build_case()


def _validate(case: ContractCase) -> dict[str, object]:
    return validate_input_contract(
        case.frames,
        case.split_report,
        case.classification_report,
        expectations=case.expectations,
    )


def test_validate_input_contract_accepts_exact_synthetic_contract(
    contract_case: ContractCase,
) -> None:
    result = _validate(contract_case)

    assert result["classification_contract"]["threshold"] == 4.0
    assert result["classification_contract"]["comparison_operator"] == ">"
    assert result["classification_contract"]["equality_boundary_counts"] == {
        "train": 2,
        "validation": 1,
        "test": 2,
    }
    provenance = result["saved_report_provenance"]
    assert not provenance["data_split_report"]["portable"]
    assert not provenance["classification_threshold_report"]["portable"]
    assert not provenance["data_split_report"]["calculation_bug_inferred"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["return_1d"]), "columns must exactly"),
        (
            lambda frame: frame[[*frame.columns[1:], frame.columns[0]]],
            "columns must exactly",
        ),
        (
            lambda frame: frame.assign(return_1d="not-numeric"),
            "numeric dtype",
        ),
        (
            lambda frame: frame.assign(return_1d=np.nan),
            "must not contain NaN",
        ),
        (
            lambda frame: frame.assign(return_1d=np.inf),
            "must not contain Infinity",
        ),
        (
            lambda frame: frame.assign(target_high_volatility=2),
            "only 0 and 1",
        ),
        (
            lambda frame: frame.assign(
                Date=frame["Date"].mask(frame.index == 1, frame["Date"].iloc[0])
            ),
            "must be unique",
        ),
        (
            lambda frame: frame.iloc[::-1].reset_index(drop=True),
            "oldest to newest",
        ),
    ],
)
def test_split_value_contract_rejects_invalid_data(
    contract_case: ContractCase,
    mutation: object,
    message: str,
) -> None:
    mutate = mutation
    contract_case.frames["train"] = mutate(  # type: ignore[operator]
        contract_case.frames["train"].copy()
    )

    with pytest.raises(ValueError, match=message):
        _validate(contract_case)


def test_load_labeled_split_rejects_unparseable_date(tmp_path: Path) -> None:
    frame = _frame("2020-01-01", [1.0])
    frame["Date"] = frame["Date"].astype(object)
    frame.loc[0, "Date"] = "not-a-date"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="could not be parsed strictly"):
        load_labeled_split(path, "train")


def test_load_labeled_split_rejects_missing_file_and_date_column(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Could not read train"):
        load_labeled_split(tmp_path / "missing.csv", "train")

    path = tmp_path / "missing-date.csv"
    pd.DataFrame({"Close": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column: Date"):
        load_labeled_split(path, "train")


def test_rejects_non_datetime_and_nat_dates(contract_case: ContractCase) -> None:
    contract_case.frames["train"]["Date"] = contract_case.frames["train"][
        "Date"
    ].dt.strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="datetime64 dtype"):
        _validate(contract_case)

    contract_case = _build_case()
    contract_case.frames["train"].loc[0, "Date"] = pd.NaT
    with pytest.raises(ValueError, match="must not contain NaT"):
        _validate(contract_case)


def test_rejects_label_that_uses_greater_than_or_equal(
    contract_case: ContractCase,
) -> None:
    contract_case.frames["train"].loc[
        contract_case.frames["train"][TARGET_COLUMN].eq(4.0), CLASSIFICATION_TARGET
    ] = 1
    contract_case.classification_report = _classification_report(
        contract_case.frames, 4.0
    )

    with pytest.raises(ValueError, match="strict Original Train Q75"):
        _validate(contract_case)


def test_validation_and_test_values_cannot_refit_train_q75(
    contract_case: ContractCase,
) -> None:
    contract_case.frames["validation"][TARGET_COLUMN] *= 1000
    contract_case.frames["test"][TARGET_COLUMN] *= 2000
    for split_name in ("validation", "test"):
        frame = contract_case.frames[split_name]
        frame[CLASSIFICATION_TARGET] = frame[TARGET_COLUMN].gt(4.0).astype("int8")
    contract_case.classification_report = _classification_report(
        contract_case.frames, 4.0
    )

    result = _validate(contract_case)

    assert result["classification_contract"]["threshold"] == 4.0


def test_rejects_overlap_even_when_each_split_is_sorted(
    contract_case: ContractCase,
) -> None:
    validation = contract_case.frames["validation"]
    validation["Date"] = pd.bdate_range("2020-01-03", periods=len(validation))
    contract_case.expectations = AuditExpectations(
        splits={
            **contract_case.expectations.splits,
            "validation": SplitAuditExpectation(4, "2020-01-03", "2020-01-08"),
        },
        gap_dates=contract_case.expectations.gap_dates,
        classification_q75=4.0,
    )

    with pytest.raises(ValueError, match="overlap or are out of order"):
        _validate(contract_case)


def test_rejects_wrong_audit_row_count(contract_case: ContractCase) -> None:
    contract_case.expectations = AuditExpectations(
        splits={
            **contract_case.expectations.splits,
            "train": SplitAuditExpectation(999, "2020-01-01", "2020-01-07"),
        },
        gap_dates=contract_case.expectations.gap_dates,
        classification_q75=4.0,
    )

    with pytest.raises(ValueError, match="row-count audit mismatch"):
        _validate(contract_case)


def test_rejects_wrong_audit_date_range(contract_case: ContractCase) -> None:
    contract_case.expectations = AuditExpectations(
        splits={
            **contract_case.expectations.splits,
            "train": SplitAuditExpectation(5, "1999-01-01", "2020-01-07"),
        },
        gap_dates=contract_case.expectations.gap_dates,
        classification_q75=4.0,
    )

    with pytest.raises(ValueError, match="date-range audit mismatch"):
        _validate(contract_case)


def test_rejects_saved_gap_date_drift(contract_case: ContractCase) -> None:
    contract_case.split_report["gap_dates"]["gap1"][0] = "2020-01-07"

    with pytest.raises(ValueError, match="gap1 date audit mismatch"):
        _validate(contract_case)


def test_rejects_saved_classification_threshold_drift(
    contract_case: ContractCase,
) -> None:
    contract_case.classification_report["threshold_decimal"] = 999.0

    with pytest.raises(ValueError, match="differs from recomputed Q75"):
        _validate(contract_case)


def test_rejects_classification_report_semantic_drift(
    contract_case: ContractCase,
) -> None:
    contract_case.classification_report["quantile_method"] = "nearest"
    with pytest.raises(ValueError, match="quantile_method is inconsistent"):
        _validate(contract_case)

    contract_case = _build_case()
    contract_case.classification_report["train"] = None
    with pytest.raises(TypeError, match="missing train stats"):
        _validate(contract_case)

    contract_case = _build_case()
    contract_case.classification_report["train"]["rows"] = 999
    with pytest.raises(ValueError, match="train stats differ"):
        _validate(contract_case)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gap_size", 4, "Split gap mismatch"),
        ("total_gap_rows", 9, "total_gap_rows"),
        ("split_rows", {}, "row counts"),
        ("date_ranges", {}, "date ranges"),
        ("positions", None, "positions must be an object"),
        ("gap_dates", None, "gap_dates must be an object"),
        ("source_input_unchanged", False, "source remained unchanged"),
    ],
)
def test_rejects_invalid_split_report_semantics(
    contract_case: ContractCase,
    field: str,
    value: object,
    message: str,
) -> None:
    contract_case.split_report[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        _validate(contract_case)


def test_rejects_incomplete_split_positions(contract_case: ContractCase) -> None:
    contract_case.split_report["positions"] = {"train": {"end": 4}}

    with pytest.raises(ValueError, match="positions are incomplete"):
        _validate(contract_case)


def test_rejects_wrong_q75_audit(contract_case: ContractCase) -> None:
    contract_case.expectations = AuditExpectations(
        splits=contract_case.expectations.splits,
        gap_dates=contract_case.expectations.gap_dates,
        classification_q75=123.0,
    )

    with pytest.raises(ValueError, match="Q75 audit mismatch"):
        _validate(contract_case)


def test_rejects_incomplete_frames_and_expectations(
    contract_case: ContractCase,
) -> None:
    frames = {"train": contract_case.frames["train"]}
    with pytest.raises(ValueError, match="frames must contain exactly"):
        validate_input_contract(
            frames,
            contract_case.split_report,
            contract_case.classification_report,
            expectations=contract_case.expectations,
        )

    incomplete_expectations = AuditExpectations(
        splits={"train": contract_case.expectations.splits["train"]},
        gap_dates=contract_case.expectations.gap_dates,
        classification_q75=4.0,
    )
    with pytest.raises(ValueError, match="cover every split"):
        validate_input_contract(
            contract_case.frames,
            contract_case.split_report,
            contract_case.classification_report,
            expectations=incomplete_expectations,
        )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _portable_path(path: Path, report: Path) -> str:
    return Path(os.path.relpath(path, report.parent)).as_posix()


def _write_reproduction(
    root: Path,
    case: ContractCase,
) -> None:
    processed = root / "data" / "processed"
    interim = root / "data" / "interim"
    reports = root / "outputs" / "reports"
    processed.mkdir(parents=True)
    interim.mkdir(parents=True)
    reports.mkdir(parents=True)
    target = interim / "qqq_regression_target.csv"
    target.write_text("synthetic reproduction input\n", encoding="utf-8")
    for name, frame in case.frames.items():
        frame.to_csv(processed / f"{name}.csv", index=False)
        frame.to_csv(processed / f"{name}_labeled.csv", index=False)

    split_path = reports / "data_split_report.json"
    split_payload = copy.deepcopy(case.split_report)
    split_payload["paths_relative_to"] = "report_directory"
    split_payload["input_path"] = _portable_path(target, split_path)
    split_payload["output_paths"] = {
        name: _portable_path(processed / f"{name}.csv", split_path)
        for name in case.frames
    }
    _write_json(split_path, split_payload)

    classification_path = reports / "classification_threshold.json"
    classification_payload = copy.deepcopy(case.classification_report)
    classification_payload["paths_relative_to"] = "report_directory"
    classification_payload["input_paths"] = {
        name: _portable_path(processed / f"{name}.csv", classification_path)
        for name in case.frames
    }
    classification_payload["output_paths"] = {
        name: _portable_path(
            processed / f"{name}_labeled.csv", classification_path
        )
        for name in case.frames
    }
    _write_json(classification_path, classification_payload)


def _write_pipeline_case(
    root: Path, case: ContractCase
) -> SpikeInputContractPaths:
    inputs = root / "authoritative"
    reports = root / "saved-reports"
    train = inputs / "train_labeled.csv"
    validation = inputs / "validation_labeled.csv"
    test = inputs / "test_labeled.csv"
    inputs.mkdir(parents=True)
    for name, path in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        case.frames[name].to_csv(path, index=False)
    split_report = reports / "data_split_report.json"
    classification_report = reports / "classification_threshold.json"
    _write_json(split_report, case.split_report)
    _write_json(classification_report, case.classification_report)
    reproduction = root / "fresh-reproduction"
    _write_reproduction(reproduction, case)
    return SpikeInputContractPaths(
        train_labeled=train,
        validation_labeled=validation,
        test_labeled=test,
        split_report=split_report,
        classification_report=classification_report,
        reproduced_root=reproduction,
        output_report=root / "generated" / "spike_input_contract.json",
    )


def test_pipeline_writes_portable_deterministic_strict_json_without_mutating_sources(
    tmp_path: Path, contract_case: ContractCase
) -> None:
    paths = _write_pipeline_case(tmp_path, contract_case)
    before = {name: _sha256(path) for name, path in paths.protected_inputs().items()}

    report = run_spike_input_contract_pipeline(
        paths, expectations=contract_case.expectations
    )
    first_bytes = paths.output_report.read_bytes()
    after = {name: _sha256(path) for name, path in paths.protected_inputs().items()}

    assert before == after
    assert report["paths_relative_to"] == "report_directory"
    assert report["required_columns"] == list(REQUIRED_COLUMNS)
    assert report["model_columns"] == list(MODEL_COLUMNS)
    assert report["frozen_rules"]["primary_iqr_multiplier"] == 3.0
    assert report["frozen_rules"]["target_backward_reach"] == 5
    assert report["frozen_rules"]["feature_forward_reach"] == 19
    assert report["saved_report_provenance"]["isolated_phase1_reproduction"][
        "all_labeled_csvs_byte_identical"
    ]
    assert all(not Path(item["path"]).is_absolute() for item in report["inputs"].values())
    assert b"NaN" not in first_bytes and b"Infinity" not in first_bytes
    json.loads(first_bytes)

    with pytest.raises(FileExistsError, match="--overwrite-generated"):
        run_spike_input_contract_pipeline(paths, expectations=contract_case.expectations)
    assert paths.output_report.read_bytes() == first_bytes

    run_spike_input_contract_pipeline(
        paths,
        expectations=contract_case.expectations,
        overwrite_generated=True,
    )
    assert paths.output_report.read_bytes() == first_bytes


def test_pipeline_rejects_output_that_aliases_protected_input(
    tmp_path: Path, contract_case: ContractCase
) -> None:
    paths = _write_pipeline_case(tmp_path, contract_case)
    unsafe = SpikeInputContractPaths(
        train_labeled=paths.train_labeled,
        validation_labeled=paths.validation_labeled,
        test_labeled=paths.test_labeled,
        split_report=paths.split_report,
        classification_report=paths.classification_report,
        reproduced_root=paths.reproduced_root,
        output_report=paths.train_labeled,
    )
    before = _sha256(paths.train_labeled)

    with pytest.raises(ValueError, match="aliases protected baseline"):
        run_spike_input_contract_pipeline(
            unsafe,
            overwrite_generated=True,
            expectations=contract_case.expectations,
        )
    assert _sha256(paths.train_labeled) == before


def test_pipeline_failure_does_not_write_or_mutate_sources(
    tmp_path: Path, contract_case: ContractCase
) -> None:
    paths = _write_pipeline_case(tmp_path, contract_case)
    broken = pd.read_csv(paths.validation_labeled)
    broken.loc[0, "return_1d"] = np.nan
    broken.to_csv(paths.validation_labeled, index=False)
    before = {name: _sha256(path) for name, path in paths.protected_inputs().items()}

    with pytest.raises(ValueError, match="must not contain NaN"):
        run_spike_input_contract_pipeline(paths, expectations=contract_case.expectations)

    after = {name: _sha256(path) for name, path in paths.protected_inputs().items()}
    assert before == after
    assert not paths.output_report.exists()


def test_pipeline_rejects_nonportable_fresh_reproduction(
    tmp_path: Path, contract_case: ContractCase
) -> None:
    paths = _write_pipeline_case(tmp_path, contract_case)
    report_path = (
        paths.reproduced_root / "outputs" / "reports" / "data_split_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.pop("paths_relative_to")
    _write_json(report_path, payload)

    with pytest.raises(ValueError, match="portable-path contract"):
        run_spike_input_contract_pipeline(paths, expectations=contract_case.expectations)


def test_pipeline_rejects_reproduction_hash_mismatch(
    tmp_path: Path, contract_case: ContractCase
) -> None:
    paths = _write_pipeline_case(tmp_path, contract_case)
    reproduced_train = (
        paths.reproduced_root / "data" / "processed" / "train_labeled.csv"
    )
    reproduced_train.write_text("different bytes\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hashes differ"):
        run_spike_input_contract_pipeline(paths, expectations=contract_case.expectations)


def test_pipeline_preflight_rejects_missing_reproduction_and_output_directory(
    tmp_path: Path, contract_case: ContractCase
) -> None:
    paths = _write_pipeline_case(tmp_path, contract_case)
    missing_reproduction = SpikeInputContractPaths(
        train_labeled=paths.train_labeled,
        validation_labeled=paths.validation_labeled,
        test_labeled=paths.test_labeled,
        split_report=paths.split_report,
        classification_report=paths.classification_report,
        reproduced_root=tmp_path / "missing-reproduction",
        output_report=paths.output_report,
    )
    with pytest.raises(FileNotFoundError, match="Missing isolated"):
        run_spike_input_contract_pipeline(
            missing_reproduction, expectations=contract_case.expectations
        )

    paths.output_report.mkdir(parents=True)
    with pytest.raises(ValueError, match="output path is a directory"):
        run_spike_input_contract_pipeline(
            paths,
            overwrite_generated=True,
            expectations=contract_case.expectations,
        )


def test_default_paths_record_requested_reproduction_root(tmp_path: Path) -> None:
    paths = SpikeInputContractPaths.defaults(tmp_path)

    assert paths.reproduced_root == tmp_path
    assert paths.output_report.name == "spike_input_contract.json"


def test_cli_reports_validation_failure(
    tmp_path: Path,
    contract_case: ContractCase,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_pipeline_case(tmp_path, contract_case)
    arguments = [
        "--reproduced-root",
        str(paths.reproduced_root),
        "--train",
        str(paths.train_labeled),
        "--validation",
        str(paths.validation_labeled),
        "--test",
        str(paths.test_labeled),
        "--split-report",
        str(paths.split_report),
        "--classification-report",
        str(paths.classification_report),
        "--output",
        str(paths.output_report),
    ]

    assert main(arguments) == 1
    assert "row-count audit mismatch" in capsys.readouterr().err


def test_cli_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "contract.json"

    def fake_pipeline(
        paths: SpikeInputContractPaths, *, overwrite_generated: bool = False
    ) -> dict[str, object]:
        assert paths.output_report == output
        assert overwrite_generated
        return {
            "inputs": {
                name: {"sha256": f"{name}-hash"}
                for name in ("train", "validation", "test")
            }
        }

    monkeypatch.setattr(
        "src.build_spike_input_contract.run_spike_input_contract_pipeline",
        fake_pipeline,
    )

    assert (
        main(
            [
                "--reproduced-root",
                str(tmp_path / "reproduction"),
                "--output",
                str(output),
                "--overwrite-generated",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "Spike input contract saved" in captured.out
    assert "train=train-hash" in captured.out


