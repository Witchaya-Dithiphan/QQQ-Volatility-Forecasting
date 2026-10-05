"""Integration contracts for the verified Phase 1 data runner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import PROJECT_ROOT, RAW_DATA_MANIFEST_PATH, RAW_DATA_PATH
from src import download_qqq_data
from src import run_data_pipeline as runner
from src.build_features import FEATURE_COLUMNS
from src.build_targets import (
    CLASSIFICATION_TARGET,
    TARGET_COLUMN,
    run_classification_target_pipeline,
)
from src.split_data import SPLIT_GAP

EXPECTED_ARTIFACTS = {
    "clean_data": Path("data/interim/qqq_clean.csv"),
    "feature_data": Path("data/interim/qqq_features.csv"),
    "regression_target_data": Path("data/interim/qqq_regression_target.csv"),
    "train_data": Path("data/processed/train.csv"),
    "validation_data": Path("data/processed/validation.csv"),
    "test_data": Path("data/processed/test.csv"),
    "train_labeled_data": Path("data/processed/train_labeled.csv"),
    "validation_labeled_data": Path("data/processed/validation_labeled.csv"),
    "test_labeled_data": Path("data/processed/test_labeled.csv"),
    "cleaning_report": Path("outputs/reports/cleaning_report.json"),
    "feature_report": Path("outputs/reports/feature_report.json"),
    "regression_target_report": Path("outputs/reports/regression_target_report.json"),
    "split_report": Path("outputs/reports/data_split_report.json"),
    "classification_threshold_report": Path(
        "outputs/reports/classification_threshold.json"
    ),
}
EXPECTED_SPLIT_ROWS = {"train": 1733, "validation": 371, "test": 373}


def _sha256(path: Path) -> str:
    """Hash exact file bytes for mutation and rerun assertions."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_hashes(paths: runner.PipelinePaths) -> dict[str, str]:
    """Hash all known generated files without changing them."""
    return {name: _sha256(path) for name, path in paths.generated_outputs().items()}


@pytest.fixture
def verified_inputs(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Copy the pinned contract into isolation and guard repository originals."""
    original_hashes = {
        RAW_DATA_PATH: _sha256(RAW_DATA_PATH),
        RAW_DATA_MANIFEST_PATH: _sha256(RAW_DATA_MANIFEST_PATH),
    }
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    snapshot = source_dir / "qqq_daily.csv"
    manifest = source_dir / "qqq_daily_snapshot.json"
    snapshot.write_bytes(RAW_DATA_PATH.read_bytes())
    manifest.write_bytes(RAW_DATA_MANIFEST_PATH.read_bytes())
    assert _sha256(snapshot) == original_hashes[RAW_DATA_PATH]
    assert _sha256(manifest) == original_hashes[RAW_DATA_MANIFEST_PATH]
    yield snapshot, manifest
    assert {path: _sha256(path) for path in original_hashes} == original_hashes


def _paths(tmp_path: Path, inputs: tuple[Path, Path]) -> runner.PipelinePaths:
    """Keep every generated output away from configured user artifacts."""
    return runner.PipelinePaths.under_output_root(
        tmp_path / "generated", snapshot=inputs[0], manifest=inputs[1]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    """Read dates consistently for chronology and frame comparisons."""
    return pd.read_csv(path, parse_dates=["Date"])


def test_run_data_pipeline_builds_labeled_chronological_splits(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pipeline preserves the artifact, leakage, and label contracts."""

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Runner attempted a live download")

    monkeypatch.setattr(download_qqq_data, "urlopen", reject_network)
    monkeypatch.setattr(download_qqq_data, "fetch_nasdaq_history", reject_network)
    paths = _paths(tmp_path, verified_inputs)
    input_hashes = (_sha256(paths.snapshot), _sha256(paths.manifest))
    result = runner.run_data_pipeline(paths)

    assert paths.generated_outputs() == {
        name: tmp_path / "generated" / relative
        for name, relative in EXPECTED_ARTIFACTS.items()
    }
    assert set(paths.generated_outputs()) == set(EXPECTED_ARTIFACTS)
    assert len(paths.generated_outputs()) == 14
    assert all(path.is_file() for path in paths.generated_outputs().values())
    assert result.split_rows == EXPECTED_SPLIT_ROWS
    assert (
        result.snapshot_sha256.lower(),
        result.manifest_sha256.lower(),
    ) == input_hashes
    assert (_sha256(paths.snapshot), _sha256(paths.manifest)) == input_hashes

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    snapshot = _read_csv(paths.snapshot)
    clean = _read_csv(paths.clean_data)
    features = _read_csv(paths.feature_data)
    regression = _read_csv(paths.regression_target_data)
    assert manifest["dataset"]["row_count"] == len(snapshot) == 2512
    assert len(clean) == len(features) == len(regression) == len(snapshot)
    assert len(FEATURE_COLUMNS) == 8
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert TARGET_COLUMN in regression.columns
    assert CLASSIFICATION_TARGET not in regression.columns
    assert regression[TARGET_COLUMN].tail(5).isna().all()
    regression_report = json.loads(paths.regression_target_report.read_text())
    assert regression_report["paths_relative_to"] == "report_directory"
    assert regression_report["input_path"] == "../../data/interim/qqq_features.csv"
    assert (
        regression_report["output_path"]
        == "../../data/interim/qqq_regression_target.csv"
    )
    assert regression_report["report_path"] == "regression_target_report.json"
    assert regression_report["target_column"] == TARGET_COLUMN
    assert regression_report["horizon_trading_days"] == 5
    assert regression_report["target_nan_count"] == 5
    assert regression_report["boundary_nan_count"] == 5
    assert regression_report["other_nan_count"] == 0
    assert regression_report["valid_target_count"] == len(regression) - 5
    assert (
        regression_report["output_checksum"]
        == _sha256(paths.regression_target_data).upper()
    )
    assert (
        regression_report["input_checksum_before"]
        == _sha256(paths.feature_data).upper()
    )
    assert (
        regression_report["input_checksum_after"] == _sha256(paths.feature_data).upper()
    )
    assert regression_report["statistics"]["mean"] == pytest.approx(
        regression[TARGET_COLUMN].mean()
    )
    assert regression[TARGET_COLUMN].iloc[0] == pytest.approx(
        np.std(features["return_1d"].iloc[1:6], ddof=1) * np.sqrt(252)
    )

    originals = {
        "train": _read_csv(paths.train_data),
        "validation": _read_csv(paths.validation_data),
        "test": _read_csv(paths.test_data),
    }
    labeled = {
        "train": _read_csv(paths.train_labeled_data),
        "validation": _read_csv(paths.validation_labeled_data),
        "test": _read_csv(paths.test_labeled_data),
    }
    for name, frame in originals.items():
        assert len(frame) == EXPECTED_SPLIT_ROWS[name]
        assert frame["Date"].is_monotonic_increasing
        assert frame["Date"].is_unique
        assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(frame.columns)
        assert CLASSIFICATION_TARGET not in frame.columns
        assert list(labeled[name].columns) == [*frame.columns, CLASSIFICATION_TARGET]
        pd.testing.assert_frame_equal(labeled[name][frame.columns], frame)

    assert originals["train"]["Date"].max() < originals["validation"]["Date"].min()
    assert originals["validation"]["Date"].max() < originals["test"]["Date"].min()
    assert sum(len(frame) for frame in originals.values()) + 2 * SPLIT_GAP == 2487

    model_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    modeling_dates = (
        regression.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=model_columns)["Date"]
        .reset_index(drop=True)
    )
    positions = pd.Series(modeling_dates.index, index=modeling_dates)
    for earlier, later in (("train", "validation"), ("validation", "test")):
        earlier_end = int(positions[originals[earlier]["Date"].iloc[-1]])
        later_start = int(positions[originals[later]["Date"].iloc[0]])
        assert later_start - earlier_end - 1 == SPLIT_GAP
    assert len(modeling_dates) == sum(map(len, originals.values())) + 2 * SPLIT_GAP

    train_q75 = float(originals["train"][TARGET_COLUMN].quantile(0.75))
    report = json.loads(paths.classification_threshold_report.read_text())
    split_report = json.loads(paths.split_report.read_text())
    assert split_report["paths_relative_to"] == "report_directory"
    assert split_report["input_path"] == "../../data/interim/qqq_regression_target.csv"
    assert split_report["output_paths"]["train"] == "../../data/processed/train.csv"
    assert report["paths_relative_to"] == "report_directory"
    assert report["input_paths"]["train"] == "../../data/processed/train.csv"
    assert report["output_paths"]["test"] == "../../data/processed/test_labeled.csv"
    assert report["threshold_source"] == "train_only"
    assert report["threshold_decimal"] == pytest.approx(train_q75)
    assert result.classification_threshold == pytest.approx(train_q75)
    assert report["comparison_rule"] == f"{TARGET_COLUMN} > threshold"
    assert split_report["gap_size"] == SPLIT_GAP
    assert split_report["split_rows"] == EXPECTED_SPLIT_ROWS
    for name, frame in originals.items():
        expected_labels = (frame[TARGET_COLUMN] > train_q75).astype("int8")
        pd.testing.assert_series_equal(
            labeled[name][CLASSIFICATION_TARGET],
            expected_labels.rename(CLASSIFICATION_TARGET),
            check_dtype=False,
        )
        equal_to_threshold = frame[TARGET_COLUMN].eq(train_q75)
        if equal_to_threshold.any():
            assert (
                labeled[name].loc[equal_to_threshold, CLASSIFICATION_TARGET].eq(0).all()
            )


@pytest.mark.parametrize("missing", ["snapshot", "manifest"])
def test_missing_protected_input_fails_before_output_creation(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
    missing: str,
) -> None:
    """Missing protected inputs stop the runner at verification."""
    paths = _paths(tmp_path, verified_inputs)
    paths = replace(paths, **{missing: tmp_path / f"missing_{missing}"})
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths)
    assert error.value.stage == "snapshot verification"
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("invalid", ["snapshot", "manifest"])
def test_invalid_protected_input_fails_before_output_creation(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
    invalid: str,
) -> None:
    """A bad manifest or mismatched snapshot cannot generate artifacts."""
    paths = _paths(tmp_path, verified_inputs)
    if invalid == "snapshot":
        paths.snapshot.write_bytes(paths.snapshot.read_bytes() + b"\n")
    else:
        payload = json.loads(paths.manifest.read_text())
        payload["dataset"]["sha256"] = "0" * 64
        paths.manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths)
    assert error.value.stage == "snapshot verification"
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("protected", ["snapshot", "manifest"])
def test_output_collision_with_protected_input_is_rejected(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
    protected: str,
) -> None:
    """Even overwrite permission cannot redirect a generated file onto input."""
    paths = _paths(tmp_path, verified_inputs)
    input_hashes = (_sha256(paths.snapshot), _sha256(paths.manifest))
    paths = replace(paths, clean_data=getattr(paths, protected))
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths, overwrite_generated=True)
    assert error.value.stage == "preflight"
    assert "protected file" in str(error.value)
    assert (_sha256(paths.snapshot), _sha256(paths.manifest)) == input_hashes
    assert not (tmp_path / "generated").exists()


def test_output_alias_to_snapshot_is_rejected(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
) -> None:
    """An existing hard link in an allowed output area cannot overwrite Raw."""
    paths = _paths(tmp_path, verified_inputs)
    paths.clean_data.parent.mkdir(parents=True)
    try:
        os.link(paths.snapshot, paths.clean_data)
    except OSError as exc:
        pytest.skip(f"Hard links unavailable on this platform: {exc}")
    checksum = _sha256(paths.snapshot)
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths, overwrite_generated=True)
    assert error.value.stage == "preflight"
    assert "protected file" in str(error.value)
    assert _sha256(paths.snapshot) == checksum
    assert not paths.feature_data.exists()


@pytest.mark.parametrize("artifact", ["clean_data", "regression_target_report"])
def test_overwrite_rejects_output_hardlinked_to_unrelated_external_file(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
    artifact: str,
) -> None:
    """Overwrite must not change an unrelated inode outside the output map."""
    paths = _paths(tmp_path, verified_inputs)
    output = paths.generated_outputs()[artifact]
    external = tmp_path / "unrelated_user_file.txt"
    external.write_bytes(b"keep unrelated content")
    output.parent.mkdir(parents=True)
    try:
        os.link(external, output)
    except OSError as exc:
        pytest.skip(f"Hard links unavailable on this platform: {exc}")
    external_hash = _sha256(external)

    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths, overwrite_generated=True)
    assert error.value.stage == "preflight"
    assert _sha256(external) == external_hash
    assert not paths.cleaning_report.exists()


@pytest.mark.parametrize("protected", ["snapshot", "manifest"])
def test_regression_report_cannot_alias_protected_input(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
    protected: str,
) -> None:
    """The new report is checked against both immutable inputs before writes."""
    paths = _paths(tmp_path, verified_inputs)
    checksums = (_sha256(paths.snapshot), _sha256(paths.manifest))
    paths = replace(paths, regression_target_report=getattr(paths, protected))
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths, overwrite_generated=True)
    assert error.value.stage == "preflight"
    assert (_sha256(paths.snapshot), _sha256(paths.manifest)) == checksums
    assert not (tmp_path / "generated").exists()


def test_symlink_output_to_manifest_is_rejected_when_supported(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
) -> None:
    """Resolved symlinks remain protected even with explicit overwrite."""
    paths = _paths(tmp_path, verified_inputs)
    paths.cleaning_report.parent.mkdir(parents=True)
    try:
        paths.cleaning_report.symlink_to(paths.manifest)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symlinks unavailable on this platform: {exc}")
    checksum = _sha256(paths.manifest)
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths, overwrite_generated=True)
    assert error.value.stage == "preflight"
    assert "protected file" in str(error.value)
    assert _sha256(paths.manifest) == checksum
    assert not paths.clean_data.exists()


def test_duplicate_generated_destinations_fail_before_writes(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
) -> None:
    """Two artifacts cannot share a path, including before either exists."""
    paths = _paths(tmp_path, verified_inputs)
    paths = replace(paths, feature_data=paths.clean_data)
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths, overwrite_generated=True)
    assert error.value.stage == "preflight"
    assert "must be distinct" in str(error.value)
    assert not (tmp_path / "generated").exists()


def test_output_root_inside_raw_is_rejected(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
) -> None:
    """The generated directory may not sit beneath the source Raw area."""
    snapshot, manifest = verified_inputs
    forbidden_root = snapshot.parent / "generated"
    paths = runner.PipelinePaths.under_output_root(
        forbidden_root, snapshot=snapshot, manifest=manifest
    )
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths)
    assert error.value.stage == "preflight"
    assert not forbidden_root.exists()


@pytest.mark.parametrize("artifact", ["test_labeled_data", "regression_target_report"])
def test_one_existing_artifact_blocks_every_stage(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
    artifact: str,
) -> None:
    """A single existing file is enough to reject the whole run at preflight."""
    paths = _paths(tmp_path, verified_inputs)
    existing_output = paths.generated_outputs()[artifact]
    existing_output.parent.mkdir(parents=True)
    existing_output.write_bytes(b"keep this user file")
    checksum = _sha256(existing_output)
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths)
    assert error.value.stage == "preflight"
    assert _sha256(existing_output) == checksum
    assert all(
        not output.exists()
        for name, output in paths.generated_outputs().items()
        if name != artifact
    )


def test_existing_artifacts_require_overwrite_and_csvs_are_repeatable(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
) -> None:
    """Preflight preserves all outputs; explicit overwrite reproduces CSVs."""
    paths = _paths(tmp_path, verified_inputs)
    input_hashes = (_sha256(paths.snapshot), _sha256(paths.manifest))
    runner.run_data_pipeline(paths)
    first_hashes = _artifact_hashes(paths)
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths)
    assert error.value.stage == "preflight"
    assert "already exist" in str(error.value)
    assert _artifact_hashes(paths) == first_hashes

    rerun = runner.run_data_pipeline(paths, overwrite_generated=True)
    assert rerun.split_rows == EXPECTED_SPLIT_ROWS
    for name, output in paths.generated_outputs().items():
        if output.suffix == ".csv":
            assert _sha256(output) == first_hashes[name]
    assert (
        _sha256(paths.regression_target_report)
        == first_hashes["regression_target_report"]
    )
    assert (_sha256(paths.snapshot), _sha256(paths.manifest)) == input_hashes


def test_classification_stage_labels_values_equal_to_train_q75_as_zero(
    tmp_path: Path,
) -> None:
    """A computed train Q75 is a strict boundary in all three saved splits."""
    targets = {
        "train": [0.10, 0.20, 0.30, 0.40, 0.40],
        "validation": [0.40, 0.41],
        "test": [0.39, 0.40, 0.50],
    }
    starts = {"train": "2020-01-01", "validation": "2021-01-01", "test": "2022-01-01"}
    input_paths: list[Path] = []
    output_paths: list[Path] = []
    for name, values in targets.items():
        frame = pd.DataFrame(
            {
                "Date": pd.date_range(starts[name], periods=len(values), freq="B"),
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Volume": 1_000_000.0,
                **{feature: 1.0 for feature in FEATURE_COLUMNS},
                TARGET_COLUMN: values,
            }
        )
        input_path = tmp_path / f"{name}.csv"
        output_path = tmp_path / f"{name}_labeled.csv"
        frame.to_csv(input_path, index=False)
        input_paths.append(input_path)
        output_paths.append(output_path)

    report_path = tmp_path / "threshold.json"
    *_, threshold = run_classification_target_pipeline(
        *input_paths, *output_paths, report_path
    )
    assert threshold == pytest.approx(0.40)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["threshold_source"] == "train_only"
    assert report["threshold_decimal"] == pytest.approx(0.40)
    for name, path in zip(targets, output_paths, strict=True):
        labeled = pd.read_csv(path)
        equal_values = labeled[TARGET_COLUMN].eq(0.40)
        assert equal_values.any()
        assert labeled.loc[equal_values, CLASSIFICATION_TARGET].eq(0).all()
        assert labeled[CLASSIFICATION_TARGET].tolist() == [
            int(value > 0.40) for value in targets[name]
        ]


def test_middle_stage_failure_stops_downstream_outputs(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed feature stage reports its name and leaves only upstream files."""
    paths = _paths(tmp_path, verified_inputs)
    input_hashes = (_sha256(paths.snapshot), _sha256(paths.manifest))

    def fail_features(**kwargs: Path) -> None:
        raise RuntimeError("simulated feature failure")

    monkeypatch.setattr(runner, "run_feature_pipeline", fail_features)
    with pytest.raises(runner.PipelineRunError) as error:
        runner.run_data_pipeline(paths)
    assert error.value.stage == "feature engineering"
    assert "simulated feature failure" in str(error.value)
    assert error.value.outputs_may_remain
    assert paths.clean_data.is_file() and paths.cleaning_report.is_file()
    assert all(
        not output.exists()
        for name, output in paths.generated_outputs().items()
        if name not in {"clean_data", "cleaning_report"}
    )
    assert (_sha256(paths.snapshot), _sha256(paths.manifest)) == input_hashes


def test_cli_success_and_preflight_failure(
    tmp_path: Path,
    verified_inputs: tuple[Path, Path],
) -> None:
    """Module CLI runs from project root and exits nonzero on existing files."""
    paths = _paths(tmp_path, verified_inputs)
    input_hashes = (_sha256(paths.snapshot), _sha256(paths.manifest))
    command = [
        sys.executable,
        "-m",
        "src.run_data_pipeline",
        "--snapshot",
        str(paths.snapshot),
        "--manifest",
        str(paths.manifest),
        "--output-root",
        str(tmp_path / "generated"),
    ]
    success = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    assert success.returncode == 0, success.stderr
    assert "train: 1733" in success.stdout
    assert all(
        output.is_file()
        for output in (
            paths.train_labeled_data,
            paths.validation_labeled_data,
            paths.test_labeled_data,
        )
    )
    first_hashes = _artifact_hashes(paths)
    failure = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    assert failure.returncode != 0
    assert "preflight" in failure.stderr
    assert "already exist" in failure.stderr
    assert _artifact_hashes(paths) == first_hashes
    assert (_sha256(paths.snapshot), _sha256(paths.manifest)) == input_hashes
