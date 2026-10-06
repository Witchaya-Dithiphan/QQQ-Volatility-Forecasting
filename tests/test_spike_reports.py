"""Tests for M6 Phase 2 report and figure generation."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd
import pytest

from src.build_spike_input_contract import _sha256
from src.build_spike_reports import SpikeReportPaths, main, run_spike_report_pipeline


def _production_paths(output_root: Path) -> SpikeReportPaths:
    return SpikeReportPaths.under_output_root(output_root)


def _assert_finite_json(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_json(child)


def test_m6_builds_traceable_report_and_readable_figures(tmp_path: Path) -> None:
    paths = _production_paths(tmp_path)
    before = {name: _sha256(path) for name, path in paths.protected_inputs().items()}

    report = run_spike_report_pipeline(paths)

    assert report["train_filtering"] == {
        "before": 1733,
        "removed": 213,
        "retained": 1520,
        "identity_valid": True,
    }
    assert report["spike_counts"] == {
        "train": {"direct": 19, "affected": 213},
        "validation": {"direct": 0, "affected": 0},
        "test": {"direct": 4, "affected": 54},
    }
    assert report["diagnostic_segments"]["test"] == {
        "full": 373,
        "non_spike": 319,
        "spike_affected": 54,
    }
    assert report["datasets"]["non_spike_train"]["class_counts"] == {
        "0": 1240,
        "1": 280,
    }
    assert report["classification_contract"]["labels_consistent"] is True
    assert report["evaluation_contract"]["flagged_copies_role"] == "diagnostic_only"
    assert "operationally non-spike-affected" in report["dataset_definition"]
    assert "Wilder RSI" in report["limitations"]["rsi"]
    assert "split boundary" in report["limitations"]["boundary"]
    _assert_finite_json(report)
    assert {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    } == before

    raw = paths.dataset_report.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == report
    first_report_bytes = paths.dataset_report.read_bytes()
    rerun = run_spike_report_pipeline(paths, overwrite_generated=True)
    assert rerun == report
    assert paths.dataset_report.read_bytes() == first_report_bytes
    for artifact in report["artifacts"].values():
        serialized = artifact["path"]
        assert not Path(serialized).is_absolute()
        resolved = (paths.dataset_report.parent / serialized).resolve()
        assert resolved.is_file()
        assert _sha256(resolved) == artifact["sha256"]
    for figure in (
        paths.daily_return_figure,
        paths.volatility_figure,
        paths.comparison_figure,
    ):
        pixels = mpimg.imread(figure)
        assert pixels.size > 0
        assert pixels.shape[0] >= 600 and pixels.shape[1] >= 900


def test_m6_report_matches_source_csvs_and_original_q75(tmp_path: Path) -> None:
    paths = _production_paths(tmp_path)
    report = run_spike_report_pipeline(paths)
    threshold = report["classification_contract"]["threshold"]
    for name, path in {
        "with_spikes_train": paths.with_spikes_train,
        "non_spike_train": paths.non_spike_train,
    }.items():
        frame = pd.read_csv(path)
        summary = report["datasets"][name]
        assert len(frame) == summary["rows"]
        assert (
            frame["target_high_volatility"]
            .astype(int)
            .equals(frame["target_volatility_5d"].gt(threshold).astype(int))
        )
        assert frame["target_high_volatility"].value_counts().to_dict() == {
            int(key): value for key, value in summary["class_counts"].items()
        }


def test_m6_default_refuses_existing_generated_outputs() -> None:
    with pytest.raises(FileExistsError, match="M6 outputs already exist"):
        run_spike_report_pipeline(SpikeReportPaths.defaults())


def test_m6_rejects_output_aliasing_protected_input(tmp_path: Path) -> None:
    original = _production_paths(tmp_path)
    paths = replace(original, dataset_report=original.train_labeled)
    with pytest.raises(ValueError, match="aliases"):
        run_spike_report_pipeline(paths)


def test_m6_rejects_stale_m5_dataset(tmp_path: Path) -> None:
    paths = _production_paths(tmp_path / "output")
    stale = tmp_path / "stale.csv"
    stale.write_bytes(paths.non_spike_train.read_bytes())
    frame = pd.read_csv(stale)
    frame.loc[0, "return_1d"] = 999.0
    frame.to_csv(stale, index=False)
    paths = SpikeReportPaths(
        paths.train_labeled,
        paths.validation_labeled,
        paths.test_labeled,
        paths.input_contract,
        paths.spike_analysis,
        paths.daily_return_figure,
        paths.with_spikes_train,
        stale,
        paths.validation_flagged,
        paths.test_flagged,
        paths.dataset_report,
        paths.volatility_figure,
        paths.comparison_figure,
    )
    with pytest.raises(AssertionError):
        run_spike_report_pipeline(paths)


def test_m6_cli_reports_collision_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--output-root", str(tmp_path)]) == 0
    assert main(["--output-root", str(tmp_path)]) == 1
    assert "partial_outputs=" in capsys.readouterr().err
