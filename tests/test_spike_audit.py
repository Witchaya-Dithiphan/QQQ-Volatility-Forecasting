"""Synthetic and pinned-snapshot tests for the M4 direct-spike audit."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from config import PROJECT_ROOT, RAW_DATA_MANIFEST_PATH, RAW_DATA_PATH
from src.audit_spikes import (
    SpikeAuditPaths,
    _preflight,
    build_spike_audit,
    run_spike_audit_pipeline,
)
from src.build_features import FEATURE_COLUMNS
from src.build_targets import TARGET_COLUMN


def _synthetic_audit_inputs() -> (
    tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
):
    """Create a consistent chronological trace with Train and Test spikes."""
    prior_date = pd.Timestamp("2024-01-01")
    split_returns = {
        "train": [0.001, -0.002, 0.003, -0.004] * 5 + [0.50],
        "validation": [0.001, -0.002, 0.003, -0.004, 0.002],
        "test": [0.001, -0.40, 0.003, -0.004, 0.002],
    }
    returns = [value for split in split_returns.values() for value in split]
    dates = pd.bdate_range(prior_date + pd.offsets.BDay(1), periods=len(returns))
    close = [100.0]
    for value in returns:
        close.append(close[-1] * (1.0 + value))
    raw_dates = pd.DatetimeIndex([prior_date, *dates])
    raw = pd.DataFrame(
        {
            "Date": raw_dates,
            "Open": close,
            "High": [value * 1.01 for value in close],
            "Low": [value * 0.99 for value in close],
            "Close": close,
            "Volume": [1_000_000 + index for index in range(len(close))],
        }
    )
    clean = raw.iloc[1:].reset_index(drop=True)
    feature = clean.copy(deep=True)
    feature["return_1d"] = returns
    for index, column in enumerate(FEATURE_COLUMNS[1:], start=1):
        feature[column] = float(index)
    target = feature.copy(deep=True)
    target[TARGET_COLUMN] = 0.02

    labeled_splits: dict[str, pd.DataFrame] = {}
    start = 0
    for split_name, values in split_returns.items():
        stop = start + len(values)
        frame = target.iloc[start:stop].copy().reset_index(drop=True)
        frame["high_volatility"] = 0
        labeled_splits[split_name] = frame
        start = stop
    stages = {
        "target": target,
        "feature": feature,
        "clean": clean,
        "raw": raw.iloc[::-1].reset_index(drop=True),
    }
    return labeled_splits, stages


def test_synthetic_trace_preserves_sign_and_handles_zero_spike_split() -> None:
    labeled, stages = _synthetic_audit_inputs()

    fitted, results, events, findings, review_items = build_spike_audit(labeled, stages)

    assert fitted.source_split == "original_train"
    assert results["train"].summary.direct_count == 1
    assert results["validation"].summary.direct_count == 0
    assert results["test"].summary.direct_count == 1
    assert set(events["split"]) == {"train", "test"}
    assert events.loc[events["split"].eq("test"), "return_1d"].item() < 0
    assert (events["abs_return_1d"] == events["return_1d"].abs()).all()
    assert set(events["data_quality_status"]) == {"market_movement"}
    assert not findings
    assert not review_items


def test_missing_trace_evidence_is_needs_review() -> None:
    labeled, stages = _synthetic_audit_inputs()
    event_date = labeled["train"].iloc[-1]["Date"]
    stages["feature"] = stages["feature"].loc[~stages["feature"]["Date"].eq(event_date)]

    _, _, events, findings, review_items = build_spike_audit(labeled, stages)
    event = events.loc[events["split"].eq("train")].iloc[0]

    assert event["data_quality_status"] == "needs_review"
    assert event["trace_status"] == "incomplete"
    assert not findings
    assert review_items[0]["missing_evidence"] == ["feature"]


@pytest.mark.parametrize("failure_mode", ["duplicate", "mismatch"])
def test_duplicate_or_mismatched_trace_is_suspected_data_error(
    failure_mode: str,
) -> None:
    labeled, stages = _synthetic_audit_inputs()
    event_date = labeled["train"].iloc[-1]["Date"]
    event_raw = stages["raw"].loc[stages["raw"]["Date"].eq(event_date)]
    if failure_mode == "duplicate":
        stages["raw"] = pd.concat([stages["raw"], event_raw], ignore_index=True)
    else:
        row_index = event_raw.index.item()
        stages["raw"].loc[row_index, "Close"] *= 1.01

    _, _, events, findings, review_items = build_spike_audit(labeled, stages)
    event = events.loc[events["split"].eq("train")].iloc[0]

    assert event["data_quality_status"] == "suspected_data_error"
    assert findings[0]["event_key"] == event["event_key"]
    assert not review_items


def _touch_protected_inputs(paths: SpikeAuditPaths) -> None:
    for path in paths.protected_inputs().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")


def test_preflight_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    paths = SpikeAuditPaths.under_roots(
        source,
        output,
        raw_snapshot=source / "raw" / "snapshot.csv",
        manifest=source / "manifest" / "snapshot.json",
        input_contract=source / "reports" / "contract.json",
    )
    _touch_protected_inputs(paths)
    paths.output_report.parent.mkdir(parents=True, exist_ok=True)
    paths.output_report.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite-generated"):
        _preflight(paths, overwrite_generated=False)


def test_preflight_rejects_hardlinked_output_alias(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    paths = SpikeAuditPaths.under_roots(
        source,
        output,
        raw_snapshot=source / "raw" / "snapshot.csv",
        manifest=source / "manifest" / "snapshot.json",
        input_contract=source / "reports" / "contract.json",
    )
    _touch_protected_inputs(paths)
    paths.output_events.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(paths.train_labeled, paths.output_events)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")

    with pytest.raises(ValueError, match="aliases protected input"):
        _preflight(paths, overwrite_generated=True)


def test_pinned_pipeline_is_deterministic_portable_and_read_only(
    tmp_path: Path,
) -> None:
    required = [
        RAW_DATA_PATH,
        RAW_DATA_MANIFEST_PATH,
        PROJECT_ROOT / "outputs" / "reports" / "spike_input_contract.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Ignored pinned inputs or M1 contract are unavailable")
    paths = SpikeAuditPaths.under_roots(PROJECT_ROOT, tmp_path)
    before = {
        name: path.read_bytes() for name, path in paths.protected_inputs().items()
    }

    first = run_spike_audit_pipeline(paths)
    first_bytes = {
        name: path.read_bytes() for name, path in paths.generated_outputs().items()
    }
    second = run_spike_audit_pipeline(paths, overwrite_generated=True)

    assert first == second
    assert {
        name: path.read_bytes() for name, path in paths.generated_outputs().items()
    } == first_bytes
    assert {
        name: path.read_bytes() for name, path in paths.protected_inputs().items()
    } == before
    assert first["source_checksum_guard"]["unchanged"] is True
    assert first["splits"]["train"]["direct_count"] == 19
    assert first["splits"]["train"]["affected_union_count"] == 213
    assert first["splits"]["validation"]["direct_count"] == 0
    assert first["splits"]["test"]["direct_count"] == 4
    assert first["splits"]["test"]["affected_union_count"] == 54
    assert first["event_count"] == 23
    assert first["status_counts"] == {"market_movement": 23}
    assert not first["audit_findings"]
    assert not first["review_items"]
    json.loads(paths.output_report.read_text(encoding="utf-8"))

    moved_root = tmp_path.parent / f"{tmp_path.name}-moved"
    shutil.copytree(tmp_path / "outputs", moved_root / "outputs")
    moved_report = moved_root / "outputs" / "reports" / "spike_analysis.json"
    moved_payload = json.loads(moved_report.read_text(encoding="utf-8"))
    for artifact in moved_payload["generated_artifacts"].values():
        assert (moved_report.parent / artifact["path"]).resolve().is_file()
