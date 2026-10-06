"""Build the M4 direct-spike audit without modifying Phase 1 inputs."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qqq-volatility-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:  # pragma: no cover - direct-script bootstrap
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import (
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    PROJECT_ROOT,
    RAW_DATA_MANIFEST_PATH,
    RAW_DATA_PATH,
    REGRESSION_TARGET_DATA_PATH,
    SPIKE_ANALYSIS_REPORT_PATH,
    SPIKE_DAILY_RETURN_FIGURE_PATH,
    SPIKE_EVENT_AUDIT_PATH,
    SPIKE_INPUT_CONTRACT_REPORT_PATH,
    TEST_LABELED_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
)
from src.build_features import FEATURE_COLUMNS
from src.build_spike_input_contract import (
    _same_path,
    _sha256,
    _strict_json_load,
    _write_and_verify_report,
    load_labeled_split,
)
from src.build_targets import TARGET_COLUMN
from src.detect_spikes import (
    ExtremeIQRThreshold,
    apply_spike_detector,
    fit_primary_spike_detector,
)
from src.download_qqq_data import verify_snapshot
from src.report_paths import report_relative_path
from src.spike_contract import AffectedMaskResult, build_affected_result

SPLIT_NAMES: Final = ("train", "validation", "test")
OHLCV_COLUMNS: Final = ("Open", "High", "Low", "Close", "Volume")
TRACE_STAGE_NAMES: Final = ("labeled", "target", "feature", "clean", "raw")
NUMERIC_ABSOLUTE_TOLERANCE: Final = 1e-12
REPORT_VERSION: Final = 1
EVENT_COLUMNS: Final = (
    "event_key",
    "split",
    "original_position",
    "Date",
    "return_1d",
    "abs_return_1d",
    "threshold",
    "comparison_rule",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "ohlcv_source_stage",
    "requested_start",
    "requested_end",
    "clipped_start",
    "clipped_end",
    "clipped_start_date",
    "clipped_end_date",
    "left_clipped",
    "right_clipped",
    "labeled_rows",
    "target_rows",
    "feature_rows",
    "clean_rows",
    "raw_rows",
    "trace_status",
    "comparison_status",
    "data_quality_status",
    "audit_note",
)


@dataclass(frozen=True, slots=True)
class SpikeAuditPaths:
    """Protected inputs and the only three M4-generated outputs."""

    raw_snapshot: Path
    manifest: Path
    clean_data: Path
    feature_data: Path
    target_data: Path
    train_labeled: Path
    validation_labeled: Path
    test_labeled: Path
    input_contract: Path
    output_report: Path
    output_events: Path
    output_figure: Path

    @classmethod
    def defaults(cls) -> SpikeAuditPaths:
        """Return project-root inputs and outputs."""
        return cls(
            raw_snapshot=RAW_DATA_PATH,
            manifest=RAW_DATA_MANIFEST_PATH,
            clean_data=CLEAN_DATA_PATH,
            feature_data=FEATURE_DATA_PATH,
            target_data=REGRESSION_TARGET_DATA_PATH,
            train_labeled=TRAIN_LABELED_DATA_PATH,
            validation_labeled=VALIDATION_LABELED_DATA_PATH,
            test_labeled=TEST_LABELED_DATA_PATH,
            input_contract=SPIKE_INPUT_CONTRACT_REPORT_PATH,
            output_report=SPIKE_ANALYSIS_REPORT_PATH,
            output_events=SPIKE_EVENT_AUDIT_PATH,
            output_figure=SPIKE_DAILY_RETURN_FIGURE_PATH,
        )

    @classmethod
    def under_roots(
        cls,
        input_root: str | Path,
        output_root: str | Path,
        *,
        raw_snapshot: str | Path = RAW_DATA_PATH,
        manifest: str | Path = RAW_DATA_MANIFEST_PATH,
        input_contract: str | Path = SPIKE_INPUT_CONTRACT_REPORT_PATH,
    ) -> SpikeAuditPaths:
        """Map Phase 1 inputs and M4 outputs below explicit roots."""
        source = Path(input_root)
        destination = Path(output_root)
        return cls(
            raw_snapshot=Path(raw_snapshot),
            manifest=Path(manifest),
            clean_data=source / "data" / "interim" / "qqq_clean.csv",
            feature_data=source / "data" / "interim" / "qqq_features.csv",
            target_data=source / "data" / "interim" / "qqq_regression_target.csv",
            train_labeled=source / "data" / "processed" / "train_labeled.csv",
            validation_labeled=(
                source / "data" / "processed" / "validation_labeled.csv"
            ),
            test_labeled=source / "data" / "processed" / "test_labeled.csv",
            input_contract=Path(input_contract),
            output_report=destination / "outputs" / "reports" / "spike_analysis.json",
            output_events=destination / "outputs" / "reports" / "spike_event_audit.csv",
            output_figure=(
                destination
                / "outputs"
                / "figures"
                / "spike_analysis"
                / "daily_return_spikes.png"
            ),
        )

    def labeled_inputs(self) -> dict[str, Path]:
        """Return labeled split paths by stable split name."""
        return {
            "train": self.train_labeled,
            "validation": self.validation_labeled,
            "test": self.test_labeled,
        }

    def protected_inputs(self) -> dict[str, Path]:
        """Return every source artifact read by M4."""
        return {
            "raw_snapshot": self.raw_snapshot,
            "manifest": self.manifest,
            "clean_data": self.clean_data,
            "feature_data": self.feature_data,
            "target_data": self.target_data,
            **{f"{name}_labeled": path for name, path in self.labeled_inputs().items()},
            "spike_input_contract": self.input_contract,
        }

    def generated_outputs(self) -> dict[str, Path]:
        """Return the three files M4 may create or explicitly replace."""
        return {
            "spike_analysis_report": self.output_report,
            "spike_event_audit": self.output_events,
            "daily_return_spikes_figure": self.output_figure,
        }


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _preflight(paths: SpikeAuditPaths, *, overwrite_generated: bool) -> None:
    """Validate all paths and overwrite rules before creating directories."""
    for name, path in paths.protected_inputs().items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected input {name}: {path}")
    outputs = paths.generated_outputs()
    for output_name, output in outputs.items():
        for source_name, source in paths.protected_inputs().items():
            if _same_path(output, source):
                raise ValueError(
                    f"Generated output {output_name} aliases protected input "
                    f"{source_name}: {source}"
                )
    output_items = list(outputs.items())
    for index, (name, path) in enumerate(output_items):
        for other_name, other_path in output_items[index + 1 :]:
            if _same_path(path, other_path):
                raise ValueError(
                    f"Generated outputs must be distinct: {name}, {other_name}"
                )
    protected_directories = {
        paths.raw_snapshot.resolve(strict=False).parent,
        paths.manifest.resolve(strict=False).parent,
    }
    for output in outputs.values():
        resolved = output.resolve(strict=False)
        if any(
            resolved == directory or _is_relative_to(resolved, directory)
            for directory in protected_directories
        ):
            raise ValueError(
                f"Generated output cannot be inside Raw/Manifest: {output}"
            )
    existing = [
        (name, path)
        for name, path in outputs.items()
        if path.exists() or path.is_symlink()
    ]
    if any(path.is_dir() for _, path in existing):
        raise ValueError("A generated output path is an existing directory")
    if existing and not overwrite_generated:
        details = ", ".join(f"{name}={path}" for name, path in existing)
        raise FileExistsError(
            "M4 outputs already exist; pass --overwrite-generated to replace only "
            f"the declared artifacts: {details}"
        )
    if overwrite_generated:
        for name, path in existing:
            if path.is_symlink() or path.stat().st_nlink > 1:
                raise ValueError(
                    f"Refusing to overwrite aliased M4 output: {name}={path}"
                )


def _load_stage(path: Path, stage_name: str, required: Sequence[str]) -> pd.DataFrame:
    """Read a trace stage and strictly parse Date without assuming row order."""
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read {stage_name}: {error}") from error
    missing = [column for column in ("Date", *required) if column not in frame.columns]
    if missing:
        raise ValueError(f"{stage_name} missing required columns: {missing}")
    try:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{stage_name} contains invalid Date values") from error
    return frame


def _one_row(frame: pd.DataFrame, event_date: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[frame["Date"].eq(event_date)]


def _numbers_match(first: Any, second: Any) -> bool:
    try:
        left = float(first)
        right = float(second)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite([left, right]).all()
        and np.isclose(left, right, rtol=0.0, atol=NUMERIC_ABSOLUTE_TOLERANCE)
    )


def _trace_event(
    split_name: str,
    position: int,
    labeled_row: pd.Series,
    window: Mapping[str, Any],
    stages: Mapping[str, pd.DataFrame],
    fitted: ExtremeIQRThreshold,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Trace one detector event and classify only from internal evidence."""
    event_date = pd.Timestamp(labeled_row["Date"]).normalize()
    matches = {name: _one_row(frame, event_date) for name, frame in stages.items()}
    counts = {name: len(rows) for name, rows in matches.items()}
    event_key = f"{split_name}:{position}:{event_date.date().isoformat()}"
    missing = [name for name, count in counts.items() if count == 0]
    duplicates = [name for name, count in counts.items() if count > 1]
    mismatches: list[str] = []
    if not missing and not duplicates:
        rows = {name: frame.iloc[0] for name, frame in matches.items()}
        for column in OHLCV_COLUMNS:
            reference = rows["raw"][column]
            for stage_name in ("clean", "feature", "target", "labeled"):
                if not _numbers_match(rows[stage_name][column], reference):
                    mismatches.append(f"{column}:{stage_name}_vs_raw")
        for column in (*FEATURE_COLUMNS, TARGET_COLUMN):
            reference = rows["labeled"][column]
            comparison_stages = (
                ("feature", "target") if column in FEATURE_COLUMNS else ("target",)
            )
            for stage_name in comparison_stages:
                if not _numbers_match(rows[stage_name][column], reference):
                    mismatches.append(f"{column}:{stage_name}_vs_labeled")
        raw_sorted = (
            stages["raw"].sort_values("Date", kind="stable").reset_index(drop=True)
        )
        raw_positions = raw_sorted.index[raw_sorted["Date"].eq(event_date)].tolist()
        if len(raw_positions) == 1 and raw_positions[0] > 0:
            raw_position = raw_positions[0]
            current_close = float(cast(Any, raw_sorted.at[raw_position, "Close"]))
            prior_close = float(cast(Any, raw_sorted.at[raw_position - 1, "Close"]))
            raw_return = current_close / prior_close - 1.0
            if not _numbers_match(raw_return, labeled_row["return_1d"]):
                mismatches.append("return_1d:labeled_vs_raw_close_pct_change")
        else:
            missing.append("raw_previous_trading_day")

    if duplicates or mismatches:
        quality_status = "suspected_data_error"
        comparison_status = "mismatch"
        note = "Internal stage inconsistency requires a separate baseline review."
    elif missing:
        quality_status = "needs_review"
        comparison_status = "not_evaluated"
        note = (
            "Evidence is incomplete; no market or data-error conclusion was inferred."
        )
    else:
        quality_status = "market_movement"
        comparison_status = "matched"
        note = (
            "All required stages traced uniquely and raw Close pct_change matched "
            "the detector return; no external market-event claim was inferred."
        )
    trace_status = "complete" if not missing and not duplicates else "incomplete"
    source_row = matches["raw"].iloc[0] if counts["raw"] == 1 else labeled_row
    signed_return = float(labeled_row["return_1d"])
    event = {
        "event_key": event_key,
        "split": split_name,
        "original_position": position,
        "Date": event_date.date().isoformat(),
        "return_1d": signed_return,
        "abs_return_1d": abs(signed_return),
        "threshold": fitted.threshold,
        "comparison_rule": fitted.comparison_rule,
        **{column: source_row[column] for column in OHLCV_COLUMNS},
        "ohlcv_source_stage": (
            "pinned_raw_snapshot" if counts["raw"] == 1 else "labeled_fallback"
        ),
        **{
            name: window[name]
            for name in (
                "requested_start",
                "requested_end",
                "clipped_start",
                "clipped_end",
                "clipped_start_date",
                "clipped_end_date",
                "left_clipped",
                "right_clipped",
            )
        },
        **{f"{name}_rows": counts[name] for name in TRACE_STAGE_NAMES},
        "trace_status": trace_status,
        "comparison_status": comparison_status,
        "data_quality_status": quality_status,
        "audit_note": note,
    }
    finding = None
    if quality_status == "suspected_data_error":
        finding = {
            "event_key": event_key,
            "status": quality_status,
            "missing_stages": missing,
            "duplicate_stages": duplicates,
            "mismatches": mismatches,
            "evidence": "stage trace and raw-derived return comparison",
        }
    review_item = None
    if quality_status == "needs_review":
        review_item = {"event_key": event_key, "missing_evidence": missing}
    return event, finding, review_item


def build_spike_audit(
    labeled_splits: Mapping[str, pd.DataFrame],
    stages: Mapping[str, pd.DataFrame],
) -> tuple[
    ExtremeIQRThreshold,
    dict[str, AffectedMaskResult],
    pd.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build M4 audit data from validated in-memory inputs without I/O."""
    if set(labeled_splits) != set(SPLIT_NAMES):
        raise ValueError(f"labeled_splits must contain exactly {SPLIT_NAMES}")
    fitted = fit_primary_spike_detector(labeled_splits["train"])
    results: dict[str, AffectedMaskResult] = {}
    events: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        frame = labeled_splits[split_name]
        direct = apply_spike_detector(frame, fitted)
        result = build_affected_result(
            direct,
            split_name=split_name,
            dates=frame["Date"],
        )
        results[split_name] = result
        windows = {window.spike_position: asdict(window) for window in result.windows}
        for position in np.flatnonzero(result.direct_flags.to_numpy(dtype=bool)):
            event, finding, review_item = _trace_event(
                split_name,
                int(position),
                frame.iloc[int(position)],
                windows[int(position)],
                {**stages, "labeled": frame},
                fitted,
            )
            events.append(event)
            if finding is not None:
                findings.append(finding)
            if review_item is not None:
                review_items.append(review_item)
    event_frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return fitted, results, event_frame, findings, review_items


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            frame.to_csv(
                temporary, index=False, lineterminator="\n", float_format="%.17g"
            )
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _write_figure(
    labeled_splits: Mapping[str, pd.DataFrame],
    results: Mapping[str, AffectedMaskResult],
    threshold: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharey=True)
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    try:
        for axis, split_name in zip(axes, SPLIT_NAMES, strict=True):
            frame = labeled_splits[split_name]
            direct = results[split_name].direct_flags
            axis.plot(
                frame["Date"],
                frame["return_1d"],
                color="0.45",
                linewidth=0.8,
                label="Daily return",
            )
            axis.scatter(
                frame.loc[direct, "Date"],
                frame.loc[direct, "return_1d"],
                color=colors[split_name],
                s=28,
                zorder=3,
                label=f"Direct spikes ({int(direct.sum())})",
            )
            axis.axhline(
                threshold,
                color="crimson",
                linestyle="--",
                linewidth=0.9,
                label="+threshold",
            )
            axis.axhline(
                -threshold,
                color="crimson",
                linestyle=":",
                linewidth=0.9,
                label="-threshold",
            )
            axis.set_title(split_name.capitalize())
            axis.set_ylabel("return_1d")
            axis.grid(alpha=0.2)
            axis.legend(loc="best")
        axes[-1].set_xlabel("Date")
        figure.suptitle("QQQ daily returns and direct spikes (affected rows not shown)")
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            figure.savefig(
                temporary_path, dpi=150, metadata={"Software": "QQQ spike audit"}
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        plt.close(figure)


def _report_payload(
    paths: SpikeAuditPaths,
    fitted: ExtremeIQRThreshold,
    results: Mapping[str, AffectedMaskResult],
    events: pd.DataFrame,
    findings: list[dict[str, Any]],
    review_items: list[dict[str, Any]],
    checksums_before: Mapping[str, str],
    checksums_after_outputs: Mapping[str, str],
) -> dict[str, Any]:
    status_counts = Counter(str(value) for value in events["data_quality_status"])
    event_records: list[dict[str, Any]] = []
    for raw_record in events.to_dict(orient="records"):
        row = cast(dict[str, Any], raw_record)
        event_records.append(
            {
                "event_key": row["event_key"],
                "split": row["split"],
                "original_position": int(row["original_position"]),
                "event_date": row["Date"],
                "requested_start": int(row["requested_start"]),
                "requested_end": int(row["requested_end"]),
                "clipped_start": int(row["clipped_start"]),
                "clipped_end": int(row["clipped_end"]),
                "clipped_start_date": row["clipped_start_date"],
                "clipped_end_date": row["clipped_end_date"],
                "left_clipped": bool(row["left_clipped"]),
                "right_clipped": bool(row["right_clipped"]),
                "data_quality_status": row["data_quality_status"],
            }
        )
    return {
        "report_version": REPORT_VERSION,
        "paths_relative_to": "report_directory",
        "primary_detector": asdict(fitted),
        "comparison_rule_identifier": "strict_greater_than",
        "numeric_comparison": {
            "exact_fields": ["Date", "split", "original_position"],
            "float_fields": list(
                dict.fromkeys((*OHLCV_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN))
            ),
            "rtol": 0.0,
            "atol": NUMERIC_ABSOLUTE_TOLERANCE,
            "reason": "CSV float serialization may differ at sub-decimal machine precision.",
        },
        "inputs": {
            name: {
                "path": report_relative_path(path, paths.output_report),
                "sha256": checksums_before[name],
            }
            for name, path in paths.protected_inputs().items()
        },
        "source_checksum_guard": {
            "before_writing_outputs": dict(sorted(checksums_before.items())),
            "after_writing_csv_and_figure": dict(
                sorted(checksums_after_outputs.items())
            ),
            "unchanged": checksums_before == checksums_after_outputs,
            "post_report_runtime_check": (
                "The runner re-hashes every protected input after writing this report "
                "and fails if any checksum changed."
            ),
        },
        "splits": {
            name: {
                "row_count": len(results[name].direct_flags),
                **asdict(results[name].summary),
            }
            for name in SPLIT_NAMES
        },
        "events": event_records,
        "event_count": len(events),
        "status_counts": dict(sorted(status_counts.items())),
        "audit_findings": findings,
        "review_items": review_items,
        "overall_data_quality_status": (
            "requires_baseline_reproduction"
            if findings
            else "needs_review" if review_items else "internally_consistent"
        ),
        "limitations": {
            "market_evidence": "No external market-event sources were consulted.",
            "boundary": results["train"].summary.boundary_limitation,
            "rsi": results["train"].summary.rsi_limitation,
        },
        "generated_artifacts": {
            "event_audit": {
                "path": report_relative_path(paths.output_events, paths.output_report),
                "sha256": _sha256(paths.output_events),
            },
            "daily_return_figure": {
                "path": report_relative_path(paths.output_figure, paths.output_report),
                "sha256": _sha256(paths.output_figure),
            },
        },
        "expected_audit_checks": {
            "role": "regression_assertions_not_calculation_inputs",
            "train_direct": 19,
            "train_affected": 213,
            "validation_direct": 0,
            "validation_affected": 0,
            "test_direct": 4,
            "test_affected": 54,
        },
    }


def run_spike_audit_pipeline(
    paths: SpikeAuditPaths,
    *,
    overwrite_generated: bool = False,
) -> dict[str, Any]:
    """Validate sources, write exactly three M4 artifacts, and recheck inputs."""
    _preflight(paths, overwrite_generated=overwrite_generated)
    checksums_before = {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    }
    verify_snapshot(paths.raw_snapshot, paths.manifest)
    contract = _strict_json_load(paths.input_contract)
    labeled = {
        name: load_labeled_split(path, name)
        for name, path in paths.labeled_inputs().items()
    }
    for name, path in paths.labeled_inputs().items():
        expected = contract["inputs"][name]["sha256"]
        if _sha256(path) != expected:
            raise ValueError(f"{name} labeled checksum does not match M1 contract")
    stages = {
        "target": _load_stage(
            paths.target_data,
            "target",
            (*OHLCV_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN),
        ),
        "feature": _load_stage(
            paths.feature_data, "feature", (*OHLCV_COLUMNS, *FEATURE_COLUMNS)
        ),
        "clean": _load_stage(paths.clean_data, "clean", OHLCV_COLUMNS),
        "raw": _load_stage(paths.raw_snapshot, "raw", OHLCV_COLUMNS),
    }
    fitted, results, events, findings, review_items = build_spike_audit(labeled, stages)
    observed = {
        "train": (
            results["train"].summary.direct_count,
            results["train"].summary.affected_union_count,
        ),
        "validation": (
            results["validation"].summary.direct_count,
            results["validation"].summary.affected_union_count,
        ),
        "test": (
            results["test"].summary.direct_count,
            results["test"].summary.affected_union_count,
        ),
    }
    expected = {"train": (19, 213), "validation": (0, 0), "test": (4, 54)}
    if observed != expected:
        raise RuntimeError(f"Pinned-snapshot spike regression drift: {observed}")
    if {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    } != checksums_before:
        raise RuntimeError("Protected inputs changed during M4 validation")
    _write_csv(events, paths.output_events)
    _write_figure(labeled, results, fitted.threshold, paths.output_figure)
    checksums_after_outputs = {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    }
    if checksums_after_outputs != checksums_before:
        raise RuntimeError("Protected inputs changed while writing M4 CSV/figure")
    report = _report_payload(
        paths,
        fitted,
        results,
        events,
        findings,
        review_items,
        checksums_before,
        checksums_after_outputs,
    )
    _write_and_verify_report(report, paths.output_report)
    checksums_after = {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    }
    if checksums_after != checksums_before:
        raise RuntimeError("Protected inputs changed while writing M4 artifacts")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the Phase 2 M4 spike audit artifacts."
    )
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=RAW_DATA_MANIFEST_PATH)
    parser.add_argument(
        "--input-contract", type=Path, default=SPIKE_INPUT_CONTRACT_REPORT_PATH
    )
    parser.add_argument("--overwrite-generated", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = SpikeAuditPaths.under_roots(
        arguments.input_root,
        arguments.output_root,
        raw_snapshot=arguments.raw,
        manifest=arguments.manifest,
        input_contract=arguments.input_contract,
    )
    try:
        report = run_spike_audit_pipeline(
            paths, overwrite_generated=arguments.overwrite_generated
        )
    except Exception as error:  # noqa: BLE001 - CLI reports a nonzero failure.
        print(f"Spike audit failed: {error}", file=sys.stderr)
        written = {
            name: path
            for name, path in paths.generated_outputs().items()
            if path.is_file()
        }
        if written:
            print("Generated artifacts present after failure:", file=sys.stderr)
            for name, path in written.items():
                print(f"  {name}: {path.resolve(strict=False)}", file=sys.stderr)
        return 1
    print("Spike audit artifacts:")
    for name, path in paths.generated_outputs().items():
        print(f"  {name}: {path.resolve(strict=False)}")
    print(f"Event status counts: {report['status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
