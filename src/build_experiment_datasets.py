"""Build M5 primary experiment CSVs from protected labeled splits."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

if __package__ in {None, ""}:  # pragma: no cover - direct-script bootstrap
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import (
    NON_SPIKE_TRAIN_PATH,
    PROJECT_ROOT,
    RAW_DATA_MANIFEST_PATH,
    RAW_DATA_PATH,
    SPIKE_ANALYSIS_REPORT_PATH,
    SPIKE_DAILY_RETURN_FIGURE_PATH,
    SPIKE_EVENT_AUDIT_PATH,
    SPIKE_INPUT_CONTRACT_REPORT_PATH,
    TEST_FLAGGED_PATH,
    TEST_LABELED_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_FLAGGED_PATH,
    VALIDATION_LABELED_DATA_PATH,
    WITH_SPIKES_TRAIN_PATH,
)
from src.audit_spikes import _write_csv
from src.build_spike_input_contract import (
    REQUIRED_COLUMNS,
    _same_path,
    _sha256,
    _strict_json_load,
    load_labeled_split,
)
from src.build_targets import CLASSIFICATION_TARGET, TARGET_COLUMN
from src.detect_spikes import (
    ExtremeIQRThreshold,
    apply_spike_detector,
    fit_primary_spike_detector,
)
from src.spike_contract import AffectedMaskResult, build_affected_result

SPLIT_NAMES: Final = ("train", "validation", "test")
DIAGNOSTIC_COLUMNS: Final = (
    "is_spike",
    "is_spike_affected",
    "diagnostic_segment",
)


@dataclass(frozen=True, slots=True)
class ExperimentDatasetPaths:
    """Protected M1/M4 inputs and the four declared M5 outputs."""

    raw_snapshot: Path
    manifest: Path
    train_labeled: Path
    validation_labeled: Path
    test_labeled: Path
    input_contract: Path
    audit_report: Path
    audit_events: Path
    audit_figure: Path
    with_spikes_train: Path
    non_spike_train: Path
    validation_flagged: Path
    test_flagged: Path

    @classmethod
    def defaults(cls) -> ExperimentDatasetPaths:
        """Return configured project paths."""
        return cls(
            raw_snapshot=RAW_DATA_PATH,
            manifest=RAW_DATA_MANIFEST_PATH,
            train_labeled=TRAIN_LABELED_DATA_PATH,
            validation_labeled=VALIDATION_LABELED_DATA_PATH,
            test_labeled=TEST_LABELED_DATA_PATH,
            input_contract=SPIKE_INPUT_CONTRACT_REPORT_PATH,
            audit_report=SPIKE_ANALYSIS_REPORT_PATH,
            audit_events=SPIKE_EVENT_AUDIT_PATH,
            audit_figure=SPIKE_DAILY_RETURN_FIGURE_PATH,
            with_spikes_train=WITH_SPIKES_TRAIN_PATH,
            non_spike_train=NON_SPIKE_TRAIN_PATH,
            validation_flagged=VALIDATION_FLAGGED_PATH,
            test_flagged=TEST_FLAGGED_PATH,
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
        audit_report: str | Path = SPIKE_ANALYSIS_REPORT_PATH,
        audit_events: str | Path = SPIKE_EVENT_AUDIT_PATH,
        audit_figure: str | Path = SPIKE_DAILY_RETURN_FIGURE_PATH,
    ) -> ExperimentDatasetPaths:
        """Map labeled inputs and experiment outputs below explicit roots."""
        source = Path(input_root)
        experiment_root = Path(output_root) / "data" / "processed" / "experiments"
        return cls(
            raw_snapshot=Path(raw_snapshot),
            manifest=Path(manifest),
            train_labeled=source / "data" / "processed" / "train_labeled.csv",
            validation_labeled=(
                source / "data" / "processed" / "validation_labeled.csv"
            ),
            test_labeled=source / "data" / "processed" / "test_labeled.csv",
            input_contract=Path(input_contract),
            audit_report=Path(audit_report),
            audit_events=Path(audit_events),
            audit_figure=Path(audit_figure),
            with_spikes_train=experiment_root / "with_spikes" / "train.csv",
            non_spike_train=experiment_root / "non_spike" / "train.csv",
            validation_flagged=(
                experiment_root / "diagnostics" / "validation_flagged.csv"
            ),
            test_flagged=experiment_root / "diagnostics" / "test_flagged.csv",
        )

    def labeled_inputs(self) -> dict[str, Path]:
        """Return authoritative labeled split paths."""
        return {
            "train": self.train_labeled,
            "validation": self.validation_labeled,
            "test": self.test_labeled,
        }

    def protected_inputs(self) -> dict[str, Path]:
        """Return every baseline or M4 artifact protected from M5 writes."""
        return {
            "raw_snapshot": self.raw_snapshot,
            "manifest": self.manifest,
            **{f"{name}_labeled": path for name, path in self.labeled_inputs().items()},
            "spike_input_contract": self.input_contract,
            "spike_analysis_report": self.audit_report,
            "spike_event_audit": self.audit_events,
            "spike_daily_return_figure": self.audit_figure,
        }

    def generated_outputs(self) -> dict[str, Path]:
        """Return exactly the four M5-generated CSV destinations."""
        return {
            "with_spikes_train": self.with_spikes_train,
            "non_spike_train": self.non_spike_train,
            "validation_flagged": self.validation_flagged,
            "test_flagged": self.test_flagged,
        }


@dataclass(frozen=True, slots=True)
class ExperimentFrames:
    """Pure in-memory M5 outputs and their canonical M3 results."""

    fitted_detector: ExtremeIQRThreshold
    affected_results: Mapping[str, AffectedMaskResult]
    with_spikes_train: pd.DataFrame
    non_spike_train: pd.DataFrame
    validation_flagged: pd.DataFrame
    test_flagged: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ExperimentDatasetRunResult:
    """Run metadata returned by M5 instead of inventing a report artifact."""

    paths: ExperimentDatasetPaths
    fitted_detector: ExtremeIQRThreshold
    input_checksums: Mapping[str, str]
    output_checksums: Mapping[str, str]
    row_counts: Mapping[str, int]
    class_counts: Mapping[str, Mapping[int, int]]
    diagnostic_counts: Mapping[str, Mapping[str, int]]
    with_spikes_byte_identical: bool
    full_evaluation_inputs: Mapping[str, Path]
    boundary_limitation: str
    rsi_limitation: str


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _preflight(
    paths: ExperimentDatasetPaths,
    *,
    overwrite_generated: bool,
) -> None:
    """Reject missing inputs, unsafe destinations, aliases, and implicit overwrite."""
    protected = paths.protected_inputs()
    outputs = paths.generated_outputs()
    for name, path in protected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected M5 input {name}: {path}")
    for output_name, output in outputs.items():
        for source_name, source in protected.items():
            if _same_path(output, source):
                raise ValueError(
                    f"M5 output {output_name} aliases protected input "
                    f"{source_name}: {source}"
                )
    output_items = list(outputs.items())
    for index, (name, path) in enumerate(output_items):
        for other_name, other_path in output_items[index + 1 :]:
            if _same_path(path, other_path):
                raise ValueError(f"M5 outputs must be distinct: {name}, {other_name}")

    allowed_root = paths.with_spikes_train.parents[1]
    resolved_root = allowed_root.resolve(strict=False)
    for output in outputs.values():
        resolved = output.resolve(strict=False)
        if resolved == resolved_root or not _is_relative_to(resolved, resolved_root):
            raise ValueError(f"M5 output must stay inside {allowed_root}: {output}")
    protected_directories = {
        paths.raw_snapshot.resolve(strict=False).parent,
        paths.manifest.resolve(strict=False).parent,
    }
    if any(
        resolved_root == directory or _is_relative_to(resolved_root, directory)
        for directory in protected_directories
    ):
        raise ValueError("M5 experiment root cannot be inside Raw/Manifest storage")

    existing = [
        (name, path)
        for name, path in outputs.items()
        if path.exists() or path.is_symlink()
    ]
    if any(path.is_dir() for _, path in existing):
        raise ValueError("An M5 output path is an existing directory")
    if existing and not overwrite_generated:
        details = ", ".join(f"{name}={path}" for name, path in existing)
        raise FileExistsError(
            "M5 outputs already exist; pass --overwrite-generated to replace only "
            f"the four declared CSVs: {details}"
        )
    if overwrite_generated:
        for name, path in existing:
            if path.is_symlink() or path.stat().st_nlink > 1:
                raise ValueError(
                    f"Refusing to overwrite aliased M5 output: {name}={path}"
                )


def _validate_m1_contract(
    frames: Mapping[str, pd.DataFrame],
    paths: ExperimentDatasetPaths,
    contract: Mapping[str, Any],
) -> None:
    """Assert the current splits still satisfy the generated M1 contract."""
    required_columns = tuple(contract["required_columns"])
    if required_columns != REQUIRED_COLUMNS:
        raise ValueError("M1 required-column contract differs from current code")
    threshold = float(contract["classification_contract"]["threshold"])
    for split_name in SPLIT_NAMES:
        frame = frames[split_name]
        expected = contract["inputs"][split_name]
        if tuple(frame.columns) != required_columns:
            raise ValueError(f"{split_name} schema differs from M1 contract")
        if _sha256(paths.labeled_inputs()[split_name]) != expected["sha256"]:
            raise ValueError(f"{split_name} checksum differs from M1 contract")
        if len(frame) != int(expected["rows"]):
            raise ValueError(f"{split_name} row count differs from M1 contract")
        if (
            frame["Date"].duplicated().any()
            or not frame["Date"].is_monotonic_increasing
        ):
            raise ValueError(f"{split_name} Date contract is invalid")
        numeric = frame.drop(columns="Date").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{split_name} contains non-finite model values")
        labels = frame[CLASSIFICATION_TARGET].astype(int)
        if not labels.isin([0, 1]).all():
            raise ValueError(f"{split_name} labels must be binary")
        expected_labels = frame[TARGET_COLUMN].gt(threshold).astype(int)
        if not labels.equals(expected_labels):
            raise ValueError(f"{split_name} labels differ from frozen Original Q75")


def _validate_m4_dependency(
    paths: ExperimentDatasetPaths,
    report: Mapping[str, Any],
    input_checksums: Mapping[str, str],
) -> None:
    """Require a complete, internally consistent M4 audit with intact artifacts."""
    if report.get("overall_data_quality_status") != "internally_consistent":
        raise RuntimeError("M4 audit is not internally consistent")
    if report.get("audit_findings") or report.get("review_items"):
        raise RuntimeError("M4 audit has unresolved findings or review items")
    report_inputs = report["inputs"]
    m4_input_names = (
        "raw_snapshot",
        "manifest",
        "train_labeled",
        "validation_labeled",
        "test_labeled",
        "spike_input_contract",
    )
    for name in m4_input_names:
        if report_inputs[name]["sha256"] != input_checksums[name]:
            raise ValueError(f"M4 audit input checksum is stale: {name}")
    artifact_paths = {
        "event_audit": paths.audit_events,
        "daily_return_figure": paths.audit_figure,
    }
    for name, path in artifact_paths.items():
        if _sha256(path) != report["generated_artifacts"][name]["sha256"]:
            raise ValueError(f"M4 generated artifact checksum mismatch: {name}")


def _flagged_copy(frame: pd.DataFrame, result: AffectedMaskResult) -> pd.DataFrame:
    flagged = frame.copy(deep=True)
    flagged["is_spike"] = result.direct_flags.to_numpy(dtype=bool)
    flagged["is_spike_affected"] = result.affected_flags.to_numpy(dtype=bool)
    flagged["diagnostic_segment"] = np.where(
        flagged["is_spike_affected"], "spike_affected", "non_spike"
    )
    return flagged


def build_primary_experiment_frames(
    labeled_splits: Mapping[str, pd.DataFrame],
) -> ExperimentFrames:
    """Build M5 frames in memory using the canonical M2 and M3 APIs."""
    if set(labeled_splits) != set(SPLIT_NAMES):
        raise ValueError(f"labeled_splits must contain exactly {SPLIT_NAMES}")
    fitted = fit_primary_spike_detector(labeled_splits["train"])
    results: dict[str, AffectedMaskResult] = {}
    for split_name in SPLIT_NAMES:
        frame = labeled_splits[split_name]
        direct = apply_spike_detector(frame, fitted)
        results[split_name] = build_affected_result(
            direct,
            split_name=split_name,
            dates=frame["Date"],
        )
    train = labeled_splits["train"]
    keep = ~results["train"].affected_flags.to_numpy(dtype=bool)
    return ExperimentFrames(
        fitted_detector=fitted,
        affected_results=results,
        with_spikes_train=train.copy(deep=True),
        non_spike_train=train.iloc[np.flatnonzero(keep)].copy().reset_index(drop=True),
        validation_flagged=_flagged_copy(
            labeled_splits["validation"], results["validation"]
        ),
        test_flagged=_flagged_copy(labeled_splits["test"], results["test"]),
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, temporary)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _class_counts(frame: pd.DataFrame) -> dict[int, int]:
    counts = frame[CLASSIFICATION_TARGET].value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def run_experiment_dataset_pipeline(
    paths: ExperimentDatasetPaths,
    *,
    overwrite_generated: bool = False,
) -> ExperimentDatasetRunResult:
    """Validate M1/M4, write four M5 CSVs, and prove sources unchanged."""
    _preflight(paths, overwrite_generated=overwrite_generated)
    checksums_before = {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    }
    contract = _strict_json_load(paths.input_contract)
    audit = _strict_json_load(paths.audit_report)
    frames = {
        name: load_labeled_split(path, name)
        for name, path in paths.labeled_inputs().items()
    }
    _validate_m1_contract(frames, paths, contract)
    _validate_m4_dependency(
        paths,
        audit,
        checksums_before,
    )
    built = build_primary_experiment_frames(frames)
    if asdict(built.fitted_detector) != audit["primary_detector"]:
        raise ValueError("M5 fitted detector metadata differs from verified M4")
    for split_name in SPLIT_NAMES:
        summary = asdict(built.affected_results[split_name].summary)
        audit_summary = audit["splits"][split_name]
        if any(audit_summary[key] != value for key, value in summary.items()):
            raise ValueError(f"M5 {split_name} mask summary differs from verified M4")

    observed = {
        "with_spikes": len(built.with_spikes_train),
        "non_spike": len(built.non_spike_train),
        "validation_full": len(built.validation_flagged),
        "validation_affected": int(built.validation_flagged["is_spike_affected"].sum()),
        "test_full": len(built.test_flagged),
        "test_affected": int(built.test_flagged["is_spike_affected"].sum()),
    }
    expected = {
        "with_spikes": 1733,
        "non_spike": 1520,
        "validation_full": 371,
        "validation_affected": 0,
        "test_full": 373,
        "test_affected": 54,
    }
    if observed != expected or _class_counts(built.non_spike_train) != {
        0: 1240,
        1: 280,
    }:
        raise RuntimeError(
            "Pinned-snapshot M5 regression drift; verify M1/M2/M3/M4 before "
            f"changing expectations: counts={observed}, "
            f"classes={_class_counts(built.non_spike_train)}"
        )

    _atomic_copy(paths.train_labeled, paths.with_spikes_train)
    _write_csv(built.non_spike_train, paths.non_spike_train)
    _write_csv(built.validation_flagged, paths.validation_flagged)
    _write_csv(built.test_flagged, paths.test_flagged)

    if paths.with_spikes_train.read_bytes() != paths.train_labeled.read_bytes():
        raise RuntimeError("With-Spike output is not byte-identical to Original Train")
    checksums_after = {
        name: _sha256(path) for name, path in paths.protected_inputs().items()
    }
    if checksums_after != checksums_before:
        raise RuntimeError("Protected M1/M4 inputs changed while writing M5 outputs")

    output_checksums = {
        name: _sha256(path) for name, path in paths.generated_outputs().items()
    }
    return ExperimentDatasetRunResult(
        paths=paths,
        fitted_detector=built.fitted_detector,
        input_checksums=checksums_before,
        output_checksums=output_checksums,
        row_counts={
            "with_spikes_train": len(built.with_spikes_train),
            "non_spike_train": len(built.non_spike_train),
            "validation_flagged": len(built.validation_flagged),
            "test_flagged": len(built.test_flagged),
        },
        class_counts={
            "with_spikes_train": _class_counts(built.with_spikes_train),
            "non_spike_train": _class_counts(built.non_spike_train),
        },
        diagnostic_counts={
            split_name: {
                "full": len(built_frame),
                "non_spike": int((~built_frame["is_spike_affected"]).sum()),
                "spike_affected": int(built_frame["is_spike_affected"].sum()),
            }
            for split_name, built_frame in (
                ("validation", built.validation_flagged),
                ("test", built.test_flagged),
            )
        },
        with_spikes_byte_identical=True,
        full_evaluation_inputs={
            "validation": paths.validation_labeled,
            "test": paths.test_labeled,
        },
        boundary_limitation=built.affected_results["train"].summary.boundary_limitation,
        rsi_limitation=built.affected_results["train"].summary.rsi_limitation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the four Phase 2 M5 primary experiment CSVs."
    )
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=RAW_DATA_MANIFEST_PATH)
    parser.add_argument(
        "--input-contract", type=Path, default=SPIKE_INPUT_CONTRACT_REPORT_PATH
    )
    parser.add_argument("--audit-report", type=Path, default=SPIKE_ANALYSIS_REPORT_PATH)
    parser.add_argument("--audit-events", type=Path, default=SPIKE_EVENT_AUDIT_PATH)
    parser.add_argument(
        "--audit-figure", type=Path, default=SPIKE_DAILY_RETURN_FIGURE_PATH
    )
    parser.add_argument("--overwrite-generated", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = ExperimentDatasetPaths.under_roots(
        arguments.input_root,
        arguments.output_root,
        raw_snapshot=arguments.raw,
        manifest=arguments.manifest,
        input_contract=arguments.input_contract,
        audit_report=arguments.audit_report,
        audit_events=arguments.audit_events,
        audit_figure=arguments.audit_figure,
    )
    try:
        result = run_experiment_dataset_pipeline(
            paths,
            overwrite_generated=arguments.overwrite_generated,
        )
    except Exception as error:  # noqa: BLE001 - CLI converts failures to exit codes.
        print(f"M5 experiment dataset build failed: {error}", file=sys.stderr)
        written = {
            name: path
            for name, path in paths.generated_outputs().items()
            if path.is_file()
        }
        if written:
            print("Generated M5 outputs present after failure:", file=sys.stderr)
            for name, path in written.items():
                print(f"  {name}: {path.resolve(strict=False)}", file=sys.stderr)
        return 1
    print("M5 experiment datasets:")
    for name, path in result.paths.generated_outputs().items():
        print(
            f"  {name}: {path.resolve(strict=False)} "
            f"rows={result.row_counts[name]} sha256={result.output_checksums[name]}"
        )
    print(f"Class counts: {dict(result.class_counts)}")
    print(f"Diagnostic counts: {dict(result.diagnostic_counts)}")
    print(
        "Full evaluation inputs remain Original labeled splits: "
        f"validation={result.full_evaluation_inputs['validation']}, "
        f"test={result.full_evaluation_inputs['test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
