"""Run the verified Phase 1 data pipeline from snapshot to labeled splits."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import (
    CLASSIFICATION_THRESHOLD_REPORT_PATH,
    CLEAN_DATA_PATH,
    CLEANING_REPORT_PATH,
    DATA_SPLIT_REPORT_PATH,
    FEATURE_DATA_PATH,
    FEATURE_REPORT_PATH,
    RAW_DATA_MANIFEST_PATH,
    RAW_DATA_PATH,
    REGRESSION_TARGET_DATA_PATH,
    REGRESSION_TARGET_REPORT_PATH,
    TEST_DATA_PATH,
    TEST_LABELED_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
)
from src.build_features import run_feature_pipeline
from src.build_targets import (
    run_classification_target_pipeline,
    run_regression_target_pipeline,
)
from src.clean_data import run_cleaning_pipeline
from src.download_qqq_data import verify_snapshot
from src.split_data import run_split_pipeline

_T = TypeVar("_T")


class PipelineRunError(RuntimeError):
    """Describe a failed pipeline stage without hiding its original exception."""

    def __init__(self, stage: str, message: str, *, outputs_may_remain: bool) -> None:
        """Initialize a stage-aware pipeline error."""
        self.stage = stage
        self.outputs_may_remain = outputs_may_remain
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    """All protected inputs, output areas, and generated pipeline artifacts."""

    snapshot: Path
    manifest: Path
    interim_root: Path
    processed_root: Path
    reports_root: Path
    clean_data: Path
    cleaning_report: Path
    feature_data: Path
    feature_report: Path
    regression_target_data: Path
    regression_target_report: Path
    train_data: Path
    validation_data: Path
    test_data: Path
    split_report: Path
    train_labeled_data: Path
    validation_labeled_data: Path
    test_labeled_data: Path
    classification_threshold_report: Path

    @classmethod
    def defaults(cls) -> PipelinePaths:
        """Build paths from the repository's current ``config.py`` values."""
        return cls(
            snapshot=RAW_DATA_PATH,
            manifest=RAW_DATA_MANIFEST_PATH,
            interim_root=CLEAN_DATA_PATH.parent,
            processed_root=TRAIN_DATA_PATH.parent,
            reports_root=CLEANING_REPORT_PATH.parent,
            clean_data=CLEAN_DATA_PATH,
            cleaning_report=CLEANING_REPORT_PATH,
            feature_data=FEATURE_DATA_PATH,
            feature_report=FEATURE_REPORT_PATH,
            regression_target_data=REGRESSION_TARGET_DATA_PATH,
            regression_target_report=REGRESSION_TARGET_REPORT_PATH,
            train_data=TRAIN_DATA_PATH,
            validation_data=VALIDATION_DATA_PATH,
            test_data=TEST_DATA_PATH,
            split_report=DATA_SPLIT_REPORT_PATH,
            train_labeled_data=TRAIN_LABELED_DATA_PATH,
            validation_labeled_data=VALIDATION_LABELED_DATA_PATH,
            test_labeled_data=TEST_LABELED_DATA_PATH,
            classification_threshold_report=CLASSIFICATION_THRESHOLD_REPORT_PATH,
        )

    @classmethod
    def under_output_root(
        cls,
        output_root: str | Path,
        *,
        snapshot: str | Path = RAW_DATA_PATH,
        manifest: str | Path = RAW_DATA_MANIFEST_PATH,
    ) -> PipelinePaths:
        """Map an isolated root to the repository's artifact directory contract."""
        root = Path(output_root)
        interim = root / "data" / "interim"
        processed = root / "data" / "processed"
        reports = root / "outputs" / "reports"
        return cls(
            snapshot=Path(snapshot),
            manifest=Path(manifest),
            interim_root=interim,
            processed_root=processed,
            reports_root=reports,
            clean_data=interim / "qqq_clean.csv",
            cleaning_report=reports / "cleaning_report.json",
            feature_data=interim / "qqq_features.csv",
            feature_report=reports / "feature_report.json",
            regression_target_data=interim / "qqq_regression_target.csv",
            regression_target_report=reports / "regression_target_report.json",
            train_data=processed / "train.csv",
            validation_data=processed / "validation.csv",
            test_data=processed / "test.csv",
            split_report=reports / "data_split_report.json",
            train_labeled_data=processed / "train_labeled.csv",
            validation_labeled_data=processed / "validation_labeled.csv",
            test_labeled_data=processed / "test_labeled.csv",
            classification_threshold_report=(reports / "classification_threshold.json"),
        )

    def generated_outputs(self) -> dict[str, Path]:
        """Return every file the runner is allowed to create or overwrite."""
        return {
            "clean_data": self.clean_data,
            "cleaning_report": self.cleaning_report,
            "feature_data": self.feature_data,
            "feature_report": self.feature_report,
            "regression_target_data": self.regression_target_data,
            "regression_target_report": self.regression_target_report,
            "train_data": self.train_data,
            "validation_data": self.validation_data,
            "test_data": self.test_data,
            "split_report": self.split_report,
            "train_labeled_data": self.train_labeled_data,
            "validation_labeled_data": self.validation_labeled_data,
            "test_labeled_data": self.test_labeled_data,
            "classification_threshold_report": (self.classification_threshold_report),
        }

    def categorized_outputs(self) -> tuple[tuple[Path, tuple[Path, ...]], ...]:
        """Associate each generated file with its permitted output area."""
        return (
            (
                self.interim_root,
                (self.clean_data, self.feature_data, self.regression_target_data),
            ),
            (
                self.processed_root,
                (
                    self.train_data,
                    self.validation_data,
                    self.test_data,
                    self.train_labeled_data,
                    self.validation_labeled_data,
                    self.test_labeled_data,
                ),
            ),
            (
                self.reports_root,
                (
                    self.cleaning_report,
                    self.feature_report,
                    self.regression_target_report,
                    self.split_report,
                    self.classification_threshold_report,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Successful runner outputs needed by callers and the command-line summary."""

    paths: PipelinePaths
    split_rows: Mapping[str, int]
    classification_threshold: float
    snapshot_sha256: str
    manifest_sha256: str


def _sha256(path: Path) -> str:
    """Calculate a file's uppercase SHA-256 digest without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _is_relative_to(path: Path, directory: Path) -> bool:
    """Return whether a resolved path is inside a resolved directory."""
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _paths_alias(first: Path, second: Path) -> bool:
    """Detect lexical, symlink, and existing hard-link aliases."""
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            return False
    return False


def _preflight(paths: PipelinePaths, *, overwrite_generated: bool) -> None:
    """Validate all destinations after snapshot verification and before writes."""
    snapshot = paths.snapshot.resolve(strict=True)
    manifest = paths.manifest.resolve(strict=True)
    protected_files = (snapshot, manifest)
    protected_directories = {
        snapshot.parent,
        manifest.parent,
        RAW_DATA_PATH.resolve(strict=False).parent,
        RAW_DATA_MANIFEST_PATH.resolve(strict=False).parent,
    }

    resolved_outputs: dict[str, Path] = {}
    for name, output in paths.generated_outputs().items():
        resolved = output.resolve(strict=False)
        for protected in protected_files:
            if _paths_alias(output, protected):
                raise ValueError(f"Output {name} aliases protected file: {protected}")
        resolved_outputs[name] = resolved

    output_items = list(resolved_outputs.items())
    for index, (name, output) in enumerate(output_items):
        for other_name, other_output in output_items[index + 1 :]:
            if _paths_alias(output, other_output):
                raise ValueError(
                    f"Generated outputs must be distinct: {name} and {other_name}"
                )

    for allowed_root, outputs in paths.categorized_outputs():
        resolved_root = allowed_root.resolve(strict=False)
        existing_ancestor = next(
            (
                candidate
                for candidate in (allowed_root, *allowed_root.parents)
                if candidate.exists() or candidate.is_symlink()
            ),
            None,
        )
        if existing_ancestor is not None and not existing_ancestor.is_dir():
            raise ValueError(
                "Generated output area has a non-directory ancestor: "
                f"{existing_ancestor}"
            )
        if any(
            resolved_root == protected or _is_relative_to(resolved_root, protected)
            for protected in protected_directories
        ):
            raise ValueError(
                f"Generated output area cannot be inside Raw or Manifest data: "
                f"{allowed_root}"
            )
        for output in outputs:
            resolved_output = output.resolve(strict=False)
            if resolved_output == resolved_root or not _is_relative_to(
                resolved_output, resolved_root
            ):
                raise ValueError(
                    f"Output must stay inside its allowed area {allowed_root}: {output}"
                )

    existing = [
        (name, output)
        for name, output in paths.generated_outputs().items()
        if output.exists() or output.is_symlink()
    ]
    directories = [(name, output) for name, output in existing if output.is_dir()]
    if directories:
        details = ", ".join(f"{name}={path}" for name, path in directories)
        raise ValueError(f"Generated output path is a directory: {details}")
    if existing and not overwrite_generated:
        details = ", ".join(f"{name}={path}" for name, path in existing)
        raise FileExistsError(
            "Generated outputs already exist; pass --overwrite-generated to replace "
            f"only these known files: {details}"
        )
    if overwrite_generated:
        for name, output in existing:
            if output.is_symlink() or output.stat().st_nlink > 1:
                raise ValueError(
                    "Refusing to overwrite generated output with another filesystem "
                    f"alias: {name}={output}"
                )


def _call_stage(stage: str, function: Callable[[], _T]) -> _T:
    """Run one stage and add stage context to any failure."""
    try:
        return function()
    except Exception as exc:
        raise PipelineRunError(
            stage,
            f"{stage} failed: {exc}",
            outputs_may_remain=stage not in {"snapshot verification", "preflight"},
        ) from exc


def _protected_checksums(
    paths: PipelinePaths,
    *,
    outputs_may_remain: bool,
) -> dict[str, str]:
    """Read protected checksums and report missing or unreadable inputs."""
    try:
        return {
            "snapshot": _sha256(paths.snapshot),
            "manifest": _sha256(paths.manifest),
        }
    except OSError as exc:
        raise PipelineRunError(
            "protected checksum verification",
            f"Could not checksum a protected input: {exc}",
            outputs_may_remain=outputs_may_remain,
        ) from exc


def _verify_protected_checksums(
    paths: PipelinePaths,
    checksums_before: Mapping[str, str],
) -> None:
    """Ensure the snapshot and manifest remain byte-identical after execution."""
    checksums_after = _protected_checksums(paths, outputs_may_remain=True)
    changed = [
        name
        for name, before in checksums_before.items()
        if checksums_after[name] != before
    ]
    if changed:
        raise PipelineRunError(
            "protected checksum verification",
            "Protected input changed during the run: " + ", ".join(changed),
            outputs_may_remain=True,
        )


def run_data_pipeline(
    paths: PipelinePaths | None = None,
    *,
    overwrite_generated: bool = False,
) -> PipelineResult:
    """Run all existing data stages after verifying and protecting the snapshot.

    Args:
        paths: Complete input/output path set. Defaults to ``config.py`` values.
        overwrite_generated: Permit replacement of only the 14 declared outputs.

    Returns:
        Split row counts, train-only threshold, checksums, and artifact paths.

    Raises:
        PipelineRunError: If verification, preflight, a stage, or the final
            protected-input checksum check fails.
    """
    selected_paths = PipelinePaths.defaults() if paths is None else paths

    _call_stage(
        "snapshot verification",
        lambda: verify_snapshot(selected_paths.snapshot, selected_paths.manifest),
    )
    checksums_before = _protected_checksums(
        selected_paths,
        outputs_may_remain=False,
    )
    _call_stage(
        "preflight",
        lambda: _preflight(selected_paths, overwrite_generated=overwrite_generated),
    )

    try:
        _call_stage(
            "cleaning",
            lambda: run_cleaning_pipeline(
                input_path=selected_paths.snapshot,
                output_path=selected_paths.clean_data,
                report_path=selected_paths.cleaning_report,
            ),
        )
        _call_stage(
            "feature engineering",
            lambda: run_feature_pipeline(
                input_path=selected_paths.clean_data,
                output_path=selected_paths.feature_data,
                report_path=selected_paths.feature_report,
            ),
        )
        _call_stage(
            "regression target",
            lambda: run_regression_target_pipeline(
                input_path=selected_paths.feature_data,
                output_path=selected_paths.regression_target_data,
                report_path=selected_paths.regression_target_report,
            ),
        )
        train, validation, test = _call_stage(
            "chronological split",
            lambda: run_split_pipeline(
                input_path=selected_paths.regression_target_data,
                train_output_path=selected_paths.train_data,
                validation_output_path=selected_paths.validation_data,
                test_output_path=selected_paths.test_data,
                report_output_path=selected_paths.split_report,
            ),
        )
        _, _, _, threshold = _call_stage(
            "classification target",
            lambda: run_classification_target_pipeline(
                train_input_path=selected_paths.train_data,
                validation_input_path=selected_paths.validation_data,
                test_input_path=selected_paths.test_data,
                train_output_path=selected_paths.train_labeled_data,
                validation_output_path=selected_paths.validation_labeled_data,
                test_output_path=selected_paths.test_labeled_data,
                threshold_report_path=(selected_paths.classification_threshold_report),
            ),
        )
    except Exception:
        _verify_protected_checksums(selected_paths, checksums_before)
        raise

    _verify_protected_checksums(selected_paths, checksums_before)
    return PipelineResult(
        paths=selected_paths,
        split_rows={
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        classification_threshold=float(threshold),
        snapshot_sha256=checksums_before["snapshot"],
        manifest_sha256=checksums_before["manifest"],
    )


def _build_parser() -> argparse.ArgumentParser:
    """Create the minimal command-line interface for the data runner."""
    parser = argparse.ArgumentParser(
        description="Build verified QQQ Phase 1 data artifacts end to end."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=RAW_DATA_PATH,
        help="Verified raw snapshot CSV (default: config.RAW_DATA_PATH).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=RAW_DATA_MANIFEST_PATH,
        help="Snapshot manifest JSON (default: config.RAW_DATA_MANIFEST_PATH).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Isolated root mapped to data/interim, data/processed, and "
            "outputs/reports. Defaults to the configured artifact paths."
        ),
    )
    parser.add_argument(
        "--overwrite-generated",
        action="store_true",
        help="Replace only the runner's 14 known generated artifacts.",
    )
    return parser


def _paths_from_arguments(arguments: argparse.Namespace) -> PipelinePaths:
    """Build default or isolated output paths from parsed CLI arguments."""
    if arguments.output_root is not None:
        return PipelinePaths.under_output_root(
            arguments.output_root,
            snapshot=arguments.snapshot,
            manifest=arguments.manifest,
        )

    defaults = PipelinePaths.defaults()
    return replace(
        defaults,
        snapshot=arguments.snapshot,
        manifest=arguments.manifest,
    )


def _print_success(result: PipelineResult) -> None:
    """Print the main artifact inventory and split counts."""
    print("Phase 1 data pipeline completed successfully.")
    print("Split rows:")
    for name, rows in result.split_rows.items():
        print(f"  {name}: {rows}")
    print(f"Train-only Q75 threshold: {result.classification_threshold:.12g}")
    print("Generated artifacts:")
    for name, path in result.paths.generated_outputs().items():
        print(f"  {name}: {path.resolve(strict=False)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a nonzero exit code with stage-aware errors."""
    arguments = _build_parser().parse_args(argv)
    paths = _paths_from_arguments(arguments)
    try:
        result = run_data_pipeline(
            paths,
            overwrite_generated=arguments.overwrite_generated,
        )
    except PipelineRunError as exc:
        print(f"Pipeline failed at stage '{exc.stage}': {exc}", file=sys.stderr)
        if exc.outputs_may_remain:
            print(
                "Generated artifacts from completed or partially completed stages "
                "may remain in place.",
                file=sys.stderr,
            )
        return 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"Pipeline failed before execution: {exc}", file=sys.stderr)
        return 1

    _print_success(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
