"""Run the verified Phase 2 spike-analysis stages without rebuilding Phase 1."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

if __package__ in {None, ""}:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import PROJECT_ROOT
from src.audit_spikes import SpikeAuditPaths, run_spike_audit_pipeline
from src.build_experiment_datasets import (
    ExperimentDatasetPaths,
    run_experiment_dataset_pipeline,
)
from src.build_spike_input_contract import (
    DEFAULT_AUDIT_EXPECTATIONS,
    SpikeInputContractPaths,
    _collect_reproduction_evidence,
    _sha256,
    _strict_json_load,
    load_labeled_split,
    run_spike_input_contract_pipeline,
    validate_input_contract,
)
from src.build_spike_reports import SpikeReportPaths, run_spike_report_pipeline
from src.download_qqq_data import verify_snapshot

_T = TypeVar("_T")


class SpikePipelineRunError(RuntimeError):
    """A stage-aware Phase 2 failure that reports whether outputs may remain."""

    def __init__(self, stage: str, message: str, *, outputs_may_remain: bool) -> None:
        self.stage = stage
        self.outputs_may_remain = outputs_may_remain
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SpikePipelinePaths:
    """Explicit baseline inputs and generated Phase 2 destinations."""

    input_root: Path
    output_root: Path
    reproduced_root: Path | None
    raw_snapshot: Path
    manifest: Path
    clean_data: Path
    feature_data: Path
    target_data: Path
    train_labeled: Path
    validation_labeled: Path
    test_labeled: Path
    split_report: Path
    classification_report: Path
    accepted_input_contract: Path
    input_contract: Path
    spike_analysis: Path
    spike_events: Path
    daily_return_figure: Path
    with_spikes_train: Path
    non_spike_train: Path
    validation_flagged: Path
    test_flagged: Path
    dataset_report: Path
    volatility_figure: Path
    comparison_figure: Path

    @classmethod
    def under_roots(
        cls,
        *,
        input_root: str | Path = PROJECT_ROOT,
        output_root: str | Path = PROJECT_ROOT,
        reproduced_root: str | Path | None = None,
    ) -> SpikePipelinePaths:
        """Map baseline reads and Phase 2 writes independently."""
        source = Path(input_root)
        destination = Path(output_root)
        reports = destination / "outputs" / "reports"
        figures = destination / "outputs" / "figures" / "spike_analysis"
        experiments = destination / "data" / "processed" / "experiments"
        return cls(
            input_root=source,
            output_root=destination,
            reproduced_root=(
                None if reproduced_root is None else Path(reproduced_root)
            ),
            raw_snapshot=source / "data" / "raw" / "qqq_daily.csv",
            manifest=source / "data" / "manifests" / "qqq_daily_snapshot.json",
            clean_data=source / "data" / "interim" / "qqq_clean.csv",
            feature_data=source / "data" / "interim" / "qqq_features.csv",
            target_data=source / "data" / "interim" / "qqq_regression_target.csv",
            train_labeled=source / "data" / "processed" / "train_labeled.csv",
            validation_labeled=(
                source / "data" / "processed" / "validation_labeled.csv"
            ),
            test_labeled=source / "data" / "processed" / "test_labeled.csv",
            split_report=source / "outputs" / "reports" / "data_split_report.json",
            classification_report=(
                source / "outputs" / "reports" / "classification_threshold.json"
            ),
            accepted_input_contract=(
                source / "outputs" / "reports" / "spike_input_contract.json"
            ),
            input_contract=reports / "spike_input_contract.json",
            spike_analysis=reports / "spike_analysis.json",
            spike_events=reports / "spike_event_audit.csv",
            daily_return_figure=figures / "daily_return_spikes.png",
            with_spikes_train=experiments / "with_spikes" / "train.csv",
            non_spike_train=experiments / "non_spike" / "train.csv",
            validation_flagged=experiments / "diagnostics" / "validation_flagged.csv",
            test_flagged=experiments / "diagnostics" / "test_flagged.csv",
            dataset_report=reports / "experiment_dataset_report.json",
            volatility_figure=figures / "volatility_spike_effect.png",
            comparison_figure=figures / "dataset_comparison.png",
        )

    def baseline_inputs(self) -> dict[str, Path]:
        """Return Phase 1 artifacts that no Phase 2 stage may modify."""
        return {
            "raw_snapshot": self.raw_snapshot,
            "manifest": self.manifest,
            "clean_data": self.clean_data,
            "feature_data": self.feature_data,
            "target_data": self.target_data,
            "train_labeled": self.train_labeled,
            "validation_labeled": self.validation_labeled,
            "test_labeled": self.test_labeled,
            "data_split_report": self.split_report,
            "classification_threshold_report": self.classification_report,
        }

    def generated_outputs(self) -> dict[str, Path]:
        """Return all and only the eleven files M1/M4/M5/M6 may write."""
        return {
            "spike_input_contract": self.input_contract,
            "spike_analysis": self.spike_analysis,
            "spike_event_audit": self.spike_events,
            "daily_return_spikes": self.daily_return_figure,
            "with_spikes_train": self.with_spikes_train,
            "non_spike_train": self.non_spike_train,
            "validation_flagged": self.validation_flagged,
            "test_flagged": self.test_flagged,
            "experiment_dataset_report": self.dataset_report,
            "volatility_spike_effect": self.volatility_figure,
            "dataset_comparison": self.comparison_figure,
        }

    def categorized_outputs(self) -> tuple[tuple[Path, tuple[Path, ...]], ...]:
        """Associate each declared output with its allowed generated area."""
        reports = self.output_root / "outputs" / "reports"
        figures = self.output_root / "outputs" / "figures" / "spike_analysis"
        experiments = self.output_root / "data" / "processed" / "experiments"
        return (
            (
                reports,
                (
                    self.input_contract,
                    self.spike_analysis,
                    self.spike_events,
                    self.dataset_report,
                ),
            ),
            (
                figures,
                (
                    self.daily_return_figure,
                    self.volatility_figure,
                    self.comparison_figure,
                ),
            ),
            (
                experiments,
                (
                    self.with_spikes_train,
                    self.non_spike_train,
                    self.validation_flagged,
                    self.test_flagged,
                ),
            ),
        )

    def m1_paths(self) -> SpikeInputContractPaths:
        if self.reproduced_root is None:
            raise ValueError("reproduced_root must be resolved before M1")
        return SpikeInputContractPaths(
            train_labeled=self.train_labeled,
            validation_labeled=self.validation_labeled,
            test_labeled=self.test_labeled,
            split_report=self.split_report,
            classification_report=self.classification_report,
            reproduced_root=self.reproduced_root,
            output_report=self.input_contract,
            experiment_root=self.with_spikes_train.parents[1],
        )

    def m4_paths(self) -> SpikeAuditPaths:
        return SpikeAuditPaths(
            self.raw_snapshot,
            self.manifest,
            self.clean_data,
            self.feature_data,
            self.target_data,
            self.train_labeled,
            self.validation_labeled,
            self.test_labeled,
            self.input_contract,
            self.spike_analysis,
            self.spike_events,
            self.daily_return_figure,
        )

    def m5_paths(self) -> ExperimentDatasetPaths:
        return ExperimentDatasetPaths(
            self.raw_snapshot,
            self.manifest,
            self.train_labeled,
            self.validation_labeled,
            self.test_labeled,
            self.input_contract,
            self.spike_analysis,
            self.spike_events,
            self.daily_return_figure,
            self.with_spikes_train,
            self.non_spike_train,
            self.validation_flagged,
            self.test_flagged,
        )

    def m6_paths(self) -> SpikeReportPaths:
        return SpikeReportPaths(
            self.train_labeled,
            self.validation_labeled,
            self.test_labeled,
            self.input_contract,
            self.spike_analysis,
            self.daily_return_figure,
            self.with_spikes_train,
            self.non_spike_train,
            self.validation_flagged,
            self.test_flagged,
            self.dataset_report,
            self.volatility_figure,
            self.comparison_figure,
        )


@dataclass(frozen=True, slots=True)
class SpikePipelineResult:
    """Successful Phase 2 run metadata for callers and the CLI summary."""

    paths: SpikePipelinePaths
    threshold: float
    row_counts: Mapping[str, int]
    diagnostic_counts: Mapping[str, Mapping[str, int]]
    protected_checksums: Mapping[str, str]
    output_checksums: Mapping[str, str]


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            return False
    return False


def _has_symlink_ancestor(path: Path, stop: Path) -> bool:
    candidate = path
    stop_resolved = stop.resolve(strict=False)
    while True:
        if candidate.is_symlink():
            return True
        if (
            candidate.resolve(strict=False) == stop_resolved
            or candidate.parent == candidate
        ):
            return False
        candidate = candidate.parent


def _discover_reproduced_root(paths: SpikePipelinePaths) -> Path:
    if paths.reproduced_root is not None:
        return paths.reproduced_root
    if not paths.accepted_input_contract.is_file():
        raise FileNotFoundError(
            "Missing accepted M1 contract; pass --reproduced-root explicitly: "
            f"{paths.accepted_input_contract}"
        )
    accepted = _strict_json_load(paths.accepted_input_contract)
    serialized = accepted["saved_report_provenance"]["isolated_phase1_reproduction"][
        "output_root"
    ]
    if not isinstance(serialized, str) or Path(serialized).is_absolute():
        raise ValueError(
            "Accepted M1 reproduction root must be a portable relative path"
        )
    return (paths.accepted_input_contract.parent / serialized).resolve()


def _preflight(paths: SpikePipelinePaths, *, overwrite_generated: bool) -> None:
    """Reject missing inputs and every unsafe output before creating directories."""
    for name, path in paths.baseline_inputs().items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected baseline input {name}: {path}")
    if paths.reproduced_root is None or not paths.reproduced_root.is_dir():
        raise FileNotFoundError(
            f"Missing Phase 1 reproduction root: {paths.reproduced_root}"
        )

    outputs = paths.generated_outputs()
    protected = paths.baseline_inputs()
    output_items = list(outputs.items())
    for output_name, output in output_items:
        for source_name, source in protected.items():
            if _paths_alias(output, source):
                raise ValueError(
                    f"Generated output {output_name} aliases protected input {source_name}"
                )
    for index, (name, path) in enumerate(output_items):
        for other_name, other_path in output_items[index + 1 :]:
            if _paths_alias(path, other_path):
                raise ValueError(f"Duplicate output destinations: {name}, {other_name}")

    resolved_output_root = paths.output_root.resolve(strict=False)
    protected_directories = {
        path.resolve(strict=False).parent for path in protected.values()
    }
    if any(
        resolved_output_root == directory
        or _is_relative_to(resolved_output_root, directory)
        for directory in protected_directories
    ):
        raise ValueError("output_root cannot be inside a protected input directory")
    for allowed_root, categorized in paths.categorized_outputs():
        resolved_allowed = allowed_root.resolve(strict=False)
        if _has_symlink_ancestor(allowed_root, paths.output_root):
            raise ValueError(
                f"Generated output area traverses a symlink: {allowed_root}"
            )
        for output in categorized:
            resolved = output.resolve(strict=False)
            if resolved == resolved_allowed or not _is_relative_to(
                resolved, resolved_allowed
            ):
                raise ValueError(
                    f"Output escapes allowed root {allowed_root}: {output}"
                )

    existing = [
        (name, path)
        for name, path in output_items
        if path.exists() or path.is_symlink()
    ]
    if any(path.is_dir() for _, path in existing):
        raise ValueError("A generated output path is an existing directory")
    if existing and not overwrite_generated:
        details = ", ".join(f"{name}={path}" for name, path in existing)
        raise FileExistsError(
            "Phase 2 outputs already exist; pass --overwrite-generated to replace "
            f"only the eleven declared files: {details}"
        )
    if overwrite_generated:
        for name, path in existing:
            if path.is_symlink() or path.stat().st_nlink > 1:
                raise ValueError(f"Refusing to overwrite aliased output: {name}={path}")


def _validate_m1_before_writes(paths: SpikePipelinePaths) -> None:
    """Run M1's read-only semantic and reproduction checks before any write."""
    m1 = paths.m1_paths()
    frames = {
        name: load_labeled_split(path, name)
        for name, path in m1.labeled_inputs().items()
    }
    validated = validate_input_contract(
        frames,
        _strict_json_load(m1.split_report),
        _strict_json_load(m1.classification_report),
        expectations=DEFAULT_AUDIT_EXPECTATIONS,
    )
    if not validated["classification_contract"]["labels_valid"]:
        raise ValueError("M1 classification labels are invalid")
    _collect_reproduction_evidence(
        m1.reproduced_root,
        {name: _sha256(path) for name, path in m1.labeled_inputs().items()},
        m1.output_report,
    )


def _call_stage(stage: str, function: Callable[[], _T]) -> _T:
    try:
        return function()
    except Exception as error:
        raise SpikePipelineRunError(
            stage,
            f"{stage} failed: {error}",
            outputs_may_remain=stage
            not in {"snapshot verification", "preflight", "M1 pre-write validation"},
        ) from error


def _checksums(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in paths.items()}


def _verify_protected(
    paths: SpikePipelinePaths,
    before: Mapping[str, str],
    *,
    stage_error: Exception | None,
) -> None:
    try:
        after = _checksums(paths.baseline_inputs())
    except OSError as error:
        message = f"Could not re-check protected inputs: {error}"
        if stage_error is not None:
            message = f"{stage_error}; additionally, {message}"
        raise SpikePipelineRunError(
            "protected checksum verification", message, outputs_may_remain=True
        ) from error
    changed = [name for name, digest in before.items() if after[name] != digest]
    if changed:
        message = "Protected baseline input changed: " + ", ".join(changed)
        if stage_error is not None:
            message = f"{stage_error}; additionally, {message}"
        raise SpikePipelineRunError(
            "protected checksum verification", message, outputs_may_remain=True
        ) from stage_error


def run_spike_pipeline(
    paths: SpikePipelinePaths | None = None,
    *,
    overwrite_generated: bool = False,
) -> SpikePipelineResult:
    """Run M1 and M4-M6; M2/M3 remain the canonical APIs used by those stages."""
    selected = SpikePipelinePaths.under_roots() if paths is None else paths
    selected = replace(selected, reproduced_root=_discover_reproduced_root(selected))
    _call_stage(
        "snapshot verification",
        lambda: verify_snapshot(selected.raw_snapshot, selected.manifest),
    )
    _call_stage(
        "preflight",
        lambda: _preflight(selected, overwrite_generated=overwrite_generated),
    )
    before = _checksums(selected.baseline_inputs())
    try:
        _call_stage(
            "M1 pre-write validation", lambda: _validate_m1_before_writes(selected)
        )
        _call_stage(
            "M1 input contract",
            lambda: run_spike_input_contract_pipeline(
                selected.m1_paths(), overwrite_generated=overwrite_generated
            ),
        )
        audit = _call_stage(
            "M4 direct-spike audit",
            lambda: run_spike_audit_pipeline(
                selected.m4_paths(), overwrite_generated=overwrite_generated
            ),
        )
        datasets = _call_stage(
            "M5 experiment datasets",
            lambda: run_experiment_dataset_pipeline(
                selected.m5_paths(), overwrite_generated=overwrite_generated
            ),
        )
        _call_stage(
            "M6 reports and figures",
            lambda: run_spike_report_pipeline(
                selected.m6_paths(), overwrite_generated=overwrite_generated
            ),
        )
    except Exception as error:
        _verify_protected(selected, before, stage_error=error)
        raise
    _verify_protected(selected, before, stage_error=None)
    output_checksums = _checksums(selected.generated_outputs())
    return SpikePipelineResult(
        paths=selected,
        threshold=float(audit["primary_detector"]["threshold"]),
        row_counts=datasets.row_counts,
        diagnostic_counts=datasets.diagnostic_counts,
        protected_checksums=before,
        output_checksums=output_checksums,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 2 offline from read-only accepted Phase 1 inputs. "
            "--output-root changes generated Phase 2 destinations only."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root containing accepted Phase 1 inputs (default: project root).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root for generated Phase 2 artifacts only; never remaps inputs.",
    )
    parser.add_argument(
        "--reproduced-root",
        type=Path,
        help="Optional accepted isolated Phase 1 reproduction; otherwise discover it from the existing M1 contract under input-root.",
    )
    parser.add_argument("--overwrite-generated", action="store_true")
    return parser


def _partial_outputs(paths: SpikePipelinePaths) -> dict[str, Path]:
    return {
        name: path for name, path in paths.generated_outputs().items() if path.is_file()
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = SpikePipelinePaths.under_roots(
        input_root=arguments.input_root,
        output_root=arguments.output_root,
        reproduced_root=arguments.reproduced_root,
    )
    try:
        result = run_spike_pipeline(
            paths, overwrite_generated=arguments.overwrite_generated
        )
    except Exception as error:  # noqa: BLE001 - CLI provides one nonzero boundary
        print(f"Phase 2 spike pipeline failed: {error}", file=sys.stderr)
        partial = _partial_outputs(paths)
        if partial:
            print("Partial generated outputs may remain:", file=sys.stderr)
            for name, path in partial.items():
                print(f"  {name}: {path.resolve(strict=False)}", file=sys.stderr)
        return 1
    print("Phase 2 spike pipeline complete (Phase 1 inputs remained read-only).")
    print(f"Input root: {result.paths.input_root.resolve(strict=False)}")
    print(f"Output root: {result.paths.output_root.resolve(strict=False)}")
    print(f"Primary threshold: {result.threshold:.16g}")
    print(f"Rows: {dict(result.row_counts)}")
    print(f"Diagnostics: {dict(result.diagnostic_counts)}")
    print("Generated artifacts:")
    for name, path in result.paths.generated_outputs().items():
        print(
            f"  {name}: {path.resolve(strict=False)} sha256={result.output_checksums[name]}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
