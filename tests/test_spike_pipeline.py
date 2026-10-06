"""Integration and safety tests for the separate Phase 2 runner."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import PROJECT_ROOT
from src.build_spike_input_contract import _sha256
from src.run_spike_analysis import (
    SpikePipelinePaths,
    SpikePipelineRunError,
    _discover_reproduced_root,
    _preflight,
    main,
    run_spike_pipeline,
)


def _production_paths(output_root: Path) -> SpikePipelinePaths:
    return SpikePipelinePaths.under_roots(
        input_root=PROJECT_ROOT,
        output_root=output_root,
    )


def _write_synthetic_inputs(root: Path) -> SpikePipelinePaths:
    reproduction = root / "accepted-reproduction"
    reproduction.mkdir(parents=True)
    paths = SpikePipelinePaths.under_roots(
        input_root=root / "baseline",
        output_root=root / "generated",
        reproduced_root=reproduction,
    )
    for name, path in paths.baseline_inputs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"protected:{name}\n", encoding="utf-8")
    return paths


def _copy_accepted_baseline(root: Path) -> tuple[Path, Path]:
    """Copy every protected Phase 1 input and return its reproduction root."""
    source_paths = _production_paths(root / "unused-output")
    baseline = root / "accepted-baseline"
    for source in source_paths.baseline_inputs().values():
        relative = source.relative_to(PROJECT_ROOT)
        destination = baseline / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    reproduced = _discover_reproduced_root(source_paths)
    return baseline, reproduced


def test_phase2_runner_happy_path_is_offline_read_only_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, reproduced = _copy_accepted_baseline(tmp_path)
    paths = SpikePipelinePaths.under_roots(
        input_root=baseline,
        output_root=tmp_path / "phase2-output",
        reproduced_root=reproduced,
    )
    protected_before = {
        name: path.read_bytes() for name, path in paths.baseline_inputs().items()
    }

    def network_forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Phase 2 runner attempted network access")

    monkeypatch.setattr("socket.create_connection", network_forbidden)
    first = run_spike_pipeline(paths)
    first_bytes = {
        name: path.read_bytes()
        for name, path in first.paths.generated_outputs().items()
    }
    unrelated = paths.output_root / "keep-me.txt"
    unrelated.write_text("not a declared output", encoding="utf-8")

    second = run_spike_pipeline(paths, overwrite_generated=True)

    assert first.threshold == pytest.approx(0.0465436445413787)
    assert first.row_counts == {
        "with_spikes_train": 1733,
        "non_spike_train": 1520,
        "validation_flagged": 371,
        "test_flagged": 373,
    }
    assert first.diagnostic_counts == {
        "validation": {"full": 371, "non_spike": 371, "spike_affected": 0},
        "test": {"full": 373, "non_spike": 319, "spike_affected": 54},
    }
    assert first.output_checksums == second.output_checksums
    assert {
        name: path.read_bytes()
        for name, path in second.paths.generated_outputs().items()
    } == first_bytes
    assert unrelated.read_text(encoding="utf-8") == "not a declared output"
    assert {
        name: path.read_bytes() for name, path in paths.baseline_inputs().items()
    } == protected_before
    assert len(first.paths.generated_outputs()) == 11
    assert all(path.is_file() for path in first.paths.generated_outputs().values())

    m1 = json.loads(paths.input_contract.read_text(encoding="utf-8"))
    experiment_root = (paths.input_contract.parent / m1["experiment_root"]).resolve()
    assert experiment_root == paths.with_spikes_train.parents[1].resolve()
    assert paths.train_labeled.is_relative_to(baseline)
    assert not paths.train_labeled.is_relative_to(paths.output_root)


def test_runner_refuses_existing_outputs_before_any_stage(tmp_path: Path) -> None:
    paths = _production_paths(tmp_path)
    paths.dataset_report.parent.mkdir(parents=True)
    paths.dataset_report.write_text("existing", encoding="utf-8")
    before = paths.dataset_report.read_bytes()

    with pytest.raises(SpikePipelineRunError, match="preflight failed") as captured:
        run_spike_pipeline(paths)

    assert captured.value.stage == "preflight"
    assert not captured.value.outputs_may_remain
    assert paths.dataset_report.read_bytes() == before
    assert not paths.input_contract.exists()


def test_preflight_rejects_duplicate_and_protected_destinations(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    duplicate = replace(paths, comparison_figure=paths.volatility_figure)
    with pytest.raises(ValueError, match="Duplicate output destinations"):
        _preflight(duplicate, overwrite_generated=False)

    protected = replace(paths, dataset_report=paths.train_labeled)
    with pytest.raises(ValueError, match="aliases protected input"):
        _preflight(protected, overwrite_generated=False)


def test_preflight_rejects_hardlink_overwrite(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("outside", encoding="utf-8")
    paths.dataset_report.parent.mkdir(parents=True)
    try:
        os.link(external, paths.dataset_report)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    checksum = _sha256(external)

    with pytest.raises(ValueError, match="Refusing to overwrite aliased output"):
        _preflight(paths, overwrite_generated=True)

    assert _sha256(external) == checksum


def test_preflight_rejects_output_escape_and_symlink_traversal(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    escaped = replace(paths, comparison_figure=tmp_path / "outside.png")
    with pytest.raises(ValueError, match="escapes allowed root"):
        _preflight(escaped, overwrite_generated=False)

    real_figures = tmp_path / "real-figures"
    real_figures.mkdir()
    figure_link = paths.output_root / "outputs" / "figures" / "spike_analysis"
    figure_link.parent.mkdir(parents=True)
    try:
        figure_link.symlink_to(real_figures, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable on this platform: {error}")
    with pytest.raises(ValueError, match="traverses a symlink"):
        _preflight(paths, overwrite_generated=False)


def test_preflight_failure_creates_no_output_directory(tmp_path: Path) -> None:
    paths = SpikePipelinePaths.under_roots(
        input_root=tmp_path / "missing-baseline",
        output_root=tmp_path / "new-output",
        reproduced_root=tmp_path / "missing-reproduction",
    )
    with pytest.raises(FileNotFoundError, match="Missing protected baseline input"):
        _preflight(paths, overwrite_generated=False)
    assert not paths.output_root.exists()


def test_stage_failure_reports_partial_outputs_and_preserves_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    before = {name: _sha256(path) for name, path in paths.baseline_inputs().items()}
    monkeypatch.setattr("src.run_spike_analysis.verify_snapshot", lambda *_: None)
    monkeypatch.setattr(
        "src.run_spike_analysis._validate_m1_before_writes", lambda *_: None
    )

    def write_m1(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        paths.input_contract.parent.mkdir(parents=True, exist_ok=True)
        paths.input_contract.write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(
        "src.run_spike_analysis.run_spike_input_contract_pipeline", write_m1
    )
    monkeypatch.setattr(
        "src.run_spike_analysis.run_spike_audit_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("audit boom")),
    )

    with pytest.raises(
        SpikePipelineRunError, match="M4 direct-spike audit failed"
    ) as captured:
        run_spike_pipeline(paths)

    assert captured.value.stage == "M4 direct-spike audit"
    assert captured.value.outputs_may_remain
    assert paths.input_contract.is_file()
    assert {
        name: _sha256(path) for name, path in paths.baseline_inputs().items()
    } == before


def test_cli_keeps_input_and_output_roots_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "accepted-baseline"
    generated = tmp_path / "isolated-phase2"

    def fake_run(
        paths: SpikePipelinePaths | None = None,
        *,
        overwrite_generated: bool = False,
    ) -> SimpleNamespace:
        assert paths is not None
        assert paths.input_root == baseline
        assert paths.output_root == generated
        assert (
            paths.train_labeled == baseline / "data" / "processed" / "train_labeled.csv"
        )
        assert (
            paths.input_contract
            == generated / "outputs" / "reports" / "spike_input_contract.json"
        )
        assert overwrite_generated
        checksums = {name: "hash" for name in paths.generated_outputs()}
        return SimpleNamespace(
            paths=paths,
            threshold=0.1,
            row_counts={},
            diagnostic_counts={},
            output_checksums=checksums,
        )

    monkeypatch.setattr("src.run_spike_analysis.run_spike_pipeline", fake_run)
    assert (
        main(
            [
                "--input-root",
                str(baseline),
                "--output-root",
                str(generated),
                "--overwrite-generated",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert f"Input root: {baseline.resolve()}" in output
    assert f"Output root: {generated.resolve()}" in output


def test_reproduction_discovery_and_required_path_errors(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    unresolved = replace(paths, reproduced_root=None)
    with pytest.raises(ValueError, match="must be resolved"):
        unresolved.m1_paths()
    with pytest.raises(FileNotFoundError, match="accepted M1 contract"):
        _discover_reproduced_root(unresolved)

    unresolved.accepted_input_contract.parent.mkdir(parents=True, exist_ok=True)
    unresolved.accepted_input_contract.write_text(
        json.dumps(
            {
                "saved_report_provenance": {
                    "isolated_phase1_reproduction": {
                        "output_root": str(tmp_path.resolve())
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="portable relative path"):
        _discover_reproduced_root(unresolved)


def test_preflight_rejects_missing_reproduction_unsafe_root_and_directory_output(
    tmp_path: Path,
) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    missing = replace(paths, reproduced_root=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="reproduction root"):
        _preflight(missing, overwrite_generated=False)

    unsafe_root = paths.raw_snapshot.parent / "generated"
    unsafe = replace(paths, output_root=unsafe_root)
    with pytest.raises(ValueError, match="inside a protected input directory"):
        _preflight(unsafe, overwrite_generated=False)

    paths.dataset_report.mkdir(parents=True)
    with pytest.raises(ValueError, match="existing directory"):
        _preflight(paths, overwrite_generated=True)


def test_changed_protected_input_is_reported_without_hiding_stage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    monkeypatch.setattr("src.run_spike_analysis.verify_snapshot", lambda *_: None)
    monkeypatch.setattr(
        "src.run_spike_analysis._validate_m1_before_writes", lambda *_: None
    )

    def corrupt_then_fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        paths.train_labeled.write_text("changed", encoding="utf-8")
        raise RuntimeError("M1 boom")

    monkeypatch.setattr(
        "src.run_spike_analysis.run_spike_input_contract_pipeline",
        corrupt_then_fail,
    )
    with pytest.raises(
        SpikePipelineRunError,
        match="M1 input contract failed.*additionally.*train_labeled",
    ) as captured:
        run_spike_pipeline(paths)
    assert captured.value.stage == "protected checksum verification"


def test_cli_failure_lists_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "output"

    def fail(paths: SpikePipelinePaths | None = None, **_: object) -> None:
        assert paths is not None
        paths.input_contract.parent.mkdir(parents=True)
        paths.input_contract.write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic CLI failure")

    monkeypatch.setattr("src.run_spike_analysis.run_spike_pipeline", fail)
    assert main(["--output-root", str(output_root)]) == 1
    error = capsys.readouterr().err
    assert "synthetic CLI failure" in error
    assert "Partial generated outputs may remain" in error
    assert "spike_input_contract" in error
