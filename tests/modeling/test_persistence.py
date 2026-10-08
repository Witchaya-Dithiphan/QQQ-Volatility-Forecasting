"""Immutable artifact storage, lifecycle completeness and resume identity."""
import copy
import json
import numpy as np
import pytest
from src.modeling import persistence as ps
from src.modeling.artifacts import write_json, file_sha256


def hashes(): return {"config": "a" * 64, "inputs": {"train": "b" * 64, "validation": "c" * 64}, "code": "d" * 64, "dependency_lock": "e" * 64, "runtime": "f" * 64}


def manifest(lifecycle="infrastructure_noop"):
    return ps.make_manifest(run_id="run-1", variant="with_spike", task="infrastructure", model="noop", implementation="dummy", lifecycle=lifecycle, hashes=hashes(), role="infrastructure_test")


def complete(tmp_path):
    result = manifest()
    ps.transition(result, "running")
    write_json(tmp_path / "config.json", {})
    ps.save_npz(tmp_path / "state.npz", {"values": np.arange(4.)}, {"kind": "noop"})
    write_json(tmp_path / "load_verification.json", {"passed": True})
    result["load_verification"] = {"passed": True}
    ps.capture_artifacts(result, tmp_path)
    ps.transition(result, "completed", run_dir=tmp_path)
    return result


def test_npz_metadata_safe_reload_and_nonoverwrite(tmp_path):
    path = tmp_path / "state.npz"
    arrays = {"weights": np.array([1., 2.]), "labels": np.array([0, 1])}
    ps.save_npz(path, arrays, {"feature_order": ["a", "b"]})
    loaded, metadata = ps.load_npz(path)
    assert metadata == {"feature_order": ["a", "b"]}
    for name in arrays: np.testing.assert_array_equal(loaded[name], arrays[name])
    with pytest.raises(FileExistsError): ps.save_npz(path, arrays, {})
    with pytest.raises(ValueError): ps.save_npz(tmp_path / "unsafe.npz", {"bad": np.array([object()])}, {})
    with pytest.raises(ValueError): ps.save_npz(tmp_path / "bad.npz", {"bad": np.array([np.nan])}, {})
    assert not (tmp_path / "bad.npz").exists()


def test_reload_equality_shapes_tolerances_and_exact_labels():
    assert ps.verify_reload([1.], [1. + 1e-11])["passed"]
    assert not ps.verify_reload([1.], [1.01])["passed"]
    assert not ps.verify_reload([1.], [[1.]])["passed"]
    assert not ps.verify_reload([np.nan], [np.nan])["passed"]
    assert not ps.verify_reload([0, 1], [0, 1.00000000001], labels=True)["passed"]


def test_run_paths_are_nonoverwriting_and_isolated(tmp_path):
    path = ps.create_run_directory(tmp_path, "with_spike", "regression", "multiple_linear", "scratch", "run-1")
    assert path == tmp_path / "with_spike/regression/multiple_linear/scratch/run-1"
    with pytest.raises(FileExistsError): ps.create_run_directory(tmp_path, "with_spike", "regression", "multiple_linear", "scratch", "run-1")
    assert ps.create_run_directory(tmp_path, "non_spike", "regression", "multiple_linear", "scratch", "run-1") != path
    with pytest.raises(ValueError): ps.create_run_directory(tmp_path, "with_spike", "regression", "../escape", "scratch", "run-1")


def test_complete_resume_and_tampered_artifacts(tmp_path):
    result = complete(tmp_path)
    assert ps.resume_check(result, hashes(), tmp_path)["can_skip"]
    (tmp_path / "config.json").write_text("changed")
    checked = ps.resume_check(result, hashes(), tmp_path)
    assert not checked["can_skip"] and checked["status"] == "incompatible"


@pytest.mark.parametrize("field", ["config", "inputs", "code", "dependency_lock", "runtime"])
def test_every_hash_blocks_resume(tmp_path, field):
    result = complete(tmp_path)
    actual = hashes()
    actual[field] = {"train": "changed"} if field == "inputs" else "0" * 64
    checked = ps.resume_check(result, actual, tmp_path)
    assert not checked["can_skip"] and field in checked["mismatches"]
    assert checked["mismatches"][field]["expected"] == result["hashes"][field]


def test_lifecycle_status_and_required_artifacts(tmp_path):
    result = manifest()
    with pytest.raises(ValueError): ps.transition(result, "completed", run_dir=tmp_path)
    ps.transition(result, "running")
    with pytest.raises(ValueError): ps.transition(result, "completed", run_dir=tmp_path)
    ps.transition(result, "failed", stage="serialize", error={"exception_type": "ValueError", "message": "bad", "traceback_path": "traceback.txt"})
    assert result["status"] == "failed" and result["error"]["message"] == "bad"
    with pytest.raises(ValueError): ps.transition(result, "running")
    skipped = manifest()
    with pytest.raises(ValueError): ps.transition(skipped, "skipped")
    ps.transition(skipped, "skipped", reason="explicit optional skip")
    assert not ps.resume_check(skipped, hashes(), tmp_path)["can_skip"]


def test_supervised_lifecycle_forbids_test_and_clustering_confusion(tmp_path):
    result = manifest("supervised_tuned")
    ps.transition(result, "running")
    write_json(tmp_path / "test_metrics.json", {})
    with pytest.raises(ValueError, match="Test"): ps.capture_artifacts(result, tmp_path)
    cluster = manifest("clustering_train")
    with pytest.raises(ValueError): ps.validate_lifecycle(cluster, {"confusion_matrix": [[1, 0], [0, 1]]})


def test_missing_reload_and_required_file_block_resume(tmp_path):
    result = complete(tmp_path)
    result["load_verification"]["passed"] = False
    assert not ps.resume_check(result, hashes(), tmp_path)["can_skip"]
    result["load_verification"]["passed"] = True
    (tmp_path / "state.npz").unlink()
    assert not ps.resume_check(result, hashes(), tmp_path)["can_skip"]


def test_reference_adapter_checksum_and_nooverwrite(tmp_path):
    path = tmp_path / "reference.joblib"
    value = {"array": np.arange(3.), "kind": "synthetic_artifact"}
    ps.save_reference(path, value)
    loaded = ps.load_reference(path, expected_sha256=file_sha256(path))
    np.testing.assert_array_equal(loaded["array"], value["array"])
    with pytest.raises(FileExistsError): ps.save_reference(path, value)
    with pytest.raises(ValueError): ps.load_reference(path, expected_sha256="0" * 64)


def test_corrupt_reload_record_blocks_resume(tmp_path):
    result = complete(tmp_path)
    (tmp_path / "load_verification.json").write_text('{"passed":false}')
    ps.capture_artifacts(result, tmp_path)
    assert not ps.resume_check(result, hashes(), tmp_path)["can_skip"]



def test_git_state_excludes_only_generated_modeling_outputs(tmp_path, monkeypatch):
    import subprocess
    import config
    original_run = subprocess.run
    original_root = config.PROJECT_ROOT
    original_run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    generated = tmp_path / "outputs/modeling/with_spike/regression/model/scratch/run/manifest.json"
    generated.parent.mkdir(parents=True)
    generated.write_text("{}")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    def git_command(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            kwargs["cwd"] = original_root
        return original_run(args, **kwargs)
    monkeypatch.setattr(subprocess, "run", git_command)
    assert ps._git_state()[0] is False
    unrelated = tmp_path / "source.py"
    unrelated.write_text("source")
    assert ps._git_state()[0] is True
    unrelated.unlink()
    (tmp_path / "notes.md").write_text("documentation")
    assert ps._git_state()[0] is True


def test_default_modeling_output_is_gitignored():
    import subprocess
    from config import PROJECT_ROOT
    result = subprocess.run(["git", "check-ignore", "--no-index", "outputs/modeling/synthetic/manifest.json"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0
