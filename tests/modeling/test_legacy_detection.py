"""Legacy run detection tests — classify_run_directory, m5_status, m6_status."""
from __future__ import annotations
import io
import json
import zipfile

import numpy as np
import pytest
from pathlib import Path

from src.modeling.persistence import classify_run_directory


def _make_minimal_manifest(**overrides) -> dict:
    """Minimal manifest dict with all keys required by read_manifest."""
    base = {
        "manifest_version": 1, "run_id": "testrun", "variant": "with_spike",
        "task": "classification", "model": "logistic", "implementation": "scratch",
        "lifecycle": "supervised_tuned", "role": "production", "pilot_only": False,
        "required": True, "status": "running", "stage": "initialized",
        "hashes": {"config": "abc", "inputs": "def", "code": "ghi", "dependency_lock": "jkl", "runtime": "mno"}, "artifact_checksums": {}, "load_verification": None,
        "test_accessed": False, "test_access": None, "selected_hyperparameters": None,
        "output_policy": None, "reason": None, "error": None, "mismatches": {},
        "git_revision": None, "git_dirty": None, "config_id": None, "input_identity": {},
        "timing": {"started_utc": None, "finished_utc": None, "elapsed_seconds": None},
    }
    base.update(overrides)
    return base


def _write_pickle_npz(path: Path) -> None:
    """Write an NPZ that contains an object-dtype array (requires allow_pickle=True)."""
    buf = io.BytesIO()
    np.save(buf, np.array([{"k": "v"}], dtype=object), allow_pickle=True)
    buf.seek(0)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("arr_0.npy", buf.read())


def _write_safe_npz(path: Path) -> None:
    buf = io.BytesIO()
    np.save(buf, np.array([1.0, 2.0]))
    buf.seek(0)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("arr_0.npy", buf.read())


# ---------------------------------------------------------------------------
# No manifest → legacy_incompatible
# ---------------------------------------------------------------------------

def test_empty_dir_is_legacy_incompatible(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    assert "missing_manifest" in result["reasons"]


def test_legacy_metadata_flagged(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text('{"model": "knn"}', encoding="utf-8")
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    assert "has_legacy_metadata" in result["reasons"]
    assert "missing_manifest" in result["reasons"]


def test_unsafe_npz_flagged(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_pickle_npz(run_dir / "model.npz")
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    assert "unsafe_object_npz" in result["reasons"]


def test_safe_npz_alone_still_legacy(tmp_path):
    """Safe NPZ without manifest is still legacy_incompatible."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_safe_npz(run_dir / "model.npz")
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    assert "missing_manifest" in result["reasons"]


# ---------------------------------------------------------------------------
# Legacy run must not be resumable or finalizable
# ---------------------------------------------------------------------------

def test_legacy_cannot_be_resumed(tmp_path):
    from src.modeling.persistence import read_manifest
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text('{"model": "knn"}', encoding="utf-8")
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    with pytest.raises(FileNotFoundError):
        read_manifest(run_dir)


def test_legacy_cannot_finalize_test(tmp_path):
    from src.modeling.persistence import _finalization_prerequisites
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text('{"model": "knn"}', encoding="utf-8")
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    with pytest.raises((FileNotFoundError, PermissionError, ValueError)):
        _finalization_prerequisites(run_dir, "finalize-test")


# ---------------------------------------------------------------------------
# Invalid/corrupt manifest → legacy_incompatible
# ---------------------------------------------------------------------------

def test_corrupt_manifest_flagged(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{bad json", encoding="utf-8")
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    assert "invalid_manifest" in result["reasons"]


# ---------------------------------------------------------------------------
# Fail-closed: manifest present but not accepted
# ---------------------------------------------------------------------------

def test_classify_running_manifest_is_incompatible(tmp_path):
    """status='running' must not pass — only 'completed' is accepted."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(_make_minimal_manifest(status="running")), encoding="utf-8")
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    assert "not_completed" in result["reasons"]


def test_classify_completed_missing_artifacts_is_incompatible(tmp_path):
    """completed manifest with no artifact files must return incomplete_artifacts."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(_make_minimal_manifest(status="completed", artifact_checksums={})),
        encoding="utf-8",
    )
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    assert "incomplete_artifacts" in result["reasons"]


def test_classify_completed_null_load_verification_is_incompatible(tmp_path):
    """completed manifest with load_verification=None must report missing_or_failed_load_verification."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(_make_minimal_manifest(status="completed", load_verification=None)),
        encoding="utf-8",
    )
    result = classify_run_directory(run_dir)
    assert result["status"] == "legacy_incompatible"
    assert "missing_or_failed_load_verification" in result["reasons"]


# ---------------------------------------------------------------------------
# m6_status integration with fail-closed classification
# ---------------------------------------------------------------------------

def test_m6_status_legacy_dir_is_incompatible(tmp_path):
    from src.modeling.m6_training import m6_status
    run_dir = tmp_path / "with_spike" / "classification" / "adaboost" / "scratch" / "run001"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text('{"model": "adaboost"}', encoding="utf-8")
    result = m6_status(tmp_path, "with_spike", "adaboost", "run001")
    assert result["status"] == "legacy_incompatible"
    assert result["can_skip"] is False


def test_m6_status_nonexistent_is_missing(tmp_path):
    from src.modeling.m6_training import m6_status
    result = m6_status(tmp_path, "with_spike", "adaboost", "run001")
    assert result["status"] == "missing"
    assert result["can_skip"] is False


# ---------------------------------------------------------------------------
# m5_status helper
# ---------------------------------------------------------------------------

def test_m5_status_legacy_dir_is_incompatible(tmp_path):
    from src.modeling.classification_training import m5_status
    run_dir = tmp_path / "with_spike" / "classification" / "logistic" / "scratch" / "run001"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text('{"model": "logistic"}', encoding="utf-8")
    result = m5_status(tmp_path, "with_spike", "logistic", "run001")
    assert result["status"] == "legacy_incompatible"
    assert result["can_skip"] is False


def test_m5_status_nonexistent_is_missing(tmp_path):
    from src.modeling.classification_training import m5_status
    result = m5_status(tmp_path, "with_spike", "logistic", "run001")
    assert result["status"] == "missing"
    assert result["can_skip"] is False


def _completed_run(path):
    from src.modeling import persistence as ps
    from src.modeling.artifacts import write_json
    path.mkdir(parents=True, exist_ok=True)
    manifest = _make_minimal_manifest(status="completed", load_verification={"passed": True})
    for name in ps.required_artifacts(manifest):
        artifact = path / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.suffix == ".npz":
            _write_safe_npz(artifact)
        else:
            artifact.write_text("{}", encoding="utf-8")
    (path / "load_verification.json").write_text(json.dumps(manifest["load_verification"]), encoding="utf-8")
    ps.capture_artifacts(manifest, path)
    write_json(path / "manifest.json", manifest)
    return manifest


def test_completed_object_npz_cannot_classify_resume_or_complete(tmp_path):
    from src.modeling import persistence as ps
    from src.modeling.artifacts import write_json
    manifest = _completed_run(tmp_path)
    assert ps.resume_check(manifest, manifest["hashes"], tmp_path)["can_skip"]
    _write_pickle_npz(tmp_path / "model.npz")
    ps.capture_artifacts(manifest, tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = ps.classify_run_directory(tmp_path)
    assert result["status"] == "legacy_incompatible"
    assert "unsafe_object_npz" in result["reasons"]
    resume = ps.resume_check(manifest, manifest["hashes"], tmp_path)
    assert resume["can_skip"] is False
    assert "unsafe_object_npz" in resume["mismatches"]["artifacts"]
    manifest["status"] = "running"
    with pytest.raises(ValueError, match="Cannot complete"):
        ps.transition(manifest, "completed", run_dir=tmp_path)


@pytest.mark.parametrize("helper,model", [("m5", "logistic"), ("m6", "adaboost")])
@pytest.mark.parametrize("damage,reason", [
    ("corrupt", "invalid_manifest"),
    ("artifacts", "incomplete_artifacts"),
    ("missing_receipt", "missing_or_failed_load_verification"),
    ("failed_receipt", "missing_or_failed_load_verification"),
    ("running", "not_completed"),
])
def test_status_classifies_every_existing_directory(tmp_path, monkeypatch, helper, model, damage, reason):
    from src.modeling import classification_training as m5, m6_training as m6
    from src.modeling.artifacts import write_json
    module = m5 if helper == "m5" else m6
    path = tmp_path / "with_spike" / "classification" / model / "scratch" / "run001"
    manifest = _completed_run(path)
    if damage == "corrupt":
        (path / "manifest.json").write_text("{bad", encoding="utf-8")
    elif damage == "artifacts":
        (path / "model.npz").unlink()
    elif damage == "missing_receipt":
        (path / "load_verification.json").unlink()
    else:
        if damage == "running":
            manifest["status"] = "running"
        else:
            manifest["load_verification"] = {"passed": False}
            (path / "load_verification.json").write_text(json.dumps({"passed": False}), encoding="utf-8")
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "resume_check", lambda *args: pytest.fail("Invalid directory reached resume"))
    result = getattr(module, helper + "_status")(tmp_path, "with_spike", model, "run001", hashes=manifest["hashes"])
    assert result["can_skip"] is False
    assert result["status"] == "legacy_incompatible"
    assert reason in result["mismatches"]["classification"]["reasons"]
