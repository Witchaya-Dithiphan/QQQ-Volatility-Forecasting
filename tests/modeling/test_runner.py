"""Test-only no-op artifact run and CLI gates; never a trained model."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from config import PROJECT_ROOT
from src.modeling.runner import run_noop, current_hashes
from src.modeling.persistence import resume_check


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
def test_dummy_e2e_reload_and_resume_without_test(tmp_path, monkeypatch, variant):
    original = Path.read_bytes
    def guard(path):
        assert not path.name.startswith("test_"), "Test input touched"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guard)
    path = run_noop(variant, tmp_path, "proof-1")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["task"] == "infrastructure" and manifest["implementation"] == "dummy"
    assert manifest["model"] == "noop" and manifest["role"] == "infrastructure_test"
    assert manifest["test_accessed"] is False
    assert manifest["load_verification"]["passed"] is True
    assert resume_check(manifest, current_hashes(variant), path)["can_skip"]
    assert (path / "runtime.json").is_file()
    with pytest.raises(FileExistsError): run_noop(variant, tmp_path, "proof-1")


def test_cli_checks_data_and_rejects_models_and_test():
    command = [sys.executable, "-m", "src.modeling.runner"]
    success = subprocess.run([*command, "check-data", "--variant", "with_spike"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert '"train_rows": 1733' in success.stdout
    for action in ("train", "tune", "evaluate", "finalize-test"):
        rejected = subprocess.run([*command, action], cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "M2" in rejected.stderr


def test_noop_failure_preserves_honest_status(tmp_path, monkeypatch):
    from src.modeling import runner
    def fail(self, X): raise ArithmeticError("synthetic failure")
    monkeypatch.setattr(runner.Standardizer, "fit", fail)
    with pytest.raises(ArithmeticError): run_noop("with_spike", tmp_path, "fail-proof")
    path = tmp_path / "with_spike/infrastructure/noop/dummy/fail-proof"
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["exception_type"] == "ArithmeticError"
    assert (path / manifest["error"]["traceback_path"]).exists()
    assert manifest["timing"]["elapsed_seconds"] >= 0
    assert manifest["load_verification"] is None


def test_noop_manifest_has_portable_input_and_git_provenance(tmp_path):
    path = run_noop("with_spike", tmp_path, "provenance-proof")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["input_identity"]["train"]["path"] == "data/processed/experiments/with_spikes/train.csv"
    assert manifest["input_identity"]["validation"]["path"] == "data/processed/validation_labeled.csv"
    assert manifest["input_identity"]["train"]["rows"] == 1733
    assert manifest["input_identity"]["validation"]["date_range"] == {"start": "2023-08-28", "end": "2025-02-19"}
    assert set(manifest["input_identity"]) == {"train", "validation"}
    assert len(manifest["git_revision"]) == 40 and isinstance(manifest["git_dirty"], bool)
    assert manifest["config_id"].startswith("cfg-")
