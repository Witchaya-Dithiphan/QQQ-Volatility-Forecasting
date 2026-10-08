"""Finalization gate tests use synthetic Test fixtures, never production Test."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.modeling import persistence as ps, runner, datasets as ds
from src.modeling.configuration import load_config
from src.modeling.artifacts import write_json, file_sha256
from src.modeling.contracts import FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET


@pytest.fixture
def tuned(tmp_path, monkeypatch):
    config = load_config()
    hashes = {"config": config["config_sha256"], "inputs": {"train": "b" * 64, "validation": "c" * 64}, "code": "d" * 64, "dependency_lock": "e" * 64, "runtime": "f" * 64}
    manifest = ps.make_manifest(run_id="synthetic", variant="with_spike", task="regression", model="multiple_linear", implementation="scratch", lifecycle="supervised_tuned", hashes=hashes, role="candidate")
    manifest["selected_hyperparameters"] = {}
    manifest["output_policy"] = {"clipping_min": 0.0}
    ps.transition(manifest, "running")
    write_json(tmp_path / "config.json", config)
    for name in ["model", "preprocessor"]: ps.save_npz(tmp_path / (name + ".npz"), {"values": np.arange(2.)}, {"synthetic_gate_fixture": True})
    for name in ["search_results.json", "validation_metrics.json"]: write_json(tmp_path / name, {})
    (tmp_path / "validation_predictions.csv").write_text("Date,target,raw_prediction,final_prediction,clipping_indicator\n2024-01-01,0,0,0,False\n")
    manifest["load_verification"] = {"passed": True}
    write_json(tmp_path / "load_verification.json", {"passed": True})
    ps.capture_artifacts(manifest, tmp_path)
    ps.transition(manifest, "completed", run_dir=tmp_path)
    write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(runner, "current_hashes", lambda variant: hashes)
    monkeypatch.setattr(ps, "_git_state", lambda: (False, "synthetic-revision"))
    return tmp_path, manifest, hashes


def replace_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("block", ["dirty", "wrong_command", "no_selection", "no_policy", "pilot", "hash", "reload", "unknown_manifest"])
def test_finalize_prerequisites_default_deny(tuned, monkeypatch, block):
    path, manifest, hashes = tuned
    if block == "dirty": monkeypatch.setattr(ps, "_git_state", lambda: (True, "rev"))
    if block == "no_selection": manifest["selected_hyperparameters"] = None
    if block == "no_policy": manifest["output_policy"] = None
    if block == "pilot": manifest["pilot_only"] = True
    if block == "hash": hashes["code"] = "0" * 64
    if block == "reload": manifest["load_verification"]["passed"] = False
    if block == "unknown_manifest": manifest["unknown"] = 1
    replace_manifest(path, manifest)
    command = "evaluate" if block == "wrong_command" else "finalize-test --run synthetic"
    with pytest.raises((PermissionError, ValueError)):
        ps.authorize_finalize_test(path, caller_command=command)


def test_authorized_synthetic_test_access_records_provenance(tuned, tmp_path, monkeypatch):
    path, manifest, hashes = tuned
    authorization = ps.authorize_finalize_test(path, caller_command="finalize-test --run synthetic")
    import pandas as pd
    frame = pd.DataFrame({"Date": ["2025-01-01", "2025-01-02"]})
    for col in ["Close", "Volume", "Open", "High", "Low", *FEATURE_COLUMNS]: frame[col] = [1., 2.]
    frame[REGRESSION_TARGET] = [.1, .3]
    frame[CLASSIFICATION_TARGET] = [0, 1]
    test_path = tmp_path / "synthetic-input.csv"
    frame.to_csv(test_path, index=False)
    digest = file_sha256(test_path)
    # Synthetic input is outside the run artifact directory to preserve inventory.
    fixture_path = tmp_path.parent / (tmp_path.name + "-fixture.csv")
    test_path.rename(fixture_path)
    monkeypatch.setattr(ds, "TEST_PATH", fixture_path)
    monkeypatch.setattr(ds, "_load_reports", lambda: ({"artifacts": {"test_labeled": {"sha256": digest}}}, {"classification_contract": {"threshold": .25}, "inputs": {"test": {"sha256": digest}}}))
    result = ds.load_test(allow_test=True, authorization=authorization)
    assert len(result.X) == 2
    record = json.loads((path / "test_access.json").read_text())
    assert record["test_accessed"] is True and record["test_input_sha256"] == digest
    assert record["config_sha256"] == hashes["config"]
    assert record["code_snapshot_sha256"] == hashes["code"]
    assert record["caller_command"].startswith("finalize-test") and record["utc_timestamp"]
    with pytest.raises((PermissionError, FileExistsError)): ds.load_test(allow_test=True, authorization=authorization)


def test_authorization_rechecked_after_issue(tuned, monkeypatch):
    path, _, _ = tuned
    authorization = ps.authorize_finalize_test(path, caller_command="finalize-test")
    monkeypatch.setattr(ps, "_git_state", lambda: (True, "rev"))
    with pytest.raises(PermissionError): authorization.validate()


def test_access_snapshot_manifest_records_actual_access(tuned):
    path, original, _ = tuned
    authorization = ps.authorize_finalize_test(path, caller_command="finalize-test")
    authorization.record_access("a" * 64)
    snapshot = json.loads((path / "test_access_manifest.json").read_text())
    assert snapshot["test_accessed"] is True
    assert snapshot["test_access"]["test_input_sha256"] == "a" * 64
    assert snapshot["status"] == "running" and snapshot["stage"] == "test_accessed"
    assert snapshot["lifecycle"] == "supervised_finalized"
    assert json.loads((path / "manifest.json").read_text())["test_accessed"] is False
