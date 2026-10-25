"""The one-shot log-HAR evaluator must refuse before it can read the held-out split."""
import json

import pytest

from src.ml.log_har import test_eval


def test_refuses_without_allow_test():
    with pytest.raises(PermissionError):
        test_eval.main([])


def test_refuses_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(test_eval, "ROOT", tmp_path)
    (tmp_path / "test_receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        test_eval.main(["--allow-test"])


def test_refuses_when_code_changed_after_the_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(test_eval, "ROOT", tmp_path)
    (tmp_path / "decision.json").write_text(json.dumps(
        {"hashes": {"code": {"model.py": "0" * 64}, "source_manifest": "", "prior": "", "models": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="code changed"):
        test_eval.main(["--allow-test"])


def test_a_crash_after_opening_still_blocks_a_second_read(tmp_path, monkeypatch):
    monkeypatch.setattr(test_eval, "ROOT", tmp_path)
    monkeypatch.setattr(test_eval, "verify_decision", lambda: {})
    # Pre-lock preparation is stubbed; it is real code that needs no Test data.
    monkeypatch.setattr(test_eval, "load_arm", lambda variant, arm: None)
    monkeypatch.setattr(test_eval, "load_classroom", lambda variant, name: None)

    def boom(**_):
        raise RuntimeError("crash while reading")
    monkeypatch.setattr(test_eval, "load_test", boom)
    with pytest.raises(RuntimeError):
        test_eval.main(["--allow-test"])
    assert json.loads((tmp_path / "test_receipt.json").read_text(encoding="utf-8"))["status"] == "opened"
    with pytest.raises(FileExistsError):
        test_eval.main(["--allow-test"])


def test_a_failure_before_the_read_leaves_test_unopened(tmp_path, monkeypatch):
    monkeypatch.setattr(test_eval, "ROOT", tmp_path)
    monkeypatch.setattr(test_eval, "verify_decision", lambda: {})

    def missing(variant, arm):
        raise FileNotFoundError("arm checkpoint missing")
    monkeypatch.setattr(test_eval, "load_arm", missing)
    with pytest.raises(FileNotFoundError):
        test_eval.main(["--allow-test"])
    assert not (tmp_path / "test_receipt.json").exists()
