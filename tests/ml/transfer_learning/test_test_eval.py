"""The one-shot evaluator must refuse before it can read the held-out split."""
import json

import pytest

from src.ml.transfer_learning import test_eval


def test_refuses_without_allow_test():
    with pytest.raises(PermissionError):
        test_eval.main([])


def test_refuses_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(test_eval, "ROOT", tmp_path)
    (tmp_path / "test_receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        test_eval.main(["--allow-test"])


def test_a_crash_after_opening_still_blocks_a_second_read(tmp_path, monkeypatch):
    monkeypatch.setattr(test_eval, "ROOT", tmp_path)
    monkeypatch.setattr(test_eval, "verify_decision", lambda: {})

    def boom(**_):
        raise RuntimeError("crash while reading")
    monkeypatch.setattr(test_eval, "load_test", boom)
    with pytest.raises(RuntimeError):
        test_eval.main(["--allow-test"])
    assert json.loads((tmp_path / "test_receipt.json").read_text(encoding="utf-8"))["status"] == "opened"
    with pytest.raises(FileExistsError):
        test_eval.main(["--allow-test"])


def test_saved_labels_are_the_labels_that_were_scored(tmp_path):
    import numpy as np
    import pandas as pd
    from src.ml.transfer_learning.experiment import write_predictions
    dates = np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[D]")
    write_predictions(tmp_path / "p.csv", dates, [0, 1], [0.03, 0.04], labels=[0, 1])
    write_predictions(tmp_path / "q.csv", dates, [0, 1], [0.2, 0.7])
    assert pd.read_csv(tmp_path / "p.csv")["final_prediction"].tolist() == [0, 1]
    assert pd.read_csv(tmp_path / "q.csv")["final_prediction"].tolist() == [0, 1]


def test_refuses_when_code_changed_after_the_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(test_eval, "ROOT", tmp_path)
    (tmp_path / "decision.json").write_text(json.dumps(
        {"hashes": {"code": {"experiment.py": "0" * 64}, "source_manifest": "", "models": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="code changed"):
        test_eval.main(["--allow-test"])
