"""Strict serialization and deterministic provenance contracts."""
import hashlib
import json
from pathlib import Path
import pytest
from src.modeling.artifacts import canonical_bytes, canonical_hash, write_json, file_sha256, code_snapshot_hash, runtime_fingerprint


def test_canonical_json_and_strict_writer(tmp_path):
    assert canonical_bytes({"é": 2, "a": 1}) == '{"a":1,"é":2}'.encode()
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert not path.exists()
    write_json(path, {"value": None, "reason": "one class"})
    assert json.loads(path.read_text()) == {"value": None, "reason": "one class"}
    with pytest.raises(FileExistsError):
        write_json(path, {})
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_order_content_and_scope(tmp_path):
    for name in ["config.py", "configs/modeling.json", "src/models/a.py", "src/modeling/b.py"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    paths = sorted(["config.py", "configs/modeling.json", "src/models/a.py", "src/modeling/b.py"])
    expected = hashlib.sha256()
    for name in paths:
        data = (tmp_path / name).read_bytes()
        expected.update(name.encode() + b"\0" + len(data).to_bytes(8, "big") + data)
    assert code_snapshot_hash(tmp_path) == expected.hexdigest()
    before = code_snapshot_hash(tmp_path)
    (tmp_path / "README.md").write_text("irrelevant")
    assert code_snapshot_hash(tmp_path) == before
    (tmp_path / "src/models/a.py").write_text("changed")
    assert code_snapshot_hash(tmp_path) != before


def test_runtime_is_strict_and_repeatable():
    first = runtime_fingerprint()
    assert first == runtime_fingerprint()
    assert set(first["values"]["versions"]) == {"numpy", "pandas", "scikit-learn", "xgboost", "joblib"}
    assert first["sha256"] == canonical_hash(first["values"])
    assert len(first["values"]["numpy_config_sha256"]) == 64


def test_prediction_csv_schema_and_nooverwrite(tmp_path):
    from src.modeling.artifacts import write_predictions
    import pandas as pd
    path = tmp_path / "validation.csv"
    write_predictions(path, dates=["2024-01-01", "2024-01-02"], target=[.1, .2], raw=[-.1, .3], task="regression")
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["Date", "target_volatility_5d", "raw_prediction", "final_prediction", "clipping_indicator"]
    assert frame["final_prediction"].tolist() == [0., .3]
    with pytest.raises(FileExistsError): write_predictions(path, dates=["2024-01-01"], target=[.1], raw=[.1], task="regression")
    classifier = tmp_path / "classifier.csv"
    write_predictions(classifier, dates=["2024-01-01", "2024-01-02"], target=[0, 1], raw=[-1, 2], task="classification", threshold=0)
    result = pd.read_csv(classifier)
    assert list(result.columns) == ["Date", "target_high_volatility", "raw_score", "final_prediction", "threshold"]
    assert result["final_prediction"].tolist() == [0, 1]


def test_invalid_prediction_csv_creates_no_artifact(tmp_path):
    from src.modeling.artifacts import write_predictions
    path = tmp_path / "bad.csv"
    with pytest.raises(ValueError): write_predictions(path, dates=["2024-01-01"], target=[1], raw=[float("nan")], task="classification", threshold=0)
    assert not path.exists()
