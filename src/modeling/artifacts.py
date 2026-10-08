"""Strict JSON, content hashes and reproducible runtime provenance."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any
import numpy as np
from config import PROJECT_ROOT


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_snapshot_hash(root: Path = PROJECT_ROOT) -> str:
    paths = [root / "config.py", root / "configs/modeling.json"]
    for directory in ("src/modeling", "src/models"):
        paths.extend((root / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def runtime_fingerprint() -> dict[str, Any]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    normalized = "\n".join(line.rstrip() for line in output.getvalue().replace("\r\n", "\n").splitlines()) + "\n"
    values = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")},
              "numpy_config_sha256": hashlib.sha256(normalized.encode()).hexdigest()}
    return {"values": values, "sha256": canonical_hash(values)}


def write_predictions(path: Path, *, dates, target, raw, task: str, threshold: float | None = None, diagnostic_segment=None) -> None:
    """Write explicit supervised outputs without inventing probability columns."""
    import pandas as pd
    from .contracts import REGRESSION_TARGET, CLASSIFICATION_TARGET
    from .metrics import _paired
    y, score = _paired(target, raw, binary=task == "classification")
    dates = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    if len(dates) != len(y) or dates.isna().any() or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Prediction dates must align and be unique/chronological")
    if task == "regression":
        if threshold is not None or (y < 0).any(): raise ValueError("Invalid regression output policy")
        columns = {"Date": dates.strftime("%Y-%m-%d"), REGRESSION_TARGET: y,
                   "raw_prediction": score, "final_prediction": np.maximum(score, 0), "clipping_indicator": score < 0}
    elif task == "classification":
        if threshold is None or not np.isfinite(threshold): raise ValueError("Classification requires a frozen finite threshold")
        columns = {"Date": dates.strftime("%Y-%m-%d"), CLASSIFICATION_TARGET: y.astype(int),
                   "raw_score": score, "final_prediction": (score >= threshold).astype(int), "threshold": threshold}
    else:
        raise ValueError("Unsupported supervised task")
    if diagnostic_segment is not None:
        if len(diagnostic_segment) != len(y) or pd.isna(diagnostic_segment).any(): raise ValueError("Invalid diagnostic segments")
        columns["diagnostic_segment"] = diagnostic_segment
    data = pd.DataFrame(columns).to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
