"""Save, load and prove that a reloaded model predicts exactly the same.

"Save-load model to test data" is a graded requirement, and a reload bug stays
silent until the moment you need the saved model — so every run verifies it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_ARRAY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _check_array(name: str, value) -> np.ndarray:
    data = np.asarray(value)
    if not _ARRAY_NAME.fullmatch(name):
        raise ValueError(f"Unsafe array name: {name!r}")
    if data.dtype.kind not in "biufcUS":
        raise ValueError(f"Unsupported dtype for {name!r}: {data.dtype}")
    if data.dtype.kind in "fc" and not np.isfinite(data).all():
        raise ValueError(f"Nonfinite values in {name!r}")
    return data


def save_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    """Write arrays to <path> and metadata to <path>.json, overwriting both.

    Overwrites rather than refusing: during development the same run gets retried
    constantly, and a FileExistsError there costs more than it protects.
    """
    path = Path(path)
    safe = {name: _check_array(name, value) for name, value in arrays.items()}
    if not safe:
        raise ValueError("Refusing to save a model with no arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with path.open("wb") as stream:
        np.savez(stream, **safe)


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return arrays, metadata


def save_reference(path: Path, value: Any) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        joblib.dump(value, stream)


def load_reference(path: Path):
    import joblib

    return joblib.load(Path(path))


def verify_reload(before, after, *, labels: bool = False, rtol: float = 1e-10, atol: float = 1e-12) -> dict:
    before, after = np.asarray(before), np.asarray(after)
    same_shape = before.shape == after.shape
    finite = bool(np.isfinite(before).all() and np.isfinite(after).all()) if before.dtype.kind in "fc" else True
    if labels:
        passed = bool(same_shape and np.array_equal(before, after))
    else:
        passed = bool(same_shape and finite and np.allclose(before, after, rtol=rtol, atol=atol))
    return {
        "passed": passed,
        "labels_exact": labels,
        "rtol": 0.0 if labels else rtol,
        "atol": 0.0 if labels else atol,
        "reason": None if passed else "shape, nonfinite value or reload output mismatch",
    }
