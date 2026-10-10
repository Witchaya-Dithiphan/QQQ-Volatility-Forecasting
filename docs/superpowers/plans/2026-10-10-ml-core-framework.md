# ML Core Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** สร้าง `src/ml/` ให้เป็นฐานที่นักพัฒนา 2 คนเขียนโมเดล scratch 19 ตัวต่อได้ทันที โดยพิสูจน์ด้วย pilot ที่รันครบลูป train → save → load → re-test → figure

**Architecture:** `src/ml/core/` เป็นชั้นกลางที่โมเดลทุกตัวใช้ร่วมกัน — โหลดข้อมูลตาม variant, standardize/PCA, grid search บน Validation, เทียบ scratch กับ sklearn แบบ strict, เซฟ/โหลดแล้วพิสูจน์ว่าทำนายเท่าเดิม, วาดรูปและสร้าง leaderboard โมเดลแต่ละตัวเป็นไฟล์เดียวที่ทำตาม `BaseModel` และไม่รู้จัก variant ไม่รู้จัก preprocessing

**Tech Stack:** Python 3.14.8 · NumPy 2.5.3 · pandas 3.0.6 · matplotlib 3.11.2 · scikit-learn 1.9.1 (reference เท่านั้น) · xgboost 3.4.1 (reference เท่านั้น) · pytest 9.1.1

**อ่านก่อนเริ่ม:** [`AGENTS.md`](../../../AGENTS.md) · [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) · [`PROJECT_STRUCTURE.md`](../../../PROJECT_STRUCTURE.md)

**คำสั่งรัน test ตลอดแผนนี้ใช้:** `.\.venv\Scripts\python.exe -m pytest ... -v`

---

## File Structure

| ไฟล์ | ความรับผิดชอบ |
| --- | --- |
| `src/ml/core/contracts.py` | ชื่อ feature/target/path ของแต่ละ variant — ค่าคงที่ล้วน |
| `src/ml/core/preprocess.py` | `Standardizer`, `PCA` |
| `src/ml/core/metrics.py` | metric/loss/curve/selection ทุกตัว |
| `src/ml/core/persist.py` | `save_npz`, `load_npz`, `verify_reload` |
| `src/ml/core/data.py` | `load(variant)`, `load_test(allow_test=)` |
| `src/ml/core/base.py` | `BaseModel` abstract interface |
| `src/ml/core/compare.py` | `assert_parity()` |
| `src/ml/core/trainer.py` | grid search + สร้าง artifact ครบชุด |
| `src/ml/core/figures.py` | ROC / confusion / performance curve / leaderboard |
| `src/ml/registry.py` | ชื่อโมเดล → คลาส |
| `src/ml/run.py` | CLI `train` / `report` / `finalize` |
| `src/ml/regression/multiple.py` | โมเดล pilot |

---

## Task 1: สร้างโครง `src/ml/` และย้าย preprocess + metrics

**Files:**

- Create: `src/ml/__init__.py`, `src/ml/core/__init__.py`, `src/ml/regression/__init__.py`, `src/ml/classification/__init__.py`, `src/ml/clustering/__init__.py`
- Create: `src/ml/core/preprocess.py`, `src/ml/core/metrics.py`, `src/ml/core/contracts.py`
- Create: `tests/ml/__init__.py`, `tests/ml/core/__init__.py`
- Create: `tests/ml/core/test_preprocess.py`, `tests/ml/core/test_metrics.py`

- [ ] **Step 1: สร้างโฟลเดอร์และ `__init__.py` ทั้งหมด**

```bash
mkdir -p src/ml/core src/ml/regression src/ml/classification src/ml/clustering
mkdir -p tests/ml/core tests/ml/regression tests/ml/classification tests/ml/clustering
for d in src/ml src/ml/core src/ml/regression src/ml/classification src/ml/clustering \
         tests/ml tests/ml/core tests/ml/regression tests/ml/classification tests/ml/clustering; do
  touch "$d/__init__.py"
done
```

- [ ] **Step 2: คัดลอกไฟล์คณิตศาสตร์มาตรง ๆ (ห้ามแก้ตรรกะ)**

```bash
cp src/modeling/preprocessing.py src/ml/core/preprocess.py
cp src/modeling/metrics.py       src/ml/core/metrics.py
cp tests/modeling/test_preprocessing.py tests/ml/core/test_preprocess.py
cp tests/modeling/test_metrics.py       tests/ml/core/test_metrics.py
```

- [ ] **Step 3: แก้ import ในไฟล์ test ที่คัดลอกมา**

เปลี่ยนทุกบรรทัดที่ขึ้นต้นด้วย `from src.modeling.preprocessing import` เป็น `from src.ml.core.preprocess import`
และ `from src.modeling.metrics import` เป็น `from src.ml.core.metrics import`

```bash
sed -i 's/from src\.modeling\.preprocessing import/from src.ml.core.preprocess import/' tests/ml/core/test_preprocess.py
sed -i 's/from src\.modeling\.metrics import/from src.ml.core.metrics import/'           tests/ml/core/test_metrics.py
grep -n "src\.modeling" tests/ml/core/*.py
```

Expected: ไม่มี output (แปลว่าไม่เหลือ import เก่า) ถ้ายังมี ให้แก้มือให้หมด

- [ ] **Step 4: เขียน `src/ml/core/contracts.py`**

```python
"""Accepted feature/target identities and dataset paths."""
from config import (
    NON_SPIKE_TRAIN_PATH,
    TEST_LABELED_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
    WITH_SPIKES_TRAIN_PATH,
)

FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "historical_volatility_5d",
    "historical_volatility_20d",
    "intraday_range",
    "sma_ratio_5_20",
    "rsi_14",
    "volume_zscore_20",
)
REGRESSION_TARGET = "target_volatility_5d"
CLASSIFICATION_TARGET = "target_high_volatility"
DATE_COLUMN = "Date"
VARIANTS = ("with_spike", "non_spike")
VARIANT_TRAIN_PATHS = {
    "with_spike": WITH_SPIKES_TRAIN_PATH,
    "non_spike": NON_SPIKE_TRAIN_PATH,
}
VALIDATION_PATH = VALIDATION_LABELED_DATA_PATH
TEST_PATH = TEST_LABELED_DATA_PATH
```

- [ ] **Step 5: รัน test ที่ย้ายมา**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core -v`
Expected: PASS ทั้งหมด (เดิมมี 9 tests ใน preprocess และ 11 tests ใน metrics)

- [ ] **Step 6: Commit**

```bash
git add src/ml tests/ml
git commit -m "feat: carry tested numerical code into the new src/ml home

preprocess.py and metrics.py are pure NumPy with no governance coupling, so they
move verbatim rather than being rewritten — their existing tests move with them
and must stay green to prove the move changed nothing."
```

---

## Task 2: `persist.py` — save / load / verify

**Files:**

- Create: `src/ml/core/persist.py`
- Create: `tests/ml/core/test_persist.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_persist.py
import numpy as np
import pytest
from src.ml.core.persist import load_npz, save_npz, verify_reload


def test_save_load_round_trip_preserves_arrays_and_metadata(tmp_path):
    arrays = {"weights": np.array([1.5, -2.25, 0.0]), "intercept": np.array([0.75])}
    metadata = {"model": "multiple", "seed": 42}
    path = tmp_path / "model.npz"

    save_npz(path, arrays, metadata)
    loaded_arrays, loaded_metadata = load_npz(path)

    assert loaded_metadata == metadata
    assert set(loaded_arrays) == set(arrays)
    for name, value in arrays.items():
        assert np.array_equal(loaded_arrays[name], value)


def test_save_npz_overwrites_so_reruns_do_not_crash(tmp_path):
    path = tmp_path / "model.npz"
    save_npz(path, {"a": np.array([1.0])}, {"run": 1})
    save_npz(path, {"a": np.array([2.0])}, {"run": 2})

    arrays, metadata = load_npz(path)
    assert arrays["a"] == np.array([2.0])
    assert metadata == {"run": 2}


def test_save_npz_rejects_nonfinite_arrays(tmp_path):
    with pytest.raises(ValueError):
        save_npz(tmp_path / "bad.npz", {"a": np.array([np.inf])}, {})


def test_verify_reload_passes_on_identical_predictions():
    before = np.array([0.1, 0.2, 0.3])
    assert verify_reload(before, before.copy())["passed"] is True


def test_verify_reload_fails_when_predictions_differ():
    result = verify_reload(np.array([0.1, 0.2]), np.array([0.1, 0.9]))
    assert result["passed"] is False
    assert result["reason"]
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_persist.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.core.persist'`

- [ ] **Step 3: เขียน `src/ml/core/persist.py`**

```python
"""Save, load and prove that a reloaded model predicts exactly the same."""
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
        raise ValueError(f"Unsafe array name: {name}")
    if data.dtype.kind not in "biufcUS":
        raise ValueError(f"Unsupported dtype for {name}: {data.dtype}")
    if data.dtype.kind in "fc" and not np.isfinite(data).all():
        raise ValueError(f"Nonfinite values in {name}")
    return data


def save_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    """Write arrays to <path> and metadata to <path>.json, overwriting both."""
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
    """Proof for the assignment's save-load-and-test-again requirement."""
    before, after = np.asarray(before), np.asarray(after)
    same_shape = before.shape == after.shape
    finite = np.isfinite(before).all() and np.isfinite(after).all()
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
```

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_persist.py -v`
Expected: PASS 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/ml/core/persist.py tests/ml/core/test_persist.py
git commit -m "feat: add model persistence with reload verification

Overwrites on save instead of refusing like the old implementation did — during
development the same run gets retried constantly, and a FileExistsError there
cost more than the accidental-overwrite protection was worth."
```

---

## Task 3: `data.py` — loader ตาม variant

**Files:**

- Create: `src/ml/core/data.py`
- Create: `tests/ml/core/test_data.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_data.py
import numpy as np
import pytest
from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.core.data import load, load_test


def test_with_spike_variant_matches_accepted_row_counts():
    data = load("with_spike")
    assert data.variant == "with_spike"
    assert data.train.X.shape == (1733, 8)
    assert data.validation.X.shape == (371, 8)
    assert data.train.X.flags.writeable is False


def test_non_spike_variant_has_the_accepted_reduced_train():
    data = load("non_spike")
    assert data.train.X.shape == (1520, 8)
    assert data.validation.X.shape == (371, 8)


def test_both_variants_share_the_same_validation_set():
    assert np.array_equal(load("with_spike").validation.X, load("non_spike").validation.X)


def test_train_ends_before_validation_begins():
    data = load("with_spike")
    assert data.train.dates[-1] < data.validation.dates[0]


def test_feature_order_is_the_accepted_order():
    assert load("with_spike").feature_names == list(FEATURE_COLUMNS)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        load("whatever")


def test_test_set_is_denied_by_default():
    with pytest.raises(PermissionError):
        load_test()


def test_test_set_loads_only_with_explicit_opt_in():
    split = load_test(allow_test=True)
    assert split.X.shape == (373, 8)
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_data.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.core.data'`

- [ ] **Step 3: เขียน `src/ml/core/data.py`**

```python
"""Read-only accepted datasets, with the Test split denied unless asked for."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXPERIMENT_DATASET_REPORT_PATH, SPIKE_INPUT_CONTRACT_REPORT_PATH

from .contracts import (
    CLASSIFICATION_TARGET,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    REGRESSION_TARGET,
    TEST_PATH,
    VALIDATION_PATH,
    VARIANT_TRAIN_PATHS,
)


@dataclass(frozen=True)
class Split:
    name: str
    dates: np.ndarray
    X: np.ndarray
    y_regression: np.ndarray
    y_classification: np.ndarray
    sha256: str


@dataclass(frozen=True)
class Dataset:
    variant: str
    train: Split
    validation: Split
    threshold: float
    feature_names: list[str]


def _readonly(values, dtype=None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _accepted_threshold() -> float:
    contract = json.loads(SPIKE_INPUT_CONTRACT_REPORT_PATH.read_text(encoding="utf-8"))
    return float(contract["classification_contract"]["threshold"])


def _expected_sha256() -> dict[str, str]:
    report = json.loads(EXPERIMENT_DATASET_REPORT_PATH.read_text(encoding="utf-8"))
    return {name: entry["sha256"].lower() for name, entry in report["artifacts"].items()}


def _read_split(path: Path, name: str, *, threshold: float, expected_sha256: str | None) -> Split:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Accepted input changed on disk: {path}")

    frame = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in (DATE_COLUMN, *FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET) if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing accepted columns in {path}: {missing}")
    if frame.isna().any().any():
        raise ValueError(f"Null values in {path}")

    dates = pd.to_datetime(frame[DATE_COLUMN], format="%Y-%m-%d", errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"Dates must be unique and chronological in {path}")

    targets = frame[REGRESSION_TARGET].to_numpy(dtype=np.float64)
    labels = frame[CLASSIFICATION_TARGET].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Classification labels must be binary in {path}")
    if not np.array_equal(labels, (targets > threshold).astype(np.int64)):
        raise ValueError(f"Labels disagree with the accepted Q75 threshold in {path}")

    return Split(
        name=name,
        dates=_readonly(dates.to_numpy()),
        X=_readonly(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)),
        y_regression=_readonly(targets),
        y_classification=_readonly(labels),
        sha256=digest,
    )


def load(variant: str = "with_spike") -> Dataset:
    if variant not in VARIANT_TRAIN_PATHS:
        raise ValueError(f"Unknown dataset variant: {variant!r}; expected one of {sorted(VARIANT_TRAIN_PATHS)}")
    threshold = _accepted_threshold()
    digests = _expected_sha256()
    train_key = "with_spikes_train" if variant == "with_spike" else "non_spike_train"

    train = _read_split(VARIANT_TRAIN_PATHS[variant], "train", threshold=threshold, expected_sha256=digests.get(train_key))
    validation = _read_split(VALIDATION_PATH, "validation", threshold=threshold, expected_sha256=digests.get("validation_labeled"))
    if train.dates[-1] >= validation.dates[0]:
        raise ValueError("Train and Validation overlap in time")

    return Dataset(variant, train, validation, threshold, list(FEATURE_COLUMNS))


def load_test(*, allow_test: bool = False) -> Split:
    """Default-deny so the Test split cannot leak into model selection."""
    if allow_test is not True:
        raise PermissionError(
            "Test split is only available during finalize; pass allow_test=True from src.ml.run finalize"
        )
    digests = _expected_sha256()
    return _read_split(TEST_PATH, "test", threshold=_accepted_threshold(), expected_sha256=digests.get("test_labeled"))
```

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_data.py -v`
Expected: PASS 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/ml/core/data.py src/ml/core/contracts.py tests/ml/core/test_data.py
git commit -m "feat: load accepted datasets by variant with a default-deny Test split

Keeps the SHA-256 and label-threshold checks that catch silent data corruption,
but drops the cross-report identity chain: it re-read the full original train set
on every non_spike load to prove subsetting, which belongs in a test, not in the
hot path of 38 training runs."
```

---

## Task 4: `base.py` — `BaseModel` interface

**Files:**

- Create: `src/ml/core/base.py`
- Create: `tests/ml/core/test_base.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_base.py
import numpy as np
import pytest
from src.ml.core.base import BaseModel


class Dummy(BaseModel):
    name = "dummy"
    task = "regression"

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def get_params(self):
        return {"scale": self.scale}

    def fit(self, X, y=None):
        self.mean_ = float(np.asarray(y).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_ * self.scale)

    def state_dict(self):
        return {"mean": np.array([self.mean_])}

    def load_state(self, arrays, metadata):
        self.mean_ = float(arrays["mean"][0])
        self.scale = metadata["params"]["scale"]
        return self


def test_subclass_can_fit_and_predict():
    model = Dummy().fit(np.zeros((3, 2)), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(model.predict(np.zeros((3, 2))), 4.0)


def test_state_round_trip_restores_predictions():
    X, y = np.zeros((3, 2)), np.array([1.0, 2.0, 3.0])
    original = Dummy(scale=3.0).fit(X, y)
    restored = Dummy().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.allclose(original.predict(X), restored.predict(X))


def test_instantiating_the_interface_directly_is_rejected():
    with pytest.raises(TypeError):
        BaseModel()


def test_classification_helpers_raise_when_not_implemented():
    with pytest.raises(NotImplementedError):
        Dummy().decision_function(np.zeros((1, 2)))
    with pytest.raises(NotImplementedError):
        Dummy().predict_proba(np.zeros((1, 2)))


def test_reference_is_optional_and_reports_absence():
    with pytest.raises(NotImplementedError):
        Dummy().reference()
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_base.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.core.base'`

- [ ] **Step 3: เขียน `src/ml/core/base.py`**

```python
"""The contract every scratch model implements. See AGENTS.md section 4."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

TASKS = ("regression", "classification", "clustering")


class BaseModel(ABC):
    name: str = ""
    task: str = ""

    @abstractmethod
    def get_params(self) -> dict:
        """Every hyperparameter, JSON-serializable, enough to rebuild this model."""

    @abstractmethod
    def fit(self, X, y=None) -> "BaseModel":
        """Clustering models receive y=None."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """regression: values | classification: 0/1 labels | clustering: cluster ids."""

    @abstractmethod
    def state_dict(self) -> dict[str, np.ndarray]:
        """Only learned parameters, as NumPy arrays."""

    @abstractmethod
    def load_state(self, arrays: dict, metadata: dict) -> "BaseModel":
        """Rebuild from state_dict() output plus {"params": get_params()}."""

    def decision_function(self, X) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not produce decision scores")

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not produce calibrated probabilities")

    def reference(self):
        raise NotImplementedError(f"{type(self).__name__} has no reference implementation yet")
```

**ข้อบังคับเพิ่มเติม:** hyperparameter ทุกตัวใน `__init__` **ต้องมีค่า default** เพราะ trainer
สร้างอินสแตนซ์เปล่าด้วย `model_class()` แล้วเรียก `load_state()` ตอนพิสูจน์ว่า reload แล้วทำนายเท่าเดิม
ถ้า `__init__` บังคับให้ส่งอาร์กิวเมนต์ ขั้นตอนนั้นจะพังทุกโมเดล

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_base.py -v`
Expected: PASS 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/ml/core/base.py tests/ml/core/test_base.py
git commit -m "feat: define the BaseModel contract both developers code against

Abstract rather than duck-typed because the two model families are written in
parallel by different agents on different branches — a missing method has to fail
at import time, not after a six-hour training run."
```

---

## Task 5: `compare.py` — strict parity harness

**Files:**

- Create: `src/ml/core/compare.py`
- Create: `tests/ml/core/test_compare.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_compare.py
import numpy as np
import pytest
from src.ml.core.compare import ParityError, align_clusters, assert_parity


class _Const:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def predict(self, X):
        return self.values


def test_identical_regression_predictions_pass():
    assert_parity(_Const([1.0, 2.0]), _Const([1.0, 2.0]), np.zeros((2, 1)), None, task="regression")


def test_regression_difference_beyond_tolerance_fails():
    with pytest.raises(ParityError) as error:
        assert_parity(_Const([1.0, 2.0]), _Const([1.0, 2.5]), np.zeros((2, 1)), None, task="regression")
    assert "max abs diff" in str(error.value)


def test_regression_difference_within_tolerance_passes():
    assert_parity(_Const([1.0, 2.0]), _Const([1.0, 2.0 + 1e-9]), np.zeros((2, 1)), None, task="regression")


def test_classification_requires_every_label_to_match():
    with pytest.raises(ParityError):
        assert_parity(_Const([0, 1, 1]), _Const([0, 1, 0]), np.zeros((3, 1)), None, task="classification")


def test_cluster_labels_are_aligned_before_comparison():
    assert np.array_equal(align_clusters(np.array([0, 0, 1]), np.array([1, 1, 0])), np.array([0, 0, 1]))


def test_clustering_parity_ignores_cluster_renaming():
    assert_parity(_Const([0, 0, 1]), _Const([1, 1, 0]), np.zeros((3, 1)), None, task="clustering")
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_compare.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.core.compare'`

- [ ] **Step 3: เขียน `src/ml/core/compare.py`**

```python
"""Strict scratch-vs-reference comparison. Policy lives in AGENTS.md section 6."""
from __future__ import annotations

import numpy as np

RTOL = 1e-5
ATOL = 1e-8


class ParityError(AssertionError):
    """Raised when a scratch model disagrees with its reference implementation."""


def _report(label: str, scratch: np.ndarray, reference: np.ndarray) -> str:
    difference = np.abs(scratch - reference)
    worst = int(np.argmax(difference))
    return (
        f"{label}: max abs diff {difference.max():.3e} at index {worst} "
        f"(scratch={scratch[worst]!r}, reference={reference[worst]!r}, "
        f"mismatching elements {int((difference > ATOL).sum())}/{len(difference)})"
    )


def align_clusters(scratch_labels, reference_labels) -> np.ndarray:
    """Rename reference clusters to the scratch naming by majority overlap."""
    scratch_labels = np.asarray(scratch_labels)
    reference_labels = np.asarray(reference_labels)
    mapping = {}
    for cluster in np.unique(reference_labels):
        members = reference_labels == cluster
        values, counts = np.unique(scratch_labels[members], return_counts=True)
        mapping[cluster] = values[np.argmax(counts)]
    return np.array([mapping[c] for c in reference_labels])


def assert_parity(scratch, reference, X, y=None, *, task: str, label: str = "predict") -> dict:
    """Raise ParityError unless the scratch model matches its reference."""
    scratch_prediction = np.asarray(scratch.predict(X), dtype=np.float64)
    reference_prediction = np.asarray(reference.predict(X), dtype=np.float64)

    if scratch_prediction.shape != reference_prediction.shape:
        raise ParityError(f"{label}: shape {scratch_prediction.shape} != {reference_prediction.shape}")

    if task == "regression":
        if not np.allclose(scratch_prediction, reference_prediction, rtol=RTOL, atol=ATOL):
            raise ParityError(_report(label, scratch_prediction, reference_prediction))

    elif task == "classification":
        if not np.array_equal(scratch_prediction, reference_prediction):
            disagreements = int((scratch_prediction != reference_prediction).sum())
            raise ParityError(f"{label}: {disagreements}/{len(scratch_prediction)} labels differ")
        for method in ("decision_function", "predict_proba"):
            try:
                mine = np.asarray(getattr(scratch, method)(X), dtype=np.float64)
            except (NotImplementedError, AttributeError):
                continue
            try:
                theirs = np.asarray(getattr(reference, method)(X), dtype=np.float64)
            except (NotImplementedError, AttributeError):
                continue
            if theirs.ndim == 2 and theirs.shape[1] == 2:
                theirs = theirs[:, 1]
            if mine.shape == theirs.shape and not np.allclose(mine, theirs, rtol=RTOL, atol=ATOL):
                raise ParityError(_report(f"{label}.{method}", mine, theirs))

    elif task == "clustering":
        aligned = align_clusters(scratch_prediction, reference_prediction)
        if not np.array_equal(scratch_prediction, aligned):
            raise ParityError(f"{label}: cluster partitions differ after aligning cluster ids")

    else:
        raise ValueError(f"Unknown task: {task!r}")

    return {"task": task, "n_samples": int(len(scratch_prediction)), "passed": True, "rtol": RTOL, "atol": ATOL}
```

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_compare.py -v`
Expected: PASS 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/ml/core/compare.py tests/ml/core/test_compare.py
git commit -m "feat: add the strict parity gate that proves each scratch model is correct

The failure message names the worst index and both values because the whole point
is diagnosing the mismatch, and 'arrays are not almost equal' sends you back to
the debugger with nothing to go on."
```

---

## Task 6: `registry.py` — ทะเบียนโมเดล

**Files:**

- Create: `src/ml/registry.py`
- Create: `tests/ml/test_registry.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/test_registry.py
import pytest
from src.ml.core.base import BaseModel
from src.ml.registry import MODELS, build, model_names


def test_every_registered_entry_is_a_base_model_subclass():
    for name, factory in MODELS.items():
        assert issubclass(factory, BaseModel), name


def test_registered_name_matches_the_class_attribute():
    for name, factory in MODELS.items():
        assert factory.name == name


def test_build_returns_a_fresh_instance():
    name = next(iter(MODELS))
    assert build(name) is not build(name)


def test_unknown_model_name_lists_what_is_available():
    with pytest.raises(KeyError) as error:
        build("does_not_exist")
    assert "available" in str(error.value)


def test_model_names_are_sorted_for_stable_merge_conflicts():
    assert model_names() == sorted(model_names())
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_registry.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.registry'`

- [ ] **Step 3: เขียน `src/ml/registry.py`**

```python
"""Model name -> class. Add exactly one alphabetically sorted line per model.

Both developers edit this file, so keep it mechanical: one import, one dict entry,
sorted by name. That keeps merge conflicts to a single obvious line.
"""
from __future__ import annotations

from .core.base import BaseModel
from .regression.multiple import MultipleRegression

MODELS: dict[str, type[BaseModel]] = {
    "multiple": MultipleRegression,
}


def model_names() -> list[str]:
    return sorted(MODELS)


def build(name: str, **hyperparams) -> BaseModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; available: {model_names()}")
    return MODELS[name](**hyperparams)


def models_for_task(task: str) -> list[str]:
    return sorted(name for name, factory in MODELS.items() if factory.task == task)
```

- [ ] **Step 4: รัน test — จะยัง fail เพราะโมเดล pilot ยังไม่มี**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_registry.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.regression.multiple'` — Task 7 จะทำให้ผ่าน

---

## Task 7: โมเดล pilot — Multiple Linear Regression

**Files:**

- Create: `src/ml/regression/multiple.py`
- Create: `tests/ml/regression/test_multiple.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/regression/test_multiple.py
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.ml.core.compare import assert_parity
from src.ml.regression.multiple import MultipleRegression


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    y = 2.5 + 1.5 * X[:, 0] - 0.75 * X[:, 1] + 0.25 * X[:, 3] + rng.normal(scale=0.05, size=120)
    return X, y


def test_recovers_known_coefficients(data):
    X, y = data
    model = MultipleRegression().fit(X, y)
    assert np.allclose(model.coef_, [1.5, -0.75, 0.0, 0.25], atol=0.05)
    assert model.intercept_ == pytest.approx(2.5, abs=0.05)


def test_normal_equation_solves_an_exactly_determined_system():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([2.0, 3.0, 5.0])
    model = MultipleRegression(fit_intercept=False).fit(X, y)
    assert np.allclose(model.coef_, [2.0, 3.0])


def test_matches_sklearn_exactly(data):
    X, y = data
    scratch = MultipleRegression().fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert_parity(scratch, reference, X, task="regression")


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = MultipleRegression().fit(X, y)
    restored = MultipleRegression().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))


def test_reference_is_sklearn_linear_regression():
    assert isinstance(MultipleRegression().reference(), LinearRegression)
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/regression/test_multiple.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.regression.multiple'`

- [ ] **Step 3: เขียน `src/ml/regression/multiple.py`**

```python
"""Multiple linear regression by least squares."""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class MultipleRegression(BaseModel):
    """Minimises ||y - Xb||^2.

    Solved with lstsq rather than inverting X'X: the features are collinear
    (condition number 35.6 on this train set) and the normal equations square
    that conditioning, while lstsq's SVD handles rank deficiency gracefully —
    which is also exactly what sklearn does, so parity comes for free.
    """

    name = "multiple"
    task = "regression"

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def get_params(self) -> dict:
        return {"fit_intercept": self.fit_intercept}

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y=None) -> "MultipleRegression":
        y = np.asarray(y, dtype=np.float64)
        solution, *_ = np.linalg.lstsq(self._design(X), y, rcond=None)
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(solution[0]), solution[1:]
        else:
            self.intercept_, self.coef_ = 0.0, solution
        self.history_ = {"iteration": [0], "loss": [float(np.mean((y - self.predict(X)) ** 2))]}
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"coef": np.asarray(self.coef_), "intercept": np.array([self.intercept_])}

    def load_state(self, arrays: dict, metadata: dict) -> "MultipleRegression":
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        self.fit_intercept = metadata["params"]["fit_intercept"]
        return self

    def reference(self):
        from sklearn.linear_model import LinearRegression

        return LinearRegression(fit_intercept=self.fit_intercept)
```

- [ ] **Step 4: รัน test ของโมเดลและของ registry ให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/regression/test_multiple.py tests/ml/test_registry.py -v`
Expected: PASS ทั้ง 10 tests

- [ ] **Step 5: Commit**

```bash
git add src/ml/registry.py src/ml/regression/multiple.py tests/ml/test_registry.py tests/ml/regression/test_multiple.py
git commit -m "feat: add the pilot model and the registry both branches extend

Multiple regression goes first because it is the simplest model that still
exercises the whole pipeline, so when the trainer lands we can tell framework
bugs from model bugs."
```

---

## Task 8: `trainer.py` — grid search + artifacts

**Files:**

- Create: `src/ml/core/trainer.py`
- Create: `tests/ml/core/test_trainer.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_trainer.py
import json

import numpy as np
import pytest

from src.ml.core.trainer import expand_grid, train_model
from src.ml.regression.multiple import MultipleRegression


def test_expand_grid_produces_every_combination_in_a_stable_order():
    combinations = expand_grid({"a": [1, 2], "b": ["x", "y"]})
    assert combinations == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_empty_grid_yields_one_default_candidate():
    assert expand_grid({}) == [{}]


def test_train_model_writes_every_required_artifact(tmp_path):
    result = train_model(
        MultipleRegression,
        grid={"fit_intercept": [True, False]},
        variant="with_spike",
        output_root=tmp_path,
    )

    directory = tmp_path / "with_spike" / "regression" / "multiple"
    for filename in (
        "config.json",
        "model.npz",
        "preprocessor.npz",
        "search_results.json",
        "validation_metrics.json",
        "validation_predictions.csv",
        "load_verification.json",
    ):
        assert (directory / filename).is_file(), filename

    assert result["load_verification"]["passed"] is True
    assert len(json.loads((directory / "search_results.json").read_text(encoding="utf-8"))) == 2


def test_selected_candidate_is_the_best_on_validation(tmp_path):
    result = train_model(
        MultipleRegression,
        grid={"fit_intercept": [True, False]},
        variant="with_spike",
        output_root=tmp_path,
    )
    rmse_values = [c["metrics"]["clipped_metrics"]["rmse"]["value"] for c in result["search_results"]]
    assert result["validation_metrics"]["clipped_metrics"]["rmse"]["value"] == pytest.approx(min(rmse_values))


def test_trainer_source_never_references_the_test_split():
    """A monkeypatch cannot catch `from .data import load_test`, so check the source."""
    import inspect

    import src.ml.core.trainer as trainer_module

    source = inspect.getsource(trainer_module)
    assert "load_test" not in source
    assert "TEST_PATH" not in source
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_trainer.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.core.trainer'`

- [ ] **Step 3: เขียน `src/ml/core/trainer.py`**

```python
"""Grid search on Validation, then save, reload and prove the model survived."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT

from . import metrics as metrics_module
from .data import load
from .persist import load_npz, save_npz, verify_reload
from .preprocess import PCA, Standardizer

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "modeling"


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product in sorted key order so runs are reproducible."""
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _write_json(path: Path, payload) -> None:
    """No `default=` on purpose: an unserializable value must fail, not be stringified.

    regression_outputs() returns NumPy arrays alongside its metrics, and silently
    writing "[0.11 0.42 ...]" into validation_metrics.json would poison the report
    that every number in the paper is traced back to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _evaluate(model, X, y_regression, y_classification, task: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (json-safe metrics, final prediction, score used for metrics)."""
    if task == "regression":
        outputs = metrics_module.regression_outputs(y_regression, model.predict(X))
        metrics = {
            "raw_metrics": outputs["raw_metrics"],
            "clipped_metrics": outputs["clipped_metrics"],
            "negative_count": outputs["negative_count"],
        }
        return metrics, outputs["final_prediction"], outputs["raw_prediction"]

    if task == "classification":
        try:
            scores = np.asarray(model.predict_proba(X), dtype=np.float64)
            kind = "probability"
        except NotImplementedError:
            scores = np.asarray(model.decision_function(X), dtype=np.float64)
            kind = "decision"
        if scores.ndim == 2 and scores.shape[1] == 2:
            scores = scores[:, 1]
        threshold = 0.5 if kind == "probability" else 0.0
        metrics = metrics_module.classification_metrics(y_classification, scores, threshold=threshold)
        return {**metrics, "score_kind": kind}, np.asarray(model.predict(X)), scores

    labels = np.asarray(model.predict(X))
    return metrics_module.clustering_metrics(X, labels), labels, labels


def train_model(model_class, *, grid: dict, variant: str, output_root: Path | None = None,
                pca_components: int | None = None, seed: int = 42) -> dict:
    data = load(variant)
    task = model_class.task
    directory = Path(output_root or OUTPUT_ROOT) / variant / task / model_class.name
    directory.mkdir(parents=True, exist_ok=True)

    standardizer = Standardizer().fit(data.train.X)
    train_X, validation_X = standardizer.transform(data.train.X), standardizer.transform(data.validation.X)
    preprocessor_arrays = {"mean": standardizer.mean_, "scale": standardizer.scale_}
    pca = None
    if pca_components:
        pca = PCA(pca_components).fit(train_X)
        train_X, validation_X = pca.transform(train_X), pca.transform(validation_X)
        preprocessor_arrays["components"] = pca.components_

    y_train = data.train.y_regression if task == "regression" else data.train.y_classification
    search_results = []
    for index, params in enumerate(expand_grid(grid)):
        candidate = model_class(**params).fit(train_X, None if task == "clustering" else y_train)
        candidate_metrics, prediction, scores = _evaluate(
            candidate, validation_X, data.validation.y_regression, data.validation.y_classification, task
        )
        search_results.append({
            "id": f"{model_class.name}-{index:02d}",
            "params": params,
            "metrics": candidate_metrics,
            "model": candidate,
            "prediction": prediction,
            "scores": scores,
        })

    best = metrics_module.select_candidate(
        [{"id": c["id"], "metrics": c["metrics"]} for c in search_results], task=task
    )
    winner = next(c for c in search_results if c["id"] == best["id"])
    model = winner["model"]

    save_npz(directory / "model.npz", model.state_dict(), {"params": model.get_params(), "seed": seed})
    save_npz(directory / "preprocessor.npz", preprocessor_arrays,
             {"pca_components": pca_components, "feature_names": data.feature_names})

    arrays, metadata = load_npz(directory / "model.npz")
    reloaded = model_class().load_state(arrays, metadata)
    labels = task != "regression"
    verification = verify_reload(model.predict(validation_X), reloaded.predict(validation_X), labels=labels)

    _write_json(directory / "config.json",
                {"model": model_class.name, "task": task, "variant": variant, "seed": seed,
                 "params": model.get_params(), "pca_components": pca_components,
                 "train_sha256": data.train.sha256, "validation_sha256": data.validation.sha256})
    _write_json(directory / "search_results.json",
                [{k: c[k] for k in ("id", "params", "metrics")} for c in search_results])
    _write_json(directory / "validation_metrics.json", winner["metrics"])
    _write_json(directory / "load_verification.json", verification)
    if getattr(model, "history_", None):
        _write_json(directory / "history.json", model.history_)

    with (directory / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Date", "target", "raw_score", "final_prediction"])
        target = data.validation.y_regression if task == "regression" else data.validation.y_classification
        for date, actual, score, prediction in zip(
            data.validation.dates, target, winner["scores"], winner["prediction"]
        ):
            writer.writerow([str(date)[:10], actual, score, prediction])

    return {"directory": directory, "model": model, "search_results": search_results,
            "validation_metrics": winner["metrics"], "load_verification": verification}
```

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_trainer.py -v`
Expected: PASS 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/ml/core/trainer.py tests/ml/core/test_trainer.py
git commit -m "feat: add the training loop that produces every required artifact

Reloads the model it just saved and compares predictions on every run, because
'save and load the model, then test again' is graded and a reload bug is silent
until the moment you need the saved model."
```

---

## Task 9: `figures.py` — รูปและ leaderboard

**Files:**

- Create: `src/ml/core/figures.py`
- Create: `tests/ml/core/test_figures.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/core/test_figures.py
import csv

import numpy as np

from src.ml.core.figures import (
    plot_confusion_matrix,
    plot_performance_curve,
    plot_roc_curves,
    write_leaderboard,
)


def test_roc_figure_is_written(tmp_path):
    path = tmp_path / "roc.png"
    plot_roc_curves({"model_a": ([0.0, 0.5, 1.0], [0.0, 0.8, 1.0], 0.85)}, path, title="ROC")
    assert path.is_file() and path.stat().st_size > 0


def test_confusion_matrix_figure_is_written(tmp_path):
    path = tmp_path / "cm.png"
    plot_confusion_matrix([[300, 20], [15, 90]], path, title="Confusion")
    assert path.is_file() and path.stat().st_size > 0


def test_performance_curve_figure_is_written(tmp_path):
    path = tmp_path / "curve.png"
    plot_performance_curve({"iteration": [0, 1, 2], "loss": [1.0, 0.5, 0.25]}, path, title="Loss")
    assert path.is_file() and path.stat().st_size > 0


def test_leaderboard_rows_are_sorted_by_the_primary_metric(tmp_path):
    path = tmp_path / "leaderboard.csv"
    write_leaderboard(
        [
            {"model": "b", "variant": "with_spike", "task": "regression", "rmse": 0.5},
            {"model": "a", "variant": "with_spike", "task": "regression", "rmse": 0.2},
        ],
        path,
        sort_by="rmse",
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [row["model"] for row in rows] == ["a", "b"]
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_figures.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.core.figures'`

- [ ] **Step 3: เขียน `src/ml/core/figures.py`**

```python
"""Every figure the assignment grades: ROC, confusion matrix, performance curve."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ไม่มี display บนเครื่องที่รัน test
import matplotlib.pyplot as plt  # noqa: E402


def _save(figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curves(curves: dict[str, tuple], path: Path, *, title: str = "ROC curve") -> None:
    """curves: {model_name: (fpr, tpr, auc)}"""
    figure, axes = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in sorted(curves.items()):
        axes.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={auc:.3f})")
    axes.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(title)
    axes.legend(fontsize="small", loc="lower right")
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_confusion_matrix(matrix, path: Path, *, title: str = "Confusion matrix",
                          labels: tuple[str, str] = ("negative", "positive")) -> None:
    figure, axes = plt.subplots(figsize=(5, 4.5))
    image = axes.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            axes.text(j, i, f"{value:,}", ha="center", va="center",
                      color="white" if value > max(max(r) for r in matrix) / 2 else "black")
    axes.set_xticks([0, 1], [f"predicted {labels[0]}", f"predicted {labels[1]}"])
    axes.set_yticks([0, 1], [f"actual {labels[0]}", f"actual {labels[1]}"])
    axes.set_title(title)
    figure.colorbar(image, ax=axes)
    _save(figure, path)


def plot_performance_curve(history: dict, path: Path, *, title: str = "Training loss") -> None:
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(history["iteration"], history["loss"], linewidth=1.5)
    axes.set_xlabel("iteration")
    axes.set_ylabel("loss")
    axes.set_title(title)
    axes.grid(alpha=0.3)
    _save(figure, path)


def write_leaderboard(rows: list[dict], path: Path, *, sort_by: str, descending: bool = False) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty leaderboard")
    ordered = sorted(rows, key=lambda row: (row.get(sort_by) is None, row.get(sort_by)), reverse=descending)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
```

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/core/test_figures.py -v`
Expected: PASS 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/ml/core/figures.py tests/ml/core/test_figures.py
git commit -m "feat: add the figure and leaderboard writers the rubric asks for

Forces the Agg backend so figures render identically in tests, in notebooks and
on a machine with no display."
```

---

## Task 10: `run.py` — CLI

**Files:**

- Create: `src/ml/run.py`
- Create: `tests/ml/test_run.py`

- [ ] **Step 1: เขียน test ที่ fail ก่อน**

```python
# tests/ml/test_run.py
import pytest
from src.ml.run import main


def test_train_a_single_model_succeeds(tmp_path):
    assert main(["train", "--model", "multiple", "--variant", "with_spike",
                 "--output-root", str(tmp_path)]) == 0
    assert (tmp_path / "with_spike" / "regression" / "multiple" / "model.npz").is_file()


def test_unknown_model_name_fails_without_traceback(tmp_path, capsys):
    assert main(["train", "--model", "nope", "--variant", "with_spike",
                 "--output-root", str(tmp_path)]) == 1
    assert "available" in capsys.readouterr().err


def test_unknown_variant_is_rejected_by_the_parser(tmp_path):
    with pytest.raises(SystemExit):
        main(["train", "--model", "multiple", "--variant", "nope", "--output-root", str(tmp_path)])


def test_finalize_requires_the_explicit_allow_test_flag(tmp_path, capsys):
    assert main(["finalize", "--output-root", str(tmp_path)]) == 1
    assert "--allow-test" in capsys.readouterr().err
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -v`
Expected: FAIL ด้วย `ModuleNotFoundError: No module named 'src.ml.run'`

- [ ] **Step 3: เขียน `src/ml/run.py`**

```python
"""CLI for the modeling layer: train / report / finalize."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.contracts import VARIANTS
from .core.trainer import OUTPUT_ROOT, train_model
from .registry import MODELS, build, model_names


def _train(args) -> int:
    names = model_names() if args.all else [args.model]
    for name in names:
        if name not in MODELS:
            print(f"Unknown model {name!r}; available: {model_names()}", file=sys.stderr)
            return 1
    for name in names:
        model_class = MODELS[name]
        grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))[name] if args.grid else {}
        result = train_model(model_class, grid=grid, variant=args.variant,
                             output_root=Path(args.output_root) if args.output_root else None)
        status = "ok" if result["load_verification"]["passed"] else "RELOAD FAILED"
        print(f"{name:>20} | {args.variant:<10} | reload={status} | {result['directory']}")
    return 0


def _report(args) -> int:
    print(f"Report for variant {args.variant} is not implemented yet", file=sys.stderr)
    return 1


def _finalize(args) -> int:
    if not args.allow_test:
        print("finalize reads the Test split and requires --allow-test", file=sys.stderr)
        return 1
    print("finalize is not implemented yet", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.ml.run", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train one model or all of them")
    group = train.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="model name from the registry")
    group.add_argument("--all", action="store_true", help="train every registered model")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--grid", help="path to a JSON file of {model: grid}")
    train.add_argument("--output-root")
    train.set_defaults(handler=_train)

    report = sub.add_parser("report", help="build figures and the leaderboard")
    report.add_argument("--variant", choices=VARIANTS, required=True)
    report.add_argument("--output-root")
    report.set_defaults(handler=_report)

    finalize = sub.add_parser("finalize", help="evaluate selected models on the Test split")
    finalize.add_argument("--allow-test", action="store_true")
    finalize.add_argument("--output-root")
    finalize.set_defaults(handler=_finalize)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ml/test_run.py -v`
Expected: PASS 4 tests

- [ ] **Step 5: รัน CLI จริงเพื่อพิสูจน์ว่า pilot ครบลูป**

Run: `.\.venv\Scripts\python.exe -m src.ml.run train --model multiple --variant with_spike`
Expected: บรรทัดเดียวที่ลงท้ายด้วย `reload=ok` และมีไฟล์ครบ 7 ไฟล์ใน `outputs/modeling/with_spike/regression/multiple/`

ตรวจเพิ่ม:

```bash
ls outputs/modeling/with_spike/regression/multiple/
cat outputs/modeling/with_spike/regression/multiple/load_verification.json
```

Expected: `"passed": true`

- [ ] **Step 6: Commit**

```bash
git add src/ml/run.py tests/ml/test_run.py
git commit -m "feat: add the modeling CLI with report and finalize stubbed out

The stubs return 1 and say so rather than pretending to work — the previous
runner's habit of silently producing empty artifacts is what made the old
documentation untrustworthy."
```

---

## Task 11: ลบ `src/modeling/` และตั้ง pytest basetemp

**Files:**

- Delete: `src/modeling/`, `src/models/`, `tests/modeling/`
- Modify: `pytest.ini`, `.gitignore`

- [ ] **Step 1: ตั้ง `pytest.ini` ให้ tmp อยู่ไดรฟ์เดียวกับโปรเจกต์ (ADR-006)**

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = --basetemp=.pytest_tmp
```

- [ ] **Step 2: เพิ่ม `.pytest_tmp/` ใน `.gitignore`**

เพิ่มบรรทัดนี้ต่อท้ายส่วน generated outputs ของ `.gitignore`:

```text
.pytest_tmp/
```

- [ ] **Step 3: ลบ package เก่าและ test ของ governance layer**

```bash
git rm -r --quiet src/modeling src/models tests/modeling
git rm --cached --quiet outputs/.test_ledger.json
```

- [ ] **Step 4: ตรวจว่าไม่มีใครยัง import ของเก่าอยู่**

Run: `grep -rn "src\.modeling\|src\.models" --include="*.py" src tests`
Expected: ไม่มี output ถ้ามี ให้แก้ให้หมดก่อนไปต่อ

- [ ] **Step 5: รัน test ทั้งชุด**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: **ผ่านทั้งหมด 0 failed** — ชั้นข้อมูลต้องยังเขียวครบ (เดิม 387 tests) บวก tests ใหม่ของ `src/ml/`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: retire the src/modeling governance layer

Removes the config schema, integrity seals, lifecycle manifests, resume gate,
test ledger and expected-runs machinery. None of it is required by the
assignment, and code_snapshot_hash() actively blocked the work: because it
hashes src/models/**, writing model number 19 invalidated the finalized runs of
models 1-18. The tested numerical code moved to src/ml/core in earlier commits.

Also pins pytest's basetemp inside the repo so tmp_path lands on the same drive
as the project — five data-layer tests fail otherwise on Windows because a
relative path cannot cross drives."
```

---

## Task 12: อัปเดตเอกสารด้วยผลรันจริง แล้วเปิด PR

**Files:**

- Modify: `PROJECT_STATUS.md`, `RUNBOOK.md`

- [ ] **Step 1: รัน test ทั้งชุดแล้วจดตัวเลขจริง**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
บันทึกบรรทัดสรุปไว้ตรง ๆ เช่น `412 passed, 3 skipped in 28.1s`

- [ ] **Step 2: อัปเดตตารางสถานะใน `PROJECT_STATUS.md`**

แก้แถว `Test suite` ให้เป็นตัวเลขจากขั้นที่ 1 และแก้แถว `โมเดล` เป็น `1/19 (pilot: multiple)`

- [ ] **Step 3: เพิ่มคำสั่ง modeling ที่ใช้ได้จริงลง `RUNBOOK.md` หัวข้อ 10**

แทนที่ข้อความ "ยังไม่มี training/evaluation CLI ที่รันได้ใน repository" ด้วย:

```bat
.\.venv\Scripts\python.exe -m src.ml.run train --model multiple --variant with_spike
```

พร้อมระบุว่า `report` และ `finalize` ยังไม่ implement

- [ ] **Step 4: Commit และเปิด PR**

```bash
git add PROJECT_STATUS.md RUNBOOK.md
git commit -m "docs: record the real test count and pilot status after the core landed"
git push -u origin feat/ml-core
gh pr create --title "feat: ML core framework and pilot model" --body "$(cat <<'BODY'
## Summary
- สร้าง `src/ml/core/` (data, preprocess, metrics, persist, base, compare, trainer, figures)
- ย้ายโค้ดคณิตศาสตร์ที่มี test ผ่านแล้วมาจาก `src/modeling/` แล้วลบ governance layer ทิ้ง
- pilot: Multiple Linear Regression ครบลูป train → save → load → verify

## Test plan
- [ ] `pytest -q` เขียวทั้งชุด
- [ ] `python -m src.ml.run train --model multiple --variant with_spike` ได้ `reload=ok`
- [ ] artifact ครบ 7 ไฟล์ และ `load_verification.json` มี `"passed": true`
- [ ] ไม่มี `src.modeling` หลงเหลือใน `src/` และ `tests/`
BODY
)"
```

- [ ] **Step 5: หลัง merge — แจ้งทีม**

บอกเพื่อนร่วมทีมให้ `git pull --rebase origin main` แล้วเริ่มเขียนโมเดลตาม `MODEL_CARDS.md`
โดยยึด interface ใน `AGENTS.md` §4 และดู `src/ml/regression/multiple.py` เป็นตัวอย่างจริง

---

## หลังจบแผนนี้

โมเดลที่เหลือ 18 ตัวไม่มีแผนรายขั้นในเอกสารนี้ เพราะแต่ละตัวมีสเปกของตัวเองอยู่ใน
[`MODEL_CARDS.md`](../../../MODEL_CARDS.md) แล้ว และทุกตัวทำตาม pattern เดียวกันเป๊ะ:

1. เขียน unit test ของกลไกภายในบนตัวอย่างที่คำนวณมือได้ → ให้ fail
2. เขียนโมเดลตาม `BaseModel` → ให้ test ผ่าน
3. เพิ่ม parity test กับ reference → แก้จนผ่านแบบ strict
4. เพิ่ม save/load round-trip test
5. เพิ่ม 1 บรรทัดใน `src/ml/registry.py`
6. รัน `src.ml.run train --model <name> --variant with_spike` แล้วตรวจ `reload=ok`
7. commit

`src/ml/regression/multiple.py` คือตัวอย่างอ้างอิงของ pattern นี้

### งานที่จงใจเลื่อนไปแผนถัดไป

`run.py report` และ `run.py finalize` ถูกวางเป็น stub ที่คืน exit code 1 และบอกเหตุผล
เพราะทั้งคู่ต้องรอให้มีโมเดลครบก่อนจึงจะมีความหมาย ตาม [`ROADMAP.md`](../../../ROADMAP.md)
จะ implement วันที่ 16–17:

- `report` — รวม `validation_metrics.json` ของทุกโมเดลเป็น leaderboard, วาด ROC ซ้อนกัน,
  confusion matrix และ performance curve รวมถึงตารางเปรียบเทียบ variant A ↔ B

- `finalize` — เรียก `load_test(allow_test=True)` ครั้งเดียว ประเมินเฉพาะโมเดลที่เลือกไว้แล้ว
  และเขียน `test_metrics.json` + `test_predictions.csv`
