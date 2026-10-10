# TASKS — Tree / Ensemble / Margin Family (8 โมเดล)

> **เอกสารนี้สำหรับผู้พัฒนาที่ใช้ Codex CLI** · branch `feat/models-tree-family`
> อ่านจบแล้วลงมือได้เลยโดยไม่ต้องถามใคร — ทุกอย่างที่ต้องรู้อยู่ในนี้หรือในลิงก์ที่ชี้ไว้
>
> อ่านก่อนเริ่ม 3 ไฟล์ตามลำดับ: [`AGENTS.md`](../../AGENTS.md) (กฎและ interface) →
> [`docs/plan/MODEL_CARDS.md`](../plan/MODEL_CARDS.md) (คณิตศาสตร์รายโมเดล) →
> เอกสารนี้ (ลำดับงานและเกณฑ์ผ่าน)

---

## 0. ตั้งต้น (ทำครั้งเดียว ~15 นาที)

```bat
git clone <repo> && cd QQQ-Volatility-Forecasting
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

ขอไฟล์ `data/raw/qqq_daily.csv` จากเพื่อนร่วมทีม (ไม่ได้อยู่ใน Git) วางแล้วตรวจ:

```bat
.\.venv\Scripts\python.exe src\download_qqq_data.py verify-snapshot
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp/phase1-reproduction" --overwrite-generated
.\.venv\Scripts\python.exe -m pytest -q
```

**เกณฑ์ผ่านข้อ 0:** `verify-snapshot` คืน `row_count: 2512` และ sha256 ขึ้นต้น `649e1db1`
· `pytest` ไม่มี failed

> ห้ามลบโฟลเดอร์ `tmp/phase1-reproduction` — มี report ผูก path นี้ไว้เป็น provenance

```bat
git checkout -b feat/models-tree-family
```

---

## 1. กฎ 7 ข้อที่ห้ามละเมิด (ย่อจาก `AGENTS.md`)

1. **ห้ามแก้ชั้นข้อมูล** — `data/**` และ data pipeline ใน `src/` อ่านได้อย่างเดียว
2. **ห้ามแตะ Test set** — ใช้ Train เทรน Validation เลือก hyperparameter เท่านั้น
3. **ทุกโมเดลเขียนเองด้วย NumPy** — sklearn/xgboost อยู่ใน `reference()` เท่านั้น
4. **ห้าม import `src/modeling/`** — package นั้นถูกยกเลิกแล้ว
5. **1 โมเดล = 1 ไฟล์** ห้ามแก้ไฟล์ของอีกฝั่ง (`src/ml/regression/**`, `src/ml/clustering/**` ไม่ใช่เขตของคุณ)
6. **ห้ามเคลมว่าเสร็จโดยไม่ได้รันคำสั่งและเห็น output จริง**
7. **ใช้ `np.random.RandomState(seed)` ไม่ใช่ `np.random.default_rng(seed)`** — sklearn สร้างบน
   legacy MT19937 ถ้าใช้ `default_rng` parity ของ `random_forest` และ `mlp` จะไม่มีทางผ่าน

---

## 1.5 ไม่ต้องรออะไรแล้ว — core เสร็จและ merge เข้า `main` ครบ

ทุกอย่างที่คุณต้องใช้พร้อมใช้งานแล้ว ยืนยันด้วยการรันจริงเมื่อ 2026-10-10:

| พร้อมใช้ | ใช้ทำอะไร |
| --- | --- |
| `core/base.py` | `BaseModel` ที่โมเดลคุณต้อง subclass |
| `core/compare.py` | `assert_parity()` ด่านตรวจความถูกต้อง |
| `core/data.py` | `load(variant)` — Train/Validation พร้อม Test แบบ default-deny |
| `core/preprocess.py` | `Standardizer`, `PCA` (ใช้ผ่าน trainer ไม่ต้องเรียกเอง) |
| `core/metrics.py` | confusion matrix · accuracy · precision · sensitivity · specificity · TNR · F1 · ROC+AUC · R² · losses |
| `core/persist.py` | `save_npz` / `load_npz` / `verify_reload` |
| `core/trainer.py` | grid search + ผลิต artifact ครบชุด |
| `core/figures.py` | ROC · confusion · performance curve · leaderboard |
| `registry.py` · `run.py` | ทะเบียนโมเดล + CLI `train` |

```bat
.\.venv\Scripts\python.exe -m src.ml.run train --model multiple_linear --variant with_spike
rem ->      multiple_linear | with_spike | candidates= 1 | selected=multiple_linear-00 | reload=ok
```

`multiple_linear` เป็น pilot ที่รันครบลูปแล้ว **ใช้ `src/ml/regression/multiple_linear.py`
เป็นตัวอย่างอ้างอิง** ว่าโมเดลที่ถูกต้องหน้าตาเป็นอย่างไร — สั้นและครบทุก method

เริ่ม T1 ได้ทันที `decision_tree` เป็นฐานของอีก 4 โมเดล ยิ่งเสร็จเร็วยิ่งดีกับทั้งสาย

---

## 2. ลำดับงาน — ทำตามลำดับนี้เท่านั้น

ลำดับนี้ไม่ได้เรียงตามความยาก แต่เรียงตาม **การพึ่งพากัน** — `decision_tree` ถูก reuse โดยอีก 4 ตัว

```text
T1  tree primitives + decision_tree   ← ฐานของทุกอย่าง ห้ามข้าม
T2  random_forest                     ← ใช้ tree จาก T1
T3  adaboost ★                        ← ใช้ weighted stump จาก T1
T4  gradient_boosting                 ← ใช้ regression tree
T5  xgboost                           ← ต่อยอดจาก T4 (เสี่ยงสุด เริ่มแต่เช้า)
T6  svm                               ← อิสระ ทำแทรกได้ถ้าติดขัดที่ T5
T7  mlp                               ← อิสระ
T8  stacking                          ← ต้องรอ logistic/knn จากอีกฝั่ง merge ก่อน
```

**ถ้าติดขัดเกิน 2 ชั่วโมงในตัวใดตัวหนึ่ง** ให้ข้ามไปตัวถัดไปแล้วกลับมาทีหลัง อย่าค้างอยู่กับตัวเดียว

---

## 3. โครงที่ทุกโมเดลต้องเขียนเหมือนกัน

```python
# src/ml/classification/<name>.py
from __future__ import annotations
import numpy as np
from ..core.base import BaseModel


class YourModel(BaseModel):
    name = "<ชื่อตรงกับ key ใน configs/modeling.json และชื่อไฟล์>"
    task = "classification"

    def __init__(self, max_depth: int = 3, random_state: int = 42):
        # hyperparameter ทุกตัวต้องมี default เพราะ trainer สร้างอินสแตนซ์เปล่าตอน reload
        self.max_depth = max_depth
        self.random_state = random_state

    def get_params(self) -> dict:
        return {"max_depth": self.max_depth, "random_state": self.random_state}

    def fit(self, X, y=None) -> "YourModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        ...
        self.history_ = {"iteration": [...], "loss": [...]}   # เฉพาะโมเดลที่เทรนวนรอบ
        return self

    def predict(self, X) -> np.ndarray:        # คืน label 0/1
        ...

    def decision_function(self, X) -> np.ndarray:   # คะแนนดิบ
        ...

    def predict_proba(self, X) -> np.ndarray:  # ถ้าให้ความน่าจะเป็นจริงไม่ได้ ไม่ต้อง override
        ...

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"...": np.asarray(...)}        # เฉพาะพารามิเตอร์ที่เรียนรู้มา

    def load_state(self, arrays: dict, metadata: dict) -> "YourModel":
        ...
        for k, v in metadata["params"].items():
            setattr(self, k, v)
        return self

    def reference(self):
        from sklearn.xxx import Yyy
        return Yyy(...)                        # ตั้งค่าให้เทียบเท่ากันตาม MODEL_CARDS
```

เสร็จแล้วเพิ่ม **1 บรรทัด** ใน `src/ml/registry.py` (เรียงตามตัวอักษร เพื่อให้ conflict แก้ง่าย):

```python
"random_forest": RandomForest,
```

---

## 4. วงจร TDD ของทุกโมเดล (ทำ 7 ขั้นนี้ซ้ำทุกตัว)

| ขั้น | ทำอะไร | คำสั่ง / เกณฑ์ |
| --- | --- | --- |
| 1 | อ่าน card ของโมเดลนั้นใน `MODEL_CARDS.md` ให้จบ **โดยเฉพาะหัวข้อ "จุดที่ parity พลาด"** | — |
| 2 | เขียน **unit test กลไกภายใน** บนตัวอย่างเล็กที่คำนวณมือได้ → ต้อง fail | `pytest tests/ml/classification/test_<name>.py -v` |
| 3 | เขียนโมเดลให้ test ผ่าน | ขั้นนี้ยังไม่ต้องสนใจ parity |
| 4 | เพิ่ม **parity test** กับ reference | `assert_parity(scratch, ref, X, y, task="classification")` |
| 5 | ไล่แก้จน parity ผ่าน **strict** | ดู §5 เมื่อไม่ผ่าน |
| 6 | เพิ่ม **save/load round-trip test** | predict ก่อน/หลัง reload ต้องเท่ากันเป๊ะ |
| 7 | เพิ่มใน `registry.py` แล้วรันจริง 2 variant | `python -m src.ml.run train --model <name> --variant with_spike` ต้องได้ `reload=ok` |

**โครง test ที่ต้องมีครบ 3 แบบ**

```python
# tests/ml/classification/test_<name>.py
import numpy as np, pytest
from src.ml.core.compare import assert_parity
from src.ml.classification.<name> import YourModel


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    return X, y


def test_<กลไกภายใน>():
    """ทดสอบหัวใจของอัลกอริทึมบนตัวอย่างที่คำนวณมือได้ — ดู card ข้อ 'Unit test'"""
    ...


def test_matches_reference(data):
    X, y = data
    scratch = YourModel().fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert_parity(scratch, reference, X, y, task="classification")


def test_state_round_trip(data):
    X, y = data
    original = YourModel().fit(X, y)
    restored = YourModel().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))
```

---

## 5. เมื่อ parity ไม่ผ่าน — ทำตามลำดับนี้

1. อ่านหัวข้อ **"จุดที่ parity พลาด"** ของ card นั้นอีกรอบ — 90% ของปัญหาถูกบันทึกไว้แล้ว
2. ตรวจว่าปิด randomness ของ reference ครบหรือยัง (`random_state=42`, `subsample=1`,
   `max_features=None`, `shuffle=False`)
3. เทียบ**ทีละชั้น** ไม่ใช่เทียบผลสุดท้าย — เช่น GB ใช้ `staged_decision_function`,
   XGBoost ใช้ `iteration_range`, RF เทียบ tree ต้นแรกต้นเดียวก่อน
4. ยังไม่ผ่าน → **หยุด** บันทึกลง card ของโมเดลนั้นว่าต่างตรงไหนเท่าไรเพราะอะไร แล้วแจ้งเจ้าของโปรเจกต์

**ห้ามเด็ดขาด:** ลดเกณฑ์เงียบ ๆ · เอาผลของ reference มาสวมแทน scratch · ข้าม parity test ไป

---

## 6. รายละเอียดรายงาน 8 tasks

ทุก task อ้าง card ใน [`docs/plan/MODEL_CARDS.md`](../plan/MODEL_CARDS.md) ซึ่งมีสมการ, grid,
reference constructor และกับดัก parity ครบแล้ว **ที่นี่บอกเฉพาะลำดับและเกณฑ์ผ่าน**

### T1 — tree primitives + `decision_tree` (card #11) · ~3 ชม. · **สำคัญที่สุด**

สร้าง `src/ml/classification/decision_tree.py` ที่ข้างในมีฟังก์ชันใช้ซ้ำได้:

```python
def _best_split(X, y, sample_weight, min_samples_leaf):
    """คืน (feature_index, threshold, gain) ที่ดีที่สุด หรือ None ถ้าไม่มี split ที่ gain > 0

    ต้องรองรับ sample_weight ตั้งแต่แรก เพราะ AdaBoost (T3) ต้องใช้ weighted Gini
    tie-breaking: gain เท่ากัน -> feature index น้อยกว่าชนะ -> threshold น้อยกว่าชนะ
    """
```

- [ ] unit test: 4 ตัวอย่างที่คำนวณ Gini มือได้ → `_best_split` คืนค่าที่ถูก
- [ ] unit test: ข้อมูลบริสุทธิ์ → ต้นไม้มีแค่ root
- [ ] unit test: `sample_weight` ไม่เท่ากัน → split เปลี่ยนไปตามน้ำหนักจริง
- [ ] parity: เทียบ `tree_.feature` และ `tree_.threshold` กับ sklearn **ทีละ node** ไม่ใช่เทียบแค่ predict
- [ ] **เกณฑ์ผ่าน T1:** structure ของต้นไม้ตรง sklearn 100% บน 3 ชุดข้อมูลสังเคราะห์

> ให้เวลากับ T1 มากกว่าที่ประมาณไว้ — อีก 4 โมเดลพึ่งโค้ดนี้ ถ้า T1 ผิด จะไปโผล่เป็นบั๊กลึก ๆ ใน T2–T5

### T2 — `random_forest` (card #12) · ~2.5 ชม.

- [ ] bootstrap ใช้ `np.random.RandomState(seed_i).randint(0, n, n)` — ตรวจว่าตรงกับ
      `sklearn.ensemble._forest._generate_sample_indices(seed_i, n, n, None)` เป๊ะ
- [ ] เฉลี่ย **probability** ไม่ใช่ majority vote
- [ ] unit test แยกบั๊ก: `n_estimators=1, bootstrap=False, max_features=None` → ต้องเท่ากับ `decision_tree` เดี่ยว
- [ ] รองรับ PCA ผ่าน trainer (ไม่ต้องเขียน PCA เอง core มีให้แล้ว)
- [ ] **เกณฑ์ผ่าน:** `assert_parity` ผ่าน strict

### T3 — `adaboost` ★ (card #15) · ~3 ชม.

โมเดลนี้โจทย์บังคับให้ **อธิบายกลไกภายในอย่างละเอียดในรายงาน** และต้องมีทั้ง scratch และ ready-to-use

- [ ] SAMME: `α = lr · log((1-err)/err)` ใช้ natural log
- [ ] จัดการเคสขอบตาม config: `err ≤ 1e-12` → หยุดแบบ perfect · `err ≥ 0.5` ตัวแรก → failed
- [ ] normalize น้ำหนักทุกรอบ
- [ ] unit test: คำนวณ `err`, `α`, `w` ใหม่ 2 รอบด้วยมือบน 4 ตัวอย่าง → เทียบทุกค่า
- [ ] parity: เทียบ `estimator_weights_` และ `estimator_errors_` กับ sklearn ทีละตัว
- [ ] **เช็ค API:** sklearn 1.9 ถอด `algorithm="SAMME.R"` แล้ว ถ้า constructor ไม่รับ `algorithm` ให้ตัดทิ้ง
- [ ] **บันทึกไว้ให้รายงาน:** AdaBoost ใช้ exponential loss จึงไวต่อ outlier ที่สุด และ target ของเรา
      มี kurtosis 16.26 → คาดว่าจะฟื้นตัวชัดที่สุดใน variant `non_spike` **ให้จดผลจริงไว้ไม่ว่าออกหัวหรือก้อย**

### T4 — `gradient_boosting` (card #13) · ~3.5 ชม.

- [ ] `F0` = log-odds ของ prior ไม่ใช่ 0
- [ ] ค่าในใบใช้ **Newton step** `Σg / Σh` ไม่ใช่ mean ของ residual ← สาเหตุอันดับ 1 ที่ไม่ตรง
- [ ] tree ภายในใช้ **squared error** ไม่ใช่ Gini
- [ ] unit test: `n_estimators=1, learning_rate=1` → `F = F0 + γ` ตรวจด้วยมือได้
- [ ] parity: เทียบทีละ stage ด้วย `staged_decision_function` ของ sklearn

### T5 — `xgboost` (card #14) · ~4 ชม. · **เสี่ยงสุด เริ่มตอนเช้า**

- [ ] `tree_method="exact"` และปิด subsample/colsample ทั้ง 4 ตัว
- [ ] `base_score=0.5` → `F0 = 0` (**ต่างจาก GB อย่าก๊อปข้ามกัน**)
- [ ] `min_child_weight=0`, `reg_alpha=0`
- [ ] unit test: คำนวณ `w* = -G/(H+λ)` และ gain ของ split เดียวด้วยมือ
- [ ] parity: เทียบ `predict(output_margin=True)` ทีละ iteration ด้วย `iteration_range`
- [ ] **แผนสำรอง:** ถ้าเกินครึ่งวันแล้วยังต่าง > 1e-3 ให้ลดเหลือกลไกแกน
      (exact greedy + gradient/hessian + leaf shrinkage) แล้วบันทึกขอบเขตลง card — ยังนับเป็น XGBoost from scratch

### T6 — `svm` (card #17) · ~3.5 ชม.

- [ ] **เกณฑ์ parity ของตัวนี้ต่างจากตัวอื่น** — เทียบที่ objective ไม่ใช่น้ำหนัก (hinge ไม่เรียบ)
      `rel_gap < 1e-3` และ `J_scratch <= J_ref·(1+1e-3)` และ label agreement ≥ 0.99
- [ ] step size ของ subgradient ต้องเป็น `1/t` ไม่ใช่ค่าคงที่
- [ ] `loss="hinge"` ไม่ใช่ `squared_hinge` ซึ่งเป็น default ของ sklearn
- [ ] unit test หลักใช้ `fit_intercept=False` เพื่อตัดปัญหา liblinear penalize intercept
- [ ] **แผนสำรอง:** ถ้า kernel SVM ไม่ทัน ให้เหลือ linear แล้วอธิบาย kernel trick ในรายงาน

### T7 — `mlp` (card #10) · ~4 ชม.

- [ ] weight init ต้องก๊อปสูตรของ sklearn: uniform ในช่วง `±sqrt(6/(fan_in+fan_out))` ด้วย `RandomState`
- [ ] อนุพันธ์ ReLU ที่ `z=0` ใช้ 0 (`z > 0` ไม่ใช่ `z >= 0`)
- [ ] L2 หารด้วยจำนวนตัวอย่าง ไม่ใช่ขนาด batch
- [ ] unit test: **gradient check** ทุกชั้นเทียบ numerical gradient ที่ 1e-6
- [ ] unit test: XOR ที่มีชั้นซ่อนต้องแก้ได้ accuracy 1.0 ← พิสูจน์ว่า backprop ถูกจริง
- [ ] reference ตั้ง `tol=0` และ `n_iter_no_change` ใหญ่กว่า `max_iter` เพื่อปิด early stopping ของ sklearn

### T8 — `stacking` (card #16) · ~3.5 ชม. · **ต้องรออีกฝั่ง**

ต้องรอ `logistic`, `decision_tree`, `knn` พร้อมก่อน (`knn` มาจากอีก branch) — ทำเป็นตัวสุดท้าย

- [ ] **purge นับตาม original trading day ไม่ใช่ row gap** — variant `non_spike` มีแถวหายไป
      การนับ 5 แถวจะข้ามเวลาจริงเกิน 5 วัน → เกิด leakage
- [ ] meta ห้ามเห็น in-sample probability เด็ดขาด
- [ ] unit test: ตรวจว่าไม่มี index ของ validation โผล่ใน training ของ fold ใด ๆ
- [ ] sklearn `StackingClassifier` ไม่มี purge → ตัวเลขจะต่าง **และนั่นถูกต้องแล้ว** ให้บันทึกเหตุผล

---

## 7. เสร็จแต่ละตัวแล้วทำอะไร

```bat
rem 1) test ของตัวเองต้องเขียว
.\.venv\Scripts\python.exe -m pytest tests/ml/classification/test_<name>.py -v

rem 2) test ทั้งชุดต้องไม่พังจากของเดิม
.\.venv\Scripts\python.exe -m pytest -q

rem 3) รันจริงทั้ง 2 variant
.\.venv\Scripts\python.exe -m src.ml.run train --model <name> --variant with_spike
.\.venv\Scripts\python.exe -m src.ml.run train --model <name> --variant non_spike

rem 4) ตรวจว่า save/load ผ่านจริง
type outputs\modeling\with_spike\classification\<name>\load_verification.json
```

**ต้องเห็น `"passed": true` ด้วยตาจริง ๆ ก่อนบอกว่าเสร็จ**

จากนั้นกลับไปแก้ card ของโมเดลนั้นใน `docs/plan/MODEL_CARDS.md` ใส่**ตัวเลขจริง**
(ผลต่างสูงสุดที่วัดได้, เวลาที่ใช้จริง, ความต่างจาก reference ถ้ามี) แล้ว commit

```bat
git add src/ml/classification/<name>.py tests/ml/classification/test_<name>.py src/ml/registry.py docs/plan/MODEL_CARDS.md
git commit -m "feat: add <name> from scratch with strict parity against <reference>"
```

**เปิด PR เมื่อจบกลุ่ม ไม่ใช่ทีละตัว** และ `git pull --rebase origin main` บ่อย ๆ

---

## 8. Definition of Done ของทั้ง branch

- [ ] 8 โมเดลผ่าน DoD ครบ 6 ข้อตาม `AGENTS.md` §5
- [ ] `pytest -q` เขียวทั้งชุด
- [ ] ทุกโมเดลมี `load_verification.json` ที่ `"passed": true` ทั้ง 2 variant
- [ ] `MODEL_CARDS.md` ของทั้ง 8 ใบอัปเดตด้วยตัวเลขจริงแล้ว
- [ ] ไม่มี `src.modeling` หลงเหลือใน import
- [ ] ไม่มีไฟล์ของอีกฝั่งถูกแก้ (`git diff --stat origin/main` ต้องมีแต่ไฟล์ในเขตตัวเอง)

---

## 9. ติดปัญหาให้ถามอะไร

ถามเจ้าของโปรเจกต์พร้อม **ข้อมูล 4 อย่างนี้เสมอ** จะได้ตอบได้เร็ว:

1. โมเดลไหน task ไหน
2. คำสั่งที่รันและ output จริงแบบ copy มาทั้งก้อน
3. ผลต่างที่วัดได้เป็นตัวเลข (`max abs diff` เท่าไร ที่ index ไหน)
4. ลองอะไรไปแล้วบ้างจากรายการใน §5
