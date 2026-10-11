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

T1 `decision_tree` ลงมือแล้วและผ่านเกณฑ์ด้านโมเดล; ผลอยู่ในหัวข้อ T1 ด้านล่าง

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

**บทเรียน T1 ที่ใช้ต่อกับ T2/T3:** sklearn สลับลำดับ feature แบบ seeded ที่ทุก node
แม้ `splitter="best"` และ `max_features=None` ถ้า gain เสมอกัน จะเลือกตัวที่เจอก่อนในลำดับนั้น
ไม่ใช่ feature index ต่ำสุด การใช้ `_best_split()` ตาม index ตรง ๆ ทำให้ Decision Tree depth 5
ต่างที่บาง node ของ Train จริงและ probability ต่างสูงสุด 0.2879; `fit()` จึงจำลองลำดับของ sklearn
ก่อนเรียก `_best_split()` และผ่าน strict parity ครบ 12 runs ของ grid จริงแล้ว

**บทเรียน metric จาก T1:** `non_spike` มี probability = 0.5 จำนวน 3 แถว; sklearn/Decision Tree
ให้ label 0 แต่ trainer เดิมนับ `score >= 0.5` เป็น label 1 ตอนคำนวณ accuracy/F1
แก้ให้ confusion metric ใช้ `model.predict()` ที่บันทึกลง prediction CSV ส่วน AUC/AP ใช้ score ตามเดิม
ผลที่ถูกหลังรันใหม่คือ accuracy 0.884097 และ F1 0.188679 (`non_spike`)

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

- [x] unit test: 4 ตัวอย่างที่คำนวณ Gini มือได้ → `_best_split` คืนค่าที่ถูก
- [x] unit test: ข้อมูลบริสุทธิ์ → ต้นไม้มีแค่ root
- [x] unit test: `sample_weight` ไม่เท่ากัน → split เปลี่ยนไปตามน้ำหนักจริง
- [x] parity: เทียบ `tree_.feature` และ `tree_.threshold` กับ sklearn **ทีละ node** ไม่ใช่เทียบแค่ predict
- [x] **เกณฑ์ผ่าน T1:** structure ของต้นไม้ตรง sklearn 100% บน 3 ชุดข้อมูลสังเคราะห์

**ผลรัน T1 (2026-10-11):** `tests/ml/classification/test_decision_tree.py` ผ่าน 21/21;
`pytest -q` ผ่าน 679/679. Strict parity ผ่านทั้ง 6 candidates × 2 variants บน Train/Validation:
label, probability, feature และ threshold ตรง reference ทุกค่าที่ตรวจ; save→load ผ่านทั้งคู่
(`load_verification.json` มี `"passed": true`).

| variant | candidate ที่เลือกบน Validation | AUC | F1 |
| --- | --- | ---: | ---: |
| `with_spike` | depth 2, min leaf 5 | 0.641779 | 0.309859 |
| `non_spike` | depth 5, min leaf 5 | 0.692355 | 0.188679 |

ผลเต็มอยู่ใน [`MODEL_PROGRESS.md`](../models/decision_tree/MODEL_PROGRESS.md) และ
`outputs/modeling/<variant>/classification/decision_tree/`. **สถานะ DoD: ค้างข้อ 5 เท่านั้น** —
leaderboard คือตารางเปรียบเทียบ metric ของทุกโมเดลจาก `validation_metrics.json`;
`src.ml.run report` ยังไม่ implement (Phase C1 ของอีกฝั่ง) จึงยังไม่มีแถว `decision_tree`
ใน `leaderboard.csv` ของทั้งสอง variant. เมื่อตัว report พร้อม ให้รันทั้งสอง variant
และตรวจแถวนี้ก่อนเปลี่ยนสถานะเป็น ✅; ไม่ต้องแตะ Test set ในขั้นนี้

> ให้เวลากับ T1 มากกว่าที่ประมาณไว้ — อีก 4 โมเดลพึ่งโค้ดนี้ ถ้า T1 ผิด จะไปโผล่เป็นบั๊กลึก ๆ ใน T2–T5

### T2 — `random_forest` (Random Forest card) · ~2.5 ชม.

- [x] bootstrap ใช้ `np.random.RandomState(seed_i).randint(0, n, n)` — ตรวจว่าตรงกับ
      `sklearn.ensemble._forest._generate_sample_indices(seed_i, n, n, None)` เป๊ะ
- [x] เฉลี่ย **probability** ไม่ใช่ majority vote
- [x] unit test แยกบั๊ก: `n_estimators=1, bootstrap=False, max_features=None` → ต้องเท่ากับ `decision_tree` เดี่ยว
- [x] รองรับ PCA ผ่าน trainer (ไม่ต้องเขียน PCA เอง core มีให้แล้ว)
- [x] **เกณฑ์ผ่าน:** `assert_parity` ผ่าน strict

**ผลรัน T2 (2026-10-11):** grid ดิบ 4 ค่า แล้วใช้ hyperparameter ที่ชนะใน grid ดิบกับ PCA(2,4,6) อีก 3 ค่า; เลือกผู้ชนะจากทั้ง 7 แถวด้วย AP ก่อน AUC. ทดสอบบังคับให้ PCA(4) ชนะแล้วตรวจว่า save, reload และ metadata ใช้ validation matrix ขนาด 4 มิติจริง. `tests/ml/core/test_trainer.py`, `test_preprocess.py`, `test_decision_tree.py`, `test_random_forest.py` ผ่านรวม 108 tests; forest parity บน synthetic และ Train/Validation จริงที่ `(50,3)` และ `(100,5)` ในทั้งสอง variant ผ่าน label, probability และโครงสร้างทุก tree ที่ตรวจ. หลัง PCA(2,4,6) ทั้งสอง variant ก็ผ่าน strict parity บน Train/Validation; bootstrap count ที่ส่งเข้าต้นไม้ตรง sklearn โดยตรง. PCA ที่ rank ไม่พอถูกบันทึกเป็น invalid candidate และ reload ใช้ preprocessor ที่บันทึกจริง.

| variant | candidate ที่เลือก | PCA | AP | AUC | F1 | reload |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `with_spike` | depth 3, 100 trees | none | 0.254788 | 0.686202 | 0.291667 | ผ่าน |
| `non_spike` | depth 5, 50 trees | none | 0.301006 | 0.676645 | 0.291667 | ผ่าน |

ผลเต็มอยู่ใน [`MODEL_PROGRESS.md`](../models/random_forest/MODEL_PROGRESS.md) และ `outputs/modeling/<variant>/classification/random_forest/`. **DoD ค้าง leaderboard ข้อ 5** เพราะ `src.ml.run report` ยังไม่ implement; ยังไม่แตะ Test set.

**บทเรียน T2:** ทางเดิน `max_features="sqrt"` ของ sklearn อาจแยก node ที่ gain เป็นศูนย์ และ near-tie ระดับ floating point อาจเปลี่ยนฟีเจอร์ที่เลือกจน probability/label ต่าง. มี regression tests ปิดทั้งสองกรณี; ทางเดิน T1 ที่ `max_features=None` ยังคงเดิม.

**บทเรียนสำหรับ T6/PCA:** ถ้า `PCA(k)` ขอ component เกิน numerical rank ให้บันทึก candidate นั้นเป็น `invalid` พร้อม AP/AUC เป็น `null` และเหตุผล แล้วเลือกจาก candidates ที่เหลือ. Error อื่นจาก PCA ต้อง raise ตามเดิม. การพิสูจน์ reload ต้องอ่าน `preprocessor.npz` กลับมาและ transform Validation ดิบด้วย mean/scale, constant mask และ components ที่บันทึกไว้ ก่อนเปรียบเทียบ prediction.

### T3 — `adaboost` ★ (card #15) · ~3 ชม.

โมเดลนี้โจทย์บังคับให้ **อธิบายกลไกภายในอย่างละเอียดในรายงาน** และต้องมีทั้ง scratch และ ready-to-use

- [x] SAMME: `α = lr · log((1-err)/err)` ใช้ natural log
- [x] จัดการเคสขอบตาม config: `err ≤ 1e-12` → หยุดแบบ perfect · `err ≥ 0.5` ตัวแรก → failed
- [x] normalize น้ำหนักทุกรอบ
- [x] unit test: คำนวณ `err`, `α`, `w` ใหม่ 2 รอบด้วยมือบน 4 ตัวอย่าง → เทียบทุกค่า
- [x] parity: เทียบ `estimator_weights_` และ `estimator_errors_` กับ sklearn ทีละตัว
- [x] **เช็ค API:** sklearn 1.9 ถอด `algorithm="SAMME.R"` แล้ว ถ้า constructor ไม่รับ `algorithm` ให้ตัดทิ้ง
- [x] **บันทึกไว้ให้รายงาน:** AdaBoost ใช้ exponential loss จึงไวต่อ outlier ที่สุด และ target ของเรา
      มี kurtosis 16.26 → คาดว่าจะฟื้นตัวชัดที่สุดใน variant `non_spike` **ให้จดผลจริงไว้ไม่ว่าออกหัวหรือก้อย**

**ผลรัน T3 (2026-10-11):** 6 candidates ต่อ variant, เลือกด้วย AP ก่อน AUC. เทียบ sklearn 1.9.1 ทีละรอบครบทั้ง grid ในทั้งสอง variant: feature/threshold ของ stump ตรงทุกตัว, `estimator_weights_`/`estimator_errors_`/`decision_function` ต่าง 0 พอดี, `predict_proba` ต่าง ≤ 1.1e-16. `test_adaboost.py` 28 tests ผ่าน; ทั้งชุด 761 tests ผ่าน.

| variant | candidate ที่เลือก | AP | AUC | F1 | reload |
| --- | --- | ---: | ---: | ---: | --- |
| `with_spike` | lr 0.5, 100 stumps | 0.236309 | 0.676776 | 0.222222 | ผ่าน |
| `non_spike` | lr 0.5, 100 stumps | 0.240748 | 0.640862 | 0.235294 | ผ่าน |

**สมมติฐาน outlier ไม่ได้รับการสนับสนุน:** AP เพิ่มเพียง +0.0044 และ AUC ลด 0.0359 เมื่อตัด spike (`random_forest` ได้ AP +0.0462). รายละเอียดและข้อจำกัดของการตีความอยู่ใน [`MODEL_PROGRESS.md`](../models/adaboost/MODEL_PROGRESS.md). **DoD ค้าง leaderboard** เช่นเดียวกับ T1/T2; ยังไม่แตะ Test set.

**บทเรียน T3 สำหรับ T4/T5:** boosting ที่อัปเดตน้ำหนักสะสมจะตรง reference ระดับ bit ได้ ถ้าทำตามลำดับ floating point เดียวกัน (clip → fit → `exp(log(w) + α·incorrect)` → หารผลรวม) และสุ่ม seed ของ tree ทีละรอบจาก `RandomState` ตัวเดียว. config section `adaboost` (`alpha_factor: 0.5`, `zero_vote_label: 1`) ขัดกับ sklearn; เลือกตาม sklearn เพราะ strict parity.

### T4 — `gradient_boosting` (card #13) · ~3.5 ชม.

- [x] `F0` = log-odds ของ prior ไม่ใช่ 0
- [x] ค่าในใบใช้ **Newton step** `Σg / Σh` ไม่ใช่ mean ของ residual ← สาเหตุอันดับ 1 ที่ไม่ตรง
- [x] tree ภายในใช้ **squared error** ไม่ใช่ Gini
- [x] unit test: `n_estimators=1, learning_rate=1` → `F = F0 + γ` ตรวจด้วยมือได้
- [x] parity: เทียบทีละ stage ด้วย `staged_decision_function` ของ sklearn

**ผลรัน T4 (2026-10-11):** 8 candidates ต่อ variant, เลือกด้วย AP ก่อน AUC. เทียบ sklearn 1.9.1 ทีละ stage ครบ 16 ชุดบน Train และ Validation: ทุกต้นไม้ตรง (node count, feature, threshold), `staged_decision_function` และ `predict_proba` ต่าง 0 พอดี. stress test นอก grid 240 รัน (depth ถึง 4, ข้อมูลมีค่าซ้ำ) ตรงระดับ bit ทั้งหมด. `test_gradient_boosting.py` 31 tests ผ่าน; ทั้งชุด 792 tests ผ่าน.

| variant | candidate ที่เลือก | AP | AUC | F1 | reload |
| --- | --- | ---: | ---: | ---: | --- |
| `with_spike` | lr 0.1, depth 1, 50 trees | 0.253083 | 0.675380 | 0.244898 | ผ่าน |
| `non_spike` | lr 0.03, depth 2, 50 trees | 0.252529 | 0.673154 | 0.266667 | ผ่าน |

ผลเต็มอยู่ใน [`MODEL_PROGRESS.md`](../models/gradient_boosting/MODEL_PROGRESS.md). regression tree อยู่ใน `gradient_boosting.py` เอง ไม่ได้แก้ `decision_tree.py`. **DoD ค้าง leaderboard** เช่นเดียวกับ T1–T3; ยังไม่แตะ Test set.

**บทเรียน T4 สำหรับ T5:**

- **1 ulp ที่จุดเริ่มต้นพลิกต้นไม้ได้.** `F0` ต้องคำนวณแบบ `scipy.special.logit` (ช่วง `0.3 ≤ p ≤ 0.65` ใช้ `log1p(s) − log1p(−s)`) ไม่งั้น residual ต่าง 1 ulp แล้ว tie ระหว่างฟีเจอร์ที่แบ่งแถวชุดเดียวกันถูกตัดสินต่างกัน — เจอที่ต้น 37 ทั้งที่ต้นก่อนหน้าตรงหมด. XGBoost ใช้ `base_score` คนละแบบ แต่หลักเดียวกัน: ถ้าโครงสร้างเริ่มต่างกลางทาง ให้เทียบ raw score แบบ**bitwise ที่ stage 0** ก่อนสงสัยต้นไม้.
- **วิธีแยกว่าต้นไม้ผิดหรือ input ผิด:** ป้อน residual ของเราให้ tree ของ reference สร้างเอง ถ้าได้ต้นเดียวกับเรา ปัญหาอยู่ที่ residual ไม่ใช่ splitter.
- **`np.sum` เป็น pairwise** ส่วน reference ที่เขียนด้วย C บวก sequential — ใช้ `np.cumsum(x)[-1]` เมื่อผลรวมถูกเอาไปเทียบกับ threshold ระดับ eps.
- `predict` ของ gradient boosting ใน sklearn ใช้ `>= 0` ตรงข้ามกับกฎ `> 0.5` ของโมเดล argmax.
- `train_score_` ของ sklearn เป็น deviance = 2 × log loss.

### T5 — `xgboost` (card #14) · ~4 ชม. · **เสี่ยงสุด เริ่มตอนเช้า**

- [x] `tree_method="exact"` และปิด subsample/colsample ทั้ง 4 ตัว
- [x] `base_score=0.5` → `F0 = 0` (**ต่างจาก GB อย่าก๊อปข้ามกัน**)
- [x] `min_child_weight=0`, `reg_alpha=0`
- [x] unit test: คำนวณ `w* = -G/(H+λ)` และ gain ของ split เดียวด้วยมือ
- [x] parity: เทียบ `predict(output_margin=True)` ทีละ iteration ด้วย `iteration_range`
- [x] **แผนสำรอง:** ถ้าเกินครึ่งวันแล้วยังต่าง > 1e-3 ให้ลดเหลือกลไกแกน
      (exact greedy + gradient/hessian + leaf shrinkage) แล้วบันทึกขอบเขตลง card — ยังนับเป็น XGBoost from scratch

**ผลรัน T5 (2026-10-11):** ไม่ต้องใช้แผนสำรอง. 8 candidates ต่อ variant, เลือกด้วย AP ก่อน AUC. เทียบ `xgboost` 3.4.1 ทีละต้นครบ 16 ชุดบน Train และ Validation (1,400 ต้น): โครงสร้างตรงทุกต้น, label ต่าง 0, margin ต่างสูงสุด 4.8e-7, probability ต่างสูงสุด 6.0e-8. 13 จาก 16 ชุดตรงระดับ bit; ที่เหลือค่าใบต่าง 1 ulp ของ float32 ซึ่งพิสูจน์แล้วว่ามาจาก `expf` ของ C runtime. `test_xgboost.py` 36 tests ผ่าน; ทั้งชุด 854 tests ผ่าน.

| variant | candidate ที่เลือก | AP | AUC | F1 | reload |
| --- | --- | ---: | ---: | ---: | --- |
| `with_spike` | n 100, depth 1, lr 0.03, λ 1, γ 0 | 0.250802 | 0.669314 | 0.244898 | ผ่าน |
| `non_spike` | n 100, depth 2, lr 0.03, λ 10, γ 0.1 | 0.265178 | 0.685285 | 0.280000 | ผ่าน |

ผลเต็มอยู่ใน [`MODEL_PROGRESS.md`](../models/xgboost/MODEL_PROGRESS.md). **DoD ค้าง leaderboard** เช่นเดียวกับ T1–T4; ยังไม่แตะ Test set.

**ความต่างจาก checklist ข้างบน:** unit test "gain ของ split เดียว" ใช้สูตรของ library คือ `score_L + score_R − score_parent` ไม่มีตัวคูณ 0.5 และ γ ถูกใช้ตอน prune หลังโตเสร็จ ไม่ใช่ตอนเลือก split. `reference()` เพิ่ม `n_jobs=1`.

**บทเรียน T5:**

- **อ่าน source ให้ตรง tag กับเวอร์ชันที่ติดตั้ง.** สูตร gain ของ 3.4.1 ต่างจาก v3.1.1 ในระดับการปัดเศษ และแค่นั้นก็พลิก split ได้.
- **reference ที่เป็น float32 ต้องเลียน float32.** `np.exp` บน float32 ไม่เท่า `expf`; คำนวณใน float64 แล้วปัดจะใกล้กว่ามาก.
- **เมื่อเหลือ ulp ที่อธิบายไม่ได้ ให้เทียบ root ของทุกต้นกับสูตรทางเลือกทีละสูตร** — root ไม่มีความกำกวมเรื่องแถว สูตรที่ถูกจะตรงเกือบ 100% ส่วนที่ผิดจะตกชัดเจน.
- grid แบบ `tuples` รองรับที่ trainer แล้ว (ดูหมายเหตุ core ข้างล่าง) โมเดลไม่ต้องรับ `tuples` เอง.

> **หมายเหตุ core (2026-10-11) — แจ้งทั้งสองฝั่ง:** `src/ml/core/trainer.py::expand_grid` และ
> `src/ml/run.py::_load_grid` ถูกแก้ให้รองรับ grid แบบ `{tuples, tuple_fields}` ตามคำสั่งเจ้าของโปรเจกต์
> เพราะ grid ของ `xgboost` ใน config เป็นรูปแบบนี้และของเดิมอ่านไม่ได้. grid แบบช่วงค่าเดินโค้ดเดิมทุกบรรทัด.
> **หลักฐานว่าไม่กระทบโมเดลอื่น:** (1) ขยาย grid ของทั้ง 19 โมเดลใน config ก่อน/หลัง — 18 ตัวได้ candidate
> เหมือนเดิมและลำดับเดิม ต่างเฉพาะ `xgboost` · (2) train โมเดลที่ลงทะเบียนทั้ง 16 ตัว × 2 variants ก่อนและหลัง
> แล้ว diff 312 ไฟล์ — 310 ไฟล์เหมือนกันทุก byte ต่างเฉพาะ `search_results.json` ของ `xgboost` 2 ไฟล์
> ซึ่งเปลี่ยนแค่ `params` จาก `{"tuples": [...]}` เป็น 5 field มีชื่อ ตัวเลข metric เท่าเดิม ·
> (3) test ใหม่ `test_configured_grid_only_names_constructor_arguments` ตรวจทุกโมเดลที่ลงทะเบียนว่า
> candidate จาก config สร้างโมเดลได้จริง — โมเดลใหม่ที่ grid ไม่ตรง constructor จะ fail ที่นี่ก่อนถึง trainer.

### T6 — `svm` (card #17) · ~3.5 ชม.

- [x] **เกณฑ์ parity ของตัวนี้ต่างจากตัวอื่น** — เทียบที่ objective ไม่ใช่น้ำหนัก (hinge ไม่เรียบ)
      `rel_gap < 1e-3` และ `J_scratch <= J_ref·(1+1e-3)` และ label agreement ≥ 0.99
- [x] step size ของ subgradient ต้องเป็น `1/t` ไม่ใช่ค่าคงที่
- [x] `loss="hinge"` ไม่ใช่ `squared_hinge` ซึ่งเป็น default ของ sklearn
- [x] unit test หลักใช้ `fit_intercept=False` เพื่อตัดปัญหา liblinear penalize intercept
- [x] **แผนสำรอง:** ถ้า kernel SVM ไม่ทัน ให้เหลือ linear แล้วอธิบาย kernel trick ในรายงาน

**ผลรัน T6 (2026-10-11):** 12 candidates ต่อ variant (C 3 ค่า × ไม่มี PCA / PCA 2, 4, 6). เทียบ objective กับ `SVC(kernel="linear")` ครบ 24 ชุดจริง: relative gap สูงสุด 5.6e-06, scratch ไม่เคยแย่กว่าเกิน 5.6e-06, label ตรงต่ำสุด 99.94%. `test_svm.py` 32 tests ผ่าน.

| variant | candidate ที่เลือก | AP | AUC | F1 | reload |
| --- | --- | ---: | ---: | ---: | --- |
| `with_spike` | C 0.1, PCA 4 | 0.251395 | 0.678565 | 0.255319 | ผ่าน |
| `non_spike` | C 10, PCA 4 | 0.250110 | 0.680398 | 0.272727 | ผ่าน |

**ความต่างจาก checklist:** reference เป็น `SVC(kernel="linear")` ไม่ใช่ `LinearSVC` — `LinearSVC(intercept_scaling=1000)` ตาม card ไม่ลู่เข้าบนข้อมูลจริง (ชน 2,000,000 รอบ, objective แย่กว่าเรา 0.003% ที่ C=0.1, 0.2% ที่ C=1, 149% ที่ C=10). "objective ลด monotone" ไม่จริงสำหรับ subgradient; โมเดลคืนจุดที่ดีที่สุดที่เคยผ่าน. kernel SVM ไม่ได้ทำตาม config (`kernel: linear`). ผลเต็ม: [`MODEL_PROGRESS.md`](../models/svm/MODEL_PROGRESS.md).

### T7 — `mlp` (card #10) · ~4 ชม.

- [x] weight init ต้องก๊อปสูตรของ sklearn: uniform ในช่วง `±sqrt(6/(fan_in+fan_out))` ด้วย `RandomState`
- [x] อนุพันธ์ ReLU ที่ `z=0` ใช้ 0 (`z > 0` ไม่ใช่ `z >= 0`)
- [x] L2 หารด้วยจำนวนตัวอย่าง ไม่ใช่ขนาด batch
- [x] unit test: **gradient check** ทุกชั้นเทียบ numerical gradient ที่ 1e-6
- [x] unit test: XOR ที่มีชั้นซ่อนต้องแก้ได้ accuracy 1.0 ← พิสูจน์ว่า backprop ถูกจริง
- [x] reference ตั้ง `tol=0` และ `n_iter_no_change` ใหญ่กว่า `max_iter` เพื่อปิด early stopping ของ sklearn

**ผลรัน T7 (2026-10-11):** 12 candidates ต่อ variant. เทียบ `MLPClassifier` ครบ 24 ชุดบน Train และ Validation: น้ำหนักทุกชั้นต่างสูงสุด 2.2e-16, probability ต่างสูงสุด 3.3e-16, label ต่าง 0. `test_mlp.py` 39 tests ผ่าน รวม gradient check ทุกชั้นและ XOR.

| variant | candidate ที่เลือก | best epoch | AP | AUC | F1 | reload |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `with_spike` | [16], l2 0, lr 0.001 | 17 | 0.266817 | 0.675685 | 0.320000 | ผ่าน |
| `non_spike` | [16, 8], l2 0, lr 0.01 | 5 | 0.280762 | 0.699249 | 0.307692 | ผ่าน |

**ความต่างจาก checklist:** "L2 หารด้วยจำนวนตัวอย่าง ไม่ใช่ขนาด batch" ตรงข้ามกับ source ของ sklearn 1.9.1 ซึ่งหารด้วยขนาด **batch** — ยึดตาม source. มี Nesterov momentum 0.9 เพราะเป็น default ของ reference ที่ card ระบุ. ใช้ protocol เดียวกับ `slp`. ผลเต็ม: [`MODEL_PROGRESS.md`](../models/mlp/MODEL_PROGRESS.md).

### T8 — `stacking` (card #16) · ~3.5 ชม. · **ต้องรออีกฝั่ง**

ต้องรอ `logistic`, `decision_tree`, `knn` พร้อมก่อน (`knn` มาจากอีก branch) — ทำเป็นตัวสุดท้าย

- [x] **purge นับตาม original trading day ไม่ใช่ row gap** — variant `non_spike` มีแถวหายไป
      การนับ 5 แถวจะข้ามเวลาจริงเกิน 5 วัน → เกิด leakage
- [x] meta ห้ามเห็น in-sample probability เด็ดขาด
- [x] unit test: ตรวจว่าไม่มี index ของ validation โผล่ใน training ของ fold ใด ๆ
- [x] sklearn `StackingClassifier` ไม่มี purge → ตัวเลขจะต่าง **และนั่นถูกต้องแล้ว** ให้บันทึกเหตุผล

**ผลรัน T8 (2026-10-11):** 1 candidate ต่อ variant. ตรวจด้วย rebuild fold ต่อ fold จาก sklearn reference ของ base แต่ละตัว: meta features ของ tree และ k-NN ต่าง 0, ของ logistic ต่างสูงสุด 3.9e-07, probability สุดท้ายต่างสูงสุด 6.2e-07, label ต่าง 0 ทั้งสอง variant. leakage probe ผ่านทั้ง 8 กรณี. `test_stacking.py` 25 tests ผ่าน.

| variant | AP | AUC | F1 | แถวที่ meta ใช้ | reload |
| --- | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 0.226248 | 0.664078 | 0.235294 | 1,040 | ผ่าน |
| `non_spike` | 0.231130 | 0.665038 | 0.196078 | 852 | ผ่าน |

**ความต่างจาก checklist:** `StackingClassifier(cv=TimeSeriesSplit(4))` ตาม card **รันไม่ได้** (sklearn ต้องการ fold ที่แบ่งครบทุกแถว) จึงใช้ `KFold` เป็น behavioural reference และย้ายการตรวจจริงไปที่ rebuild. เรื่อง purge: บน `non_spike` การนับวันกับการนับแถวต่างกันจริง 166 แถว แต่ทิศทางไม่ใช่ leakage อย่างที่ checklist เขียน — นับแถวจะ purge **มากกว่า** ที่ผิดคือขอบ block เลื่อน. ผลเต็ม: [`MODEL_PROGRESS.md`](../models/stacking/MODEL_PROGRESS.md).

**สิ่งที่ T8 เจอเกี่ยวกับ `logistic` — ผลเล็ก ไม่ได้แก้:** ด้วยค่าตาม config (`learning_rate=0.05, max_iter=5000`) โมเดล `logistic` ไม่ลู่เข้าบนข้อมูลจริง. ทั้ง 12 การรัน (6 candidates × 2 variants) หยุดเพราะครบ 5,000 รอบ ไม่ใช่เพราะถึง `tol`. เทียบกับโมเดลที่รันจนลู่เข้า บน Validation: probability ต่างสูงสุด 0.0079, AP ต่างสูงสุด 0.0007, AUC ต่างสูงสุด 0.0012, label ต่างสูงสุด 1 จาก 371 แถว; ผู้ชนะทั้งสอง variant ได้ AP/AUC เท่าเดิมถึงทศนิยม 4 ตำแหน่ง. stacking รัน base logistic จนลู่เข้าเองเพราะ fold ที่เล็กกว่าห่างจาก optimum 0.03–0.07 ซึ่งมากพอจะทำให้การเทียบ meta features ไม่มีความหมาย.

**แก้ไขหลัง merge (2026-10-11) — purge ของ `slp` และ `mlp`:** เพิ่ม purge 5 วันทำการก่อน tail ของ early stopping ตาม
`neural.purge_original_trading_days: 5` ในทั้งสองโมเดล (แก้ `slp.py` ของฝั่ง linear ด้วย ตามคำสั่งเจ้าของโปรเจกต์).
ไม่ได้แก้ trainer — ใช้ `positions` ที่ส่งให้อยู่แล้ว. ผล: ผู้ชนะของ `slp` ทั้งสอง variant และ `mlp` บน `with_spike`
ไม่เปลี่ยนเลย; `mlp` บน `non_spike` เปลี่ยนจาก `[32], lr 0.001` (AP 0.2740) เป็น `[16, 8], lr 0.01` (AP 0.2808)
เพราะ candidate นั้นเลื่อนจาก 1 เป็น 5 epochs. parity ของ mlp วัดซ้ำครบ 24 ชุดได้ 2.2e-16 เท่าเดิม.
`scaler_fit: subtrain` ยังไม่ได้ทำ. โมเดล `logistic` ไม่ได้แก้ — เพิ่มเฉพาะหมายเหตุในเอกสารของมัน.

**บทเรียน T6–T8:**

- **reference ก็ผิดได้.** สองใน สาม reference ของ card ใช้ไม่ได้ตามที่เขียน: `LinearSVC` ไม่ลู่เข้า และ `StackingClassifier` + `TimeSeriesSplit` รันไม่ได้. เมื่อ scratch "ชนะ" reference ให้ดู `n_iter_` ของ reference ก่อนฉลอง.
- **parity ที่รันจนลู่เข้าพิสูจน์อัลกอริทึม ไม่ได้บอกว่า production ลู่เข้า.** ทั้งสองคำถามมีค่า แต่ต้องวัดข้อหลังบนข้อมูลจริงด้วย — `logistic` ชน max_iter ทุกการรันโดยที่ fixture ของ test หยุดที่รอบ 550.
- **โมเดลประกอบ ตรวจด้วยการประกอบ reference ทีละชิ้นบน fold เดียวกัน** ได้ผลเข้มกว่าเทียบ ensemble สำเร็จรูป และบอกได้ทันทีว่าชิ้นไหนต่าง.
- **leakage ตรวจได้โดยไม่ต้องมี reference:** เขียนทับปัจจุบันและอนาคตแล้วดูว่า output ของอดีตขยับไหม.

> **หมายเหตุ core รอบ 2 (2026-10-11) — แจ้งทั้งสองฝั่ง:** `src/ml/core/trainer.py` ถูกแก้อีก 2 จุดตามคำสั่งเจ้าของโปรเจกต์:
> (1) **PCA search ของ `svm`** ไขว้ทุก C กับ PCA 2/4/6 ให้ครบ 12 candidates ตาม config; ทางของ `random_forest` (ลอง PCA เฉพาะค่าที่ชนะ) ไม่เปลี่ยน ·
> (2) **`positions` และ `base_specific`** โมเดลที่ `fit` รับ `positions` จะได้ index วันทำการบน timeline เดิม (ตอนนี้มีแค่ `stacking`) และ preprocess แบบ `base_specific` ไม่ถูก standardize โดย trainer.
> **หลักฐานว่าไม่กระทบโมเดลอื่น:** train ทุกโมเดลที่ลงทะเบียน × 2 variants ก่อนและหลังแต่ละจุด แล้วเทียบทีละไฟล์ —
> หลังจุด (1): 312 จาก 312 ไฟล์ของ 16 โมเดลเดิมเหมือนกันทุก byte (รวม `random_forest` ที่ใช้ทาง PCA เดียวกัน) ·
> หลังจุด (2): 332 จาก 332 ไฟล์ของ 17 โมเดลเหมือนกันทุก byte · artifact ของ `svm`/`mlp`/`stacking` ใน repo
> ตรงกับการรันใหม่ทุกไฟล์ (reproducible).

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

จากนั้น **อัปเดตเอกสาร 2 ไฟล์** ก่อน commit

1. **`docs/models/<name>/MODEL_PROGRESS.md`** — stub รออยู่แล้ว เขียนทับด้วยผลจริง:
   metric ที่วัดได้ · เกณฑ์ parity ที่ใช้และค่าที่ได้ · การตัดสินใจที่ชั่งแล้ว ·
   **"ปัญหาที่เจอและแก้แล้ว"** (ส่วนที่มีค่าที่สุด — อาการ → สาเหตุจริง → วิธีแก้) · สิ่งที่ค้าง
   รูปแบบเต็มอยู่ใน [`docs/models/README.md`](../models/README.md)
2. **card ของโมเดลนั้นใน `docs/plan/MODEL_CARDS.md`** — ใส่ตัวเลขจริง
   (ผลต่างสูงสุดที่วัดได้, เวลาที่ใช้จริง, ความต่างจาก reference ถ้ามี)

> ถ้า pitfall ที่เจอจะเกิดกับโมเดลตัวอื่นของคุณด้วย **ให้ยกขึ้นมาไว้ §5 ของไฟล์นี้**
> ไม่ใช่ปล่อยไว้ในไฟล์ของโมเดลเดียว — T1 `decision_tree` เป็นฐานของอีก 4 ตัว
> บทเรียนจากมันจึงมีค่ากับ T2–T5 ทั้งหมด

```bat
git add src/ml/classification/<name>.py tests/ml/classification/test_<name>.py src/ml/registry.py docs/models/<name>/MODEL_PROGRESS.md docs/plan/MODEL_CARDS.md
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
