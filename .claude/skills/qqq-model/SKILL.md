---
name: qqq-model
description: Use when implementing, debugging, or verifying any of the 19 from-scratch models in this QQQ volatility project — the distilled working contract (BaseModel interface, strict-parity rules, TDD loop, sklearn pitfalls, DoD) so you do not have to re-read AGENTS.md and MODEL_CARDS.md every time.
---

# QQQ model implementation — working contract

เอกสารนี้คือฉบับกลั่นของ `AGENTS.md` + `docs/plan/MODEL_CARDS.md` **พอสำหรับเขียนโมเดล 1 ตัวจนจบ**
กลับไปอ่านต้นฉบับเฉพาะเมื่อ: ต้องการสมการเต็มของโมเดลตัวนั้น (card) หรือสงสัยเหตุผลเชิงสถาปัตยกรรม (ADR)

## ก่อนเริ่มทุกตัว (อ่าน 2 ไฟล์เท่านั้น)

1. `docs/tasks/claude-model-progress.md` — ตารางสถานะ ดูว่าตัวไหนเสร็จ/ติดอะไร
2. `docs/models/<name>/MODEL_PROGRESS.md` — ถ้ามี (ตัวที่เคยลองแล้ว)
3. section ของโมเดลนั้นใน `docs/plan/MODEL_CARDS.md` — สมการ + grid + pitfalls

**ห้ามอ่าน** card ของโมเดลอื่น, progress ของโมเดลอื่น, หรือ implementation ของอีก agent
เว้นแต่มี dependency ที่บันทึกไว้

## วงจร 7 ขั้น (TDD — ห้ามข้ามขั้น)

```text
1. เขียน test ก่อน       tests/ml/<task>/test_<name>.py   → รันให้เห็น FAIL จริง
2. implement            src/ml/<task>/<name>.py
3. รัน test              pytest tests/ml/<task>/test_<name>.py -v
4. เพิ่ม 1 บรรทัดใน       src/ml/registry.py  (เรียงตามตัวอักษร)
5. รัน CLI 2 variant      src.ml.run train --model <name> --variant with_spike / non_spike
6. ตรวจ                  load_verification.json ต้องมี "passed": true
7. อัปเดต progress 2 ไฟล์ + commit
```

test ต้องมีครบ 3 ชนิด: **unit test กลไกภายใน** (ตัวอย่างเล็กที่คำนวณมือได้) ·
**parity กับ reference** · **save → load → predict เท่าเดิม**

## Interface (ตายตัว)

```python
from ..core.base import BaseModel

class YourModel(BaseModel):
    name = "<ตรงกับ key ใน configs/modeling.json และชื่อไฟล์>"
    task = "regression" | "classification" | "clustering"

    def __init__(self, hp=default): ...        # ทุก hyperparameter ต้องมี default
    def get_params(self) -> dict: ...          # JSON-serializable
    def fit(self, X, y=None) -> Self: ...      # clustering ส่ง y=None
    def predict(self, X) -> np.ndarray: ...    # reg: ค่า | cls: 0/1 | clustering: cluster id
    def decision_function(self, X): ...        # cls เท่านั้น — คะแนนดิบ
    def predict_proba(self, X): ...            # cls ที่ให้ความน่าจะเป็นได้จริง
    def state_dict(self) -> dict[str, np.ndarray]: ...
    def load_state(self, arrays, metadata) -> Self: ...
    def reference(self): ...                   # sklearn estimator ที่ตั้งค่าเทียบเท่า
```

**กฎที่พลาดบ่อย**

- `X` จาก loader เป็น **read-only** → `.copy()` ก่อนเขียนทับ
- โมเดลที่วนรอบต้องเก็บ `self.history_ = {"iteration": [...], "loss": [...]}` (โจทย์บังคับ performance curve)
- preprocessing **ไม่ใช่หน้าที่โมเดล** — trainer จัดการ standardize/PCA ให้แล้ว
- hyperparameter ต้องมี default ครบ เพราะ trainer เรียก `model_class()` เปล่า ๆ ตอน verify reload
- `random_state` default `42`

## Parity — เกณฑ์เดียวกันทุกตัว

| task | เกณฑ์ |
| --- | --- |
| regression | `allclose(rtol=1e-5, atol=1e-8)` |
| classification | label ตรง **100%** + score `allclose(rtol=1e-5)` |
| clustering | partition ตรงหลัง align cluster id |

ยกเว้นตัวเดียวคือ `svm` (hinge ไม่เรียบ → เทียบ objective) **ห้ามขยายข้อยกเว้นเอง ต้องถามก่อน**

**ถ้าไม่ผ่าน:** ห้ามลดเกณฑ์เงียบ ๆ · ห้ามเอาผล reference มาสวมแทน scratch · บันทึกสาเหตุแล้วแจ้ง

### RNG — สาเหตุอันดับ 1 ที่ parity พังทั้งที่อัลกอริทึมถูก

sklearn สร้างบน **legacy MT19937** ใช้ `np.random.RandomState(seed)`
**ห้ามใช้ `np.random.default_rng()`** (PCG64 คนละ stream) — กระทบ `random_forest`, `mlp`, `slp`, `kmeans`

### sklearn 1.9 API ที่เปลี่ยน

| เดิม | ใช้แทน |
| --- | --- |
| `penalty="l2"` | `l1_ratio=0` |
| `penalty=None` | `C=np.inf` |
| `AdaBoostClassifier(algorithm="SAMME.R")` | ถอดออกแล้ว `SAMME` เป็น default |

### Pitfalls ที่พิสูจน์แล้ว

| เรื่อง | กฎ |
| --- | --- |
| `LogisticRegression` | `C = 1/(λ·Σw)` ไม่ใช่ `1/λ` — C คูณกับ loss ไม่ใช่ penalty (ยืนยัน 1e-8) |
| least squares | ใช้ `lstsq`/`pinv` ไม่ใช่ normal equation — sklearn ใช้ SVD เหมือนกัน parity มาเอง |
| tree ทุกชนิด | ตั้ง `max_features=None` ชัดเจน ไม่งั้น sklearn สุ่มฟีเจอร์ |
| `Perceptron` | `tol=None` ไม่งั้น sklearn early-stop คนละรอบ |
| `GaussianNB` | `var_smoothing` คูณ **variance สูงสุดของทั้งชุด** ไม่ใช่ของแต่ละคลาส |
| `KNeighborsClassifier` | `algorithm="brute"` + `argsort(kind="stable")` ให้ tie-break ตรง |
| `ElasticNet` | objective หาร `1/(2n)` · `selection="cyclic"` |
| ข้อมูล near-singular | อย่าเขียน test ที่ cond สูงเกินจริง — คำตอบ LS ไม่ unique แล้ว solver คนละตัวให้คนละคำตอบโดยไม่มีใครผิด (ข้อมูลจริง cond(X) = 6.0) |
| **ระยะทางของ sklearn** | ใช้สูตรกระจาย `‖a‖²+‖b‖²−2a·b` → ระยะจากจุดถึง**ตัวเอง** ได้ `4.2e-08` ไม่ใช่ 0 (วัดแล้ว) ส่วน `norm(x−x)` ได้ 0 พอดี → **ทดสอบ parity บน held-out points เสมอ** ซึ่งเป็นสิ่งที่ trainer ทำจริง ไม่ใช่ predict ทับ train · กระทบ `knn` และ `kmeans` |
| **RNG ไม่ตรง sklearn** | วินิจฉัยก่อนแก้: เพิ่ม `n_init`/budget แล้ว gap ปิดไหม — ถ้าปิด = อัลกอริทึมถูก ปัญหาอยู่ที่ init · k-means++ ของ sklearn ใช้ `rng.choice(n, p=w/w.sum())` สำหรับ center แรก (ไม่ใช่ `randint`), `rng.uniform(size=...)` แต่ละรอบ, และ **ลบ column mean ก่อน cluster** — ต่างข้อใดข้อหนึ่ง stream เพี้ยนตั้งแต่ center แรก |
| **tie ที่ proba = 0.5** | `predict` ต้องใช้ `> 0.5` ไม่ใช่ `>= 0.5` — sklearn ใช้ `argmax` ซึ่งเลือกคลาส index ต่ำกว่าเมื่อเสมอ |
| **preprocess** | trainer อ่าน `preprocess` จาก config แล้ว (7 โมเดลเป็น `none`) ส่ง `preprocess=` override ได้ใน test · PCA บังคับ standardize เสมอเพราะต้องการ input ที่ centered |

## Grid

มาจาก `configs/modeling.json` → `models.<name>.grid` เสมอ **ห้ามแต่งเอง**
CLI อ่านให้อัตโนมัติ ไม่ต้องส่ง `--grid`

grid ปกติเป็นช่วงค่าที่ trainer คูณกันทุกคู่ · ถ้า config ใช้ `tuples` (ตอนนี้มีแค่ `xgboost`)
trainer จะแปลงแต่ละ tuple เป็น 1 candidate ตาม `parameters.tuple_fields` ให้เอง
**โมเดลไม่ต้องรับ `tuples` ใน constructor**

## คำสั่ง

```bat
.\.venv\Scripts\python.exe -m pytest tests/ml -q
.\.venv\Scripts\python.exe -m src.ml.run train --model <name> --variant with_spike
.\.venv\Scripts\python.exe -m src.ml.run train --model <name> --variant non_spike
```

ใช้ `.\.venv\Scripts\python.exe` เสมอ (Python 3.14.8) · basetemp ตั้งใน `pytest.ini` แล้ว ไม่ต้องส่ง flag

## ข้อมูล

8 features (ลำดับตายตัว): `return_1d`, `return_5d`, `historical_volatility_5d`,
`historical_volatility_20d`, `intraday_range`, `sma_ratio_5_20`, `rsi_14`, `volume_zscore_20`

targets: `target_volatility_5d` (reg) · `target_high_volatility` (cls)
variants: `with_spike` train 1,733 · `non_spike` train 1,520 · validation 371 · test 373 (เหมือนกันทั้งคู่)
Q75 = `0.2530580184684854` · class balance train 1300/433

### ⚠️ Accuracy หลอกตาในชุดนี้ — อย่ารายงานเป็นตัวหลัก

validation มี **337 neg / 34 pos (positive 9.2%)** → ทายว่า negative ทุกแถวได้ accuracy
**0.9084** ซึ่งสูงกว่าโมเดลจริงหลายตัว (kNN ได้ 0.8922)

ใช้ **AUC กับ F1 เป็นตัวตัดสิน** (ตรงกับ `selection.classification` ใน config ที่ใช้
average_precision → roc_auc) และรายงานต้องอธิบายประเด็นนี้ให้ชัด

## ห้ามแตะ

`data/**` · data pipeline ใน `src/` (cleaning/features/targets/split/spike) ·
**Test set จนถึง finalize** · ไฟล์โมเดลของอีก agent (`decision_tree`, `random_forest`,
`gradient_boosting`, `xgboost`, `adaboost`, `stacking`, `svm`, `mlp`)

`src/ml/core/**` freeze แล้วและอีก agent ใช้อยู่ — แก้ได้เฉพาะเมื่อพิสูจน์ว่าเป็นปัญหาร่วม **และแจ้งก่อน**

## DoD — ติ๊กครบ 6 ข้อถึงเรียกว่าเสร็จ

- [ ] ไฟล์โมเดลทำตาม interface ครบ
- [ ] test ผ่านครบ 3 ชนิด
- [ ] รัน trainer ได้ทั้ง 2 variant
- [ ] artifact ครบใน `outputs/modeling/<variant>/<task>/<name>/`
- [ ] `load_verification.json` → `"passed": true`
- [ ] อัปเดต `docs/models/<name>/MODEL_PROGRESS.md` + ตารางใน master index ด้วย **ตัวเลขจริง**

**ห้ามเคลมว่าเสร็จโดยไม่ได้รันคำสั่งและเห็น output จริง**
