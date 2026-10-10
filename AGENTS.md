# AGENTS.md — contract กลางสำหรับ AI agent ทุกตัวในโปรเจกต์นี้

> เอกสารนี้คือ **แหล่งความจริงเดียว** สำหรับวิธีเขียนโค้ดในโปรเจกต์นี้
> Codex CLI อ่านไฟล์นี้โดยอัตโนมัติ · `CLAUDE.md` ชี้มาที่ไฟล์นี้ · ห้ามสร้างเอกสารคู่ขนานที่อาจขัดกันเอง
>
> ทีมมี 2 คนทำงานคนละ branch ด้วย agent คนละตัว **คุณจะไม่เห็นงานของอีกฝั่งจนกว่าจะ merge**
> ทุกอย่างในไฟล์นี้จึงเป็นข้อผูกพัน ไม่ใช่คำแนะนำ

---

## 1. กฎที่ห้ามละเมิดเด็ดขาด

| # | กฎ | เหตุผล |
| --- | --- | --- |
| 1 | **ห้ามแก้ชั้นข้อมูล** — `data/**`, `src/clean_data.py`, `build_features.py`, `build_targets.py`, `split_data.py`, `detect_spikes.py`, `build_experiment_datasets.py`, `run_data_pipeline.py`, `run_spike_analysis.py` | ผ่านการตรวจรับและผูก SHA-256 ไว้แล้ว อ่านได้อย่างเดียว |
| 2 | **ห้ามแตะ Test set** จนกว่าจะถึงขั้นตอน finalize | ต้องเลือก hyperparameter จาก Validation เท่านั้น ไม่งั้นผลที่รายงานเป็นโมฆะ |
| 3 | **ทุกโมเดลต้องเขียนเอง (from scratch) ด้วย NumPy** | ข้อบังคับของโจทย์: *"Every model must be built from scratch"* |
| 4 | ใช้ sklearn/XGBoost ได้เฉพาะเป็น **reference เพื่อยืนยันผล** และต้องอยู่ในไฟล์ `reference()` เท่านั้น | โจทย์หน้า 5: *"อนุญาตให้มีโมเดลสำเร็จรูปประกอบด้วยได้เพื่อยืนยันผล"* |
| 5 | **ห้าม import `src/modeling/`** — package นี้ถูกยกเลิกแล้ว | ดู ADR-001 ใน `ARCHITECTURE.md` |
| 6 | ห้ามเคลมว่างานเสร็จโดยไม่ได้รันคำสั่งและแสดง output จริง | เอกสารรอบก่อนเคลมตัวเลขที่พิสูจน์ซ้ำไม่ได้ ทำให้ต้อง audit ใหม่ทั้งโปรเจกต์ |
| 7 | 1 โมเดล = 1 ไฟล์ ห้ามแก้ไฟล์โมเดลของอีกฝั่ง | เลี่ยง merge conflict ระหว่าง 2 branch |

---

## 2. Environment

```bat
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Python ที่ freeze ไว้คือ **3.14.8** ใช้ `.\.venv\Scripts\python.exe` เสมอ อย่าเรียก `python` ลอย ๆ

**ครั้งแรกหลัง clone** ต้องสร้าง Phase 1 reproduction ด้วย ไม่งั้น Phase 2 test จะ fail:
```bat
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp/phase1-reproduction" --overwrite-generated
```
ห้ามเปลี่ยนชื่อโฟลเดอร์นี้และห้ามลบทิ้ง — `outputs/reports/spike_input_contract.json` เก็บ path นี้ไว้เป็น provenance

---

## 3. ข้อมูลที่ใช้

**Features 8 ตัว** (ลำดับนี้ตายตัว):
`return_1d`, `return_5d`, `historical_volatility_5d`, `historical_volatility_20d`,
`intraday_range`, `sma_ratio_5_20`, `rsi_14`, `volume_zscore_20`

**Targets:** `target_volatility_5d` (regression) · `target_high_volatility` (classification, Q75 จาก Original Train เท่านั้น)

**Variants 2 ชุด** — เป็น config switch เท่านั้น **ห้าม hard-code variant ในไฟล์โมเดล**

| variant | train | rows | Validation / Test |
| --- | --- | ---: | --- |
| `with_spike` | `data/processed/experiments/with_spikes/train.csv` | 1,733 | เหมือนกันทั้งสอง variant: |
| `non_spike` | `data/processed/experiments/non_spike/train.csv` | 1,520 | validation 371 · test 373 (full original) |

ค่าที่ถูกต้อง: Q75 = `0.2530580184684854` · class balance train = 1300/433 · spike threshold = `0.0465436445413787`

---

## 4. Interface ที่ทุกโมเดลต้องทำตาม

```python
# src/ml/core/base.py
class BaseModel:
    name: str          # "elastic_net" — ต้องตรงกับชื่อไฟล์
    task: str          # "regression" | "classification" | "clustering"

    def __init__(self, **hyperparams) -> None: ...
    def get_params(self) -> dict: ...                  # hyperparameter ทั้งหมด (JSON-serializable)

    def fit(self, X, y=None) -> "BaseModel": ...       # clustering ส่ง y=None
    def predict(self, X) -> np.ndarray: ...            # reg: ค่าจริง | cls: label 0/1 | clustering: cluster id

    # classification เท่านั้น
    def decision_function(self, X) -> np.ndarray: ...  # คะแนนดิบ (margin / logit)
    def predict_proba(self, X) -> np.ndarray: ...      # ถ้าโมเดลให้ความน่าจะเป็นได้จริง ไม่งั้น raise NotImplementedError

    # save / load — ข้อบังคับของโจทย์
    def state_dict(self) -> dict[str, np.ndarray]: ... # เฉพาะพารามิเตอร์ที่เรียนรู้มา
    def load_state(self, arrays: dict, metadata: dict) -> "BaseModel": ...

    # reference เพื่อยืนยันผล
    def reference(self): ...                           # คืน sklearn/xgboost estimator ที่ตั้งค่าให้เทียบเท่ากัน
```

**กฎเพิ่มเติม**

- โมเดลที่เทรนแบบวนรอบ (SLP, MLP, Perceptron, GB, XGBoost, AdaBoost, Elastic Net) ต้องเก็บ
  `self.history_ = {"iteration": [...], "loss": [...]}` ระหว่าง fit — โจทย์บังคับให้มี **performance curve**
- **hyperparameter ทุกตัวใน `__init__` ต้องมีค่า default** เพราะ trainer สร้างอินสแตนซ์เปล่า
  ด้วย `model_class()` แล้วเรียก `load_state()` ตอนพิสูจน์ว่า save→load แล้วทำนายเท่าเดิม
  ถ้า `__init__` บังคับให้ส่งอาร์กิวเมนต์ ขั้นตอนนั้นจะพัง
- `X` ที่ได้จาก loader เป็น **read-only** (`setflags(write=False)`) ห้ามเขียนทับ ให้ `.copy()` ก่อนถ้าจำเป็น
- ห้ามใช้ `np.random` แบบไม่ตั้ง seed — รับ `random_state` ผ่าน `__init__` (default `42`)
- **ใช้ `np.random.RandomState(seed)` ไม่ใช่ `np.random.default_rng(seed)`** สำหรับโมเดลที่ต้องเทียบกับ
  sklearn เพราะ sklearn ทั้งไลบรารีสร้างบน legacy MT19937 ซึ่งเป็นคนละ stream กับ PCG64 ของ `default_rng`
  พิสูจน์แล้ว: `RandomState(42).randint(0, 200, 200)` ให้ `[102 179 92 14 ...]` ตรงกับ
  `sklearn.ensemble._forest._generate_sample_indices(42, 200, 200, None)` เป๊ะ ขณะที่
  `default_rng(42).integers(0, 200, 200)` ให้ `[17 154 130 87 ...]` — ถ้าใช้ตัวหลัง parity
  ของ Random Forest, MLP, SLP และ k-Means จะไม่มีทางผ่านไม่ว่าอัลกอริทึมจะถูกแค่ไหน
- preprocessing (standardize / PCA) **ไม่ใช่หน้าที่ของโมเดล** — trainer เป็นคนจัดการ fit บน train แล้ว transform ที่เหลือ

---

## 5. Definition of Done ของ "หนึ่งโมเดล"

โมเดลจะถือว่าเสร็จก็ต่อเมื่อครบทั้ง 6 ข้อ และ **คุณรันแล้วเห็น output จริง**

1. ไฟล์ `src/ml/<task>/<name>.py` มีคลาสที่ทำตาม interface ข้อ 4 ครบ
2. `tests/ml/<task>/test_<name>.py` ผ่าน ประกอบด้วย
   - **unit test กลไกภายใน** — ทดสอบส่วนที่เป็นหัวใจของอัลกอริทึมบนตัวอย่างเล็กที่คำนวณมือได้
     (เช่น gini/entropy ของ split, gradient+hessian ของ GB, margin/KKT ของ SVM, gradient check ของ MLP)
   - **parity test กับ reference** — strict (ดูข้อ 6)
   - **save → load → predict ซ้ำได้ผลเท่าเดิม**
3. รันผ่าน trainer ได้ทั้ง 2 variant
4. มี artifact ครบใน `outputs/modeling/<variant>/<task>/<name>/`
5. ลงแถวใน leaderboard แล้ว
6. มี model card ใน `MODEL_CARDS.md` ที่อัปเดตค่าจริงแล้ว (ไม่ใช่ค่าที่วางแผนไว้)

---

## 6. Strict parity — นิยามที่ตกลงกันแล้ว

**ทุกโมเดลใช้เกณฑ์เดียวกัน ไม่มีการผ่อนปรนรายตัว**

```python
# src/ml/core/compare.py
assert_parity(scratch, reference, X, y, task=...)
```

| task | เกณฑ์ |
| --- | --- |
| regression | `np.allclose(scratch.predict(X), reference.predict(X), rtol=1e-5, atol=1e-8)` |
| classification | label ตรงกัน **100%** และ `decision_function`/`predict_proba` `allclose(rtol=1e-5)` |
| clustering | partition ตรงกันหลัง align cluster id (ชื่อ cluster สลับได้ แต่การแบ่งกลุ่มต้องเหมือน) |

ทดสอบบน **synthetic data ที่ควบคุมได้** เป็นด่านแรก แล้วตามด้วย **train set จริง**

**ถ้า parity ไม่ผ่าน:** ห้ามลดเกณฑ์เงียบ ๆ และห้ามเอาผลของ reference มาสวมแทน scratch
ให้หยุด บันทึกสาเหตุที่วิเคราะห์ได้ลง model card แล้วแจ้งเจ้าของโปรเจกต์
(ความต่างที่พบบ่อยและต้องจัดการให้ตรง ไม่ใช่ปล่อยผ่าน: tie-breaking ตอนเลือก split,
ลำดับการ update ของ coordinate descent, การ bin ค่าต่อเนื่อง, ค่าเริ่มต้นของ weight, นิยาม regularization ที่หาร n หรือไม่หาร)

---

## 7. Artifact layout

```
outputs/modeling/<variant>/<task>/<model>/
    config.json                 hyperparameter ที่ใช้ + seed + variant
    model.npz (+ .json)         พารามิเตอร์ที่เรียนรู้ (save_npz)
    preprocessor.npz (+ .json)  Standardizer / PCA ที่ fit จาก train
    search_results.json         ทุก candidate ใน grid พร้อม validation metric
    validation_metrics.json     metric ของ candidate ที่ชนะ
    validation_predictions.csv  Date, target, raw_score, final_prediction
    load_verification.json      ผลพิสูจน์ว่า save→load แล้ว predict เท่าเดิม
    history.json                loss ต่อ iteration (เฉพาะโมเดลที่เทรนวนรอบ)
```
ตอน finalize จึงเพิ่ม `test_metrics.json`, `test_predictions.csv`

---

## 8. Git workflow

```
main
 ├── feat/ml-core                   ← framework กลาง (ต้อง merge ก่อนใครเริ่มเขียนโมเดล)
 ├── feat/models-linear-family      ← Claude Code
 └── feat/models-tree-family        ← Codex CLI
```

- commit เล็ก ๆ ข้อความอธิบาย **ทำไม** ไม่ใช่ **ทำอะไร**
- ห้าม force push · ห้าม rewrite history บน `main`
- `rebase` จาก `main` บ่อย ๆ เพื่อให้ core ล่าสุดอยู่กับตัว
- เปิด PR เมื่อจบกลุ่มโมเดล ไม่ใช่ทีละตัว
- **ห้าม commit:** `.venv/`, `tmp/`, `outputs/modeling/**`, ไฟล์ข้อมูลที่ generate ได้

**การแบ่งงาน** (ห้ามข้ามเขต)

**ชื่อโมเดลต้องตรงกับ key ใน `configs/modeling.json` เป๊ะ** (ชื่อไฟล์ = ชื่อ key = `BaseModel.name`)

| Claude Code — `feat/models-linear-family` | Codex CLI — `feat/models-tree-family` |
| --- | --- |
| `regression/simple_linear.py` | `classification/decision_tree.py` |
| `regression/multiple_linear.py` | `classification/random_forest.py` (+ PCA) |
| `regression/polynomial.py` | `classification/gradient_boosting.py` |
| `regression/elastic_net.py` ★ | `classification/xgboost.py` |
| `classification/logistic.py` | `classification/adaboost.py` ★ |
| `classification/gaussian_nb.py` | `classification/stacking.py` |
| `classification/knn.py` | `classification/svm.py` (+ PCA) |
| `classification/perceptron.py` | `classification/mlp.py` |
| `classification/slp.py` | |
| `clustering/kmeans.py` | |
| `clustering/agglomerative.py` | |
| รวม 11 | รวม 8 |

★ = โมเดล extracurricular ที่โจทย์บังคับให้ทำ **ทั้ง scratch และ ready-to-use** และต้องอธิบายกลไกภายในอย่างละเอียดในรายงาน

---

## 9. คำสั่งที่ใช้บ่อย

```bat
rem รัน test ทั้งหมด (basetemp ต้องอยู่ไดรฟ์เดียวกับโปรเจกต์)
.\.venv\Scripts\python.exe -m pytest -q

rem เทรนโมเดลเดียว
.\.venv\Scripts\python.exe -m src.ml.run train --model elastic_net --variant with_spike

rem เทรนทั้งชุด
.\.venv\Scripts\python.exe -m src.ml.run train --all --variant with_spike

rem ตาราง leaderboard + รูปทั้งหมด
.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike

rem finalize (แตะ Test ครั้งเดียวตอนจบ)
.\.venv\Scripts\python.exe -m src.ml.run finalize --allow-test
```

---

## 10. สิ่งที่โจทย์ให้คะแนน (อย่าลืมระหว่างเขียนโค้ด)

- **Metrics ที่ต้องมีครบ:** Loss · Confusion matrix · Accuracy · Precision · Sensitivity ·
  Specificity · TNR · F1 · ROC + AUC · Performance curve · R² (regression)
- **Benchmarking + analysis + discussion** — ไม่ใช่แค่พิมพ์ตัวเลข ต้องมีการเปรียบเทียบและอภิปราย
- **save-load model แล้ว test ซ้ำ** เป็นข้อบังคับที่เขียนไว้ตรง ๆ ในโจทย์
- โมเดล extracurricular ต้อง **อธิบายกลไกภายในอย่างละเอียด**

รายละเอียดเต็มอยู่ใน `MODEL_CARDS.md`, `ARCHITECTURE.md` และ `REPORT_AND_SLIDES_OUTLINE.md`
