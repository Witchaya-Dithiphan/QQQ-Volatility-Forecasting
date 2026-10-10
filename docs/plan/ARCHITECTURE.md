# ARCHITECTURE — ชั้น modeling ของ QQQ Volatility Forecasting

> ขอบเขตของเอกสารนี้คือ **ชั้น modeling เท่านั้น** ชั้นข้อมูล (Phase 1/2) เสร็จและ freeze แล้ว
> สถานะจริงที่เอกสารนี้ตั้งอยู่บน: `docs/audit/REALITY_AUDIT.md`
> contract ที่ต้องทำตามตอนเขียนโค้ด: `AGENTS.md`

---

## 1. ภาพรวม

```
                    data/processed/**  (frozen, read-only, SHA-256 ผูกไว้)
                              │
                    ┌─────────▼──────────┐
                    │  core/data.py      │  load(variant) → Train / Validation
                    │                    │  load_test(allow_test=True) default-deny
                    └─────────┬──────────┘
                              │ X (read-only), y_reg, y_cls, dates
                    ┌─────────▼──────────┐
                    │ core/preprocess.py │  Standardizer → PCA  (fit บน Train เท่านั้น)
                    └─────────┬──────────┘
                              │
        ┌─────────────────────▼─────────────────────┐
        │              core/trainer.py              │
        │  grid search บน Validation → เลือก 1 ตัว   │
        │  → save → load → re-test → artifacts      │
        └──────┬───────────────────────┬────────────┘
               │                       │
     ┌─────────▼────────┐    ┌─────────▼─────────┐
     │  โมเดล scratch   │    │   reference()     │
     │  (NumPy ล้วน)    │◄──►│  sklearn/xgboost  │  core/compare.py: strict parity
     └─────────┬────────┘    └───────────────────┘
               │
     ┌─────────▼──────────────────────────────────┐
     │ core/metrics.py → core/figures.py → report │
     │  ROC/AUC, confusion, perf curve, leaderboard│
     └────────────────────────────────────────────┘
```

**หลักการ:** โมเดลรู้จักแค่ `fit/predict` ของตัวเอง ทุกอย่างรอบตัว (โหลดข้อมูล, แปลงค่า, ค้นหา
hyperparameter, วัดผล, เซฟ, วาดรูป) เป็นหน้าที่ของ `core/` ที่เขียนครั้งเดียวใช้กับทั้ง 19 โมเดล
ผลคือเพิ่มโมเดลใหม่ 1 ตัว = เขียนไฟล์เดียว ไม่ต้องแตะ core

---

## 2. โมดูลใน `src/ml/core/`

| ไฟล์ | หน้าที่ | ที่มา |
| --- | --- | --- |
| `base.py` | `BaseModel` interface (ดู `AGENTS.md` §4) | เขียนใหม่ |
| `data.py` | โหลด Train/Validation ตาม variant, ตรวจ SHA-256 + schema, Test แบบ default-deny | ย้ายจาก `modeling/datasets.py` แล้วตัด report-identity chain |
| `preprocess.py` | `Standardizer`, `PCA` | **ย้ายตรง ๆ** จาก `modeling/preprocessing.py` (มี 9 tests ผ่านแล้ว) |
| `metrics.py` | regression/classification/clustering metrics, losses, ROC/PR curve, threshold & candidate selection | **ย้ายตรง ๆ** จาก `modeling/metrics.py` (มี 11 tests ผ่านแล้ว) |
| `persist.py` | `save_npz`, `load_npz`, `save_reference`, `load_reference`, `verify_reload` | ย้ายจาก `modeling/persistence.py` บรรทัด 31–83 เท่านั้น |
| `trainer.py` | grid search บน Validation, เลือก candidate, สร้าง artifact ครบชุด, save→load→verify | เขียนใหม่ |
| `compare.py` | `assert_parity()` strict scratch ↔ reference | เขียนใหม่ |
| `figures.py` | ROC curve, confusion matrix, performance curve, leaderboard + รูปเปรียบเทียบ A↔B | เขียนใหม่ |
| `registry.py` | ทะเบียนชื่อโมเดล → คลาส เพื่อให้ CLI เรียกด้วยชื่อได้ | เขียนใหม่ |

`src/ml/run.py` คือ CLI เดียวของชั้นนี้ (`train` / `report` / `finalize`)

---

## 3. Decision records

### ADR-001 — เลิกใช้ `src/modeling/` ทั้ง package

**สถานะ:** ตัดสินใจแล้ว 2026-10-10

**บริบท:** commit `0e8b360` เพิ่มโค้ด 1,930 บรรทัดเป็น infrastructure ก่อนที่จะมีโมเดลแม้ตัวเดียว
audit พบว่าในจำนวนนั้นมีของที่ตอบโจทย์จริงราว 300 บรรทัด ส่วนที่เหลือเป็น governance layer
(config schema 984 บรรทัด, integrity seal 3 ชั้น, lifecycle state machine, resume gate,
finalize-test gate, test ledger, expected-runs 688 แถว, dependency smoke)

**ปัญหาที่พิสูจน์แล้วว่าบล็อกงานจริง ไม่ใช่ความเสี่ยงเชิงทฤษฎี:**

1. `artifacts.py:37` — `code_snapshot_hash()` รวม `src/models/**/*.py` เข้า hash ที่ใช้ตัดสินว่า
   run เดิมยังใช้ได้ไหม → **เขียนโมเดลตัวใหม่ทุกครั้งจะทำให้ผลของโมเดลก่อนหน้าทั้งหมดใช้ไม่ได้**
   พอเขียนครบ 19 ตัวต้องเทรนใหม่ทั้ง 38 runs
2. `test_gate.require_clean_git = true` + `--untracked-files=all` → notebook ที่เพิ่งรันหรือรูป
   ที่เพิ่งสร้าง 1 ไฟล์ทำให้ finalize ล็อกตัวเอง และ `configuration.py:59` เขียนกันไว้ไม่ให้แก้ flag นี้
3. แก้ hyperparameter grid 1 ค่า = ต้อง re-seal hash 3 ชั้นด้วย `config_sync --write-plan`
   ไม่งั้น `load_config()` โยน `ValueError` ทุกจุดเรียก
4. ไม่มีข้อใดในโจทย์ขอ provenance ledger หรือ integrity hash

**ทางเลือกที่พิจารณา:** (ก) เก็บทั้งหมดแล้วแก้เฉพาะจุดที่บล็อก (ข) trim เหลือแกน (ค) เลิกใช้ทั้ง package

**มติ:** (ค) — สร้าง `src/ml/` เป็นบ้านใหม่ที่ไม่ import `src/modeling` เลย แต่ **ย้ายโค้ดคณิตศาสตร์
ที่ทดสอบแล้ว** (`metrics.py`, `preprocessing.py`, save/load 5 ฟังก์ชัน, loader) มาเป็นของตัวเอง
แล้วลบ `src/modeling/` ทิ้ง เหตุผลที่ไม่เลือก (ก): ต่อให้ปิดสวิตช์ที่บล็อกได้ ต้นทุนการผลิต
artifact 8–12 ไฟล์ + manifest 25 ฟิลด์ต่อ run × 38 runs ยังอยู่ และไม่มีคะแนนผูกกับมัน

**ผลที่ยอมรับ:** เสีย provenance machinery ที่ละเอียด แลกกับเวลาที่เอาไปเขียนโมเดลและรูปซึ่งมีคะแนนจริง
`outputs/.test_ledger.json` ที่ถูก commit ไว้ (ผิดกฎ `.gitignore` ของโปรเจกต์เอง) จะถูกลบออกจาก git

---

### ADR-002 — Strict parity เกณฑ์เดียวกันทุกโมเดล

**บริบท:** ทางเลือกคือแบ่ง tier (closed-form เข้ม / iterative หลวม) หรือใช้เกณฑ์เดียว

**มติ:** เกณฑ์เดียวทุกตัวตาม `AGENTS.md` §6

**เหตุผล:** การแบ่ง tier เปิดช่องให้ "ปรับเกณฑ์จนผ่าน" ซึ่งทำลายคุณค่าทั้งหมดของการมี reference
ความต่างที่มักถูกอ้างว่าเลี่ยงไม่ได้ (tie-breaking, ลำดับ update, การ bin, ค่าเริ่มต้น weight)
แท้จริงแล้ว**กำหนดให้ตรงกันได้ทุกข้อ** ถ้าตั้ง seed และ hyperparameter ให้ตรงจริง — การไล่ให้ตรง
คือสิ่งที่พิสูจน์ว่าเราเข้าใจอัลกอริทึม ซึ่งตรงกับสิ่งที่โจทย์ให้คะแนน

**ผลที่ยอมรับ:** XGBoost, kernel SVM และ MLP จะใช้เวลา debug นานกว่าตัวอื่นมาก ถือเป็นความเสี่ยงอันดับ 1
ใน `RISKS_AND_DOD.md` และมีแผนสำรองกำกับไว้

---

### ADR-003 — Variant เป็น config switch ไม่ใช่โค้ดคนละชุด

**มติ:** `load(variant)` สลับเฉพาะ path ของ train CSV · Validation/Test เป็น full original เสมอ ·
ไฟล์โมเดลห้ามรู้จักคำว่า variant

**เหตุผล:** ทำให้ A↔B เป็น paired comparison ที่เปลี่ยนตัวแปรเดียวจริง ๆ และการรัน variant B
= เปลี่ยน flag แล้วรันซ้ำ ไม่ใช่งานเขียนโค้ดรอบใหม่

---

### ADR-004 — Elastic Net + AdaBoost เป็นโมเดล extracurricular

**บริบท:** โจทย์บังคับโมเดลนอกห้องเรียน 2 ตัว (regression 1 + classification 1 ต้องไม่ซ้ำกัน)
และบังคับว่า 2 ตัวนี้ต้องทำ **ทั้ง scratch และ ready-to-use** ทางเลือกที่ชั่งคือ GARCH / ELM

**มติ:** Elastic Net (regression) + AdaBoost (classification)

**เหตุผล:**
- ทั้งคู่มีใน sklearn → ทำข้อบังคับ "scratch + ready-to-use" ได้โดยไม่ต้องเพิ่ม dependency
  นอก `requirements-lock.txt` ที่เพิ่ง freeze (โจทย์ระบุ reference ว่า Scikit-learn/PyTorch เจาะจง)
- GARCH ตกไปเพราะไม่มีใน sklearn/PyTorch, ต้องลง `arch` เพิ่ม, และพยากรณ์จาก return series
  ไม่ใช่ 8 features ของเรา → benchmark เทียบโมเดลอื่นแบบ apple-to-apple ไม่ได้ และขัดข้อกำหนด
  "ใช้ feature ≥ 5" ของโจทย์
- มีเหตุผลเชิงข้อมูลหนุน (วัดจาก train set จริง): condition number = **35.6**,
  `historical_volatility_5d ~ 20d` |r| = **0.804** → Elastic Net แก้ multicollinearity ตรงจุด;
  target มี kurtosis = **16.26** → AdaBoost ที่ใช้ exponential loss เป็นโมเดลที่ทฤษฎีทำนายว่า
  จะเสียหายจาก spike มากที่สุด ทำให้มันเป็นตัวเอกของการเปรียบเทียบ A↔B โดยธรรมชาติ

---

### ADR-005 — เก็บ default-deny ของ Test ไว้ แต่ทิ้ง gate machinery

**มติ:** `load_test()` ต้องส่ง `allow_test=True` อย่างชัดแจ้งถึงจะอ่านได้ และเรียกได้จาก
`run.py finalize` เท่านั้น — แต่ไม่มี authorization object, ledger หรือ clean-git check

**เหตุผล:** คุณค่าจริงของ gate คือกัน Test รั่วเข้า loop เลือกโมเดล ซึ่ง flag เดียวก็ได้ผลนั้นแล้ว
ส่วน ledger/clean-git เป็นพิธีกรรมที่ล็อกตัวเองบ่อยกว่าที่มันกันอุบัติเหตุได้จริง

---

### ADR-006 — pytest basetemp ต้องอยู่ไดรฟ์เดียวกับโปรเจกต์

**บริบท:** โปรเจกต์อยู่ `D:` แต่ pytest ใช้ tmp บน `C:` ทำให้ 5 tests fail ด้วย
`Report and referenced artifact must be on the same drive` ซึ่ง**ไม่ใช่ bug** แต่เป็นข้อจำกัดของ
`os.path.relpath` บน Windows ที่คำนวณ path ข้ามไดรฟ์ไม่ได้

**มติ:** ตั้ง `addopts = --basetemp=.pytest_tmp` ใน `pytest.ini` และ gitignore โฟลเดอร์นั้น

**เหตุผล:** เป็น path สัมพัทธ์ในรีโป จึงใช้ได้กับทุกเครื่องของทีมโดยไม่ต้องตั้งค่าเอง
พิสูจน์แล้วว่าทำให้ผลเป็น **459 passed, 3 skipped, 0 failed**

---

## 4. การไหลของงานหนึ่งรอบ (trainer)

```
1. load(variant)                      → Train, Validation
2. Standardizer.fit(Train.X)          → transform Train + Validation
   [ถ้าโมเดลใช้ PCA] PCA.fit(Z_train) → transform ทั้งคู่
3. for แต่ละชุด hyperparameter ใน grid:
       model.fit(Train)  →  predict(Validation)  →  metrics
       เก็บลง search_results.json
4. select_candidate()                 → ผู้ชนะ 1 ตัว (เกณฑ์: regression=rmse, classification=AP→AUC)
5. [classification] select_threshold() บน Validation เท่านั้น
6. save_npz(model.state_dict())  +  save_npz(preprocessor)
7. load_npz → สร้างโมเดลใหม่ → predict ซ้ำ → verify_reload()  → load_verification.json
8. เขียน validation_metrics.json, validation_predictions.csv, history.json
```

Test ไม่ปรากฏในลูปนี้เลย — เข้าได้เฉพาะขั้น `finalize` หลังเลือกโมเดลเสร็จแล้วทุกตัว

---

## 5. สิ่งที่ตั้งใจ **ไม่** ทำ (YAGNI)

ไม่ทำ cross-validation (ข้อมูลเป็น time series ใช้ chronological split ตามที่ตรวจรับไว้แล้ว) ·
ไม่ทำ hyperparameter search อัตโนมัติแบบ Bayesian (ใช้ grid เล็กที่กำหนดไว้ล่วงหน้า) ·
ไม่ทำ model registry/versioning ·
ไม่ทำ experiment tracking server ·
ไม่ทำ CI pipeline ·
ไม่ทำ resume/checkpoint (รันใหม่ทั้งตัวถูกกว่าการจัดการ state)
