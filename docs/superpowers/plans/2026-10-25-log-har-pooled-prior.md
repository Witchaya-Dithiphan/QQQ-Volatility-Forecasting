# แผน implementation: Log-HAR + pooled-prior transfer (regression นอกห้องเรียน)

> ต่อจาก [SSL → DAPT → Fine-tune Transformer](2026-10-25-ssl-dapt-transformer.md) ซึ่งพบว่า pretraining ไม่ช่วยงาน classification จึงย้ายมาลอง transfer กับงาน regression ด้วยโมเดลเชิงเส้นที่อธิบายกลไกได้

**Goal:** หาโมเดล regression นอกห้องเรียนที่ทำนาย `target_volatility_5d` ได้ดีกว่า Elastic Net (Test RMSE 0.1174 / 0.1230; Validation R² เพียง 0.017 / 0.091) โดยใช้ข้อมูลจากตลาดอื่นแบบ transfer learning

**Architecture:** HAR-RV (Corsi, 2009) ในสเกล log สร้างจาก 8 ฟีเจอร์เดิม โดยไม่แก้ชั้นข้อมูล แล้วประมาณสัมประสิทธิ์ของ QQQ ด้วย **transfer ridge** ที่ดึงเข้าหาสัมประสิทธิ์ร่วม β_pool จาก ETF 10 ตัว: `min ‖y − a − Xβ‖² + λ‖β − β_pool‖²` ทั้งหมดเป็น closed form ด้วย NumPy (`lstsq`) จึงไม่มี seed และเทียบ parity กับ `LinearRegression`/`Ridge` ของ scikit-learn ได้ตรง

**Tech stack:** Python 3.14, NumPy, pandas, pytest (scikit-learn เป็น reference เท่านั้น)

## ทำไมเลือกแนวนี้ (หลักฐานจากงานที่ทำแล้ว)

- **Target เบ้มาก:** skew ของ Train 2.80 และของ Validation 0.66 การ regression ค่าความผันผวนตรง ๆ ด้วย MSE จึงถูกไม่กี่วันสุดโต่งครอบงำ log ช่วยให้กระจายใกล้ปกติ
- **Elastic Net ไม่ได้ชนะชัด:** บน Test มีโมเดลในห้องเรียนที่ RMSE ดีกว่าในทั้งสอง variant ช่วงความเชื่อมั่นซ้อนกันหมด
- **ไม่ใช่ deep learning:** GRU/Transformer ไม่ได้ประโยชน์จาก pretraining ในงาน classification แต่โมเดลเชิงเส้นที่มีพารามิเตอร์น้อยได้ประโยชน์จากข้อมูลเพิ่ม 10 เท่าโดยตรง เพราะลดความแปรปรวนของสัมประสิทธิ์
- **พร้อมใช้:** ข้อมูล ETF 10 ตัวพร้อม cutoff และ manifest มีอยู่แล้วจากการทดลองก่อน (`src/ml/transfer_learning/sources.py`)

## การออกแบบที่ล็อกไว้ก่อนดูผล

เจ้าของโปรเจกต์ยืนยันแล้ว (2026-10-25): เลือก λ และตัดสินด้วย **RMSE** และ β_pool มาจาก **ETF 10 ตัวเท่านั้น**

**ฟีเจอร์ HAR (คำนวณในโมเดลจาก 8 ฟีเจอร์เดิม)**

| ชื่อ | สูตร | ความหมาย |
| --- | --- | --- |
| `d` | `log(max(abs(return_1d)·√252, 0.01))` | ความผันผวนรายวัน (floor 1% ต่อปี เพราะ Train มีวันที่ผลตอบแทนเป็น 0 พอดี 1 วัน) |
| `w` | `log(historical_volatility_5d)` | ความผันผวนรายสัปดาห์ |
| `m` | `log(historical_volatility_20d)` | ความผันผวนรายเดือน |
| `r` | `log(intraday_range)` | ความผันผวนจากช่วงราคาในวัน (range-based) |
| `lev` | `min(return_5d, 0)` | leverage effect: ผลตอบแทนติดลบทำให้ความผันผวนสูงขึ้น (Corsi & Renò, 2012) |

- target ที่ใช้ fit คือ `log(target_volatility_5d)`
- แปลงกลับเป็นสเกลเดิมด้วย `exp(ŷ + s²/2)` โดย s² คือ residual variance ของ QQQ Train (แก้ bias ของ log-normal)
- metric ทุกตัววัดบนสเกลเดิมเพื่อเทียบกับ 19 โมเดลได้

**Pooled prior (β_pool)**

- ใช้ ETF 10 ตัว **เท่านั้น** (ไม่รวม QQQ) เพื่อให้ prior เป็นความรู้จากตลาดอื่นจริง ๆ ส่วนข้อมูล QQQ เข้ามาผ่าน likelihood
- ใช้เฉพาะแถวที่ label สิ้นสุดก่อน 2023-08-18 ตาม cutoff เดิม
- ประมาณด้วย within estimator: ลบค่าเฉลี่ยของแต่ละ ETF ออกจากฟีเจอร์และ target แล้ว OLS โดยไม่มี intercept เพราะระดับความผันผวนเฉลี่ยของแต่ละตลาดต่างกัน แต่ความคงทน (persistence) ใกล้กัน
- prior เหมือนกันทั้งสอง variant

**Transfer ridge บน QQQ**

- `β = (XcᵀXc + λI)⁻¹(Xcᵀyc + λβ_pool)` โดย Xc, yc คือข้อมูล QQQ Train ที่ลบค่าเฉลี่ยแล้ว และ intercept ไม่ถูก penalize
- `λ = κ·n_QQQ` โดยเลือก κ ∈ {0, 0.03, 0.1, 0.3, 1, 3, 10, ∞} จาก **Validation RMSE** (ตรงกับ metric หลักของ regression ในโครงงาน)
- κ = 0 คือ OLS บน QQQ อย่างเดียว ส่วน κ = ∞ คือใช้ β_pool แล้วประมาณแค่ intercept จาก QQQ ทั้งสองปลายจึงเป็น ablation ในตัว

**Arms (ประกาศล่วงหน้า)**

| arm | ฟีเจอร์ | วิธี | คำถามที่ตอบ |
| --- | --- | --- | --- |
| `H3-qqq` | d, w, m | OLS บน QQQ | HAR คลาสสิกในสเกล log ดีกว่า Elastic Net ไหม |
| `HX-qqq` | d, w, m, r, lev | OLS บน QQQ | ฟีเจอร์ range และ leverage ช่วยไหม |
| `HX-pool` | d, w, m, r, lev | β_pool + intercept ของ QQQ | ความรู้จากตลาดอื่นอย่างเดียวพอไหม |
| **`HX-prior`** | d, w, m, r, lev | transfer ridge, κ จาก Validation | **โมเดลหลัก**: transfer ช่วยเพิ่มไหม |

**เทียบกับ** Elastic Net (headline), multiple_linear, polynomial, train mean และ persistence จาก artifact เดิม บนวันเดียวกัน ด้วย paired moving-block bootstrap (ก้อน 20 วัน, 2,000 รอบ) ของผลต่าง RMSE

**เกณฑ์ตัดสิน (ตั้งไว้ก่อนดูผล)**

- **"transfer ช่วย":** RMSE บน Validation ของ `HX-prior` ต่ำกว่า `HX-qqq` และช่วงของผลต่างไม่คร่อมศูนย์
- **"ดีกว่า Elastic Net":** RMSE ต่ำกว่าบน Validation และช่วงไม่คร่อมศูนย์
- รายงาน MAE, R² และ QLIKE (Patton, 2011) เป็นตัวชี้วัดรอง โดย QLIKE เป็น loss มาตรฐานของงานพยากรณ์ความผันผวนที่ไม่ถูกครอบงำด้วยวันสุดโต่ง

**มาตรฐานเดียวกับ 19 โมเดล:** รันทั้ง `with_spike` และ `non_spike`, เลือกบน Validation, ล็อก `decision.json` พร้อม hash ก่อนเปิด Test, เปิด Test ครั้งเดียวหลังได้รับอนุมัติ (ผลเป็น post-hoc) และประเมิน holdout 39 วันหลัง snapshot ซึ่งเป็นค่าต่อเนื่องทุกวัน จึงมีประโยชน์กว่าตอน classification

## File map

- `src/ml/log_har/__init__.py`
- `src/ml/log_har/features.py` — `har_features(X)`, `log_target(y)`, `back_transform(pred, s2)`
- `src/ml/log_har/model.py` — `LogHAR(BaseModel)` รองรับ `prior=None|β_pool`, `lam`; `pooled_prior(frames)`; `state_dict`/`load_state`; `reference()`
- `src/ml/log_har/experiment.py` — CLI `validate` (ทุก arm × 2 variants) และ `summarize` → `validation_summary.json`, `decision.json`, `report/coefficients.csv`, `report/lambda_path.csv`
- `src/ml/log_har/test_eval.py` — CLI `--allow-test` ครั้งเดียว + holdout + receipt (เพิ่ม path เข้า `TEST_SPLIT_ALLOWED`)
- `src/ml/log_har/figures.py` — สัมประสิทธิ์ QQQ-only เทียบ β_pool และ β_prior, Validation RMSE ตาม κ, predicted vs actual และ residual ตามเวลา
- `tests/ml/log_har/test_{features,model,experiment,test_eval}.py`
- outputs ที่ `outputs/experiments/log_har/` ส่วนเอกสารที่ `docs/models/log_har/EXPERIMENT.md`

reuse ของเดิม: `transfer_learning.sources.load_source` / `qqq_extended`, `core.data.load` / `load_test`, `core.bootstrap`, `core.report.regression_scores` / `baselines`, `core.preprocess` (ไม่จำเป็นเพราะฟีเจอร์เป็น log แล้ว)

## ลำดับงาน, เวลาโดยประมาณ และ skill

> กติกา: ประกาศ skill ก่อนเริ่มทุก task, ใช้ `.\.venv\Scripts\python.exe`, ห้าม commit โดยไม่สรุปให้เจ้าของโปรเจกต์ดูก่อน, ห้ามเคลมว่าเสร็จโดยไม่มี output จริง

### Task 0 — เตรียม (15 นาที)

- [ ] commit งาน transfer learning, notebook และรายงานที่ค้างอยู่ก่อน (ขออนุญาตก่อน) เพื่อแยก diff ของงานนี้
- [ ] **Skills:** `superpowers:finishing-a-development-branch` (หรือทำเทียบเท่าด้วยมือ)

### Task 1 — ฟีเจอร์และโมเดล (≈2 ชม.)

- [ ] เขียน test ก่อน:
  - floor ของ `d` ใช้กับวันที่ผลตอบแทนเป็น 0
  - `lev` เป็น 0 เมื่อผลตอบแทนเป็นบวก
  - back-transform ของค่าคงที่ให้ค่าเฉลี่ยของ log-normal ถูกต้อง
  - closed form ตรงกับ `lstsq`
  - κ = 0 ตรงกับ OLS ของ QQQ และ κ ใหญ่มากลู่เข้า β_pool
  - parity: κ = 0 เทียบ `LinearRegression`; κ > 0 เทียบ `Ridge(alpha=λ)` บน target `y − Xβ_pool` แล้วบวก β_pool กลับ (เกณฑ์ `allclose(rtol=1e-5, atol=1e-8)`)
  - save → load → predict ได้ค่าเท่าเดิม
- [ ] implement `features.py`, `model.py` จนผ่าน
- [ ] **Skills:** `qqq-model` (interface, parity, DoD), `superpowers:test-driven-development`, `fullstack-dev-skills:python-pro`

### Task 2 — pooled prior จาก ETF (≈45 นาที)

- [ ] สร้าง HAR features และ log target ของ ETF ทั้ง 10 ตัวจาก `load_source` (ตรวจ cutoff เดิม)
- [ ] within estimator → β_pool พร้อมบันทึกจำนวนแถวต่อ ETF และสัมประสิทธิ์ราย ETF (ใช้ดูว่า persistence ใกล้กันจริงไหม)
- [ ] test: ไม่มีแถวที่ label เลย cutoff, ค่าเฉลี่ยต่อ ETF หลัง demean เป็นศูนย์, β_pool ตรงกับ `LinearRegression(fit_intercept=False)` บนข้อมูลที่ demean แล้ว
- [ ] **Skills:** `fullstack-dev-skills:pandas-pro`, `validate-data`

### Task 3 — Validation และล็อก decision (≈1 ชม.)

- [ ] รัน 4 arms × 2 variants รวม κ path ทั้ง 8 ค่า แล้วบันทึก predictions, config, model และ load verification
- [ ] `summarize`:
  - leaderboard ที่มี RMSE, MAE, R² และ QLIKE พร้อม CI
  - paired difference เทียบ Elastic Net, multiple_linear, persistence และ `HX-qqq`
  - criteria และ `decision.json` ที่ล็อก hash ของ code, source และ model
- [ ] ตรวจ residual ว่าไม่มีรูปแบบผิดปกติ และดู κ path ว่าไม่ได้เลือกปลายเพราะ noise
- [ ] **Skills:** `fullstack-dev-skills:ml-pipeline`, `data:statistical-analysis`, `fullstack-dev-skills:the-fool` (หา leakage และ overclaim ก่อนล็อก)

### Task 4 — Gate + Test ครั้งเดียว + holdout (≈45 นาที)

- [ ] `code-review` บน diff, `pytest -q` ทั้ง suite, `git status` ต้องไม่มีการแก้ข้อมูลหลักหรือ `outputs/modeling/`
- [ ] dry-run `test_eval` ด้วย Test ปลอมจาก Validation ก่อน (แบบเดียวกับรอบ transfer learning)
- [ ] **ขออนุมัติเจ้าของโปรเจกต์** แล้วรัน `python -m src.ml.log_har.test_eval --allow-test` ครั้งเดียว
- [ ] คำนวณ RMSE/MAE/R² ซ้ำจาก CSV ด้วย scikit-learn
- [ ] **Skills:** `code-review`, `superpowers:verification-before-completion`, `validate-data`

### Task 5 — รายงานผล (≈2 ชม.)

- [ ] `docs/models/log_har/EXPERIMENT.md` + ลิงก์จาก `docs/README.md` และ `docs/models/PROGRESS_INDEX.md`
- [ ] notebook 03 (หัวข้อใหม่บน Validation) และ notebook 05 (หัวข้อใหม่บน Test/holdout) อ่าน artifact เท่านั้น พร้อม test ใน `tests/test_notebooks.py`
- [ ] รายงาน:
  - ภาคผนวก ฉ ในรูปแบบเดียวกับภาคผนวก จ (การเตรียมข้อมูล, การเทรน, ผล และรูปจาก notebook ผ่าน `export_notebook_figures.py`)
  - ปรับข้อเสนอแนะ 5.4.2
  - เพิ่มบรรณานุกรม Corsi & Renò (2012) และ Patton (2011)
  - สร้าง docx/pdf ใหม่ และไม่ทำสไลด์
- [ ] **Skills:** `dataviz` (กราฟ), `anthropic-skills:docx` (ถ้าต้องตรวจไฟล์ Word), `superpowers:verification-before-completion`

## ความเสี่ยงที่รู้ล่วงหน้า

- **Validation สงบมาก:** Validation มี skew 0.66 และไม่มี spike ส่วน Test มีช่วง spike โมเดลที่ชนะบน Validation อาจไม่ชนะบน Test (เกิดแล้วกับ Random Forest ของ classification) จึงต้องรายงาน Test แยกช่วงด้วย
- **การแปลงกลับจาก log:** การแปลงกลับจาก log ทำให้ทำนายวันสุดโต่งต่ำกว่าจริงได้ ถ้าทำให้ RMSE บน Test แย่ลงแม้ QLIKE ดีขึ้น ต้องรายงานตรง ๆ ไม่สลับ metric หลังเห็นผล
- **holdout เล็ก:** มี 39 วัน ช่วงความเชื่อมั่นยังกว้าง

## Verification

1. `pytest tests/ml/log_har -q` ผ่าน (features, closed form, parity, save/load, cutoff, การปฏิเสธของ test_eval)
2. `pytest -q` ทั้ง suite ผ่าน
3. `python -m src.ml.log_har.experiment validate` แล้ว `summarize` ได้ Validation 371 แถวครบทุก arm และ `decision.json`
4. หลังได้รับอนุมัติ: `test_eval --allow-test` ครั้งเดียว คำนวณซ้ำด้วย scikit-learn ตรงกัน และรันซ้ำถูกปฏิเสธ
5. `git status` ไม่มีการแก้ `data/**`, `outputs/modeling/**` หรือ `deliverables/slides.pptx`
