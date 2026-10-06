# QQQ Volatility Forecasting

โปรเจกต์นี้ใช้ข้อมูล QQQ รายวันเพื่อสร้าง features และพยากรณ์ realized volatility ล่วงหน้า 5 trading days ข้อมูลแบ่งเป็นสอง workflow ที่มีวัตถุประสงค์ต่างกันอย่างชัดเจน

## Historical snapshot สำหรับ reproducibility

ผลใน reports และ Notebook ปัจจุบันอ้างอิง Snapshot เดิมจำนวน **2,512 แถว** ช่วง **2016-08-31 ถึง 2026-08-28** ที่ `data/raw/qqq_daily.csv` เท่านั้น ไฟล์นี้ต้องมี SHA-256:

```text
649e1db1b79c0990ea947a4ea162d6836b9bbdda2af24bc5ab6ae66cb83b6b13
```

Manifest อยู่ที่ `data/manifests/qqq_daily_snapshot.json` ตรวจ Snapshot โดยไม่เรียก Live API ได้ด้วย:

```powershell
python src/download_qqq_data.py verify-snapshot
```

ห้ามเปิดแล้วบันทึกใหม่ แปลง format เรียงแถวใหม่ หรือนำข้อมูลจากคนละรอบมาต่อกัน เพราะการเปลี่ยน byte ใด ๆ จะทำให้ checksum และ provenance ต่างจาก Dataset ที่ใช้สร้างผลการทดลอง

## Live downloader สำหรับ dataset refresh

ใช้คำสั่งต่อไปนี้เมื่อต้องการข้อมูลล่าสุด:

```powershell
python src/download_qqq_data.py refresh-latest
```

ข้อมูลจะถูกบันทึกแยกเป็น `data/raw/qqq_daily_latest.csv` และสร้าง report ที่ `outputs/reports/qqq_download_report.json` ซึ่งมีเวลาที่ดาวน์โหลด ช่วงวันที่ จำนวนแถว columns และ SHA-256 ของไฟล์ที่ได้ Downloader ปฏิเสธการเขียนผล refresh ทับ `qqq_daily.csv` เสมอ แม้ใช้ `--overwrite`

Nasdaq จำกัดข้อมูลย้อนหลังเป็น rolling window ดังนั้น [Live Historical Data](https://www.nasdaq.com/market-activity/etf/qqq/historical) ไม่รับประกันว่าจะสร้าง Historical Snapshot เดิมซ้ำได้ จำนวนแถวและวันเริ่มต้นของ latest dataset จึงไม่ถูกกำหนดตายตัว และ latest dataset ไม่ใช่ตัวแทนของ Snapshot เดิม

## Phase 1 baseline data pipeline

ติดตั้ง dependencies แล้วรันจาก project root ซึ่งมี `config.py`:

```powershell
python -m pip install -r requirements.txt
python -m src.run_data_pipeline
```

Runner ตรวจ Snapshot และ Manifest ก่อนเขียนไฟล์ จากนั้นเรียก cleaning → features → regression target → chronological split → classification labels ตามลำดับเดิม โดยไม่ดาวน์โหลดข้อมูล Features และ target คำนวณบน timeline เต็มก่อนเตรียม modeling rows; split ใช้ gap 5 วันซื้อขายระหว่าง Train/Validation และ Validation/Test ค่า Q75 สำหรับ label คำนวณจาก Train เท่านั้น และใช้เงื่อนไข `target_volatility_5d > Q75` แบบ strict `>` Baseline นี้ยังเก็บ spike ทุกแถวไว้

ถ้าต้องการทดลองโดยไม่แตะ generated outputs ที่ project root ให้กำหนด output root ใหม่:

```powershell
python -m src.run_data_pipeline --output-root tmp/phase1-baseline
```

ค่า `--output-root` จะสร้าง `data/interim`, `data/processed` และ `outputs/reports` ใต้ root ที่ระบุ โดยยังใช้ Snapshot/Manifest ค่า default จาก `config.py` หาก output ที่จะเขียนมีอยู่แล้ว runner จะหยุดที่ preflight ก่อนเขียนทับ หากตั้งใจรันซ้ำใน root เดิม ให้ใช้ `--overwrite-generated` ซึ่งอนุญาตเฉพาะ 14 generated artifacts ที่ runner ประกาศไว้:

```powershell
python -m src.run_data_pipeline --output-root tmp/phase1-baseline --overwrite-generated
```

หากต้องการอัปเดต generated outputs ที่ project root โดยตรง ให้รัน `python -m src.run_data_pipeline --overwrite-generated` หลังตรวจว่า outputs เดิมเขียนทับได้ ไม่มี flag ใดอนุญาตให้เขียนทับ Snapshot หรือ Manifest และ `--output-root` ใต้ Raw/Manifest จะถูกปฏิเสธ หาก stage กลางล้มเหลว runner หยุดทันที พร้อมแจ้งชื่อ stage; artifacts ของ stage ก่อนหน้าอาจค้างอยู่

### Artifacts ที่ได้

เมื่อใช้ค่า default paths ใน `config.py` จะได้ไฟล์ต่อไปนี้; เมื่อใช้ `--output-root` ให้เติม root นั้นไว้หน้าทุก path ในตาราง

| ประเภท | Files |
| --- | --- |
| Interim CSV | `data/interim/qqq_clean.csv`, `data/interim/qqq_features.csv`, `data/interim/qqq_regression_target.csv` |
| Original splits | `data/processed/train.csv`, `data/processed/validation.csv`, `data/processed/test.csv` |
| Labeled splits | `data/processed/train_labeled.csv`, `data/processed/validation_labeled.csv`, `data/processed/test_labeled.csv` |
| Reports | `outputs/reports/cleaning_report.json`, `outputs/reports/feature_report.json`, `outputs/reports/regression_target_report.json`, `outputs/reports/data_split_report.json`, `outputs/reports/classification_threshold.json` |

สำหรับ Snapshot ที่ pin อยู่ คาดว่า Train/Validation/Test มี **1,733 / 371 / 373 แถว** ตามลำดับ `regression_target_report.json` อธิบาย target ที่ใช้ `return_1d` ของวัน `t+1` ถึง `t+5`, มี 2,507 ค่า valid และ NaN 5 แถวท้ายเพราะข้อมูลอนาคตไม่ครบ Report นี้เป็นสถิติเชิงพรรณนาของ regression target; classification threshold อยู่ใน `classification_threshold.json` และ fit จาก Train เท่านั้น

Path ภายใน regression target, split และ classification reports เป็น path ที่อ้างจากโฟลเดอร์ของ report (`"paths_relative_to": "report_directory"`) เช่น `../../data/processed/train.csv` จึงย้าย artifact tree ทั้งชุดไปตำแหน่งอื่นได้โดย path ยังชี้ถูก Reports เก่าที่สร้างก่อนการเปลี่ยนนี้อาจยังมี absolute path ของเครื่องเดิม; ให้รัน pipeline ใน output root ใหม่เพื่อสร้าง reports รูปแบบปัจจุบัน

ตรวจ test suite ได้ด้วย:

```powershell
python -m pytest -q
```

## Phase 2 spike-analysis readiness

ก่อน implement experiment datasets ได้ตรวจ Phase 1 ใหม่ใน isolated output root แล้ว พบว่า reports ที่สร้างจากโค้ดปัจจุบันใช้ portable relative paths และ labeled CSV ทั้งสาม split มี SHA-256 ตรงกับชุดที่ project root แบบ byte-for-byte รายละเอียด hashes, authoritative inputs, boundary policy และคำสั่ง reproduce อยู่ใน `PHASE2_SPIKE_READINESS.md`

M1 Baseline Input Contract implement แล้วใน `src/build_spike_input_contract.py` โดยอ่าน baseline labeled splits และ saved reports โดยไม่แก้ไข ตรวจ schema, numeric/finite values, dates, split gaps, Original Train Q75, strict labels และ SHA-256 จากนั้นเขียนเฉพาะ generated `spike_input_contract.json` พร้อมบันทึก saved-report portability discrepancy โดยไม่แก้ reports เดิม ก่อนรันให้สร้าง Phase 1 reproduction ใน output root ใหม่ แล้วสร้าง generated contract ดังนี้:

```powershell
python -m src.run_data_pipeline --output-root tmp/phase2-m1-baseline
python -m src.build_spike_input_contract --reproduced-root tmp/phase2-m1-baseline
```

ผลอยู่ที่ `outputs/reports/spike_input_contract.json` และถูก ignore โดย Git คำสั่งจะไม่ overwrite โดย default; หากตั้งใจสร้าง contract ซ้ำใช้ `--overwrite-generated` ซึ่งอนุญาตเฉพาะไฟล์ contract ที่ประกาศไว้

M2 direct detector implement แล้วใน `src/detect_spikes.py`: fit Q1/Q3/IQR/threshold จาก M1-verified Original Train และ apply fitted threshold เดิมกับ split ใดก็ได้แบบ strict `>` โดยไม่ refitหรือแก้ input ส่วน `src/spike_contract.py` เป็นเจ้าของ affected window `[s-5, s+19]` เพียงแห่งเดียวและ re-export Series API เดิมเพื่อ compatibility ปัจจุบันยังไม่มี With-Spike/Non-Spike datasets, market-event/data-quality audit, Phase 2 experiment runner/reports/figures หรือ model training

## Original Course Requirements

Model plan ด้านล่างยึด requirement summary ที่ระบุแหล่งอ้างอิงเป็น
`_69-1-01076641_Project-description.pdf` ดังนี้:

- หน้า 2: รายการโมเดลในรายวิชา ได้แก่ Linear/Multiple/Polynomial Regression
  และโมเดล Classification ที่ระบุไว้ รวมทั้งกำหนดให้มี better/extracurricular
  model
- หน้า 3: ต้องมี new Regression model และ new Classification model ที่ต่างกัน,
  อธิบาย internal mechanism, มีทั้ง from-scratch และ ready-to-use
  implementation, quantitative benchmarking และ discussion; outputs ที่กำหนด
  รวม Loss, Confusion Matrix, Accuracy, Precision, Sensitivity, Specificity,
  True Negative Rate, F1-score, ROC curve, AUC, performance curve และ R-Square
- หน้า 5: Regression, Classification และ better models ต้องสร้าง from scratch;
  ใช้ Scikit-learn/PyTorch ประกอบเพื่อยืนยันและ benchmark ได้ และ rubric ให้คะแนน
  Accuracy, Loss, F1 และ performance curve

> **Source-verification limitation:** PDF ต้นฉบับทั้ง
> `_69-1-01076641_Project-description.pdf`, `Machine Learning (1).pdf` และ
> `Machine Learning (2).pdf` ไม่ได้อยู่ใน repository หรือ attachment ที่ตรวจได้ใน
> รอบนี้ จึงยังไม่สามารถตรวจข้อความตามหน้าหรือค้นคำใน Course PDFs โดยตรงได้
> Page references ข้างต้นมาจาก requirement summary ที่ส่งมากับงานและต้องตรวจซ้ำกับ
> PDF ก่อนส่งงานขั้นสุดท้าย โดยเฉพาะสถานะ extracurricular model ห้ามตีความว่าได้รับ
> การอนุมัติจากอาจารย์แล้ว

สถานะหลักฐานที่ใช้ในส่วนนี้มีความหมายดังนี้:

- `Mandatory requirement`: requirement summary ระบุว่าเอกสารต้นฉบับบังคับ
- `Baseline selected from listed models`: เลือกจากรายการโมเดลหน้า 2
- `Selected extracurricular candidate`: candidate ที่ทีมเลือก แต่ยังไม่ใช่การอนุมัติ
- `Team-selected additional metric`: metric ที่ทีมเพิ่มเอง
- `Optional diagnostic`: การวิเคราะห์เสริม ไม่ใช้แทนผลหลัก
- `Pending instructor confirmation`: ต้องยืนยันกับอาจารย์หรือเอกสารต้นฉบับ
- `Planned`: มีแผน แต่ยังไม่มี implementation ที่ตรวจพบ
- `Implemented`: มี source implementation ที่ตรวจได้
- `Verified by saved results`: มี persisted results ที่ตรวจย้อนกลับได้

จากหลักฐาน repository ปัจจุบัน data preparation เป็น `Implemented` และ split/label
มี saved reports แต่ model training ทั้งสี่ยังเป็น `Planned`: ไฟล์
`src/models/regression.py` และ `src/models/classification.py` มีเพียง module
docstring และยังไม่มี saved model metrics ใน `outputs/metrics/` หรือ model reports
ใน `outputs/reports/`

## Selected Models

Multiple Linear Regression และ Logistic Regression เป็น baseline เพราะอยู่ใน
listed models ตาม requirement summary ส่วน Elastic Net และ AdaBoost เป็น
`Selected extracurricular candidate` เนื่องจากไม่อยู่ใน model list ที่สรุปจากหน้า 2
และมีกลไกการเรียนรู้ต่างจาก baseline อย่างไรก็ตาม การยืนยันว่าเป็น “beyond
classroom learning” ยังเป็น `Pending instructor confirmation` จนกว่าจะตรวจ Course
PDFs และได้รับคำยืนยันจากอาจารย์

### Regression Models

#### Multiple Linear Regression - baseline

- **Requirement status:** `Baseline selected from listed models`; `Planned`
- **Target:** `target_volatility_5d` เป็นค่าต่อเนื่อง จึงตรงกับ regression task
- **Mechanism:** ใช้ feature เชิงตัวเลขหลายตัวสร้าง
  \(\hat{y}=\beta_0+\sum_j\beta_jx_j\) และหา coefficients ที่ลด Sum of Squared
  Errors (SSE) หรือ Mean Squared Error (MSE)
- **Why selected:** เป็น baseline ที่ตีความทิศทางและขนาดของ coefficients ได้ง่าย
  และใช้ตรวจว่าการ regularize เพิ่มประสิทธิภาพนอก sample หรือไม่
- **Limitations:** สมมติความสัมพันธ์เชิงเส้น, ไวต่อ multicollinearity และ extreme
  values และไม่สร้าง nonlinear interactions เอง ควรตรวจ residual diagnostics แต่
  residual normality ไม่ใช่เงื่อนไขจำเป็นของการพยากรณ์ทุกกรณี

#### Elastic Net - new regression candidate

- **Requirement status:** `Selected extracurricular candidate`; `Pending instructor
  confirmation`; `Planned`
- **Target and features:** รองรับ continuous target และ numerical features ชุดเดียวกับ
  baseline
- **Mechanism:** ลด squared-error loss พร้อม L1 และ L2 penalties; L1 ช่วย shrink
  coefficients บางตัวไปใกล้หรือเท่าศูนย์ ส่วน L2 ช่วยควบคุม coefficients ของ features
  ที่สัมพันธ์กัน เช่น volatility windows ระยะสั้นและยาว
- **Planned hyperparameters:** `alpha`, `l1_ratio`, จำนวน iterations, optimization
  settings/learning rate ของ from-scratch implementation และ convergence tolerance
- **Why selected:** เป็นการเปรียบเทียบ regularized linear model กับ unregularized
  baseline และอาจช่วยควบคุม overfitting แต่จะไม่ถือว่าดีกว่าจนกว่าจะมี
  Validation/Test evidence
- **Limitations:** ต้อง scale features ก่อน regularization, ผลขึ้นกับ hyperparameters
  และยังคงเป็น linear model ใน transformed feature space ที่กำหนด

### Classification Models

#### Logistic Regression - baseline

- **Requirement status:** `Baseline selected from listed models`; `Planned`
- **Target:** `target_high_volatility` เป็น binary target จากเงื่อนไข
  `target_volatility_5d > Train Q75`
- **Mechanism:** แปลง linear score ผ่าน logistic function เป็นความน่าจะเป็นของ
  Positive/High-Volatility class และ minimize Binary Cross-Entropy หรือ Log Loss
- **Why selected:** probability และ coefficients ตีความได้ และเป็น linear baseline
  ที่ชัดเจนสำหรับเปรียบเทียบกับ ensemble
- **Limitations:** decision boundary เป็นเชิงเส้นใน feature space, ไวต่อ feature
  scaling และ class imbalance; หากไม่ใช้ threshold 0.5 ต้องเลือก threshold จาก
  Validation เท่านั้น

#### AdaBoost - new classification candidate

- **Requirement status:** `Selected extracurricular candidate`; `Pending instructor
  confirmation`; `Planned`
- **Target and features:** รองรับ binary target และ numerical features
- **Mechanism:** train weak learners ตามลำดับ, เพิ่ม sample weights ให้ observations
  ที่จำแนกผิด, คำนวณ learner weights และรวมผลด้วย weighted voting จนเป็น strong
  classifier
- **Planned hyperparameters:** `n_estimators`, `learning_rate`, base-learner type และ
  complexity รวมทั้ง random seed เมื่อ implementation ใช้ randomness
- **Why selected:** เปรียบเทียบ linear baseline กับ ensemble ที่จับ nonlinear
  relationships และ feature interactions ได้
- **Limitations:** ไวต่อ noisy labels, extreme observations และ class imbalance;
  objective/loss จะระบุให้ตรงกับ AdaBoost variant ที่ implement จริง โดยยังไม่อ้าง
  SAMME, SAMME.R หรือ exponential loss ล่วงหน้า

Model plan ในภาพที่แนบมาถูกต้องในส่วนรายชื่อโมเดล แต่รายการ metrics ยังไม่ครบ
requirement summary: ต้องเพิ่ม Regression loss/objective และ performance/diagnostic
curves; Classification ต้องเพิ่ม Accuracy, loss/objective และ performance curve
พร้อมระบุว่า Sensitivity คือ Recall และ True Negative Rate คือ Specificity

## Why These Models Fit the Targets

โปรเจกต์สร้าง eight numerical features ใน `src/build_features.py` ได้แก่
`return_1d`, `return_5d`, `historical_volatility_5d`,
`historical_volatility_20d`, `intraday_range`, `sma_ratio_5_20`, `rsi_14` และ
`volume_zscore_20` ส่วน `src/build_targets.py` สร้าง continuous
`target_volatility_5d` จาก sample standard deviation ของ returns ที่ `t+1` ถึง
`t+5` และ binary `target_high_volatility` จาก Train-only Q75 แบบ strict `>`

| Model | Target compatibility | Feature compatibility | Scaling | Main strength | Main limitation |
| --- | --- | --- | --- | --- | --- |
| Multiple Linear Regression | Continuous `target_volatility_5d` | Numerical predictors หลายตัว | Recommended เพื่อเปรียบเทียบ coefficients | ตีความง่ายและเป็น baseline ชัดเจน | Linear assumption, multicollinearity และ outliers |
| Elastic Net | Continuous `target_volatility_5d` | Numerical predictors รวมทั้ง correlated features | Required ก่อน regularization | รวม L1/L2 shrinkage | ต้อง tune penalties และยังมี linear form |
| Logistic Regression | Binary `target_high_volatility` | Numerical predictors | Required/recommended | ให้ probability และ interpretable coefficients | Linear boundary และ class imbalance |
| AdaBoost | Binary `target_high_volatility` | Numerical predictors; weak learner ต้องรองรับ sample weights | โดยทั่วไปไม่จำเป็นสำหรับ tree-based AdaBoost | จับ nonlinear interactions ด้วย ensemble | ไวต่อ noise และ extreme observations |

Scaler และ preprocessing ทุกชนิดต้อง fit จาก Training case ของตนเองเท่านั้น แล้ว
apply ไปยัง Validation/Test โดยห้าม fit ใหม่ เพื่อป้องกัน leakage

## From-Scratch and Reference Implementations

ทั้งสี่โมเดลต้องมี from-scratch implementation; Scikit-learn ใช้เฉพาะ reference
implementation สำหรับ verification และ quantitative benchmarking ไม่ใช้แทนงาน
from scratch การเปรียบเทียบต้องใช้ splits, features, target definitions,
hyperparameters และ metrics ที่สอดคล้องกันเท่าที่ implementation อนุญาต

สำหรับแต่ละโมเดล เอกสารและโค้ดใน phase modeling ต้องอธิบายและทดสอบ:

1. mathematical objective และ loss/objective ที่ optimize
2. parameter/sample-weight initialization
3. forward calculation
4. loss/objective calculation และ parameter/weight update
5. stopping condition และ convergence tolerance
6. hyperparameters
7. prediction/probability procedure
8. computational limitations
9. comparison กับ Scikit-learn บน protocol เดียวกัน

แผน objective ปัจจุบันคือ SSE/MSE สำหรับ Multiple Linear Regression,
penalized squared error สำหรับ Elastic Net และ Binary Cross-Entropy/Log Loss สำหรับ
Logistic Regression ส่วน AdaBoost objective รอ freeze variant ก่อน implement

## Evaluation Metrics

Metric เดียวไม่เพียงพอสำหรับทั้งสอง tasks โดยเฉพาะ Classification ซึ่งสัดส่วน
Positive class ใน saved splits ไม่สมดุล (`classification_threshold.json` รายงาน
Train/Validation/Test ประมาณ 24.99%/9.16%/19.57%)

| Task | Metric/Artifact | Requirement status | Why used | Interpretation |
| --- | --- | --- | --- | --- |
| Regression | R-Square (`R²`) | `Mandatory requirement` | วัดสัดส่วนความแปรปรวนที่อธิบายได้เทียบ mean baseline | ค่าสูงกว่าดีกว่าและอาจติดลบบน Validation/Test |
| Regression | Training loss/objective | `Mandatory requirement` | แสดงสิ่งที่โมเดล optimize จริง | MSE/SSE หรือ penalized squared error ตามโมเดล |
| Regression | MAE | `Team-selected additional metric` | error เฉลี่ยในหน่วยเดียวกับ target และไวต่อ error ใหญ่น้อยกว่า RMSE | ต่ำกว่าดีกว่า |
| Regression | RMSE | `Team-selected additional metric` | ลงโทษ error ใหญ่แรงกว่า เหมาะกับการพลาดช่วง volatility สูง | ต่ำกว่าดีกว่าและอยู่ในหน่วย target |
| Regression | Performance/diagnostic curves | `Mandatory requirement` สำหรับ performance curve; รายละเอียดกราฟเป็น `Optional diagnostic` | ตรวจ error pattern และ generalization | อย่างน้อย loss curve, predicted-vs-actual, residual หรือ model-error comparison ตาม implementation |
| Classification | Confusion Matrix | `Mandatory requirement` | แสดง TP, TN, FP, FN และเป็นฐานคำนวณ metrics | เป็น evaluation artifact ไม่ใช่ scalar metric |
| Classification | Accuracy | `Mandatory requirement` | สัดส่วนคำทำนายถูกทั้งหมด | ห้ามใช้ค่าเดียวเพราะ class imbalance |
| Classification | Precision | `Mandatory requirement` | วัดว่าจากวันที่ทำนายว่า High Volatility มีสัดส่วนถูกเท่าใด | `TP / (TP + FP)` |
| Classification | Sensitivity/Recall | `Mandatory requirement` | วัดว่าสามารถตรวจพบวัน High Volatility จริงได้เท่าใด | Measure เดียวกันสำหรับ Positive class: `TP / (TP + FN)` |
| Classification | Specificity/True Negative Rate | `Mandatory requirement` | วัดว่าสามารถจำแนกวัน Normal Volatility จริงได้เท่าใด | Measure เดียวกัน: `TN / (TN + FP)` |
| Classification | F1-score | `Mandatory requirement` | สรุปสมดุล Precision กับ Recall | มีประโยชน์เมื่อ class ไม่สมดุล |
| Classification | ROC curve | `Mandatory requirement` | แสดง TPR/FPR ข้ามหลาย thresholds | เป็นกราฟ ไม่ใช่ scalar |
| Classification | ROC-AUC | `Mandatory requirement` | สรุป ranking ability ข้าม thresholds | scalar summary ของ ROC curve |
| Classification | Training loss/objective | `Mandatory requirement` | แสดง objective ที่ implementation optimize | Log Loss สำหรับ Logistic; AdaBoost รอ variant ที่ implement |
| Classification | Performance curve | `Mandatory requirement` | ตรวจ training/validation behavior หรือ threshold trade-off | curve ต้องระบุแกนและ dataset ชัดเจน |
| Classification | PR-AUC | `Team-selected additional metric`; `Optional diagnostic` | เน้นคุณภาพ Positive class เมื่อมีสัดส่วนน้อย | ใช้เพิ่ม ไม่ใช้แทน ROC-AUC, Precision, Recall หรือ F1 |

### Regression Metrics

MAE และ RMSE เป็น metrics ที่ทีมเลือกเพิ่ม ไม่ใช่ mandatory metrics ที่ระบุชื่อใน
requirement summary ส่วน `R²`, Loss และ performance curve เป็น mandatory ต้องแยก
training objective ออกจาก reporting metric: ห้ามใช้ MAE/RMSE แทน SSE/MSE หรือ
Elastic Net penalized objective โดยไม่อธิบาย

### Classification Metrics

`Sensitivity = Recall` ของ Positive/High-Volatility class และ
`Specificity = True Negative Rate`; ชื่อคู่เหล่านี้ไม่ใช่คนละสูตร ROC curve เป็นกราฟ
ส่วน ROC-AUC เป็น scalar summary และ Confusion Matrix เป็น artifact ไม่ใช่ scalar
metric ผลหลักต้องรายงาน metrics ครบ ไม่สรุปจาก Accuracy เพียงอย่างเดียว

## Evaluation Protocol

1. ใช้ chronological Train/Validation/Test split เดิมแบบ 70%/15%/15% และห้าม
   random shuffle
2. รักษา purging gaps 5 rows ระหว่าง Train/Validation และ Validation/Test ตาม
   `src/split_data.py`; saved report ปัจจุบันมี 1,733/371/373 rows
3. สร้าง features/targets บน complete timeline ตาม implementation เดิม และใช้
   definitions เดียวกันทุกโมเดล
4. Fit preprocessing จาก Train ของแต่ละ experiment case เท่านั้น แล้ว apply ไปยัง
   Validation/Test โดยห้าม fit ใหม่
5. เลือก hyperparameters, decision threshold และ model configuration จาก
   Train/Validation เท่านั้น ห้ามเลือกจาก Test
6. Freeze model configuration ก่อนใช้ Test สำหรับ final evaluation ครั้งสุดท้าย
7. ใช้ metrics เดียวกันสำหรับโมเดลใน task เดียวกัน และใช้ random seed เดียวกันเมื่อ
   algorithm มี randomness
8. From-scratch และ Scikit-learn reference ต้องใช้ข้อมูลและ protocol ที่สอดคล้องกัน
9. With-Spike และ Non-Spike ใช้ main Validation/Test periods เดียวกัน
10. รายงาน Full Test เป็นผลหลัก; Non-Spike และ Spike-Affected segments เป็น
    diagnostics และต้องไม่แทนผล Full Test

## With-Spike and Non-Spike Experiments

โมเดลทุกตัวต้อง train เป็นคู่ภายใต้ protocol เดียวกัน เพื่อแยกผลของ spike filtering
ออกจากผลของ model choice:

| Task | Model | With-Spike | Non-Spike |
| --- | --- | --- | --- |
| Regression | Multiple Linear Regression | Required | Required |
| Regression | Elastic Net | Required | Required |
| Classification | Logistic Regression | Required | Required |
| Classification | AdaBoost | Required | Required |

With-Spike ใช้ Original Train ส่วน Non-Spike จะตัดเฉพาะ Train modeling rows ที่
`is_spike_affected == True` หลังสร้าง features/targets แล้ว โดยไม่เปลี่ยน Original
Validation/Test, classification Q75, models, metrics หรือ search protocol ปัจจุบัน
workflow โดยรวมยังเป็น `Planned`: direct detector มี implementation แล้ว แต่ยังไม่มี
experiment datasets หรือ saved spike/model metrics จึงยังใช้คำว่า `Implemented`
กับ workflow ทั้งชุดหรือ `Verified by saved results` กับผลโมเดลไม่ได้

## Model Selection Risks and Open Questions

- ต้องแนบและตรวจ `_69-1-01076641_Project-description.pdf` หน้า 2, 3 และ 5 ก่อน
  ยืนยัน citations ใน README
- ต้องค้นคำว่า `Elastic Net`, `AdaBoost` และ `Adaptive Boosting` ใน
  `Machine Learning (1).pdf` และ `Machine Learning (2).pdf`; ขณะนี้ PDF ไม่พร้อมให้
  ตรวจ จึงห้ามเขียนว่า “Definitely not taught in class”
- ต้องให้อาจารย์ยืนยันว่า Elastic Net และ AdaBoost นับเป็น extracurricular models;
  ปัจจุบันเป็น candidates ไม่ใช่ approved choices
- ต้อง freeze AdaBoost variant/base learner/objective ก่อนเริ่ม from-scratch
  implementation
- ต้อง freeze hyperparameter search spaces, random seeds, scaling policy และ
  prediction-threshold policy ก่อนใช้ Test
- With-Spike/Non-Spike workflow และ model implementations ยังเป็น `Planned` และยัง
  ไม่มี performance claims

### Traceable sources

- Requirement summary อ้าง `_69-1-01076641_Project-description.pdf`, หน้า 2
  (listed models), หน้า 3 (new models/evaluation) และหน้า 5 (rubric/from-scratch)
  แต่ PDF ยังต้องแนบเพื่อตรวจข้อความโดยตรง
- `src/build_features.py`: รายการและนิยาม features
- `src/build_targets.py`: `target_volatility_5d`, Train-only Q75 และ
  `target_high_volatility`
- `src/split_data.py` และ `outputs/reports/data_split_report.json`: chronological
  split และ 5-row purging gaps
- `outputs/reports/classification_threshold.json`: threshold และ class distributions
- Scikit-learn reference documentation:
  [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html),
  [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html),
  [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  และ
  [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
