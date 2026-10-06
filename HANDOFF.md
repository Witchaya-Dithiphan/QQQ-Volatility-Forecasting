# QQQ Volatility Forecasting - เอกสารส่งต่องาน

เอกสารนี้อัปเดตล่าสุด ณ วันที่ **2026-10-06** บน branch
`feature/spike-analysis` โดย M1 commit แล้วที่ `399fecb`, M2 commit แล้วที่
`04c165c` และปัจจุบันมี M3 implementation/tests/docs ที่ยังไม่ได้ commit

สถานะที่ใช้ตลอดเอกสาร:

- **Implemented** - มี source implementation ใน repository
- **Verified** - ตรวจด้วย artifact หรือคำสั่งที่รันจริงในรอบนี้
- **Planned** - มีข้อกำหนดหรือแผน แต่ยังไม่มี implementation ครบ
- **Not recorded** - **ยังไม่ได้บันทึก** ใน repository หรือหลักฐานที่ตรวจได้

## 1. เป้าหมายและขอบเขตโปรเจกต์

โปรเจกต์ใช้ข้อมูล QQQ รายวันแบบ OHLCV (`Date`, `Close`, `Volume`, `Open`,
`High`, `Low`) เพื่อเตรียมข้อมูลสำหรับสองงาน:

| งาน | Target | ความหมาย | สถานะข้อมูล |
| --- | --- | --- | --- |
| Regression | `target_volatility_5d` | realized volatility แบบ annualized ของ `return_1d` ในอนาคต `t+1` ถึง `t+5` | **Implemented / Verified** |
| Classification | `target_high_volatility` | `1` เมื่อ `target_volatility_5d` มากกว่า Q75 ที่ fit จาก Original Train | **Implemented / Verified** |

Historical volatility features เช่น `historical_volatility_5d` และ
`historical_volatility_20d` ใช้ returns ที่เกิดขึ้นแล้วจนถึงวัน `t` จึงเป็น input
เชิงสาเหตุ ส่วน `target_volatility_5d` ใช้ returns หลังวัน `t` และมีไว้เป็นคำตอบที่
โมเดลต้องทำนาย ห้ามนำ target หรือข้อมูลอนาคตกลับไปสร้าง feature ณ วันเดียวกัน

สถานะงานโดยสรุป:

- **ทำแล้ว:** Snapshot verification, cleaning, feature engineering, regression
  target, chronological split พร้อม gaps, Train-only Q75 labels, labeled splits,
  end-to-end data runner, reports และ automated tests
- **ข้อมูลพร้อม แต่โค้ดโมเดลยังไม่มี:** Original/With-Spike Train,
  Validation และ Test พร้อมเป็น tabular inputs แต่ `src/models/` ยังไม่มี model
  implementation
- **วางแผนไว้:** Non-Spike experiment datasets ขั้น M5, paired model training,
  evaluation, forecast และ persisted
  model/metric artifacts

คำว่า **With-Spike** ในเอกสารนี้หมายถึง Original Train ที่ยังเก็บ extreme events
ไว้ ไม่ได้หมายความว่าโมเดล baseline ถูก train แล้ว ส่วน **Non-Spike** เป็น experiment
ที่วางแผนให้กรอง `is_spike_affected` เฉพาะสำเนา Train; ยังไม่มี code หรือ artifacts
ของ workflow นี้

## 2. สิ่งที่มีอยู่ใน repository แล้ว

ไม่สามารถพิสูจน์จาก repository เพียงอย่างเดียวว่าใครเป็นผู้สร้างแต่ละส่วน จึงใช้คำว่า
“มีอยู่ใน repository” แทนการระบุว่า Codex หรือบุคคลใดเป็นผู้แก้

| ส่วนงาน | สถานะ | หลักฐาน |
| --- | --- | --- |
| Immutable Snapshot verification | **Implemented / Verified** | `src/download_qqq_data.py`, `data/manifests/qqq_daily_snapshot.json`, tests และผล `verify-snapshot` รอบนี้ |
| Cleaning/validation | **Implemented / Verified** | `src/clean_data.py`, `outputs/reports/cleaning_report.json` |
| 8 causal features | **Implemented / Verified** | `src/build_features.py`, `outputs/reports/feature_report.json` |
| 5-day regression target | **Implemented / Verified** | `src/build_targets.py`; isolated run รอบนี้สร้าง `regression_target_report.json` สำเร็จ |
| Chronological split 70/15/15 + gaps | **Implemented / Verified** | `src/split_data.py`, `outputs/reports/data_split_report.json` |
| Train-only Q75 classification labels | **Implemented / Verified** | `src/build_targets.py`, `outputs/reports/classification_threshold.json` |
| End-to-end data runner | **Implemented / Verified** | `src/run_data_pipeline.py`; isolated run รอบนี้สำเร็จและสร้าง 14 artifacts |
| Tests | **Implemented / Verified** | `tests/`; รอบนี้ `300 passed, 1 skipped` รวม M2 detector 33 tests, M3 boundary/audit 36 tests และ M1 input-contract 39 tests |
| Regression/Classification models | **Planned** | `src/models/regression.py` และ `classification.py` มีเพียง module docstring |
| Spike/Non-Spike pipeline | **Partially implemented** | M1 input contract, M2 direct detector และ M3 split-local mask/boundary metadata verified; ยังไม่มี experiment datasets/event audit/experiment runner/figures |
| Model results/figures | **Not recorded** | `outputs/models/`, `outputs/metrics/`, `outputs/figures/` มีเพียง `.gitkeep` |

**Phase 1 - Baseline Data Preparation:** implementation ใช้งานได้และผ่าน isolated
validation รอบนี้ อย่างไรก็ตาม generated reports ที่ project root เป็นชุดเก่าบางส่วน:
`regression_target_report.json` ไม่มีอยู่ที่ root และ `data_split_report.json` กับ
`classification_threshold.json` ยังเก็บ absolute paths แม้ runner ปัจจุบันสร้าง
report-relative paths ได้แล้ว

**Phase 2 - Spike Analysis/Experiment Dataset Preparation:** **M1–M4 verified / experiment datasets planned** มี input contract, pure Train-fitted direct detector, split-local affected mask, JSON-safe boundary metadata และ direct-spike/data-quality audit artifacts แล้ว แต่ยังไม่มี generated With-Spike/Non-Spike experiment datasets หรือ complete Phase 2 runner ห้ามถือ M4 audit เป็น completed experiment pipeline ดูหลักฐานที่ `PHASE2_SPIKE_READINESS.md`

## 3. แผนผังโครงสร้างไฟล์

| Path | Git/การจัดหา | หน้าที่ | Input | Output หรือ side effect |
| --- | --- | --- | --- | --- |
| `config.py` | Tracked | Default paths ของ snapshot, intermediate data, splits และ reports | ไม่มี | ไม่มีการเขียนไฟล์ |
| `src/load_data.py` | Tracked | โหลดและ validate schema ของ QQQ CSV | CSV | DataFrame/validation error |
| `src/download_qqq_data.py` | Tracked | ตรวจ Snapshot แบบ offline หรือ refresh rolling data จาก Nasdaq | Snapshot + Manifest หรือ network | Verification summary; live CSV/report เมื่อ refresh |
| `src/clean_data.py` | Tracked | Validate/clean OHLCV โดยรักษา source | Raw Snapshot | `qqq_clean.csv`, cleaning report |
| `src/build_features.py` | Tracked | สร้าง 8 causal features | Clean CSV | `qqq_features.csv`, feature report |
| `src/build_targets.py` | Tracked | สร้าง future-volatility target และ Train-only Q75 labels | Feature CSV หรือ splits | Regression-target CSV/report และ labeled splits/report |
| `src/split_data.py` | Tracked | กรอง modeling rows แล้วแบ่งตามเวลา พร้อมสอง gaps | Regression-target CSV | Train/Validation/Test และ split report |
| `src/run_data_pipeline.py` | Tracked | Orchestrate Phase 1 ตั้งแต่ verification ถึง labels | Snapshot + Manifest | 14 generated artifacts ใน output root |
| `src/build_spike_input_contract.py` | Tracked | อ่านและ validate/freeze Phase 2 M1 baseline inputs โดยไม่แก้ไข | Labeled splits + saved reports + isolated reproduction | เขียนเฉพาะ generated portable `spike_input_contract.json` |
| `src/detect_spikes.py` | Tracked | Canonical pure direct detector: validation, Q1/Q3/IQR, fit/apply และ strict flags | M1-verified Original Train; fitted metadata + split DataFrame | Immutable metadata/boolean flagsในหน่วยความจำ; ไม่มี filesystem side effect |
| `src/spike_contract.py` | Tracked | Canonical affected-window/mask primitives, split isolation, boundary metadata และ detector compatibility exports | Direct flags + optional aligned Dates | Boolean flags/window/summary metadata ในหน่วยความจำ; ไม่มี filesystem side effect |
| `src/audit_spikes.py` | Tracked | M4 runner: reuse M2/M3, trace direct events และตรวจ data quality/checksums | M1 contract + labeled/upstream protected inputs | generated report, event CSV และ direct-spike figure ใต้ output root |
| `src/report_paths.py` | Tracked | สร้าง path ใน report แบบ relative ต่อ report directory | Artifact/report paths | Portable path string |
| `src/models/` | Tracked | ตำแหน่งสำหรับ model code | ยังไม่มี | **Planned; ไม่มี model side effect** |
| `tests/` | Tracked | Tests สำหรับ baseline pipeline และ Phase 2 spike/boundary/audit contract | Source code + synthetic fixtures | Pytest result; มี M4 artifact-runner integration test แล้ว แต่ยังไม่มี model tests |
| `notebooks/01_data_cleaning.ipynb` | Tracked | Cleaning review; มี 21 cells และ saved outputs | Local generated data | Notebook outputs ภายในไฟล์ |
| `notebooks/02_eda_and_features.ipynb` | Tracked | EDA/features/targets; มี 32 cells และ saved outputs | Local generated data/reports | Notebook outputs ภายในไฟล์ |
| `notebooks/03_regression.ipynb` | Tracked | Placeholder สำหรับ regression | ไม่มี | 0 cells; **Planned** |
| `notebooks/04_classification.ipynb` | Tracked | Placeholder สำหรับ classification | ไม่มี | 0 cells; **Planned** |
| `data/manifests/qqq_daily_snapshot.json` | Tracked | Contract/checksum ของ historical Snapshot | ไม่มี | ใช้ตรวจ Snapshot |
| `data/raw/qqq_daily.csv` | Ignored; ผู้รับงานต้องมีเอง | Historical pinned Snapshot | ส่งต่อแยกต่างหาก | Input หลัก; runner ห้ามเขียนทับ |
| `data/raw/qqq_daily_latest.csv` | Ignored/generated | Live rolling refresh; ไม่ใช่ Snapshot เดิม | Network | สร้างเมื่อ `refresh-latest` |
| `data/interim/` | `.gitkeep` tracked; contents ignored | Clean, feature และ regression-target datasets | Raw/generated stages | 3 CSVs |
| `data/processed/` | `.gitkeep` tracked; contents ignored | Original และ labeled splits | Modeling-ready data | 6 CSVs |
| `outputs/reports/` | `.gitkeep` tracked; contents ignored | JSON audit reports | Pipeline stages | 5 reports เมื่อ runner ปัจจุบันรันครบ |
| `outputs/figures/` | `.gitkeep` tracked; contents ignored | พื้นที่ figure exports | ยังไม่มี producer | ไม่มี exported figure ปัจจุบัน |
| `outputs/models/` | `.gitkeep` tracked; contents ignored | พื้นที่ persisted models | ยังไม่มี model training | ไม่มี model files ปัจจุบัน |
| `outputs/metrics/` | `.gitkeep` tracked; contents ignored | พื้นที่ persisted metrics/predictions | ยังไม่มี evaluation | ไม่มี metric files ปัจจุบัน |
| `README.md` | Tracked | Runbook และ model/evaluation plan | หลักฐานใน repo + requirement summary | Documentation |
| `PROJECT_STATUS.md` | Tracked | Phase status, spike plan และ open questions | Audit ก่อนหน้า | Documentation; อาจมีข้อความเชิงแผน |
| `requirements.txt` | Tracked | Dependency names แบบไม่ pin version | pip | ติดตั้ง environment ที่ไม่ล็อกเวอร์ชัน |

หลัง clone จะได้ Manifest และ `.gitkeep` แต่จะ **ไม่ได้** Raw Snapshot, generated
data, reports, figures, models หรือ metrics เพราะ `.gitignore` กันไฟล์เหล่านี้ไว้

## 4. ติดตั้งจากเครื่องใหม่

### เวอร์ชันและ dependencies

- Python version ที่โครงการกำหนดอย่างเป็นทางการ: **ยังไม่ได้บันทึก** ไม่มี
  `.python-version`, `pyproject.toml`, CI หรือ environment file ที่กำหนดเวอร์ชัน
- Python ที่ใช้ตรวจรอบนี้: **3.11.9** จาก `.venv/Scripts/python.exe`; เป็นข้อมูล
  environment รอบนี้ ไม่ใช่ required version
- `requirements.txt` ระบุ `numpy`, `pandas`, `pytest`, `matplotlib`, `seaborn`,
  `nbformat`, `nbconvert`, `ipykernel` โดย **ไม่ pin versions**
- ไม่มี lockfile (`poetry.lock`, `Pipfile.lock` หรือ equivalent)
- เวอร์ชันที่ติดตั้งใน environment รอบนี้: NumPy 2.4.6, pandas 3.0.6,
  pytest 9.1.1, Matplotlib 3.11.2, seaborn 0.13.2, nbformat 5.11.1,
  nbconvert 7.17.1 และ ipykernel 7.4.0
- Scikit-learn ถูกกล่าวถึงใน model plan แต่ **ไม่อยู่ใน `requirements.txt` และไม่ถูก
  ติดตั้งใน environment รอบนี้**; version ที่จะใช้: **ยังไม่ได้บันทึก**

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python --version
python -m pip check
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python --version
python -m pip check
```

`pip install` ต้องใช้ network หรือ package cache ผู้รับงานต้องได้รับ
`data/raw/qqq_daily.csv` แยกต่างหาก แล้วตรวจว่า SHA-256 ตรง Manifest; ไฟล์นี้ไม่ได้
track ใน Git และ live refresh ไม่สามารถรับประกันว่าจะสร้าง Snapshot เดิมได้

## 5. แหล่งข้อมูลและ provenance

| รายการ | ค่าที่ตรวจได้ | แหล่งหลักฐาน |
| --- | --- | --- |
| Source | Nasdaq QQQ Historical Data | `data/manifests/qqq_daily_snapshot.json` |
| Source page | `https://www.nasdaq.com/market-activity/etf/qqq/historical` | Manifest |
| Symbol | `QQQ` | Manifest |
| Snapshot path | `data/raw/qqq_daily.csv` | Manifest/config |
| Date range | 2016-08-31 ถึง 2026-08-28 | Manifest + verification รอบนี้ |
| Rows | 2,512 | Manifest + verification รอบนี้ |
| Schema | `Date, Close, Volume, Open, High, Low` | Manifest + verification รอบนี้ |
| SHA-256 | `649e1db1b79c0990ea947a4ea162d6836b9bbdda2af24bc5ab6ae66cb83b6b13` | Manifest + verification รอบนี้ |
| Original retrieval date/time | **ยังไม่ได้บันทึก** (`retrieved_at: null`) | Manifest |
| Archive modified time | 2026-08-31 04:34:02 +07:00 | Manifest; ไม่ใช่ retrieval time |
| Dataset license/terms | **ยังไม่ได้บันทึก** | ต้องเพิ่มแหล่งอ้างอิง terms/license ที่ตรวจย้อนกลับได้ |

Historical Snapshot เป็น input ที่ pin ด้วย checksum สำหรับ reproduce ผลเดิม ส่วน
`refresh-latest` เรียก Nasdaq ผ่าน network และเขียนแยกเป็น
`data/raw/qqq_daily_latest.csv` พร้อม `qqq_download_report.json` แหล่ง live เป็น
rolling window จึงอาจมีช่วงวันที่/จำนวนแถวต่างจาก Snapshot

กฎความปลอดภัย: ห้ามแก้หรือเขียนทับ `qqq_daily.csv` และ Manifest ห้ามนำ live rows
มาต่อแล้วอ้างว่าเป็น Snapshot เดิม Runner ตรวจ checksum ก่อนและหลัง execution และ
ปฏิเสธ output root ที่อยู่ใต้ Raw/Manifest directories

## 6. วิธีเตรียมข้อมูลและสร้าง `train_labeled.csv` ใหม่

ลำดับจริงใน `src/run_data_pipeline.py` คือ:

1. ตรวจ Snapshot กับ Manifest
2. Cleaning/validation โดยไม่แก้ Raw
3. สร้าง 8 features บน complete timeline
4. สร้าง `target_volatility_5d`
5. กรอง modeling copy เฉพาะ rows ที่ feature/target มี NaN/Infinity
6. แบ่งตามเวลา Train/Validation/Test พร้อม gap 5 rows สองช่วง
7. Fit Q75 จาก Original Train เท่านั้น
8. สร้าง `target_high_volatility` ด้วย strict `>` ให้ทั้งสาม splits

8 features จาก `src/build_features.py` คือ:

```text
return_1d
return_5d
historical_volatility_5d
historical_volatility_20d
intraday_range
sma_ratio_5_20
rsi_14
volume_zscore_20
```

คำสั่งปลอดภัยจาก project root โดยใช้ output root ใหม่:

```powershell
python src/download_qqq_data.py verify-snapshot
python -m src.run_data_pipeline --output-root tmp/phase1-handoff-check
```

ถ้า artifact ใดใน output root มีอยู่แล้ว runner จะหยุดที่ preflight โดยไม่เขียนทับ
ใช้ชื่อ output root ใหม่เป็นค่าเริ่มต้นที่ปลอดภัย `--overwrite-generated` อนุญาตให้
แทนที่เฉพาะ 14 outputs ที่ runner ประกาศไว้ และยังไม่อนุญาตให้แตะ Snapshot/Manifest
หาก stage หลัง preflight ล้มเหลว outputs จาก stage ที่สำเร็จหรือสำเร็จบางส่วนอาจค้าง
อยู่ ต้องตรวจและเลือก output root ใหม่ก่อน retry

Labeled split paths แบบ default:

```text
data/processed/train_labeled.csv
data/processed/validation_labeled.csv
data/processed/test_labeled.csv
```

เมื่อใช้ตัวอย่าง `--output-root tmp/phase1-handoff-check` paths จะเป็น:

```text
tmp/phase1-handoff-check/data/processed/train_labeled.csv
tmp/phase1-handoff-check/data/processed/validation_labeled.csv
tmp/phase1-handoff-check/data/processed/test_labeled.csv
```

Runner ปัจจุบันสร้าง 14 artifacts:

- Interim: `qqq_clean.csv`, `qqq_features.csv`, `qqq_regression_target.csv`
- Original splits: `train.csv`, `validation.csv`, `test.csv`
- Labeled splits: `train_labeled.csv`, `validation_labeled.csv`,
  `test_labeled.csv`
- Reports: `cleaning_report.json`, `feature_report.json`,
  `regression_target_report.json`, `data_split_report.json`,
  `classification_threshold.json`

จาก Snapshot ที่ pin: cleaning คง 2,512 rows, regression target valid 2,507 rows,
modeling-ready 2,487 rows, gaps รวม 10 rows และ Train/Validation/Test เท่ากับ
1,733/371/373 rows

## 7. วิธี train, evaluate และ forecast - ตามสถานะจริง

**ยังไม่มีคำสั่ง train/evaluate/forecast ที่รันได้ใน repository**

- `src/models/regression.py` และ `src/models/classification.py` มีเพียง docstring
- Notebook 03 และ 04 มี 0 cells
- ไม่พบ training/evaluation CLI หรือ scripts
- ไม่พบ persisted model, prediction, metric หรือ exported figure
- ไม่พบ scikit-learn dependency หรือ code เปรียบเทียบ from-scratch กับ scikit-learn

Original/With-Spike splits พร้อมเป็น inputs สำหรับเริ่มพัฒนา model แต่ยังไม่ควรอ้าง
ผลทดลอง และ paired conclusion กับ Non-Spike ต้องรอ Phase 2 artifacts

### แผนโมเดลและเกณฑ์ประเมิน

แหล่งที่มาของแผนส่วนนี้คือ **ภาพแผนโมเดลที่ผู้ใช้แนบ** ภาพไม่ได้อยู่ใน repository
และไม่ใช่หลักฐานว่า implement, train หรือ evaluate แล้ว

#### Regression

| โมเดล | Target | Metrics | สถานะโมเดล | สถานะ metrics |
| --- | --- | --- | --- | --- |
| Multiple Linear Regression | `target_volatility_5d` | MAE, RMSE, R² | **Planned** - ยังไม่มี from-scratch หรือ reference implementation | กำหนดไว้ในแผน; ผลคำนวณจริง **ยังไม่ได้บันทึก** |
| Elastic Net | `target_volatility_5d` | MAE, RMSE, R² | **Planned** - ยังไม่มี from-scratch หรือ scikit-learn comparison | กำหนดไว้ในแผน; ผลคำนวณจริง **ยังไม่ได้บันทึก** |

#### Classification

| โมเดล | Target | Metrics/artifacts | สถานะโมเดล | สถานะ metrics |
| --- | --- | --- | --- | --- |
| Logistic Regression | `target_high_volatility` | Confusion Matrix, Precision, Recall, Specificity, F1, ROC-AUC | **Planned** - ยังไม่มี from-scratch หรือ reference implementation | กำหนดไว้ในแผน; ผลคำนวณจริง **ยังไม่ได้บันทึก** |
| AdaBoost | `target_high_volatility` | Confusion Matrix, Precision, Recall, Specificity, F1, ROC-AUC | **Planned** - ยังไม่มี from-scratch หรือ scikit-learn comparison | กำหนดไว้ในแผน; ผลคำนวณจริง **ยังไม่ได้บันทึก** |

README ระบุ metrics เพิ่มจากภาพตาม requirement summary ได้แก่ Accuracy,
training loss/objective และ performance curves; metrics เหล่านี้ยังเป็น **Planned**
เช่นกัน ไม่มี model metric ใดเป็น **Verified** ณ ตอนส่งต่อ

เมื่อเริ่ม modeling ต้อง fit scaler/preprocessing แยกจาก Train ของแต่ละ case,
เลือก hyperparameters และ classification decision threshold ด้วย Validation, freeze
configuration แล้วใช้ Test สำหรับ final evaluation ทั้ง With-Spike และ Non-Spike
ต้องใช้ Validation/Test ชุดเดียวกันและ protocol เดียวกัน

ROC-AUC คำนวณจาก predicted scores/probabilities ไม่ใช่ hard labels และต้องตรวจว่า
evaluation subset มีทั้งสอง classes ก่อน หากไม่มีทั้งสอง classes ให้บันทึกว่า
คำนวณไม่ได้พร้อมเหตุผล ห้ามใส่ค่าทดแทน

## 8. Runbook คำสั่งตามลำดับ

ทุกคำสั่งรันจาก project root หลังมี Snapshot ที่ checksum ตรง Manifest

| ลำดับ | คำสั่ง copy/run | ผลลัพธ์ที่ควรเกิด | สถานะตรวจยืนยัน |
| --- | --- | --- | --- |
| 1 | `py -3 -m venv .venv` (Windows) หรือ `python3 -m venv .venv` (macOS/Linux) | สร้าง virtual environment | **Not recorded** - ไม่สร้าง environment ใหม่ในรอบนี้ |
| 2 | `python -m pip install -r requirements.txt` หลัง activate | ติดตั้ง dependencies; ต้องใช้ network/cache | **Not recorded** - environment มีอยู่แล้ว; requirements ไม่ pin |
| 3 | `python --version` และ `python -m pip check` | ตรวจ interpreter และ dependency consistency | **Verified:** Python 3.11.9 และ `No broken requirements found.` |
| 4 | `python src/download_qqq_data.py verify-snapshot` | แสดง 2,512 rows, date range และ SHA-256 ตรง Manifest | **Verified** รอบนี้; offline |
| 5 | `python -m src.run_data_pipeline --output-root tmp/phase1-handoff-check` | สร้าง 14 artifacts ใต้ isolated root | **Verified** รอบนี้ด้วย output root ชั่วคราวชื่ออื่น; offline |
| 6 | `Get-ChildItem tmp/phase1-handoff-check -Recurse -File` | เห็น interim 3, processed 6, reports 5 files | **Verified** กับ isolated artifacts รอบนี้ |
| 7 | `python -m pytest -q -rs` | Baseline และ Phase 2 M1–M3 tests ผ่าน; แสดงเหตุผลของ skipped tests | **Verified:** `300 passed, 1 skipped` |
| 8 | Train/evaluate/forecast | ไม่มีคำสั่งให้รัน | **Planned** - ห้ามสร้าง command สมมติ |

คำสั่งที่ใช้ network แยกจาก baseline reproduction:

```powershell
python src/download_qqq_data.py refresh-latest
```

คำสั่งนี้สร้าง rolling dataset แยกและไม่ควรใช้แทน pinned Snapshot ในการ reproduce
ผลเดิม การติดตั้ง packages ก็อาจต้องใช้ network ส่วน verification, pipeline และ
tests สามารถรัน offline ได้เมื่อ environment, Snapshot และ Manifest พร้อม

## 9. Assumptions และ design decisions

- Target: sample standard deviation (`ddof=1`) ของ `return_1d` ที่ `t+1...t+5`
  คูณ `sqrt(252)`; horizon 5 trading days
- Features ใช้ข้อมูลถึงวัน `t`; target ใช้อนาคตหลัง `t`
- NaN/Infinity: สร้างบน complete timeline ก่อน แล้วแปลง Infinity เป็น NaN และ
  `dropna` เฉพาะ selected features/target ใน modeling copy; 25 rows ถูกกรอง
- Split: chronological 70%/15%/15%, ไม่ shuffle, gap 5 rows สองช่วง อัตราส่วนใช้
  หลังสำรอง gaps แล้ว ตรงกับภาพแผนโมเดล
- Classification: Q75 (`interpolation="linear"`) fit จาก Original Train เท่านั้น;
  label ใช้ strict `target_volatility_5d > threshold`
- Snapshot/Manifest เป็น protected inputs และต้อง byte-identical ก่อน/หลัง runner
- Default runner ไม่ overwrite existing outputs; `--overwrite-generated` จำกัดที่
  14 known artifacts
- Runner ปัจจุบันเขียน report paths แบบ relative ต่อ report directory แต่ reports
  เก่าที่ root บางไฟล์ยังเป็น absolute paths
- Stage failure หลังเริ่มเขียนอาจทิ้ง partial outputs; ข้อความ error ระบุ stage

Spike design ที่ตกลงไว้ โดย direct detector ขั้น M2 เป็น **Implemented / Verified** และขั้น artifact/dataset ยังเป็น **Planned**:

- Detection variable: `abs(return_1d)`
- Fit Extreme-IQR จาก Original Train เท่านั้น: `Q3 + 3 * IQR`
- Direct spike ใช้ strict `>`
- Affected window รอบ spike ตำแหน่ง `s`: `s-5` ถึง `s+19`
- สร้าง features/targets ก่อนกรอง; ห้ามลบ Raw rows แล้ว recompute returns
- Filter เฉพาะสำเนา Train; คง Original Validation/Test และ Original Train Q75
- With-Spike/Non-Spike ต้องใช้ models, metrics, search space และ seeds เดียวกัน

Open questions:

- Wilder `rsi_14` ใช้ recursive smoothing จึงมีผลแบบ decay ต่อหลัง `s+19`; ต้อง
  ตัดสินใจว่าจะยอมรับ operational window, กำหนด tolerance หรือขยาย affected rule
- ต้องกำหนด boundary policy ของ affected window เมื่อชนต้น/ท้าย split และห้ามให้
  influence ข้าม split/gap โดยไม่ตั้งใจ
- AdaBoost variant/base learner, model hyperparameter spaces, scaling/imbalance
  policy และ random seeds: **ยังไม่ได้บันทึก**

## 10. ผลลัพธ์และ metrics ปัจจุบัน

### Data pipeline results

| ผลลัพธ์ | ค่า | หลักฐาน |
| --- | --- | --- |
| Raw/Clean rows | 2,512 / 2,512 | Manifest, `cleaning_report.json`, verification รอบนี้ |
| Raw date range | 2016-08-31 ถึง 2026-08-28 | Manifest/verification |
| Valid regression targets | 2,507; 5 trailing NaN | `src/build_targets.py`; isolated `regression_target_report.json` รอบนี้ |
| Modeling-ready rows | 2,487; removed 25 | `data_split_report.json` |
| Gaps | 5 + 5 rows | `data_split_report.json` |
| Train | 1,733 rows, 2016-09-29 ถึง 2023-08-18 | `data_split_report.json` |
| Validation | 371 rows, 2023-08-28 ถึง 2025-02-19 | `data_split_report.json` |
| Test | 373 rows, 2025-02-27 ถึง 2026-08-21 | `data_split_report.json` |
| Train-only Q75 | 0.2530580184684854 (25.3058018468%) | `classification_threshold.json` |
| Class counts Train | Normal 1,300 / High 433 | `classification_threshold.json` |
| Class counts Validation | Normal 337 / High 34 | `classification_threshold.json` |
| Class counts Test | Normal 300 / High 73 | `classification_threshold.json` |
| M2 detector audit | Q1 `0.0029797377830751`, Q3 `0.0138707144726510`, IQR `0.0108909766895759`, threshold `0.0465436445413787`, direct Train spikes 19 | Public M2 fit/apply API + M1 contract |
| Tests รอบนี้ | 300 passed, 1 skipped | `python -m pytest -q` outputรอบล่าสุด |

Skipped test คือ symlink-safety case ที่ `tests/test_data_pipeline.py:355` เพราะ
Windows แจ้ง `WinError 1314` ว่า process ไม่มี privilege สร้าง symlink ไม่ใช่ model
test และไม่ใช่ test failure

### Model evaluation metrics

| Metric | สถานะ |
| --- | --- |
| Regression MAE | **ยังไม่ได้บันทึก** |
| Regression RMSE | **ยังไม่ได้บันทึก** |
| Regression R² | **ยังไม่ได้บันทึก** |
| Classification Confusion Matrix/Accuracy/Precision/Recall/Specificity/F1 | **ยังไม่ได้บันทึก** |
| Classification ROC-AUC/PR-AUC | **ยังไม่ได้บันทึก** |
| Training/validation loss และ performance curves | **ยังไม่ได้บันทึก** |

ห้ามตีความ data statistics, Q75 หรือ class ratios เป็น model performance

## 11. ปัญหาคงค้างและงานถัดไป

ลำดับแนะนำ:

1. หลัง clone ให้หา Snapshot จากผู้ส่ง ตรวจ checksum และรัน verification/tests
2. ใช้ M1 contract ที่ freeze baseline schema, checksums และ split boundaries แล้ว
3. ใช้ M2 Train-only Extreme-IQR detector ที่ implement/test แล้ว
4. ใช้ M3 affected-mask/boundary metadata API ที่ implement/test แล้ว
5. ใช้ M4 event/data-quality audit ที่ verified แล้วเป็น gate ก่อนสร้าง With-Spike/Non-Spike Train artifacts ใน M5
6. Flag Validation/Test เป็น diagnostic segments ด้วย Train threshold เดิม
7. สร้าง experiment reports/checksums/class distributions และ figures ที่เหลือ
8. Freeze model algorithms, from-scratch/reference contract, dependencies,
   hyperparameters, seeds, preprocessing และ metrics
9. Implement paired regression/classification models แล้วเลือกด้วย Validation
10. ประเมิน Test ครั้งสุดท้ายและบันทึก models, predictions, metrics และ configs

เริ่ม prototype กับ Original/With-Spike Train ได้ทันทีหลัง Phase 1 verification แต่
ยังสรุปผลเปรียบเทียบผลของ spike filtering ไม่ได้จนกว่า Non-Spike artifacts และ
protocol จะพร้อม

ข้อจำกัดที่ตรวจพบ:

- Symlink test ถูก skip บน Windows เนื่องจากไม่มี privilege
- Repository-wide Ruff ยังมี 21 errors ใน Phase 1 modules/notebooks (`I001`,
  `F401`, `RUF046`); targeted M1–M3 lint ผ่าน และรายละเอียดรายไฟล์อยู่ใน
  `PHASE2_SPIKE_READINESS.md` โดยยังไม่แก้นอก scope
- Root reports บางไฟล์เป็น historical format ที่มี absolute paths
- Root ไม่มี `regression_target_report.json` แม้ README และ runner ปัจจุบันระบุไฟล์นี้;
  isolated runner สร้างได้สำเร็จ
- ไม่มี exported figures, model files หรือ metric files
- Scikit-learn ยังไม่อยู่ใน dependencies
- Python required version, dependency pins/lockfile, original retrieval time และ
  dataset license/terms **ยังไม่ได้บันทึก**

เกณฑ์ตรวจทันทีหลัง clone:

- branch/commit ตรงกับงานที่รับมาและ `git status` ไม่มี change ที่ไม่รู้ที่มา
- Manifest ถูก track แต่ Snapshot ต้องได้รับแยกและ checksum ต้องตรง
- สร้าง environment ได้และ `pip check` ผ่าน
- `verify-snapshot`, isolated runner และ tests ให้ผลสอดคล้องกับตารางข้างต้น
- ห้ามนำ generated/ignored artifacts เก่ามาปะปนโดยไม่ตรวจ provenance

## 12. ตรวจข้อมูลลับก่อนส่งต่อ

ตรวจชื่อ tracked files และค้นคำที่สื่อถึง API key, token, password, credential,
secret และ private key โดยไม่พิมพ์ค่าที่อาจเป็นความลับ ไม่พบ `.env`, credential
file, PEM/private key หรือค่า secret ในไฟล์ config/examples ที่เอกสารนี้อ้างถึง

`.gitignore` กัน `.env`, `.env.*` (ยกเว้น `.env.example`), `*.pem` และ `*.key`
แต่ผู้รับงานยังต้องตรวจ staged diff ก่อน commit ทุกครั้ง:

```powershell
git status --short
git diff --cached --name-only
```

หากภายหลังพบไฟล์ลับ ให้หยุดการส่งต่อ, remove จาก staging/history ตามขั้นตอนของทีม,
rotate credential และส่งค่าใหม่ผ่าน secret manager ห้ามวางค่าไว้ใน `HANDOFF.md`,
README, notebooks, reports หรือ commit

## หลักฐานที่ตรวจในรอบนี้

ตรวจอย่างน้อย: `.gitignore`, Git history/status, `README.md`, `PROJECT_STATUS.md`,
`config.py`, `requirements.txt`, Snapshot Manifest/CSV, `src/` ทุกส่วนที่เกี่ยวข้อง,
`tests/`, Notebook 01-04, generated datasets, `outputs/reports/` และ output
directories สำหรับ figures/models/metrics

คำสั่งที่รันจริง:

- `.venv/Scripts/python.exe src/download_qqq_data.py verify-snapshot` - สำเร็จ
- `.venv/Scripts/python.exe -m src.run_data_pipeline --output-root <isolated-temp-root>`
  - สำเร็จ สร้าง 14 artifacts แล้วลบเฉพาะ validation root ชั่วคราว
- `.venv/Scripts/python.exe -m pip check` - `No broken requirements found.`
- `.venv/Scripts/python.exe -m src.audit_spikes --output-root <isolated-temp-root>` - สำเร็จ; 23 events trace ครบ, ไม่มี finding
- `.venv/Scripts/python.exe -m pytest -q` - `307 passed, 1 skipped`

สถานะปลายทาง: Phase 1 และ Phase 2 M1–M4 **Implemented / Verified**; experiment
datasets, model train/evaluate และ forecast ยังเป็น **Planned**
