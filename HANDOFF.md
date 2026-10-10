# QQQ Volatility Forecasting - เอกสารส่งต่องาน

> **คู่มือการติดตั้งและรันหลัก:** [RUNBOOK.md](RUNBOOK.md) ใช้ Python 3.14.8
> (re-frozen 2026-10-10; รอบตรวจรับเดิมใช้ 3.11.9 และผ่าน 3.12.3 มาแล้ว ดู audit
> trail ใน RUNBOOK.md) และแยกขั้นตอนสำหรับ clone ใหม่, การรันซ้ำที่ project root
> และ isolated verification คำสั่งในส่วนหลักฐานด้านล่างเป็นบันทึกของรอบตรวจรับเดิม
>
> **Modeling handoff ล่าสุด:** [MODEL_TRAINING_PLAN.md](MODEL_TRAINING_PLAN.md)
> เป็น source of truth สำหรับ inventory 19 รายการ, With-Spike-first, scratch/reference,
> milestones และ Non-Spike reruns

เมื่อ workspace มี Snapshot, Phase 1 artifacts และ accepted Phase 2 contract ครบแล้ว:

```bat
.\.venv\Scripts\python.exe -m src.run_data_pipeline --overwrite-generated
.\.venv\Scripts\python.exe -m src.run_spike_analysis --overwrite-generated
.\.venv\Scripts\python.exe -m pytest -q
```

หลักฐาน M8 เดิมบันทึกวันที่ **2026-10-07** บน branch `feature/spike-analysis`
ที่ commit `ef3ff2d` ส่วน audit เอกสารล่าสุดวันที่ **2026-10-08** ตรวจ checkout
`main` ที่ `03dfada`; ต้องแยกสองช่วงเวลานี้และไม่เขียนผลเดิมเหมือนเพิ่งรันใหม่

สถานะที่ใช้ตลอดเอกสาร:

- **Implemented** - มี source implementation ใน repository
- **Verified** - มี artifact/command evidence โดยต้องดูวันที่กำกับ; audit เอกสาร
  2026-10-08 ไม่ได้รัน full suite ใหม่
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
- **ข้อมูลพร้อม แต่โค้ดโมเดลยังไม่มี:** Original/With-Spike/Non-Spike Train,
  Validation และ Test พร้อมเป็น tabular inputs แต่ `src/models/` ยังไม่มี model
  implementation
- **วางแผนไว้:** 19-item model inventory, With-Spike-first training, Non-Spike
  fresh reruns, evaluation และ persisted model/metric artifacts

คำว่า **With-Spike** ในเอกสารนี้หมายถึง Original Train ที่ยังเก็บ extreme events
ไว้ ไม่ได้หมายความว่าโมเดล baseline ถูก train แล้ว ส่วน **Non-Spike** เป็น experiment
ที่กรอง `is_spike_affected` เฉพาะสำเนา Train ใน M5; artifacts พร้อมแล้วแต่ยังไม่มี
model training หรือผลเปรียบเทียบ

## 2. สิ่งที่มีอยู่ใน repository แล้ว

ไม่สามารถพิสูจน์จาก repository เพียงอย่างเดียวว่าใครเป็นผู้สร้างแต่ละส่วน จึงใช้คำว่า
“มีอยู่ใน repository” แทนการระบุว่า Codex หรือบุคคลใดเป็นผู้แก้

| ส่วนงาน | สถานะ | หลักฐาน |
| --- | --- | --- |
| Immutable Snapshot verification | **Implemented / Verified** | `src/download_qqq_data.py`, Manifest, tests และ recorded verification 2026-10-07 |
| Cleaning/validation | **Implemented / Verified** | `src/clean_data.py`, `outputs/reports/cleaning_report.json` |
| 8 causal features | **Implemented / Verified** | `src/build_features.py`, `outputs/reports/feature_report.json` |
| 5-day regression target | **Implemented / Verified** | `src/build_targets.py`; recorded isolated run 2026-10-07 และ root reportที่ตรวจ 2026-10-08 |
| Chronological split 70/15/15 + gaps | **Implemented / Verified** | `src/split_data.py`, `outputs/reports/data_split_report.json` |
| Train-only Q75 classification labels | **Implemented / Verified** | `src/build_targets.py`, `outputs/reports/classification_threshold.json` |
| End-to-end data runner | **Implemented / Verified** | `src/run_data_pipeline.py`; recorded isolated run 2026-10-07 สร้าง 14 artifacts |
| Tests | **Implemented / Verified** | recorded 2026-10-07: `334 passed, 2 skipped`; M7 runner 11 passed, 1 symlink skipped, coverage 96% |
| Regression/Classification/Clustering models | **Planned** | inventory 4/12/2 ใน `MODEL_TRAINING_PLAN.md`; model filesยังเป็น docstring |
| Spike/Non-Spike pipeline | **Implemented / Verified through M7** | event audit, experiment CSVs, reports, primary figures และ separate runner พร้อม |
| Model results/figures | **Not recorded** | มี M4/M6 spike-analysis figures แล้ว แต่ยังไม่มี model/metric artifacts |

**Phase 1 - Baseline Data Preparation:** implementation ใช้งานได้ตาม recorded
verification และ audit นี้พบ root `regression_target_report.json` แล้ว รวมทั้ง
`data_split_report.json`/`classification_threshold.json` รุ่นปัจจุบันที่ใช้
report-relative paths ประวัติที่เอกสารเก่าเคยพบ absolute paths ยังคงเป็นหลักฐานของ
artifact รุ่นก่อน ไม่ใช่สถานะปัจจุบันของไฟล์ root

**Phase 2 - Spike Analysis/Experiment Dataset Preparation:** **M1–M8 verified; Gate OPEN** มี input contract, detector, affected mask, direct-spike audit, With-Spike/Non-Spike Train, diagnostic Validation/Test copies, reports, figures, separate runner และ verification gate แล้ว แต่ยังไม่มี model training/evaluation ดูหลักฐานที่ `PHASE2_SPIKE_READINESS.md` และหัวข้อ M8 ด้านล่าง

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
| `src/build_experiment_datasets.py` | Tracked | M5 builder: ตรวจ M1/M4 gates แล้วสร้าง primary experiment/diagnostic copies โดย reuse M2/M3 | Original labeled splits + M1/M4 artifacts | 4 generated CSVs ใต้ `data/processed/experiments/` |
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
| `outputs/figures/` | `.gitkeep` tracked; contents ignored | พื้นที่ figure exports | Phase 2 report runner | มี spike-analysis figures 3 ไฟล์เมื่อรัน M7; ยังไม่มี model-performance figures |
| `outputs/models/` | `.gitkeep` tracked; contents ignored | พื้นที่ persisted models | ยังไม่มี model training | ไม่มี model files ปัจจุบัน |
| `outputs/metrics/` | `.gitkeep` tracked; contents ignored | พื้นที่ persisted metrics/predictions | ยังไม่มี evaluation | ไม่มี metric files ปัจจุบัน |
| `README.md` | Tracked | Runbook และ model/evaluation plan | หลักฐานใน repo + requirement summary | Documentation |
| `PROJECT_STATUS.md` | Tracked | Phase status, spike plan และ open questions | Audit ก่อนหน้า | Documentation; อาจมีข้อความเชิงแผน |
| `requirements.txt` | Tracked | Dependency names แบบไม่ pin version | pip | ติดตั้ง environment ที่ไม่ล็อกเวอร์ชัน |

หลัง clone จะได้ Manifest และ `.gitkeep` แต่จะ **ไม่ได้** Raw Snapshot, generated
data, reports, figures, models หรือ metrics เพราะ `.gitignore` กันไฟล์เหล่านี้ไว้

## 4. ติดตั้งจากเครื่องใหม่

### เวอร์ชันและ dependencies

- Python **3.11.9** เป็นเวอร์ชันของหลักฐานตรวจรับเดิม; `.venv` ใน checkout ที่
  audit วันที่ 2026-10-08 รายงาน **Python 3.12.3** M2 freeze เดิมตาม
  `configs/modeling.json`/`config_schema.py`; audit วันที่ 2026-10-10 re-freeze
  เป็น **Python 3.14.8** หลังยืนยัน pipeline/tests ให้ผลตรงกับหลักฐานเดิม (ดู
  audit trail ใน [RUNBOOK.md](RUNBOOK.md)) ปัจจุบันยังไม่มี `.python-version`,
  `pyproject.toml`, CI หรือ environment file บังคับเวอร์ชันเดียวแบบ machine-readable
  นอกจาก modeling config gate
- `requirements.txt` ระบุ `numpy`, `pandas`, `pytest`, `matplotlib`, `seaborn`,
  `nbformat`, `nbconvert`, `ipykernel` โดย **ไม่ pin versions**
- ไม่มี lockfile (`poetry.lock`, `Pipfile.lock` หรือ equivalent)
- เวอร์ชัน packages ที่บันทึกจาก environment ในรอบ 2026-10-07: NumPy 2.4.6, pandas 3.0.6,
  pytest 9.1.1, Matplotlib 3.11.2, seaborn 0.13.2, nbformat 5.11.1,
  nbconvert 7.17.1 และ ipykernel 7.4.0
- Scikit-learn และ XGBoost ถูกกล่าวถึงใน model plan แต่ **ไม่อยู่ใน
  `requirements.txt`**; version/submission environment ที่จะใช้ยังต้อง freeze

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
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
| Date range | 2016-08-31 ถึง 2026-08-28 | Manifest + recorded verification 2026-10-07 |
| Rows | 2,512 | Manifest + recorded verification 2026-10-07 |
| Schema | `Date, Close, Volume, Open, High, Low` | Manifest + recorded verification 2026-10-07 |
| SHA-256 | `649e1db1b79c0990ea947a4ea162d6836b9bbdda2af24bc5ab6ae66cb83b6b13` | Manifest + recorded verification 2026-10-07 |
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
.\.venv\Scripts\python.exe src\download_qqq_data.py verify-snapshot
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root tmp\phase1-handoff-check
.\.venv\Scripts\python.exe -m src.run_spike_analysis --output-root tmp\phase2-m8-verification-20261007
```

ถ้า artifact ใดใน output root มีอยู่แล้ว runner จะหยุดที่ preflight โดยไม่เขียนทับ
ใช้ชื่อ output root ใหม่เป็นค่าเริ่มต้นที่ปลอดภัย `--overwrite-generated` อนุญาตให้
แทนที่เฉพาะ 14 outputs ที่ runner ประกาศไว้ และยังไม่อนุญาตให้แตะ Snapshot/Manifest
หาก stage หลัง preflight ล้มเหลว outputs จาก stage ที่สำเร็จหรือสำเร็จบางส่วนอาจค้าง
อยู่ ต้องตรวจและเลือก output root ใหม่ก่อน retry

M8 ตรวจรับ Phase 2 command ข้างต้นแล้ว: `--output-root` เปลี่ยนเฉพาะ 11 generated Phase 2 artifacts และยังอ่าน accepted baseline จาก project root หากต้องการ baseline อื่นต้องใช้ `--input-root` แยก Default ไม่ overwrite; rerun ที่ root เดิมโดยไม่มี flag ถูกทดสอบให้ fail ที่ preflight และ protected checksums ยังคงเดิม

Phase 2 must-have inventory มี reports 4, experiment CSVs 4 และ figures 3 รวม 11 ไฟล์ ผลตรวจคือ Train direct/affected 19/213, With-Spike/Non-Spike 1,733/1,520, Validation/Test affected 0/54 และ Non-Spike classes 1,240/280 Full Validation/Test ยังคงเป็น Original labeled splits ส่วน flagged copies เป็น diagnostics เท่านั้น

**M8 gate: OPEN สำหรับเริ่ม paired model training ขั้นถัดไป** — spike-specific `142 passed, 1 skipped`, full suite `334 passed, 2 skipped`, targeted Ruff/mypy ผ่าน, inventory 11/11 และ protected checksum drift = 0 Skips ทั้งสองเป็น Windows symlink privilege ไม่ใช่ passed และไม่ใช่ must-have failure Repository-wide Ruff ยังมี known issues เดิม 21 รายการนอก Phase 2 scope

ข้อจำกัดที่ต้องส่งต่อ: Wilder RSI อาจได้รับอิทธิพลหลัง `s+19`; window
`[s-5,s+19]` และ within-split clipping เป็น operational policy ไม่ใช่หลักฐานว่า
cross-boundary/spike influence เป็นศูนย์ทุก feature ณ รอบ M8 root reports ยังเป็น
historical format; checkout ที่ audit 2026-10-08 มี portable root reportsแล้ว

ยังไม่มี model training, metrics หรือ performance results จาก M8

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

With-Spike/Non-Spike splits และ Phase 2 artifacts พร้อมเป็น inputs แต่ยังไม่ควรอ้าง
ผลทดลอง ลำดับคือ With-Spike-first แล้ว fit ใหม่สำหรับ Non-Spike; paired conclusion
ต้องรอ model resultsครบทั้งสอง variants

### แผนโมเดลและเกณฑ์ประเมิน

แผน authoritative อยู่ใน [MODEL_TRAINING_PLAN.md](MODEL_TRAINING_PLAN.md) และอ้าง
Course PDF ที่ตรวจครบ 5 หน้าแล้ว Inventory มี Regression 4,
Classification 13 และ Clustering 2 โดยแยก Perceptron และ SLP เป็นคนละรายการ
ตาม course interpretation ที่ผู้ใช้ยืนยันเมื่อ 2026-10-08

#### Regression

| โมเดล | Target | Metrics | สถานะโมเดล | สถานะ metrics |
| --- | --- | --- | --- | --- |
| Simple Linear, Multiple Linear, Polynomial, Elastic Net | `target_volatility_5d` | Loss/MSE, MAE, RMSE, R² และ baselines | **Planned** - ยังไม่มี scratch/reference implementation | ผลจริง **ยังไม่ได้บันทึก** |

#### Classification

| โมเดล | Target | Metrics/artifacts | สถานะโมเดล | สถานะ metrics |
| --- | --- | --- | --- | --- |
| 12 classifiers ตาม inventory | `target_high_volatility` | Confusion Matrix, Accuracy, Precision, Recall, Specificity, F1, ROC/PR metrics | **Planned** - ยังไม่มี scratch/reference implementation | ผลจริง **ยังไม่ได้บันทึก** |

Clustering มี k-Means และ Agglomerative แยกจาก supervised metrics; ต้องใช้ internal
metrics/stability และห้าม fit ใหม่บน Test แล้วอ้างว่าเป็น forecast

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
| 1 | `py -3 -m venv .venv` (Windows) หรือ `python3 -m venv .venv` (macOS/Linux) | สร้าง virtual environment | **Not recorded** - รอบ 2026-10-07 ไม่ได้สร้าง environment ใหม่ |
| 2 | `python -m pip install -r requirements.txt` หลัง activate | ติดตั้ง dependencies; ต้องใช้ network/cache | **Not recorded** - environment มีอยู่แล้ว; requirements ไม่ pin |
| 3 | `python --version` และ `python -m pip check` | ตรวจ interpreter และ dependency consistency | **Recorded 2026-10-07:** Python 3.11.9 และ `No broken requirements found.`; audit 2026-10-08 พบ `.venv` เป็น 3.12.3 แต่ไม่ได้รัน `pip check` ใหม่ |
| 4 | `python src/download_qqq_data.py verify-snapshot` | แสดง 2,512 rows, date range และ SHA-256 ตรง Manifest | **Recorded 2026-10-07:** verified offline |
| 5 | `python -m src.run_data_pipeline --output-root tmp/phase1-handoff-check` | สร้าง 14 artifacts ใต้ isolated root | **Recorded 2026-10-07:** สำเร็จ; offline |
| 6 | `Get-ChildItem tmp/phase1-handoff-check -Recurse -File` | เห็น interim 3, processed 6, reports 5 files | **Recorded 2026-10-07:** verified |
| 7 | `python -m pytest -q -rs` | Baseline และ Phase 2 M1–M8 tests ผ่าน; แสดงเหตุผลของ skipped tests | **Recorded 2026-10-07:** `334 passed, 2 skipped` |
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
- Runner และ root reports ที่ตรวจใน audit นี้ใช้ paths แบบ relative ต่อ report directory;
  ประวัติ reports รุ่นเก่าที่เคยเป็น absolute pathsบันทึกไว้ใน readiness evidence
- Stage failure หลังเริ่มเขียนอาจทิ้ง partial outputs; ข้อความ error ระบุ stage

Spike design ต่อไปนี้เป็น **Implemented / Verified through M8 ตาม recorded evidence**;
modeling ที่ใช้ artifacts เหล่านี้ยังเป็น **Planned**:

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
| Raw/Clean rows | 2,512 / 2,512 | Manifest, `cleaning_report.json`, recorded verification 2026-10-07 |
| Raw date range | 2016-08-31 ถึง 2026-08-28 | Manifest/verification |
| Valid regression targets | 2,507; 5 trailing NaN | `src/build_targets.py`; recorded isolated report 2026-10-07 |
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
| Tests | 334 passed, 2 skipped | Recorded M8 evidence วันที่ 2026-10-07; audit เอกสารนี้ไม่ได้รันใหม่ |

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
2. ยืนยันว่า Phase 1/2 accepted artifacts/checksumsตรงกับ contract; ไม่สร้าง Phase 2 ซ้ำโดยไม่จำเป็น
3. ใช้ `MODEL_TRAINING_PLAN.md` freeze 19 algorithms, scratch/reference contract,
   dependencies, hyperparameters, seeds, preprocessing และ metrics
4. ทำ Train/Validation-only pilotหนึ่งโมเดลทั้งสอง variantsเพื่อพิสูจน์ reuse โดยไม่เปิด Test
5. หยุด Non-Spikeหลัง pilot แล้วทำ With-Spike inventory/submission packageให้ครบ
6. Fit Non-Spikeใหม่ด้วย fresh scaler/PCA/model หลัง With-Spike milestone
7. ทำ paired comparisonเมื่อ model resultsครบทั้งสอง variantsเท่านั้น

With-Spike/Non-Spike data artifactsพร้อมแล้ว แต่ยังสรุปผลของ spike filteringไม่ได้
จนกว่า model training/evaluationครบทั้งสอง variantsภายใต้ frozen protocol

ข้อจำกัดที่ตรวจพบ:

- Symlink test ถูก skip บน Windows เนื่องจากไม่มี privilege
- Repository-wide Ruff ยังมี 21 errors ใน Phase 1 modules/notebooks (`I001`,
  `F401`, `RUF046`); targeted M1–M8 lint ผ่าน และรายละเอียดรายไฟล์อยู่ใน
  `PHASE2_SPIKE_READINESS.md` โดยยังไม่แก้นอก scope
- Root reports ปัจจุบันมี portable paths แต่ readiness docsยังเก็บประวัติ format เก่า
- Root มี `regression_target_report.json` แล้ว
- มี Phase 2 spike-analysis figures 3 ไฟล์ แต่ยังไม่มี model-performance figures,
  model files หรือ metric files
- Scikit-learn ยังไม่อยู่ใน dependencies
- Recorded verification ใช้ Python **3.11.9**; `.venv` ที่ audit 2026-10-08 เป็น
  **3.12.3** และ submission environment/dependency pins/lockfile,
  original retrieval time และ dataset license/terms **ยังไม่ได้บันทึก**

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

## หลักฐาน recorded จากรอบตรวจรับ 2026-10-07

ตรวจอย่างน้อย: `.gitignore`, Git history/status, `README.md`, `PROJECT_STATUS.md`,
`config.py`, `requirements.txt`, Snapshot Manifest/CSV, `src/` ทุกส่วนที่เกี่ยวข้อง,
`tests/`, Notebook 01-04, generated datasets, `outputs/reports/` และ output
directories สำหรับ figures/models/metrics

คำสั่งที่บันทึกว่ารันจริงในรอบ 2026-10-07 (audit เอกสาร 2026-10-08 ไม่ได้รันซ้ำ):

- `.venv/Scripts/python.exe src/download_qqq_data.py verify-snapshot` - สำเร็จ
- `.venv/Scripts/python.exe -m src.run_data_pipeline --output-root <isolated-temp-root>`
  - สำเร็จ สร้าง 14 artifacts แล้วลบเฉพาะ validation root ชั่วคราว
- `.venv/Scripts/python.exe -m pip check` - `No broken requirements found.`
- `.venv/Scripts/python.exe -m src.audit_spikes --output-root <isolated-temp-root>` - สำเร็จ; 23 events trace ครบ, ไม่มี finding
- `.venv/Scripts/python.exe -m src.build_experiment_datasets --output-root <isolated-temp-root>` - สำเร็จ; สร้าง 4 CSV โดยไม่แก้ protected inputs
- `.venv/Scripts/python.exe -m src.run_spike_analysis --output-root tmp/phase2-m7-validation-20261007` - สำเร็จ; สร้าง 11 declared Phase 2 artifacts โดยอ่าน baseline จาก project root
- `.venv/Scripts/python.exe -m pytest -q` - ดูผลล่าสุดใน `PHASE2_SPIKE_READINESS.md`

สถานะปลายทาง: Phase 1 และ Phase 2 M1–M8 **Implemented / Verified; Gate OPEN**;
model train/evaluate และ forecast ยังเป็น **Planned**
