# QQQ Volatility Forecasting - คู่มือการติดตั้งและรัน Pipeline

เอกสารนี้เป็นวิธีรันหลักสำหรับ Windows Terminal จาก project root โดย frozen
submission environment ปัจจุบันคือ **Python 3.14.8** (ดู audit trail ด้านล่าง)
คำสั่งใช้ Python ใน `.venv` โดยตรง เพื่อลดความเสี่ยงจากการเรียก
Python หรือ packages คนละ environment

เอกสารที่เกี่ยวข้อง:

- [README.md](../../README.md) — ภาพรวมโปรเจกต์และ Quick Start
- [PROJECT_GUIDE_TH.md](PROJECT_GUIDE_TH.md) — คำอธิบายทั้งโปรเจกต์ตั้งแต่ข้อมูลจนถึงผล
- [DATA_PIPELINE_AUDIT.md](../audit/DATA_PIPELINE_AUDIT.md) — ผลตรวจชั้นข้อมูล

> ⚠️ **แก้ไข 2026-10-11:** บันทึกด้านล่างเป็นประวัติ `src/modeling/config_schema.py`,
> `tests/modeling/test_dependencies.py` และ `python -m src.modeling.config_sync` ถูกลบไปแล้ว
> (คำสั่งนั้นจะ fail ด้วย `ModuleNotFoundError`) และคำสั่ง `src.ml.run train` **ใช้งานได้แล้ว** ตามข้อ 10
> คำสั่ง `report`, `compare`, `analyze` และ `finalize` ใช้งานได้แล้ว (ข้อ 10) · hash ที่เก็บใน report ใช้ได้บน Windows เท่านั้น
> (ดู [PATH_AUDIT.md](../audit/PATH_AUDIT.md))
>
> หลักฐานตรวจรับเดิมใช้ Python 3.11.9 (recorded 2026-10-07); audit วันที่ 2026-10-08
> พบว่า `.venv` รายงาน Python 3.12.3 แทน M2 จึง freeze submission/modeling environment
> ไว้ที่ 3.12.3 ใน `configs/modeling.json` และ `src/modeling/config_schema.py`
>
> **อัปเดต 2026-10-10:** re-freeze เป็น **Python 3.14.8** (เวอร์ชันล่าสุดที่มีในเครื่อง
> ขณะตรวจสอบ) หลังยืนยันว่าสร้าง `.venv` ใหม่, ติดตั้ง `requirements-lock.txt`,
> `pip check`, `run_data_pipeline`, `run_spike_analysis` และ `pytest` ให้ผลตรงกับ
> ค่าที่บันทึกไว้ในคู่มือนี้ทุกจุด (Rows: 2512, Train/Validation/Test: 1733/371/373,
> Primary threshold: 0.0465436445413787) ก่อน commit การเปลี่ยนเวอร์ชันนี้ ได้ sync
> `configs/modeling.json`, `src/modeling/config_schema.py` และ
> `tests/modeling/test_dependencies.py` ให้ตรงกันแล้วด้วย
> `python -m src.modeling.config_sync --write-plan` ห้ามตีความว่าคำสั่ง
> train/evaluate มีอยู่แล้ว — ดูข้อ 10

## 1. สิ่งที่ต้องมี

- Windows และ Command Prompt หรือ VS Code Terminal
- Python 3.14.8
- Repository นี้
- Raw Snapshot `data/raw/qqq_daily.csv` (⚠️ แก้ไข 2026-10-11: อยู่ใน Git แล้ว ได้มาพร้อม clone)
- แหล่งดาวน์โหลด/อ้างอิง Raw Dataset: [Google Drive](https://drive.google.com/file/d/1x4PBm9WrHYX8uxSn_pVoLY7_QOHrTWCR/view?usp=sharing) — ใช้สำหรับเข้าถึงต้นทางเท่านั้น ห้ามนำมาเขียนทับ snapshot ที่ตรึงไว้โดยไม่ตรวจ manifest

`.venv` และ generated artifacts ใน `data/interim`, `data/processed` ไม่ถูก track ใน Git ผู้รับงานจึงต้อง
สร้าง environment แล้วรัน data pipeline เองหลัง clone (Raw Snapshot มากับ clone แล้ว)

## 2. สร้าง virtual environment

รันจาก project root:

```bat
py -3.14 -m venv .venv
```

ตรวจว่า environment ใช้ Python 3.14.8:

```bat
.\.venv\Scripts\python.exe --version
```

ผลที่คาดหวัง:

```text
Python 3.14.8
```

จะ activate environment ก่อนก็ได้:

```bat
.\.venv\Scripts\activate.bat
```

คู่มือนี้ยังคงใช้ `.\.venv\Scripts\python.exe` ในทุกคำสั่ง เพื่อระบุ interpreter
ให้ชัดเจนแม้ไม่ได้ activate

## 3. ติดตั้ง dependencies

```bat
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`pip check` ควรแสดง `No broken requirements found.`

## 4. เตรียมและตรวจ Raw Snapshot

ไฟล์ `qqq_daily.csv` มากับ clone แล้ว (track แบบ `-text` เพื่อคง byte เดิม) และต้องอยู่ที่:

```text
data\raw\qqq_daily.csv
```

Raw Dataset ต้นทางดาวน์โหลดได้จาก [Google Drive](https://drive.google.com/file/d/1x4PBm9WrHYX8uxSn_pVoLY7_QOHrTWCR/view?usp=sharing) แต่ pipeline ที่ทำซ้ำผลต้องใช้ snapshot ใน Git และผ่าน `verify-snapshot` ก่อนเสมอ

จากนั้นตรวจ Snapshot กับ tracked manifest:

```bat
.\.venv\Scripts\python.exe src\download_qqq_data.py verify-snapshot
```

ค่าหลักที่คาดหวัง:

```text
Rows: 2512
Date range: 2016-08-31 ถึง 2026-08-28
SHA-256: 649e1db1b79c0990ea947a4ea162d6836b9bbdda2af24bc5ab6ae66cb83b6b13
```

ห้ามแทน Snapshot ด้วย live download แล้วอ้างว่าเป็นชุดข้อมูลเดียวกัน เพราะแหล่งข้อมูล
แบบ rolling อาจให้ช่วงวันที่และจำนวนแถวต่างจาก Snapshot ที่ pin ไว้

## 5. ตั้งค่าครั้งแรกหลัง clone

### 5.1 สร้าง Phase 1 artifacts หลัก

```bat
.\.venv\Scripts\python.exe -m src.run_data_pipeline --overwrite-generated
```

ผลหลักที่คาดหวัง:

```text
Train: 1733 rows
Validation: 371 rows
Test: 373 rows
```

Artifacts จะอยู่ใต้ `data/interim/`, `data/processed/` และ `outputs/reports/`

### 5.2 สร้าง isolated Phase 1 reproduction

Phase 2 ต้องมีหลักฐานว่า Phase 1 สร้างซ้ำได้ตรงกับ accepted inputs สำหรับ clone ใหม่
ให้สร้าง reproduction แยกดังนี้:

```bat
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp\phase1-reproduction" --overwrite-generated
```

### 5.3 รัน Phase 2 ครั้งแรก

```bat
.\.venv\Scripts\python.exe -m src.run_spike_analysis --reproduced-root "tmp\phase1-reproduction" --overwrite-generated
```

Artifacts หลักจะอยู่ที่:

```text
data\processed\experiments\with_spikes\train.csv
data\processed\experiments\non_spike\train.csv
data\processed\experiments\diagnostics\validation_flagged.csv
data\processed\experiments\diagnostics\test_flagged.csv
outputs\reports\
outputs\figures\spike_analysis\
```

ผลหลักที่คาดหวัง:

```text
Primary threshold: 0.0465436445413787
With-Spikes Train: 1733 rows
Non-Spike Train: 1520 rows
Validation: 371 rows
Test: 373 rows
```

## 6. รันซ้ำใน workspace ที่ตั้งค่าแล้ว

เมื่อ Snapshot, Phase 1 artifacts และ accepted Phase 2 contract มีครบแล้ว ใช้สองคำสั่งนี้:

```bat
.\.venv\Scripts\python.exe -m src.run_data_pipeline --overwrite-generated
.\.venv\Scripts\python.exe -m src.run_spike_analysis --overwrite-generated
```

`--overwrite-generated` อนุญาตให้แทนที่เฉพาะ outputs ที่ runner ประกาศไว้ ไม่เขียนทับ
Raw Snapshot หรือ manifest

## 7. รัน Tests

```bat
.\.venv\Scripts\python.exe -m pytest -q
```

บน Windows อาจมี test บางข้อถูก skip เพราะต้องใช้สิทธิ์สร้าง symlink ซึ่งเป็นข้อจำกัดของระบบ
ไม่ใช่ข้อผิดพลาดของโค้ด

## 8. รันแบบ isolated โดยไม่แก้ artifacts หลัก

ใช้ output root ที่ยังไม่มีไฟล์ หรือใช้ชื่อรอบใหม่ ตัวอย่างสำหรับ Command Prompt:

```bat
set run=20261008-01
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp\phase1-%run%"
.\.venv\Scripts\python.exe -m src.run_spike_analysis --reproduced-root "tmp\phase1-%run%" --output-root "tmp\phase2-%run%"
```

ใน Command Prompt ตัวแปรใช้ `%run%` ส่วน PowerShell ใช้ `$run` หากพิมพ์ `$run`
ใน Command Prompt ระบบจะสร้างโฟลเดอร์ที่มีชื่อ `$run` ตามตัวอักษรจริง

เมื่อกำหนด `--output-root "tmp\phase2-%run%"` โฟลเดอร์ experiments จะอยู่ใต้:

```text
tmp\phase2-%run%\data\processed\experiments\
```

ไม่ได้อยู่ที่ `data\processed\experiments\` ของ project root

## 9. การแก้ปัญหาที่พบบ่อย

### Generated outputs already exist

Runner ปฏิเสธการเขียนทับโดยค่าเริ่มต้น เลือกอย่างใดอย่างหนึ่ง:

- ใช้ output root ชื่อใหม่สำหรับ isolated run
- เพิ่ม `--overwrite-generated` เมื่อตั้งใจอัปเดต declared artifacts เดิม

### No module named numpy หรือ matplotlib

กำลังใช้ Python ผิด environment หรือยังไม่ได้ติดตั้ง requirements ตรวจด้วย:

```bat
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### ไม่พบ experiments ที่ project root

ตรวจค่าที่แสดงหลัง `Output root:` หากกำหนด `--output-root tmp\...` outputs ทั้งหมด
จะอยู่ใต้ root นั้น หากต้องการอัปเดต `data\processed\experiments\` ที่ project root
ให้รัน Phase 2 โดยไม่กำหนด `--output-root`

## 10. Modeling commands

ชั้น modeling อยู่ที่ `src/ml/` และ CLI ใช้งานได้แล้ว (ยืนยันด้วยการรันจริง 2026-10-10)

```bat
rem เทรนโมเดลเดียว
.\.venv\Scripts\python.exe -m src.ml.run train --model multiple_linear --variant with_spike

rem เทรนทุกโมเดลที่ลงทะเบียนไว้
.\.venv\Scripts\python.exe -m src.ml.run train --all --variant with_spike

rem ใช้ hyperparameter grid จากไฟล์ {model: grid}
.\.venv\Scripts\python.exe -m src.ml.run train --model multiple_linear --variant non_spike --grid grids.json
```

ไฟล์ grid เขียนได้ 2 รูปแบบ ห้ามผสมกันในโมเดลเดียว:

```json
{
  "gradient_boosting": {"learning_rate": [0.03, 0.1], "max_depth": [1, 2]},
  "xgboost": {
    "tuples": [[50, 1, 0.1, 1, 0], [100, 2, 0.03, 10, 0.1]],
    "tuple_fields": ["n_estimators", "max_depth", "learning_rate", "reg_lambda", "gamma"]
  }
}
```

แบบแรกลองทุกคู่ (2 × 2 = 4 candidates) แบบที่สองลองเฉพาะชุดที่เขียน (2 candidates)
ใน `configs/modeling.json` ชื่อ field ของแบบที่สองอยู่ที่ `models.<name>.parameters.tuple_fields`

ผลที่คาดหวัง — บรรทัดสรุปต่อโมเดล ต้องลงท้ายด้วย `reload=ok`:

```text
     multiple_linear | with_spike | candidates= 1 | selected=multiple_linear-00 | reload=ok
```

Artifact จะอยู่ที่ `outputs/modeling/<variant>/<task>/<model>/` ประกอบด้วย `config.json`,
`model.npz`, `preprocessor.npz`, `search_results.json`, `validation_metrics.json`,
`validation_predictions.csv`, `load_verification.json` และ `history.json`

**`load_verification.json` ต้องมี `"passed": true`** — เป็นหลักฐานว่า save → load → ทำนายซ้ำ
ได้ผลเท่าเดิม ซึ่งเป็นข้อบังคับของโจทย์

### 10.1 สรุปผลจาก run ที่เทรนแล้ว (ไม่เทรนใหม่ ไม่แตะ Test)

```bat
rem leaderboard + รูปบน Validation — รันทั้งสอง variant (ราว 15 วินาทีต่อ variant)
.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike
.\.venv\Scripts\python.exe -m src.ml.run report --variant non_spike

rem เทียบ with_spike กับ non_spike แบบ paired (ราว 15 วินาที)
.\.venv\Scripts\python.exe -m src.ml.run compare

rem feature importance + walk-forward robustness (ราว 3–4 นาทีต่อ variant)
.\.venv\Scripts\python.exe -m src.ml.run analyze --variant with_spike
.\.venv\Scripts\python.exe -m src.ml.run analyze --variant non_spike
```

| คำสั่ง | เขียนไฟล์ที่ | สิ่งที่ควรเห็น |
| --- | --- | --- |
| `report` | `outputs/modeling/report/<variant>/leaderboard_*.{csv,md}` และ `outputs/figures/modeling/<variant>/*.png` | พิมพ์ path 12 บรรทัด · ตาราง classification มี 15 แถว (13 โมเดล + 2 baseline) |
| `compare` | `outputs/modeling/report/comparison_a_vs_b.{csv,md}` และรูป `comparison_delta_*.png` | พิมพ์ path 4 บรรทัด · ตาราง 19 แถว |
| `analyze` | `feature_importance.csv`, `linear_coefficients.csv`, `walk_forward.csv` ใต้ `outputs/modeling/report/<variant>/` | พิมพ์ path 4 บรรทัด |

ช่วงความเชื่อมั่นทุกตัวมาจาก moving-block bootstrap (ก้อน 20 วัน, 2,000 รอบ, seed คงที่) จึงได้ค่าเดิมทุกครั้งที่รัน

### 10.2 finalize — อ่าน Test ครั้งเดียว

```bat
rem ขั้นที่ 1: ล็อก hash ของ 38 run และชื่อ headline (ไม่เปิดไฟล์ Test)
.\.venv\Scripts\python.exe -m src.ml.run finalize --dry-run

rem ขั้นที่ 2: อ่าน Test หนึ่งครั้ง — ต้องได้รับอนุมัติจากเจ้าของโปรเจกต์ก่อน
.\.venv\Scripts\python.exe -m src.ml.run finalize --allow-test
```

> `finalize --allow-test` เป็นจุดเดียวที่อ่าน Test set มันจะปฏิเสธถ้ายังไม่มี `selection_lock.json`,
> ถ้า artifact ใดเปลี่ยนไปจากที่ล็อกไว้ หรือถ้ามี `outputs/modeling/finalize_receipt.json` อยู่แล้ว
> (แปลว่าเคยรันไปแล้ว) การรันซ้ำต้องใส่ `--rerun-reason "<เหตุผล>"` และรายงานต้องเปิดเผยทุกรอบ

## 11. ครั้งแรกหลัง clone — Phase 1 reproduction

Phase 2 test ต้องการโฟลเดอร์ reproduction ที่ `outputs/reports/spike_input_contract.json`
บันทึก provenance ไว้ ถ้าไม่มีจะ fail ด้วย `Missing Phase 1 reproduction root`

```bat
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp/phase1-reproduction" --overwrite-generated
```

ใช้ชื่อ `tmp/phase1-reproduction` เท่านั้นและห้ามลบทิ้ง หากเคยรันด้วยชื่ออื่นแล้วลบโฟลเดอร์นั้นไป
ให้รันคำสั่งข้างบนแล้วตามด้วย `run_spike_analysis --reproduced-root "tmp/phase1-reproduction" --overwrite-generated`
เพื่อเขียน provenance ใหม่

## 12. จาก clone เปล่าถึงผลครบ

ลำดับนี้ถูกทดสอบจริงเมื่อ 2026-10-11 โดย clone repo ไปโฟลเดอร์ใหม่แล้วรันตามทีละคำสั่ง
(รอบนั้นใช้ Python ของ environment เดิม จึงไม่ได้ทดสอบขั้นติดตั้ง library ในข้อ 2–3)

ทดสอบซ้ำเมื่อ 2026-10-13 บน clone ใหม่ของ commit `a9ae105` โดย **สร้าง virtual environment ใหม่และติดตั้ง library
จาก `requirements.txt` ใหม่ทั้งหมด** (บนเครื่องเดิม) ผลคือ `pip check` สะอาด ข้อมูลที่สร้างใหม่ 13 ไฟล์เหมือนของเดิมทุกไบต์
notebook 01–02 รันจบและได้รูปเหมือนเดิม และ `pytest -q` ได้ 1,267 passed

ทดสอบอีกครั้งเมื่อ 2026-10-13 หลังเพิ่ม notebook 06 โดยส่งออกไฟล์ที่ git ติดตามทั้ง 753 ไฟล์ไปโฟลเดอร์ว่าง
(ใช้ environment ที่สร้างจาก `requirements.txt` ในรอบก่อน) รันขั้น 3–5 และขั้น 10 แล้วรัน notebook 06 จาก kernel ใหม่
ได้ตารางและรูปเหมือนที่บันทึกไว้ทุกไบต์ และ `pytest -q` ได้ 1,308 passed

| ขั้น | คำสั่ง | เวลาโดยประมาณ | สิ่งที่ควรเห็น |
| --- | --- | --- | --- |
| 1 | สร้าง environment และติดตั้ง library ตามข้อ 2–3 | หลายนาที | `pip check` ไม่มีข้อผิดพลาด |
| 2 | `.\.venv\Scripts\python.exe src\download_qqq_data.py verify-snapshot` | ไม่กี่วินาที | SHA-256 ขึ้นต้นด้วย `649e1db1` |
| 3 | `.\.venv\Scripts\python.exe -m src.run_data_pipeline --overwrite-generated` | ไม่กี่วินาที | พิมพ์ path ของไฟล์ที่สร้าง |
| 4 | `.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp\phase1-reproduction" --overwrite-generated` | ไม่กี่วินาที | เหมือนขั้น 3 แต่อยู่ใต้ `tmp\` |
| 5 | `.\.venv\Scripts\python.exe -m src.run_spike_analysis --reproduced-root "tmp\phase1-reproduction" --overwrite-generated` | ไม่กี่วินาที | พิมพ์ path พร้อม SHA-256 ของรูป |
| 6 | `.\.venv\Scripts\python.exe -m pytest -q` | ราว 10 นาที | ทุกข้อ passed |
| 7 | `.\.venv\Scripts\python.exe -m src.ml.run report --variant with_spike` (และ `non_spike`) | ราว 15 วินาทีต่อครั้ง | พิมพ์ path 12 บรรทัด |
| 8 | `.\.venv\Scripts\python.exe -m src.ml.run compare` | ราว 15 วินาที | พิมพ์ path 4 บรรทัด |
| 9 | `.\.venv\Scripts\python.exe -m src.ml.run analyze --variant with_spike` (และ `non_spike`) | 3–4 นาทีต่อครั้ง | พิมพ์ path 4 บรรทัด |
| 10 | `.\.venv\Scripts\python.exe -m src.ml.docs_sync` | ราว 30 วินาที | ไม่พิมพ์อะไร (เอกสารตรงกับไฟล์ผลอยู่แล้ว) |
| 11 | เปิด `notebooks\03_regression.ipynb`, `04_classification.ipynb`, `05_results_comparison.ipynb`, `06_spike_analysis.ipynb` | — | ผลถูกบันทึกไว้ในไฟล์แล้ว รันซ้ำได้ |

**ไม่ต้องเทรนโมเดลใหม่** ผลของทั้ง 38 ชุดมากับ repo ใต้ `outputs\modeling\` ขั้น 7–10 สร้างตารางและรูปซ้ำจากไฟล์เหล่านั้น
และได้ผลเท่าเดิมทุกครั้งเพราะใช้ seed คงที่

**สิ่งที่อาจทำให้สงสัย**

- หลังขั้น 3–5 และขั้น 7–10 คำสั่ง `git status` อาจแสดงไฟล์ใต้ `outputs\reports\` และ `outputs\modeling\report\`
  ว่าถูกแก้ สาเหตุคือรูปแบบการขึ้นบรรทัดใหม่ (Windows เขียน CRLF ส่วน git เก็บ LF) เนื้อหาไม่ต่าง
  ตรวจได้ด้วย `git diff` ซึ่งจะไม่แสดงอะไร (ตรวจซ้ำเมื่อ 2026-10-13: `docs_sync` ทำให้ขึ้น 34 ไฟล์
  และ `report` 6 ไฟล์ โดย `git diff` ว่างทั้งสองกรณี)
- `finalize --allow-test` จะจบด้วยข้อความ `The held-out split was already evaluated` และ exit code 1 ซึ่ง **ถูกต้อง**
  เพราะ Test ถูกอ่านไปแล้วหนึ่งครั้ง (ข้อ 10.2)
- ถ้าเทรนโมเดลใหม่ด้วย `train` ไฟล์โมเดลจะไม่ตรงกับ `outputs\modeling\selection_lock.json` อีกต่อไป
  ใช้ `git checkout -- outputs/modeling` เพื่อกลับไปใช้ไฟล์ที่ล็อกไว้
