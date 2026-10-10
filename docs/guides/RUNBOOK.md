# QQQ Volatility Forecasting - คู่มือการติดตั้งและรัน Pipeline

เอกสารนี้เป็นวิธีรันหลักสำหรับ Windows Terminal จาก project root โดย frozen
submission environment ปัจจุบันคือ **Python 3.14.8** (ดู audit trail ด้านล่าง)
คำสั่งใช้ Python ใน `.venv` โดยตรง เพื่อลดความเสี่ยงจากการเรียก
Python หรือ packages คนละ environment

เอกสารที่เกี่ยวข้อง:

- [README.md](../../README.md) — ภาพรวมโปรเจกต์และ Quick Start
- [HANDOFF.md](../status/HANDOFF.md) — รายละเอียดการส่งต่องานและ data contracts
- [PHASE2_SPIKE_READINESS.md](../archive/PHASE2_SPIKE_READINESS.md) — หลักฐานการตรวจรับ Phase 2
- [PROJECT_STATUS.md](../status/PROJECT_STATUS.md) — สถานะ implementation
- [MODEL_TRAINING_PLAN.md](../archive/MODEL_TRAINING_PLAN.md) — modeling requirements,
  inventory 19 รายการ และ implementation milestones; คำสั่ง modeling ในแผนยังไม่พร้อมใช้

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
- Raw Snapshot `data/raw/qqq_daily.csv` ที่ได้รับแยกจาก Git

`.venv`, Raw Snapshot และ generated artifacts ไม่ถูก track ใน Git ผู้รับงานจึงต้อง
สร้าง environment และรับ Snapshot แยกหลัง clone

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

นำ `qqq_daily.csv` ที่ได้รับจากชุดส่งมอบไปวางที่:

```text
data\raw\qqq_daily.csv
```

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

Windows อาจ skip tests ที่ต้องใช้ symlink privilege ตามข้อจำกัดที่บันทึกใน
[PHASE2_SPIKE_READINESS.md](../archive/PHASE2_SPIKE_READINESS.md)

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

ผลที่คาดหวัง — บรรทัดสรุปต่อโมเดล ต้องลงท้ายด้วย `reload=ok`:

```text
     multiple_linear | with_spike | candidates= 1 | selected=multiple_linear-00 | reload=ok
```

Artifact จะอยู่ที่ `outputs/modeling/<variant>/<task>/<model>/` ประกอบด้วย `config.json`,
`model.npz`, `preprocessor.npz`, `search_results.json`, `validation_metrics.json`,
`validation_predictions.csv`, `load_verification.json` และ `history.json`

**`load_verification.json` ต้องมี `"passed": true`** — เป็นหลักฐานว่า save → load → ทำนายซ้ำ
ได้ผลเท่าเดิม ซึ่งเป็นข้อบังคับของโจทย์

คำสั่ง `report` และ `finalize` ยังไม่ implement (คืน exit 1 พร้อมบอกเหตุผล) เพราะต้องรอให้มี
โมเดลครบก่อน ดู [`docs/tasks/claude-linear-family.md`](../tasks/claude-linear-family.md) Phase C

> `finalize` เป็นจุดเดียวที่อ่าน Test set และต้องส่ง `--allow-test` อย่างชัดแจ้ง
> ห้ามรันก่อนเลือก hyperparameter ของทุกโมเดลจาก Validation เสร็จ

## 11. ครั้งแรกหลัง clone — Phase 1 reproduction

Phase 2 test ต้องการโฟลเดอร์ reproduction ที่ `outputs/reports/spike_input_contract.json`
บันทึก provenance ไว้ ถ้าไม่มีจะ fail ด้วย `Missing Phase 1 reproduction root`

```bat
.\.venv\Scripts\python.exe -m src.run_data_pipeline --output-root "tmp/phase1-reproduction" --overwrite-generated
```

ใช้ชื่อ `tmp/phase1-reproduction` เท่านั้นและห้ามลบทิ้ง หากเคยรันด้วยชื่ออื่นแล้วลบโฟลเดอร์นั้นไป
ให้รันคำสั่งข้างบนแล้วตามด้วย `run_spike_analysis --reproduced-root "tmp/phase1-reproduction" --overwrite-generated`
เพื่อเขียน provenance ใหม่
