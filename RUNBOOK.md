# QQQ Volatility Forecasting - คู่มือการติดตั้งและรัน Pipeline

เอกสารนี้เป็นวิธีรันหลักสำหรับ Windows Terminal จาก project root โดยตรวจรับกับ
**Python 3.11.9** คำสั่งใช้ Python ใน `.venv` โดยตรง เพื่อลดความเสี่ยงจากการเรียก
Python หรือ packages คนละ environment

เอกสารที่เกี่ยวข้อง:

- [README.md](README.md) — ภาพรวมโปรเจกต์และ Quick Start
- [HANDOFF.md](HANDOFF.md) — รายละเอียดการส่งต่องานและ data contracts
- [PHASE2_SPIKE_READINESS.md](PHASE2_SPIKE_READINESS.md) — หลักฐานการตรวจรับ Phase 2
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — สถานะ implementation

## 1. สิ่งที่ต้องมี

- Windows และ Command Prompt หรือ VS Code Terminal
- Python 3.11.9
- Repository นี้
- Raw Snapshot `data/raw/qqq_daily.csv` ที่ได้รับแยกจาก Git

`.venv`, Raw Snapshot และ generated artifacts ไม่ถูก track ใน Git ผู้รับงานจึงต้อง
สร้าง environment และรับ Snapshot แยกหลัง clone

## 2. สร้าง virtual environment

รันจาก project root:

```bat
py -3.11 -m venv .venv
```

ตรวจว่า environment ใช้ Python 3.11.9:

```bat
.\.venv\Scripts\python.exe --version
```

ผลที่คาดหวัง:

```text
Python 3.11.9
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
[PHASE2_SPIKE_READINESS.md](PHASE2_SPIKE_READINESS.md)

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
