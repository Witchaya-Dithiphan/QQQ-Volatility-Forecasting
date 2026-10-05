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
