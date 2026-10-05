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

## Pipeline เดิม

ติดตั้ง dependencies แล้วรันตามลำดับ:

```powershell
python -m pip install -r requirements.txt
python src/download_qqq_data.py verify-snapshot
python src/clean_data.py
python src/build_features.py
python src/build_targets.py
python src/split_data.py
python -c "from src.build_targets import run_classification_target_pipeline; run_classification_target_pipeline()"
python -m pytest -q -p no:cacheprovider
```

`src/build_targets.py` ปัจจุบันสร้าง regression target ผ่าน CLI ส่วน classification labels เรียกผ่าน `run_classification_target_pipeline()` ตาม contract ใน source code จนกว่าจะมี end-to-end runner ใน Phase 1
