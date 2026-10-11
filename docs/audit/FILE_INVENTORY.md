# FILE INVENTORY — 2026-10-11

> ส่วนหนึ่งของ [`DATA_PIPELINE_AUDIT.md`](DATA_PIPELINE_AUDIT.md) · รายชื่อไฟล์ในโจทย์ audit บางส่วนล้าสมัย
> (`src/modeling/datasets.py`, `preprocessing.py`, `config_schema.py` ถูกลบแล้ว; เอกสารย้ายเข้า `docs/`)
> ตารางนี้ยึดตำแหน่งจริง

## Config

| ไฟล์ | คำตัดสิน |
| --- | --- |
| `config.py` | ใช้งาน · path 39 ตัว absolute ทั้งหมด |
| `configs/modeling.json` | ใช้งาน · block `persistence` ล้าสมัย (X5) |
| `pytest.ini` | ใช้งาน · `--basetemp=.pytest_tmp` แก้ปัญหา cross-drive |
| `requirements.in` / `.txt` / `-lock.txt` | ใช้งาน · 69 pin ตรง `.venv` |
| `.gitattributes` | แก้ใน audit นี้ (เพิ่ม csv/npz/png) |
| `.gitignore` | ใช้งาน · ตัด `data/raw/*` ยกเว้น snapshot ที่ track แล้ว (A13) |

## Data pipeline (`src/*.py`)

| ไฟล์ | บรรทัด | คำตัดสิน |
| --- | --- | --- |
| `download_qqq_data.py` | 435 | ใช้งาน · เพิ่ม guard manifest/report ใน audit นี้ |
| `load_data.py` | 81 | ใช้งาน · คืนข้อมูลใหม่สุดก่อน (B4) |
| `clean_data.py` | 277 | ถูกต้อง · เรียงและเปลี่ยนรูปแบบเท่านั้น |
| `build_features.py` | 376 | ถูกต้อง · causal ทั้ง 8 features |
| `build_targets.py` | 673 | ถูกต้อง · target t+1…t+5, Q75 จาก Train |
| `split_data.py` | 418 | ถูกต้อง · chronological + gap 5 |
| `run_data_pipeline.py` | 555 | ถูกต้อง · preflight กัน raw และ manifest |
| `report_paths.py` | 26 | ใช้งาน · บังคับไดรฟ์เดียวกัน |
| `detect_spikes.py` | 178 | ถูกต้อง · fence บน `abs(return_1d)` |
| `spike_contract.py` | 314 | ถูกต้อง · หน้าต่างนับภายใน split (F5) |
| `build_spike_input_contract.py` | 931 | ใช้งาน · ตรวจแบบ grep ไม่ได้อ่านทุกบรรทัด |
| `audit_spikes.py` | 780 | ใช้งาน · ตรวจแบบ grep |
| `build_experiment_datasets.py` | 570 | ถูกต้อง · กรองเฉพาะ Train |
| `build_spike_reports.py` | 514 | ใช้งาน · ตรวจแบบ grep |
| `run_spike_analysis.py` | 582 | ใช้งาน · ไม่ได้รันใน audit นี้ |
| `src/ml/` (35 ไฟล์) | 3,540 | ตัวโหลดข้อมูลคือ `core/data.py`, `core/contracts.py` |
| `src/modeling/` | — | ตาย · เหลือ `.pyc` 13 ไฟล์ |

## Tests (data layer)

| ไฟล์ | จำนวน | สิ่งที่ assert จริง |
| --- | --- | --- |
| `test_data_pipeline.py` | 21 | รัน pipeline จริงบน snapshot จริง: 14 artifact, ขนาด split, gap 5, label, checksum · แข็งแรง |
| `test_experiment_datasets.py` | 49 | `with_spikes` byte-identical, 1,520 แถว, 1240/280 |
| `test_spike_contract.py` | 36 | ขอบหน้าต่าง, clip, union, แยกต่อ split |
| `test_spike_detection.py` | 33 | fit บน abs, strictness, ข้อมูลนอก Train ไม่เปลี่ยน fit |
| `test_features.py` | 30 | สูตรแบบ mirror (C8) + RSI seed คำนวณมือ |
| `test_targets.py` | 30 | แข็งแรง: เทียบ `np.std` อิสระ, perturbation |
| `test_classification_target.py` | 29 | แก้ 2 test ที่ไม่มีทาง fail ใน audit นี้ |
| `test_split_data.py` | 27 | ข้อมูลสังเคราะห์ 100 แถว: gap, ไม่ overlap, ไม่ shuffle |
| `test_download_qqq_data.py` | 24 | mock เฉพาะเครือข่าย |
| `test_clean_data.py` | 23 | fixture ไม่ใช้รูปแบบ raw จริง (B8) |
| `test_spike_pipeline.py` / `test_spike_audit.py` / `test_spike_reports.py` | 12 / 7 / 6 | ความปลอดภัยของ I/O และ determinism |
| `test_load_data.py` / `test_report_paths.py` / `test_leakage.py` | 6 / 2 / 1 | `test_leakage` มี cutoff เดียวบนข้อมูลสังเคราะห์ (C9) |
| **`test_audit_guards.py`** (ใหม่) | 27 | ผูกกับ artifact จริง: causality, split, threshold, spike mask, static scan |
| **`test_snapshot_write_guards.py`** (ใหม่) | 5 | writer ปฏิเสธ snapshot และ manifest |
| `test_cleaning.py` | 0 | ลบแล้ว (ว่างเปล่า) |
| `tests/ml/` (34 ไฟล์) | 634 | ไม่อยู่ในขอบเขต audit นี้ |
| `tests/modeling/` | — | ตาย · เหลือ `.pyc` 11 ไฟล์ |

## Data และ artifact บนดิสก์

| ที่อยู่ | เนื้อหา | git |
| --- | --- | --- |
| `data/raw/qqq_daily.csv` | 2,512 แถว, 124,154 bytes | track แบบ `-text` (A13 แก้แล้ว) |
| `data/manifests/qqq_daily_snapshot.json` | SHA และ metadata ของ snapshot | track |
| `data/interim/` | `qqq_clean.csv`, `qqq_features.csv`, `qqq_regression_target.csv` | ignore |
| `data/processed/` | `train/validation/test.csv` + `_labeled` | ignore |
| `data/processed/experiments/` | `with_spikes/train.csv`, `non_spike/train.csv`, `diagnostics/*_flagged.csv` | ignore |
| `outputs/reports/` | report 10 ไฟล์ (hash 26 รายการตรงไฟล์จริง) | track |
| `outputs/figures/` | 16 ไฟล์ · รวม `data_preparation/` 13 รูปจาก notebook 01/02 | track |
| `outputs/modeling/` | 38 ชุด, 370 ไฟล์ | track |
| `outputs/models/`, `outputs/metrics/` | `.gitkeep` เท่านั้น | ตาย |
| `tmp/phase1-reproduction/` | 14 ไฟล์ เหมือนชุดตรวจรับ · **ห้ามลบ** | ignore |
| `tmp/audit-repro/` | ผลรันซ้ำของ audit นี้ · ลบได้ | ignore |

## Notebooks

| ไฟล์ | cells | คำตัดสิน |
| --- | --- | --- |
| `01_data_cleaning.ipynb` | 21 | ตัวเลขและ SHA ตรงของจริง · รันใหม่บน Python 3.14.8 เมื่อ 2026-10-11 |
| `02_eda_and_features.ipynb` | 34 | รันใหม่ 2026-10-11 · hash ตรงแล้ว · เพิ่ม §11b Data Summary · อ่าน Test เชิงพรรณนา |
| `03_regression.ipynb`, `04_classification.ipynb` | 0 | ว่างเปล่า ส่งแบบนี้ไม่ได้ |

## เอกสาร

40 ไฟล์ใน `docs/` + `README.md`, `AGENTS.md`, `CLAUDE.md` ที่ root · ตัวเลขชั้นข้อมูลถูกทุกจุด
ส่วนสถานะโมเดลและจำนวน test ล้าสมัยในหลายไฟล์ และแก้ใน audit นี้โดยทำเครื่องหมาย `⚠️ แก้ไข 2026-10-11`
