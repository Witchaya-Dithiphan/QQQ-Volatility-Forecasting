# Phase 2 Spike Analysis — Baseline Provenance and Readiness

ตรวจล่าสุด: 2026-10-06 บน branch `feature/spike-analysis`, commit ตั้งต้น `9e80b22`

## Scope และสถานะ

รอบนี้ตรวจ Phase 1 provenance, freeze กลไก primary spike/boundary contract และเพิ่ม synthetic contract tests เท่านั้น ยังไม่ได้สร้าง With-Spike/Non-Spike datasets, Phase 2 reports/figures, runner หรือ model artifacts ดังนั้น **Phase 2 ยังไม่เสร็จ**

Primary rule ไม่เปลี่ยน:

- Detection variable: `abs(return_1d)`
- Fit source: Original Train เท่านั้น
- Threshold: `Q3 + 3 × IQR`
- Comparison: strict `>`
- Affected window: inclusive `[s-5, s+19]`
- Mask scope: clip ภายใน split; ไม่ propagate ข้าม gap/split
- Filter scope: สำเนา Original Train หลัง features/targets/labels พร้อมแล้ว
- Full Validation/Test และ Original Train Q75 คงเดิม

## 1. Repository และ artifact inventory

### Tracked contract/source

- `.gitignore`
- `config.py`
- `src/report_paths.py`
- `src/run_data_pipeline.py`
- `src/clean_data.py`
- `src/build_features.py`
- `src/build_targets.py`
- `src/split_data.py`
- `tests/`
- `README.md`, `PROJECT_STATUS.md`, `HANDOFF.md`
- `data/manifests/qqq_daily_snapshot.json`

### Generated/ignored inputs และ outputs

`.gitignore` ไม่ track `data/raw/*`, `data/interim/*`, `data/processed/*`, `outputs/reports/*` และ `outputs/figures/*` ยกเว้น `.gitkeep` ดังนั้น Snapshot และ labeled CSV ต้องถูกจัดหา/สร้างใหม่และตรวจ checksum หลัง clone ห้าม commit generated data เพียงเพื่อแก้ provenance

Phase 2 authoritative inputs สำหรับ snapshot ปัจจุบันคือ:

- `data/processed/train_labeled.csv`
- `data/processed/validation_labeled.csv`
- `data/processed/test_labeled.csv`

เหตุผล: ทั้งสามไฟล์ผ่าน direct schema/value validation และมี byte hashes ตรงกับผล reproduction ใหม่จาก verified Snapshot ทุกไฟล์ ไม่ได้อาศัย path หรือ checksum จาก saved report เก่าเพียงอย่างเดียว

## 2. Saved-report provenance discrepancy

### ข้อเท็จจริง

Saved reports เดิมที่ project root มี provenance fields แบบเก่า:

- `data_split_report.json`
  - `input_path` และ `output_paths.*` เป็น absolute Windows paths
  - ไม่มี `paths_relative_to`
- `classification_threshold.json`
  - `input_paths.*` และ `output_paths.*` เป็น absolute Windows paths
  - ไม่มี `paths_relative_to`

Fields เหล่านี้กระทบ portability/provenance แต่ไม่ใช่สูตรหรือค่าข้อมูลโดยตรง Fields ต่อไปนี้ยังตรวจเทียบกับ CSV/reproduction ได้:

- split schema, row counts, positions, gaps และ date ranges
- target/feature names
- Classification Q75, quantile method, strict comparison rule และ class counts
- source checksums ของ unlabeled splits

ข้อสรุปที่หลักฐานรองรับคือ saved reports ถูกสร้างด้วย report serialization รุ่นก่อน contract ปัจจุบัน ไม่ใช่หลักฐานว่า calculation code ปัจจุบันผิด และไม่ใช่เหตุให้เปลี่ยน spike rule

### หลักฐานจาก isolated reproduction

รันจาก project root:

```powershell
.venv\Scripts\python.exe src\download_qqq_data.py verify-snapshot
.venv\Scripts\python.exe -m src.run_data_pipeline --output-root tmp/phase2-readiness-baseline
```

ผล reproduction:

- สำเร็จโดยไม่ใช้ network
- สร้าง 14 Phase 1 artifacts ใต้ isolated output root
- Train/Validation/Test = 1,733/371/373
- Train-only Q75 = `0.2530580184684854`
- Reports ใหม่มี `"paths_relative_to": "report_directory"`
- Relative input/output links ทุก path resolve ไปยังไฟล์ที่มีอยู่จริง
- ไม่ได้เขียนทับ saved reports หรือ baseline CSV เดิม

ตัวอย่าง path contract ใหม่:

- Split input: `../../data/interim/qqq_regression_target.csv`
- Split train output: `../../data/processed/train.csv`
- Classification train input: `../../data/processed/train.csv`
- Classification test output: `../../data/processed/test_labeled.csv`

Report hashes ต่างจาก saved reports เดิมตามที่คาด เพราะ path serialization และ report schema ต่างกัน:

| Report | Saved root SHA-256 | Reproduced SHA-256 |
| --- | --- | --- |
| `data_split_report.json` | `EF479B6B5731E0C4AA14A375C436B00C760AFBC5DEE285CEB6F4105B002026E0` | `7E9B47C1DE7AA0D66BA6B4CEFEF02816DEFF31FC93AD247E1755DF5E9C7E440F` |
| `classification_threshold.json` | `E0670FB96B8AD0826D05D4ED888DDB9704C9379FF5F14DD6AB67D624A92F090D` | `FEB1C9A0C01448A2485B5A20837DFFC572F49450619F26AB818E0B93ED22AC53` |

ไม่ควรเปรียบเทียบ report bytes เพื่อสรุป data equality; ให้เปรียบเทียบ CSV และ semantic fields

## 3. Baseline equality และ Phase 2 input contract

### Protected source hashes

| Artifact | SHA-256 ก่อนและหลัง reproduction |
| --- | --- |
| `data/raw/qqq_daily.csv` | `649E1DB1B79C0990EA947A4EA162D6836B9BBDDA2AF24BC5AB6AE66CB83B6B13` |
| `data/manifests/qqq_daily_snapshot.json` | `5D6A4F0316F20FAB675EFED6009A49DFA5670B72781247984CF9E6FAD583F8A7` |

Snapshot hash ตรงกับค่าใน Manifest และทั้ง Snapshot/Manifest ไม่เปลี่ยนหลัง run

### Labeled CSV equality

| Split | Rows | Date range | Original SHA-256 | Reproduced SHA-256 | Result |
| --- | ---: | --- | --- | --- | --- |
| Train | 1,733 | 2016-09-29 — 2023-08-18 | `424F1BDA5C4211AC58546DA992408D373D874A0F65A509965B06D8224194E64C` | same | Byte-identical |
| Validation | 371 | 2023-08-28 — 2025-02-19 | `B84A5FA31FE1AFC6841528F8E5F0C8532B07E2BDA8B80BD2CA1DE664CF42F6EC` | same | Byte-identical |
| Test | 373 | 2025-02-27 — 2026-08-21 | `5E8AC488CB895A8899ABD9B3AE9193B4DC6EBDA2E77B79982CEF4F5EA9BF7295` | same | Byte-identical |

เพราะ hashes เหมือนกันจึงไม่ต้องทำ cell-level diff เพิ่ม หาก hashes ต่างในอนาคตต้องตรวจ schema, values, dtypes, float serialization, line endings และ date formatting ก่อนยอมรับ

### Direct input validations

ทุก split ผ่าน contract ต่อไปนี้:

- Schema ตรงกันและเรียง 16 columns ตาม baseline
- Date unique และเรียงจากเก่าไปใหม่
- Split date ranges ไม่ overlap
- Model columns เป็น finite ทั้งหมด
- Classification labels มีค่า 0/1
- Q75 คำนวณจาก Original Train เท่านั้นด้วย linear interpolation
- Recomputed Q75 = `0.2530580184684854`
- Labels ทุกแถวตรง `target_volatility_5d > Q75`
- Train มีหนึ่งแถวเท่ากับ Q75 และถูก label เป็น 0 ยืนยัน strict `>`
- Purging gaps จาก fresh split report เท่ากับ 5 rows ระหว่าง Train–Validation และ Validation–Test

Expected row/count/threshold values ใช้เป็น audit regression checks เท่านั้น Future Phase 2 codeต้อง derive ผลจาก inputs และ fail เมื่อ contract เปลี่ยนอย่างไม่คาดหมาย ห้าม hard-code expected valuesเพื่อกลบ dataset drift

## 4. Frozen boundary-policy implementation และ synthetic tests

เพิ่ม pure primitives ใน `src/spike_contract.py`:

- Fit Extreme-IQR metadata จาก caller-supplied Original Train returns
- Direct flags ใช้ strict `abs(return_1d) > threshold`
- Affected-window metadata เก็บ requested/clipped positions
- Affected mask รวม overlapping windowsแบบไม่ double-count
- Split-map API สร้าง mask แยกแต่ละ splitและไม่ propagate ข้าม gap

เพิ่ม `tests/test_spike_contract.py` ด้วย synthetic fixtures:

- left clipping เมื่อ spike อยู่ต้น split
- right clipping เมื่อ spike อยู่ท้าย split
- overlapping windows นับ union เพียงครั้งเดียว
- direct spike ทุกแถวต้องเป็น affected
- Train spike ใกล้ boundary ไม่ flag Validation/Test
- ใช้ original positions ก่อนกรอง
- strict equality ไม่เป็น direct spike
- absolute-return symmetry
- invalid/empty/non-finite inputs
- invalid/non-boolean/missing direct flags

Contract นี้ยังไม่สร้าง experiment CSV และไม่ถือว่า full detector/Phase 2 runner เสร็จ

## 5. Supplementary 1.5×IQR Test boundary audit

Sensitivity นี้ fit จาก Original Train เท่านั้น ใช้ strict `>` และไม่เปลี่ยน primary 3×IQR rule

- Threshold: `0.0302071795070148`
- Test rows: 373
- Direct spikes: 14
- Union affected rows: 177
- Non-affected rows: 196

| Position | Spike date | Requested positions | Clipped positions | Clipped dates | Right clipped |
| ---: | --- | --- | --- | --- | --- |
| 7 | 2025-03-10 | `[2,26]` | `[2,26]` | 2025-03-03 — 2025-04-04 | No |
| 25 | 2025-04-03 | `[20,44]` | `[20,44]` | 2025-03-27 — 2025-05-01 | No |
| 26 | 2025-04-04 | `[21,45]` | `[21,45]` | 2025-03-28 — 2025-05-02 | No |
| 29 | 2025-04-09 | `[24,48]` | `[24,48]` | 2025-04-02 — 2025-05-07 | No |
| 30 | 2025-04-10 | `[25,49]` | `[25,49]` | 2025-04-03 — 2025-05-08 | No |
| 51 | 2025-05-12 | `[46,70]` | `[46,70]` | 2025-05-05 — 2025-06-09 | No |
| 156 | 2025-10-10 | `[151,175]` | `[151,175]` | 2025-10-03 — 2025-11-06 | No |
| 273 | 2026-03-31 | `[268,292]` | `[268,292]` | 2026-03-24 — 2026-04-28 | No |
| 319 | 2026-06-05 | `[314,338]` | `[314,338]` | 2026-05-29 — 2026-07-06 | No |
| 323 | 2026-06-11 | `[318,342]` | `[318,342]` | 2026-06-04 — 2026-07-10 | No |
| 325 | 2026-06-15 | `[320,344]` | `[320,344]` | 2026-06-08 — 2026-07-14 | No |
| 330 | 2026-06-23 | `[325,349]` | `[325,349]` | 2026-06-15 — 2026-07-21 | No |
| 356 | 2026-07-30 | `[351,375]` | `[351,372]` | 2026-07-23 — 2026-08-21 | Yes |
| 359 | 2026-08-04 | `[354,378]` | `[354,372]` | 2026-07-28 — 2026-08-21 | Yes |

ผลนี้เป็น supplementary data-retention/boundary audit เท่านั้น ห้ามใช้ Test resultเลือก primary threshold, hyperparameters หรือ model configuration

## 6. Reproducibility contract สำหรับ generated/ignored artifacts

### สิ่งที่ต้องมีหลัง clone

1. Python environment ติดตั้ง `requirements.txt`
2. Tracked Manifest ที่ `data/manifests/qqq_daily_snapshot.json`
3. Historical Snapshot ที่ `data/raw/qqq_daily.csv` ซึ่ง SHA-256 ต้องตรง Manifest
4. Tracked source/config/tests จาก branch ที่ต้องการตรวจ

### คำสั่ง Phase 1 verification

```powershell
.venv\Scripts\python.exe src\download_qqq_data.py verify-snapshot
.venv\Scripts\python.exe -m src.run_data_pipeline --output-root tmp/phase2-readiness-baseline
.venv\Scripts\python.exe -m pytest -q -rs
```

- เลือก output root ใหม่ที่ยังไม่มีอยู่
- Default runner ต้องหยุดหาก generated output มีอยู่แล้ว
- ใช้ `--overwrite-generated` เฉพาะเมื่อตั้งใจแทนที่ declared generated outputs ใน isolated root
- ห้ามชี้ output root ไปใต้ Raw/Manifest
- ห้าม overwrite project-root baseline เพียงเพื่อทำให้ reports ดูใหม่

### Phase 2 runner contract ที่ต้อง implement ต่อ

- ใช้ runner แยกจาก Phase 1
- อ่าน labeled CSV สามชุดเป็น protected inputs
- Preflight paths/checksums ก่อน write
- Reports ใช้ portable relative paths
- Reports บันทึก input/output SHA-256
- Default ไม่ overwrite
- Explicit overwrite จำกัดเฉพาะ declared Phase 2 artifacts
- Generated experiment data/reports/figures ยังคง ignored และ reproduce จาก command ได้

## 7. Test evidence

คำสั่งที่รัน:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_spike_contract.py -q
.venv\Scripts\python.exe -m pytest -q -rs
.venv\Scripts\python.exe -m ruff check config.py src\spike_contract.py tests\test_spike_contract.py
```

ผลล่าสุดหลังแก้ source:

- Spike contract tests: 14 passed
- Full suite: 206 passed, 1 skipped
- Skipped: Windows symlink privilege (`WinError 1314`), ไม่ใช่ test failure
- Ruff: all checks passed

## 8. Readiness decision

**พร้อมเริ่ม Phase 2 implementation ขั้นถัดไปแบบมีเงื่อนไข** เพราะ:

- Fresh Phase 1 reportsพิสูจน์ portable-path contract แล้ว
- Current และ reproduced labeled CSVs byte-identical
- Authoritative Phase 2 inputs และ hashesระบุชัด
- Snapshot/Manifest ไม่เปลี่ยน
- Primary rule และ boundary policyมี executable synthetic tests

งานที่ยังไม่เสร็จและห้ามอ้างว่าเสร็จ:

- Full input-contract/report generator สำหรับ Phase 2
- Direct market-event/data-quality audit
- With-Spike/Non-Spike datasets
- Validation/Test diagnostic artifacts
- Phase 2 runner, reports และ figures
- Optional sensitivity report artifact
- Model training/evaluation
