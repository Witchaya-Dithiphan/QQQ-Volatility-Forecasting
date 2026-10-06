# Phase 2 — Spike Analysis Plan

## สถานะและขอบเขต

เอกสารนี้เป็น implementation plan สำหรับ Phase 2 เท่านั้น ปัจจุบัน M1 Baseline Input Contract และ M2 pure primary detector เสร็จแล้ว โดย `src/detect_spikes.py` เป็นเจ้าของ direct-detector logic และ `src/spike_contract.py` เป็นเจ้าของ affected-window logic แต่ยังไม่มี experiment datasets, market-event/data-quality audit, Phase 2 experiment runner/reports/figures หรือ model artifacts ดังนั้นงาน **Spike Analysis / experiment dataset preparation ยังไม่เสร็จ** และยังไม่มีการ train หรือประเมินโมเดลจาก Phase 2

คำว่า Non-Spike ในโครงการนี้หมายถึงสำเนาของ Original Train ที่ตัดแถวตาม operational affected window ออก ไม่ได้หมายความว่าข้อมูลปราศจากอิทธิพลของ spike ในทุก feature อย่างสมบูรณ์

## ผลตรวจความสอดคล้องกับโค้ดและข้อมูลปัจจุบัน

### สิ่งที่สอดคล้องกับมติ

- [x] `return_1d` สร้างจาก `Close.pct_change(1)` บน complete timeline ก่อน split
- [x] Regression target ที่แถว `t` ใช้ `return_1d` ของ `t+1` ถึง `t+5`
- [x] `rsi_14` ใช้ Wilder average แบบ recursive จริง จึงอาจได้รับอิทธิพลหลัง `s+19`
- [x] Baseline split ใช้ลำดับเวลาและ purging gap 5 แถวสองช่วง
- [x] Classification Q75 fit จาก Original Train และใช้ strict `>`
- [x] Original labeled splits มี schema เดียวกัน 16 columns
- [x] Original Train มี 1,733 แถว ช่วง 2016-09-29 ถึง 2023-08-18
- [x] Original Validation มี 371 แถว ช่วง 2023-08-28 ถึง 2025-02-19
- [x] Original Test มี 373 แถว ช่วง 2025-02-27 ถึง 2026-08-21
- [x] Primary 3×IQR audit ให้ threshold `0.0465436445413787`
- [x] Primary audit พบ Train direct spikes 19 แถว
- [x] Primary audit พบ Train affected rows 213 แถว
- [x] Primary audit เหลือ Non-Spike Train 1,520 แถว
- [x] Primary Non-Spike Train class 0/1 เท่ากับ 1,240/280
- [x] เมื่อนำ Train threshold ไปใช้กับชุดอื่น Primary audit พบ Validation direct/affected = 0/0 และ Test direct/affected = 4/54

ค่าข้างต้นเป็น observed/expected checks สำหรับ regression testing ห้ามนำไป hard-code เป็นผลลัพธ์ของ detector

### ข้อขัดแย้งหรือข้อควรระวังที่พบ

- [x] Saved `data_split_report.json` และ `classification_threshold.json` ใน working directory ยังเก็บ absolute paths และไม่มี `paths_relative_to`
- [x] Fresh isolated reproduction พิสูจน์ว่าโค้ดปัจจุบันสร้าง portable relative paths และ links resolve ได้จริง
- [x] Labeled CSV เดิมและ reproduced CSV มี SHA-256 ตรงกันทุก split จึงจัด discrepancy เป็น artifact provenance ไม่ใช่ data mismatch
- [x] ไม่เปลี่ยนมติ spike เพื่อชดเชยความเก่าของ report
- [x] คำนวณ checksum ของ labeled CSV inputs โดยตรงโดยไม่อาศัย saved report เก่าเพียงอย่างเดียว
- [x] Primary 3×IQR ไม่มี direct spike ใกล้ต้นหรือท้าย split ในข้อมูลปัจจุบัน จึงไม่มี real-data clipping case สำหรับกฎหลัก
- [x] เพิ่ม synthetic fixtures ทดสอบ left/right clipping, overlap และ cross-split isolation
- [x] Sensitivity 1.5×IQR มี Test spikes วันที่ 2026-07-30 และ 2026-08-04 ที่ถูก right-clip; บันทึกเป็น supplementary audit โดยไม่เปลี่ยน primary rule
- [x] Generated data/reports/figures ถูก `.gitignore` ไว้ และ readiness report ระบุคำสั่ง reproduce/checksums แล้ว

## มติการทดลองที่ Freeze แล้ว

### Primary spike rule

- [x] Detection variable คือ `abs(return_1d)`
- [x] Fit จาก Original Train เท่านั้น
- [x] `Q1 = percentile 25`
- [x] `Q3 = percentile 75`
- [x] `IQR = Q3 - Q1`
- [x] `threshold = Q3 + 3 × IQR`
- [x] `is_spike = abs(return_1d) > threshold`
- [x] ใช้ strict `>`; ค่าเท่ากับ thresholdไม่เป็น spike
- [x] ห้าม hard-code threshold หรือ expected counts
- [x] ใช้ Train threshold ค่าเดียวกันตรวจ Validation/Test
- [x] ห้าม fit threshold ใหม่จาก Validation/Test
- [x] Freeze rule นี้ก่อนดู model scores
- [x] ห้ามอ้างว่า 3×IQR เป็นเกณฑ์ที่ดีที่สุดโดยพิสูจน์แล้ว

### Primary affected-window rule

- [x] ใช้ตำแหน่งวันซื้อขายบน Original split ก่อนกรอง
- [x] สำหรับ spike ตำแหน่ง `s` ใช้ inclusive window `[s-5, s+19]`
- [x] `s-5` ถึง `s-1` ครอบคลุมแถวที่ target ในอนาคต 5 วันใช้ return ของ spike
- [x] `s` ถึง `s+19` ครอบคลุม direct return และ rolling features หลักถึง 20 วัน
- [x] รวม overlapping windows ก่อนนับ affected rows
- [x] Clip window ภายใน split ที่กำลังวิเคราะห์
- [x] ห้ามคำนวณ position ใหม่หลังกรอง Non-Spike Train
- [x] ห้ามขยาย mask ข้าม purging gap หรือข้าม split

### RSI policy

- [x] เลือก Policy A เป็นกฎหลัก
- [x] ยอมรับ `[s-5, s+19]` เป็น operational definition
- [x] ไม่ใช้ RSI decay tolerance เป็นกฎหลัก
- [x] ไม่ขยาย window เพื่อครอบคลุม RSI decay เป็นกฎหลัก
- [x] Reports และเอกสารต้องระบุว่า Wilder RSI อาจยังได้รับอิทธิพลหลัง `s+19`
- [x] ห้ามกล่าวว่า Non-Spike Train ปราศจากอิทธิพลของ spike ในทุก featureอย่างสมบูรณ์

### Dataset and evaluation policy

- [x] With-Spike Train คือ Original Train ทุกแถว
- [x] Non-Spike Train คือ Original Train ที่ลบ `is_spike_affected == True`
- [x] กรองหลังสร้าง features, regression target และ classification labels แล้วเท่านั้น
- [x] ห้ามแก้ Raw, Snapshot, Manifest, Clean, Feature, Target หรือ Original splits
- [x] ห้ามคำนวณ returns, features, targets หรือ Classification Q75 ใหม่หลังกรอง
- [x] แถวที่เหลือต้องรักษา Date, ลำดับเวลา และค่าทุกคอลัมน์เหมือนต้นฉบับ
- [x] With-Spike และ Non-Spike ใช้ Full Validation/Test ชุดเดิม
- [x] ห้าม re-split, shuffle, ย้ายแถว หรือกรอง Validation/Test สำหรับผลหลัก
- [x] Non-Spike และ Spike-Affected Validation/Test segments เป็น diagnostics เท่านั้น

---

# Must-have สำหรับ Phase 2

## M1. Freeze Baseline Input Contract

**Dependencies:** Phase 1 labeled splits และ saved reports ต้องมีอยู่

**Artifacts:**

- `config.py`
- `src/build_spike_input_contract.py`
- `outputs/reports/spike_input_contract.json`
- `tests/test_experiment_datasets.py`

**Checklist:**

- [x] เพิ่ม Phase 2 input/output paths ใน `config.py`
- [x] เพิ่มค่าคงที่ primary multiplier `3.0`
- [x] เพิ่ม target-backward reach `5`
- [x] เพิ่ม feature-forward reach `19`
- [x] ระบุ comparison rule เป็น strict `>`
- [x] ระบุ RSI policy เป็น `operational_window_s_minus_5_to_s_plus_19`
- [x] อ่าน `train_labeled.csv`, `validation_labeled.csv`, `test_labeled.csv` แบบ read-only
- [x] Validate required 16-column schema
- [x] Validate numeric dtypes ของ features, target และ label
- [x] Validate Date parse, uniqueness และ ascending order
- [x] Validate ไม่มี NaN/Infinity ใน model columns
- [x] Validate labels เป็น 0/1 เท่านั้น
- [x] Validate row counts และ date ranges ปัจจุบัน
- [x] Validate split ไม่ overlap
- [x] Validate gap dates/size จาก baseline split report
- [x] Validate Original Classification Q75 และ label consistency
- [x] คำนวณ SHA-256 ของ labeled splits โดยตรง
- [x] บันทึก source checksums ก่อนเริ่มเขียน output
- [x] บันทึกความไม่สอดคล้องของ saved report portability
- [x] ตรวจ Phase 1 reproduction ใน isolated output root หรือบันทึกเหตุผลหากยังไม่ทำ
- [x] ห้าม rewrite baseline artifacts จาก Phase 2 runner

**Definition of Done:**

- [x] Contract JSON เป็น strict JSON และใช้ portable paths
- [x] Schema, rows, dates, gaps, labels และ checksums ถูก assert
- [x] ความต่างระหว่าง current code กับ saved reports ถูกบันทึก ไม่ถูกแก้เงียบ ๆ
- [x] ไม่มี source file ใดเปลี่ยน checksum

## M2. Implement Pure Primary Spike Detector

**Dependencies:** M1

**Artifacts:**

- `src/detect_spikes.py`
- `tests/test_spike_detection.py`

**Checklist:**

- [x] สร้าง immutable/config metadata สำหรับ fitted detector
- [x] Validate ว่า input มี `return_1d`
- [x] Validate `return_1d` เป็น numeric และ finite
- [x] ปฏิเสธ empty input
- [x] ปฏิเสธ invalid multiplier
- [x] คำนวณ `abs(return_1d)` โดยไม่แก้ input
- [x] Fit Q1 จาก Original Train เท่านั้น
- [x] Fit Q3 จาก Original Train เท่านั้น
- [x] คำนวณ IQR
- [x] คำนวณ threshold จาก multiplier 3.0
- [x] แยก API `fit` ออกจาก API `apply`
- [x] Apply ด้วย strict `>`
- [x] ค่าเท่ากับ thresholdต้องเป็น `False`
- [x] Negative return ใช้ absolute value อย่างถูกต้อง
- [x] ใช้ fitted threshold เดียวกันกับทุก split
- [x] คืนผล deterministic
- [x] ห้าม hard-code threshold/count/dates
- [x] คืน metadata: source split, quantile method, Q1, Q3, IQR, multiplier, threshold และ comparison rule

**Tests:**

- [x] Train-only fit test
- [x] Validation mutation ไม่เปลี่ยน threshold
- [x] Test mutation ไม่เปลี่ยน threshold
- [x] Strict equality boundary test
- [x] Positive/negative symmetry test
- [x] NaN/Infinity/nonnumeric/empty input tests
- [x] Input immutability test
- [x] Deterministic rerun test
- [x] Expected audit values ใช้ `pytest.approx`/assert count เป็น regression checks เท่านั้น

**Definition of Done:**

- [x] Detector เป็น pure API และไม่มี filesystem side effect
- [x] Primary audit ได้ค่าตรง expected checks โดยไม่ hard-code
- [x] Unit tests ผ่าน

## M3. Implement Affected Mask และ Boundary Policy

**Dependencies:** M2

**Artifacts:**

- `src/spike_contract.py`
- `tests/test_spike_contract.py`
- Boundary metadata ใน `spike_analysis.json`

**Checklist:**

- [ ] รับ direct flags บน Original split positions
- [ ] สร้าง inclusive window `[s-5, s+19]`
- [ ] รวม overlapping windows ด้วย boolean OR ก่อนนับ
- [ ] Clip lower bound ที่ position 0
- [ ] Clip upper bound ที่ positionสุดท้ายของ split
- [ ] ห้ามใช้ index ที่เกิดหลัง Non-Spike filtering
- [ ] ห้ามขยาย Train mask ไป Validation
- [ ] ห้ามขยาย Validation mask ไป Test
- [ ] ห้ามขยาย mask ผ่าน purging gaps
- [ ] สร้าง `is_spike` และ `is_spike_affected` เป็น boolean
- [ ] ตรวจ direct spike ทุกแถวเป็น affected
- [ ] บันทึกต่อ spike: original position, event date, requested bounds, clipped bounds และ clipped date range
- [ ] บันทึกต่อ split: left/right clipping count, overlap count และ union affected count
- [ ] บันทึก `mask_scope = within_split`
- [ ] บันทึก `cross_split_propagation = false`
- [ ] บันทึกข้อจำกัดว่า features/targets ถูกสร้างก่อน split และอาจอ้างข้อมูลก่อนขอบ split
- [ ] ระบุว่า mask clipping เป็น experimental rule ไม่ใช่หลักฐานว่า cross-boundary influence เป็นศูนย์

**Boundary-reporting policy:**

- [ ] Full Validation/Test ต้องคงเดิมเสมอ แม้พบ boundary case
- [ ] Diagnostic flags ใช้ local split positions และแสดง clipped/unclipped bounds
- [ ] Primary real data ไม่มี clipping case ให้รายงานว่า `observed_clipping = 0`
- [ ] Synthetic fixtures ต้องพิสูจน์ left clip, right clip, overlap และ cross-split isolation
- [ ] Optional 1.5×IQR Test right-clipping caseรายงานแยกจาก primary results

**Tests:**

- [ ] Spike ที่ position แรก
- [ ] Spike ที่ positionสุดท้าย
- [ ] Spike ก่อนขอบ 5 แถว
- [ ] Spikeก่อนท้าย 19 แถว
- [ ] Overlapping windows
- [ ] หลาย spike ที่ window ไม่ overlap
- [ ] Empty direct mask
- [ ] Train/Validation/Test isolation
- [ ] Original positions ไม่เปลี่ยนหลังสร้าง Non-Spike copy

**Definition of Done:**

- [ ] Mask ตรง `[s-5, s+19]` แบบ inclusive
- [ ] Primary Train affected count ตรง expected 213 โดยไม่ hard-code
- [ ] Boundary metadata ครบและ Full Validation/Test ไม่เปลี่ยน

## M4. Direct Spike Audit และ Data-quality Review

**Dependencies:** M2, M3

**Artifacts:**

- `outputs/reports/spike_analysis.json`
- `outputs/reports/spike_event_audit.csv`
- `outputs/figures/spike_analysis/daily_return_spikes.png`

**Checklist:**

- [ ] ดึง direct spike dates จาก detector output
- [ ] บันทึก Date, signed return และ absolute return
- [ ] บันทึก Open, High, Low, Close และ Volume
- [ ] Trace event date กลับไปยัง labeled, target, feature, clean และ raw timeline
- [ ] ตรวจค่าระหว่าง stages ว่าสอดคล้องกัน
- [ ] แยกสถานะ `market_movement`, `suspected_data_error` หรือ `needs_review`
- [ ] บันทึก audit note และ source checksum ต่อ artifact
- [ ] บันทึก expected 19 events เป็น verification ไม่ใช่ input
- [ ] หากพบ data error ให้สร้าง finding แยก
- [ ] ห้ามแก้ Raw/Snapshot/Manifest จาก Phase 2 script
- [ ] ห้ามเปลี่ยน baseline หรือ detector rule อัตโนมัติ
- [ ] หากต้องแก้ source data ให้หยุดการสรุปผลและเปิด baseline-reproduction workstream แยก

**Tests:**

- [ ] ทุก reported date ตรงกับ detector flags
- [ ] ทุก event trace กลับ source date ได้หนึ่งแถวพอดี
- [ ] Report count ตรง event table
- [ ] Source checksums ก่อน/หลังเหมือนกัน
- [ ] Strict JSON ไม่มี NaN/Infinity

**Definition of Done:**

- [ ] Direct spike ทุกแถว trace กลับต้นทางได้
- [ ] Data-quality classification ถูกบันทึกโดยไม่แก้ source
- [ ] Audit artifacts ตรวจซ้ำจาก inputs เดิมได้

## M5. Build Primary Experiment Datasets

**Dependencies:** M1–M4

**Artifacts:**

- `src/build_experiment_datasets.py`
- `data/processed/experiments/with_spikes/train.csv`
- `data/processed/experiments/non_spike/train.csv`
- `data/processed/experiments/diagnostics/validation_flagged.csv`
- `data/processed/experiments/diagnostics/test_flagged.csv`
- `tests/test_experiment_datasets.py`

**With-Spike checklist:**

- [ ] สร้างจาก Original Train แบบ deep copy
- [ ] เก็บ Original Train ครบทุกแถว
- [ ] รักษา schema, Date, order และค่าทุกคอลัมน์
- [ ] บันทึก checksum ของ source และ output

**Non-Spike checklist:**

- [ ] เริ่มจาก Original Train ก่อนกรอง
- [ ] Attach primary `is_spike` และ `is_spike_affected`
- [ ] Filter เฉพาะ `is_spike_affected == True` ออกจากสำเนา Train
- [ ] ห้าม recompute `return_1d` หรือ `return_5d`
- [ ] ห้าม recompute volatility, RSI, SMA หรือ volume features
- [ ] ห้าม recompute regression target
- [ ] ห้าม recompute Classification Q75 หรือ labels
- [ ] รักษาค่าทุก source column ของ kept rows แบบ exact/defined tolerance
- [ ] รักษาลำดับ Date
- [ ] ตรวจว่าไม่มี affected row เหลือใน filtered artifact
- [ ] ตรวจ expected rows 1,520 และ class 0/1 = 1,240/280 โดยไม่ hard-codeเป็น logic

**Validation/Test checklist:**

- [ ] Full Validation/Test ยังคงเป็น Original labeled splits
- [ ] สร้าง flagged copies แยกเพื่อ diagnostics เท่านั้น
- [ ] ใช้ primary Train threshold ค่าเดียวกัน
- [ ] ไม่ filter หรือ overwrite Original Validation/Test
- [ ] ระบุ Full, Non-Spike และ Spike-Affected segment membership
- [ ] Primary expected diagnostic counts: Validation 371/0 affected, Test 319/54 non-affected/affected

**Tests:**

- [ ] Original inputs ไม่เปลี่ยน checksum
- [ ] With-Spike values ตรง Original Train
- [ ] Non-Spike เป็น subset ตาม Date ที่ถูกต้อง
- [ ] Kept source-column values ตรง Original Train
- [ ] ไม่มี return ข้ามวันที่จากการกรอง
- [ ] Labels ตรง Original Q75 เดิม
- [ ] Full Validation/Test row count และ checksumไม่เปลี่ยน
- [ ] Diagnostic copies ไม่ถูกใช้แทน Full split โดย API contract

**Definition of Done:**

- [ ] Primary With-Spike/Non-Spike artifacts สร้างซ้ำได้
- [ ] Original artifacts ไม่เปลี่ยน
- [ ] Full Validation/Test ยังคงเดิมทั้งสอง experiment cases

## M6. Reports และ Primary Figures

**Dependencies:** M1–M5

**Artifacts:**

- `outputs/reports/spike_analysis.json`
- `outputs/reports/experiment_dataset_report.json`
- `outputs/figures/spike_analysis/daily_return_spikes.png`
- `outputs/figures/spike_analysis/volatility_spike_effect.png`
- `outputs/figures/spike_analysis/dataset_comparison.png`

**Checklist:**

- [ ] ใช้ strict JSON และ portable relative paths
- [ ] บันทึก detector formula/configuration
- [ ] บันทึก Q1, Q3, IQR, multiplier และ threshold
- [ ] บันทึก source split และ strict comparison rule
- [ ] บันทึก direct count/dates
- [ ] บันทึก requested/clipped affected bounds
- [ ] บันทึก affected union counts
- [ ] บันทึก before/after rows และ date ranges
- [ ] บันทึก class counts/ratios ก่อนและหลังกรอง
- [ ] บันทึก Full/Non-Spike/Spike-Affected diagnostic segment counts
- [ ] บันทึก source/output checksums
- [ ] บันทึก Original Q75 และยืนยันว่าไม่ได้ recompute
- [ ] บันทึก RSI operational-definition limitation
- [ ] ใช้ถ้อยคำว่า `operationally non-spike-affected` เมื่อสรุป Non-Spike Train
- [ ] ห้ามใช้ถ้อยคำว่า “ปราศจาก spike influence ทุก feature”
- [ ] Figures ต้อง trace กลับ verified data และ report values ได้
- [ ] ตรวจ title, axes, units, legend และ date coverage

**Tests:**

- [ ] Report counts ตรง CSV
- [ ] Report checksums ตรงไฟล์
- [ ] Report paths portable
- [ ] Report rerun deterministic เมื่อ inputs/configเหมือนเดิม
- [ ] Figures ทั้งหมดถูกสร้างและไม่เป็นไฟล์ว่าง

**Definition of Done:**

- [ ] Reports และ primary figures ครบ
- [ ] ทุกตัวเลข trace กลับ source artifacts ได้
- [ ] RSI และ boundary limitations ปรากฏในทั้ง report และ documentation

## M7. Separate Phase 2 Runner

**Dependencies:** M1–M6 APIs

**Recommended artifact:** `src/run_spike_analysis.py`

เลือก runner แยกเพราะ `src/run_data_pipeline.py` มีหน้าที่ rebuild Phase 1 ตั้งแต่ Snapshot ถึง labeled splits และมี declared outputs/safety lifecycle ของ Phase 1 อยู่แล้ว การแยก runner ทำให้ Phase 2 อ่าน baseline แบบ protected inputs และ overwrite ได้เฉพาะ generated Phase 2 artifacts

**Checklist:**

- [ ] สร้าง `SpikePipelinePaths` หรือโครง path contractเทียบเท่า
- [ ] รองรับ default project paths
- [ ] รองรับ isolated `--output-root`
- [ ] อ่าน labeled splits เป็น protected inputs
- [ ] ตรวจ baseline contract ก่อน write ใด ๆ
- [ ] ตรวจ input checksums ก่อน run
- [ ] Preflight generated output paths
- [ ] ป้องกัน path alias กับทุก protected baseline artifact
- [ ] ป้องกัน duplicate output destinations
- [ ] ป้องกัน output หลุดจาก allowed roots
- [ ] ไม่ overwrite โดย default
- [ ] รองรับ `--overwrite-generated` เฉพาะ declared Phase 2 outputs
- [ ] ปฏิเสธ symlink/hard-link overwrite
- [ ] เรียก detector → mask → audit → datasets → reports → figures ตามลำดับ
- [ ] ตรวจ protected checksums ซ้ำเมื่อสำเร็จและเมื่อ stage ล้มเหลว
- [ ] ระบุ stage ใน error message
- [ ] แจ้งว่า partial generated outputs อาจเหลือหาก stage ล้มเหลว
- [ ] พิมพ์ summary paths/counts/checksums
- [ ] ห้ามเรียก model training

**Artifacts:**

- `src/run_spike_analysis.py`
- `tests/test_spike_pipeline.py`
- README run command

**Tests:**

- [ ] End-to-end run ใต้ temporary output root
- [ ] Default no-overwrite behavior
- [ ] Explicit overwrite จำกัดเฉพาะ Phase 2 outputs
- [ ] Protected-input alias tests
- [ ] Duplicate-output tests
- [ ] Symlink/hard-link safety testsตาม platform capability
- [ ] Failure-stage reporting
- [ ] Source checksum verification หลัง failure
- [ ] Offline execution test
- [ ] Deterministic rerun test

**Definition of Done:**

- [ ] Phase 2 reproduce ได้ด้วย documented command เดียว
- [ ] Runner ไม่แก้หรือ rebuild Phase 1 inputs
- [ ] Integration tests ผ่าน

## M8. Verification และ Documentation Gate

**Dependencies:** M1–M7

**Artifacts:**

- `README.md`
- `HANDOFF.md`
- `PROJECT_STATUS.md`
- Phase 2 run/test evidence

**Checklist:**

- [ ] รัน spike-specific unit/integration tests
- [ ] รัน existing full test suite
- [ ] ตรวจ generated artifacts inventory
- [ ] ตรวจ source checksums ก่อน/หลัง
- [ ] ตรวจ report/CSV/figure consistency
- [ ] บันทึกคำสั่ง reproduce
- [ ] บันทึก expected outputs
- [ ] บันทึก RSI limitation
- [ ] บันทึก boundary clipping policy
- [ ] บันทึก saved-report portability mismatch และ resolution/status
- [ ] อัปเดตสถานะ Phase 2 ตามหลักฐานจริงเท่านั้น
- [ ] ห้ามระบุว่า model training เสร็จ
- [ ] ห้ามระบุผล model performance ใน Phase 2

**Definition of Done:**

- [ ] Must-have tests ทั้งหมดผ่าน
- [ ] Must-have artifacts ครบและตรวจสอบได้
- [ ] Documentation ตรงกับ implementation และ generated results
- [ ] จึงเปิด gate ให้ Phase ถัดไปเริ่ม paired model training

---

# Optional Sensitivity Analysis สำหรับ Phase 2

Sensitivity เป็นการวิเคราะห์ผลต่อจำนวนข้อมูล ไม่แทน primary 3×IQR dataset และไม่ใช้เลือก primary ruleย้อนหลังจาก model/Test scores

## S1. Q3 + 1.5×IQR

**Dependencies:** M1–M3

**Checklist:**

- [ ] Fit Q1/Q3/IQR จาก Original Train เท่านั้น
- [ ] ใช้ threshold `Q3 + 1.5 × IQR`
- [ ] ใช้ strict `>`
- [ ] Apply Train threshold ค่าเดียวกับ Validation/Test
- [ ] สร้าง affected mask ด้วย `[s-5, s+19]`
- [ ] รายงาน direct, affected, remaining rows และ class distribution
- [ ] รายงาน Test right-boundary clipping ที่เกิดขึ้นจริง
- [ ] ห้ามสร้างเป็น primary Non-Spike dataset
- [ ] ห้าม train model variant นี้ใน Phase 2

**Observed checks:**

- Threshold ประมาณ `0.0302071795070148`
- Train direct/affected/remaining ประมาณ 91/690/1,043
- Remaining Train class 0/1 ประมาณ 991/52

## S2. Train percentile 99

**Dependencies:** M1–M3

**Checklist:**

- [ ] Fit p99 จาก `abs(return_1d)` ของ Original Train เท่านั้น
- [ ] Freeze quantile interpolation methodใน report
- [ ] ใช้ strict `>`
- [ ] Apply Train threshold ค่าเดียวกับ Validation/Test
- [ ] สร้าง affected mask ด้วย `[s-5, s+19]`
- [ ] รายงาน direct, affected, remaining rows และ class distribution
- [ ] ห้ามสร้างเป็น primary Non-Spike dataset
- [ ] ห้าม train model variant นี้ใน Phase 2

**Observed checks:**

- Threshold ประมาณ `0.0487619470246741`
- Train direct/affected/remaining ประมาณ 18/211/1,522
- Remaining Train class 0/1 ประมาณ 1,241/281

## S3. Sensitivity Artifacts และ Tests

**Artifacts:**

- `outputs/reports/spike_sensitivity.json`
- `outputs/figures/spike_analysis/threshold_comparison.png`

**Tests:**

- [ ] ทุก sensitivity threshold fit จาก Train เท่านั้น
- [ ] ทุก rule ใช้ strict `>`
- [ ] Equality boundary ไม่ถูก flag
- [ ] Validation/Test mutation ไม่เปลี่ยน fitted threshold
- [ ] Counts ใน report ตรง detector/mask outputs
- [ ] Primary config และ primary artifacts ไม่เปลี่ยนเมื่อเปิด sensitivity

**Definition of Done:**

- [ ] รายงานผลเป็น data-retention/class-distribution diagnostics
- [ ] แยก primary กับ sensitivity ชัดเจน
- [ ] ไม่มี model training หรือ Test-score-based rule selection

---

# Future Work — ไม่ใช่ Phase 2 Must-have

- [ ] Robust Z-score detector
  - [ ] Freeze median/MAD formula
  - [ ] Freeze scaling constant
  - [ ] Freeze cutoff
  - [ ] กำหนด `MAD = 0` policy
  - [ ] เพิ่ม strict-comparison และ leakage tests
- [ ] RSI decay sensitivity
  - [ ] นิยาม tolerance เชิงตัวเลข
  - [ ] ประเมิน window ที่ยาวกว่า `s+19`
  - [ ] เปรียบเทียบกับ Policy A โดยไม่เปลี่ยน primary artifacts
- [ ] Train models สำหรับ sensitivity variants
  - [ ] ทำใน modeling phase เท่านั้น
  - [ ] ใช้ paired protocol, seeds, metrics และ Full Test เดียวกัน
  - [ ] ห้ามเลือก detector ย้อนหลังจาก Test performance
- [ ] Sequence-model rules
  - [ ] ห้ามสร้าง sequence ข้ามวันที่ที่ถูกกรอง
  - [ ] เพิ่ม contiguous-segment contract และ tests
- [ ] Source-data correction workstream หาก direct audit พบ data error
  - [ ] แก้ baseline แยกจาก Phase 2
  - [ ] Reproduce Phase 1 ใหม่
  - [ ] Update manifests/checksums อย่างตรวจสอบได้
  - [ ] Re-run Phase 2 หลัง baseline review

---

# Phase 2 Definition of Done รวม

Phase 2 ถือว่าเสร็จเมื่อ Must-have M1–M8 ครบทั้งหมด:

- [ ] Primary detector fit จาก Original Train เท่านั้น
- [ ] Primary threshold ใช้ strict `>` และไม่ hard-code
- [ ] Primary affected mask ใช้ `[s-5, s+19]` แบบ inclusive
- [ ] RSI Policy A และข้อจำกัดถูกบันทึกทุกจุดที่เกี่ยวข้อง
- [ ] Boundary clipping/isolation ถูกทดสอบและรายงาน
- [ ] Direct spikes ถูก trace กลับ source
- [ ] With-Spike และ Non-Spike Train reproducible
- [ ] Full Validation/Test เป็น Original splits เดิม
- [ ] Diagnostic segments ไม่แทนผล Full Validation/Test
- [ ] Original Raw/Snapshot/Manifest/Clean/Features/Targets/Splits ไม่เปลี่ยน
- [ ] Original Classification Q75 และ labels ไม่เปลี่ยน
- [ ] Reports, figures และ checksums ตรง artifacts
- [ ] Separate Phase 2 runner ผ่าน integration tests
- [ ] Existing full test suite ผ่าน
- [ ] Saved-report portability mismatch ได้รับการตรวจและบันทึกผล
- [ ] README/HANDOFF/PROJECT_STATUS ตรงกับหลักฐานจริง
- [ ] ยังไม่มีการอ้างว่า model training หรือ model evaluation เสร็จ

Optional sensitivity ไม่เป็นเงื่อนไขปิด Phase 2 เว้นแต่ทีมยกระดับเป็น requirement ภายหลัง ส่วน Robust Z-score, RSI decay alternatives และ model training เป็น Future Work

## ลำดับ Implementation ที่แนะนำ

1. M1 — Freeze และ verify baseline input contract
2. M2 — Implement pure primary detector
3. M3 — Implement affected mask และ boundary metadata
4. M4 — ทำ direct spike/data-quality audit
5. M5 — สร้าง primary experiment datasets และ diagnostic copies
6. M6 — สร้าง reports และ primary figures
7. M7 — ประกอบเป็น separate Phase 2 runner
8. M8 — รัน verification และอัปเดต documentation
9. S1–S3 — ทำ optional data-retention sensitivity หากเวลาเพียงพอ
10. เปิด modeling phase หลัง Must-have gate ผ่านเท่านั้น
