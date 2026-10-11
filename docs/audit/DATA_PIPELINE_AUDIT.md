# DATA PIPELINE AUDIT — 2026-10-11

> ตรวจบน branch `audit/data-pipeline` (ฐาน `b4293e8`) · ลำดับความเชื่อถือ: ไฟล์ข้อมูลจริง > โค้ด > test > เอกสาร
> ทุกตัวเลขคำนวณซ้ำด้วย pandas/NumPy โดยไม่เรียกโค้ดของ pipeline เว้นแต่ระบุไว้
> เอกสารประกอบ: [`DATA_FLOW.md`](DATA_FLOW.md) · [`PATH_AUDIT.md`](PATH_AUDIT.md) · [`FILE_INVENTORY.md`](FILE_INVENTORY.md)

## คำตัดสิน: ข้อมูลปลอดภัยสำหรับการ train — **ใช่**

1. ไม่พบ 🔴 ที่กระทบแถวของ Train/Validation/Test, target, Q75 หรือ spike mask จึงไม่ต้อง regenerate ข้อมูล และผลของโมเดลทั้ง 38 ชุดยังใช้ได้
2. Snapshot ตรง manifest (SHA-256 `649e1db1…b6b13`, 2,512 แถว, 2016-08-31 → 2026-08-28) และตรงปฏิทิน NYSE พอดี ไม่มีวันขาดหรือเกิน
3. ทั้ง 8 features เป็น causal: build บนประวัติที่ตัดที่ k แถว ได้ผล bit-identical กับ build เต็ม
4. Target ใช้เฉพาะ return ที่ t+1…t+5 และ feature ไม่เห็นค่าเหล่านั้น
5. Split เป็น chronological 1,733 / 371 / 373 มี purge gap 5 แถวสองจุด หน้าต่าง target ของแถวสุดท้ายจบก่อน split ถัดไปเริ่ม
6. Q75, spike threshold, standardizer และ PCA fit จาก Train เท่านั้น
7. `with_spikes` เท่ากับ Train เดิมทุก byte · `non_spike` คือ Train ลบ 213 แถว โดยไม่คำนวณ feature/label ใหม่ · ทั้งสอง variant ใช้ Validation/Test ไฟล์เดียวกัน
8. Pipeline Phase 1 สร้างซ้ำได้ byte-identical ทั้ง 14 ไฟล์ และ `src/ml` ยังไม่เคยอ่าน Test
9. ความเสี่ยงที่เหลือเป็นเรื่อง **การตีความผล** ไม่ใช่ความถูกต้องของข้อมูล: ตัวอย่างไม่อิสระ (D13), Validation เป็น regime เดียว (E10), selection bias (E9), การเทียบ A↔B ถูก confound (F15)
10. 🔴 เดียวที่พบคือ X1 (hash พังบน Windows clone ใหม่) ซึ่งเป็นปัญหา line ending ไม่ใช่ข้อมูล และแก้แล้ว

## ผลการรันจริง

| รายการ | ค่า |
| --- | --- |
| Data-layer tests ก่อน audit | 336 passed, 0 skipped (ผ่านเช่นกันเมื่อใช้ `-W error`) |
| Data-layer tests หลัง audit | 368 passed |
| Reproducibility | รัน `run_data_pipeline --output-root` 2 รอบ: 14/14 ไฟล์เหมือนกัน และเหมือนไฟล์ที่ตรวจรับแล้ว |
| Stored hashes | 26 รายการใน `outputs/reports/*.json` + manifest ตรงไฟล์บนดิสก์ · config ของโมเดล 38 ชุดชี้ hash ปัจจุบัน |
| Coverage / ruff / mypy | ไม่ได้ติดตั้งและไม่ได้ตั้งค่า จึงไม่มีตัวเลข |

## Findings

สถานะ: ✅ แก้แล้ว · 📝 บันทึกในเอกสาร · ⏳ รอตัดสินใจ · — ไม่ต้องทำอะไร

### Track A — Download / raw

| ID | ระดับ | หลักฐาน | การแก้ | สถานะ |
| --- | --- | --- | --- | --- |
| A1–A7, A9 | OK | 2,512 แถวตรง regex `MM/DD/YYYY`; 2,608 วันธรรมดา − 96 วันหยุด = 2,512; volume=0: 0; ราคา≤0: 0; OHLC ผิด: 0; return สูงสุด 15 วันเป็นเหตุการณ์จริง; `Close` ไม่ปรับ; แถวสุดท้ายเป็นวันเต็ม | — | — |
| A8 | 🟡 | ราคาปี 2026 ตรวจกับแหล่งภายนอกไม่ได้ ตรวจได้แค่ความสอดคล้องภายใน | สุ่มเทียบราคาปิด 2–3 วันกับ Nasdaq ด้วยมือ | ⏳ |
| A10 | 🟠 | `write_csv_atomically` กันแค่ `RAW_DATA_PATH`; `--output <manifest> --overwrite` เขียนทับ manifest ได้ | เพิ่ม guard ครอบ manifest | ✅ `tests/test_snapshot_write_guards.py` |
| A11 | 🟠 | `write_json_atomically` ไม่มี guard; `--report` ชี้ snapshot/manifest ได้ | guard ทั้งสอง path | ✅ |
| A12 | 🟡 | `clean_data.py:205,223` กันแค่ raw (runner กันครบอยู่แล้ว) | รวม helper เดียว | ⏳ ไม่แตะโค้ด clean |
| A13 | 🟠 | `.gitignore:41` ตัด `data/raw/*`; Nasdaq เป็น rolling 10 ปี ไฟล์หายแล้วโหลดชุดเดิมไม่ได้ | commit CSV 124 KB หรือแนบกับงานส่ง | ⏳ เจ้าของตัดสิน |
| A14–A15 | 🟡 | refresh เขียนไฟล์ข้าง snapshot; `verify_snapshot` ไม่อ่าน `byte_length` | cosmetic | — |

### Track B — Clean

| ID | ระดับ | หลักฐาน | การแก้ | สถานะ |
| --- | --- | --- | --- | --- |
| B1–B2 | OK | 2,512 → 2,512; max abs diff 0.0 ทุกคอลัมน์; สร้างซ้ำได้ byte-identical | — | — |
| B3 | 🟡 | เปลี่ยนแค่รูปแบบวันที่และ trailing zero; CRLF จาก `to_csv` บน Windows | ดู X2 | 📝 |
| B4 | 🟠 แฝง | `load_qqq_data` คืนข้อมูลใหม่สุดก่อน; ตอนนี้ไม่มีใครคำนวณ return จาก raw ตรง ๆ | ขั้นถัดไป assert `is_monotonic_increasing` (ทำอยู่แล้วใน `split_data.py:85-88`) | 📝 |
| B5–B6 | 🟡 | cleaner ไม่ตรวจ Close ∈ [Low, High]; มีทาง drop เงียบที่ไม่ทำงานบนข้อมูลจริง | เพิ่ม mask/ให้ runner fail เมื่อ `rows_removed > 0` | ⏳ |
| B7 | 🟡 | `tests/test_cleaning.py` ว่าง | ลบ | ✅ |
| B8–B9 | 🟡 | fixture ไม่ใช้รูปแบบ raw จริง; 2 test ใช้ path relative | เพิ่ม fixture รูปแบบจริง | ⏳ |

### Track C — Features

| ID | ระดับ | หลักฐาน | การแก้ | สถานะ |
| --- | --- | --- | --- | --- |
| C1 | OK | truncation ที่ k = 15…2512 ได้ diff 0.0 ทุกค่า; ไม่มี `shift(-`, `center=`, `bfill`, `ewm` | — | — |
| C2 | OK | คำนวณซ้ำจาก raw ต่างไม่เกิน 1.8e-13; warm-up 20 แถว; inf: 0 | — | — |
| C3 | 🟡 | ใช้ **simple return** ไม่ใช่ log | รายงานต้องเขียนให้ตรง | 📝 |
| C4 | 🟡 | `volume_zscore_20` รวมวัน t ในหน้าต่าง ค่าถูกจำกัดที่ 4.25 | เขียนในรายงาน | 📝 |
| C5–C7 | 🟡 | hv5 มาจาก 5 ค่า (noise สูง); RSI แบบ Wilder seed ด้วย SMA; feature รู้ได้หลังปิดตลาดวัน t | เขียนในรายงาน | 📝 |
| C8 | 🟠 | test สูตรใช้ expression เดียวกับโค้ด ไม่มี test บนข้อมูลจริง | เพิ่ม recompute อิสระ | ✅ `test_stored_features_match_an_independent_numpy_recompute` |
| C9 | 🟠 | `test_leakage.py` มี cutoff เดียวบนข้อมูลสังเคราะห์ | truncation หลายค่า k บนข้อมูลจริง | ✅ 8 ค่า k |
| C10 | 🟡 | static grep จับแค่ `shift(-` กับ `center=True` | พึ่ง truncation test | ✅ |

### Track D — Targets

| ID | ระดับ | หลักฐาน | การแก้ | สถานะ |
| --- | --- | --- | --- | --- |
| D1–D3 | OK | `build_targets.py:182-189`; diff 3.3e-14; หน้าต่างอื่น (t…t+4, log, ddof=0) ไม่ตรงทุกแถว; 5 แถวท้าย NaN | — | ✅ pin ด้วย test |
| D6 | 🟠 info | `historical_volatility_5d[t+5]` = `target[t]` (corr 1.0) → purge gap ไม่มีส่วนเผื่อ | test `gap ≥ horizon` | ✅ |
| D7, D9–D11 | OK | Q75 = 0.2530580184684854 (pooled จะได้ 0.2376); label mismatch 0 ใน 7 ไฟล์; มี 1 แถวเท่ากับ threshold พอดีและได้ label 0 | — | ✅ pin ด้วย test |
| D8 | 🟡 | n=1733 ทำให้ Q75 เป็น order statistic พอดี จึงยืนยัน `linear` จากข้อมูลจริงไม่ได้ | หมายเหตุ | 📝 |
| D12 | 🟠 | positive rate 25.0% / 9.2% / 19.6% (regime shift ไม่ใช่ bug) | รายงาน PR-AUC + CI | 📝 |
| D13 | 🟠 | ACF lag 1–5 = 0.92, 0.85, 0.78, 0.70, 0.64; ESS ≈ 50–200 จาก 1,733 | ใช้ block bootstrap; ห้ามอ้าง n=1,733 เป็นอิสระ | 📝 |
| D14 | 🟠 | shuffled 5-fold ให้ R² 0.83 โดยไม่ใช้ feature; โค้ดไม่ได้ shuffle | คง purged expanding folds | ✅ static scan |
| D16 | 🟡 | 2 test ใน `test_classification_target.py` ไม่มีทาง fail | เขียนใหม่ผ่าน pipeline ด้วยข้อมูล poison | ✅ |

### Track E — Chronological split

| ID | ระดับ | หลักฐาน | การแก้ | สถานะ |
| --- | --- | --- | --- | --- |
| E1–E2 | OK | 2512 − 25 = 2487; − 10 = 2477 = 1733+371+373; drop NaN ก่อน split (`split_data.py:165-170`); split ตามตำแหน่ง (`:252-263`) | — | ✅ pin ด้วย test |
| E3–E8 | OK | ทุก threshold และ preprocessor fit จาก Train; fold ของ `slp`/`mlp`/`stacking` เป็น chronological + purge 5; ไม่พบ `train_test_split`, `permutation`, `default_rng` | — | ✅ static scan |
| E9 | 🟠 | เลือก hyperparameter บน Validation แล้วรายงานเลขเดียวกัน (`trainer.py:227-230, 279`); SD ของ AP จาก block bootstrap ≈ 0.10–0.12 ขณะที่ top-5 ห่างกัน 0.251–0.267 | เรียกเลข Validation ว่า selection score; ไม่อ้างอันดับใน top-5; Test คือเลขไม่ลำเอียงตัวเดียว | 📝 |
| E10 | 🟠 | Validation: target std 0.068 (Train 0.144), max 0.41 (Train 1.50); 20/34 positive อยู่ใน 2024Q3 | ตาราง walk-forward ภายใน Train เป็น robustness | ⏳ |
| E11 | 🟡 | trainer ไม่ refit บน Train+Val | ตัดสินใจก่อน finalize | ⏳ |
| E12 | OK | `load_test` ไม่มีผู้เรียก; `finalize` เป็น stub; ไม่มีวันที่ช่วง Test ใน `outputs/modeling` | — | ✅ pin ด้วย test |
| E13 | 🟡 | Test ถูกอ่านนอก `src/ml`: notebook 02 (สรุป split), ชั้นข้อมูล (label/flag ด้วย threshold จาก Train) | เปิดเผยในรายงาน | 📝 |
| E14–E15 | 🟡 | guard ของ Test เป็นแค่ convention; ไม่มี test ผูกกับไฟล์จริง | static scan + positional test | ✅ |

### Track F — Spike / reduced-spike

| ID | ระดับ | หลักฐาน | การแก้ | สถานะ |
| --- | --- | --- | --- | --- |
| F1, F3 | OK | threshold 0.0465436445413787; 19 / 0 / 4 spike; ตรง `spike_event_audit.csv` | — | ✅ pin ด้วย test |
| F2 | 🟡 | fence คำนวณจาก quantile ของ **`abs(return_1d)`** ไม่ใช่ signed (signed จะได้ 21 spike) | รายงานต้องเขียน "Q1/Q3 ของ abs(return_1d)" | 📝 |
| F4, F6 | OK | mask = union [s−5, s+19] = 213 แถว; perturb ราคายืนยันขอบ −5/+19 พอดี | — | — |
| F5 | 🟠 แฝง | หน้าต่างนับภายใน split และไม่สแกน 10 แถวใน gap; ข้อมูลวันนี้ไม่มี spike ตรงนั้น | guard test เทียบ mask เต็มปฏิทิน | ✅ |
| F7–F8, F10, F12 | OK | `with_spikes` SHA เท่า `train_labeled`; `non_spike` 1,520 แถว 1240/280 ค่าเท่าเดิม; Val/Test ร่วมกัน; flag ตรง | — | ✅ pin ด้วย test |
| F9 | 🟡 | `non_spike/train.csv` ไม่มีคอลัมน์เกิน ใหญ่กว่าเพราะ `%.17g` | — | — |
| F11 | 🟡 | `data.py:112,115,132` ใช้ `digests.get(key)` ถ้า key หายจะข้ามการตรวจ hash เงียบ ๆ | ใช้ `digests[key]` | ⏳ |
| F13 | 🟠 | ไฟล์ `*_flagged.csv` ไม่ถูกใช้ใน `src/ml` เลย ยังไม่มี metric แยกช่วง spike/สงบ | ทำใน `finalize` (Test 54 แถว affected) | ⏳ |
| F15 | 🟠 | 213 แถวที่ตัดเป็น positive 71.8% (35% ของ positive ทั้งหมด); positive rate 25.0% → 18.4%; target max 1.50 → 0.55; Validation ไม่มี spike เลย | เขียนว่าผลต่าง A↔B เป็นผลของ "นโยบายตัดแถวทั้งชุด" | 📝 |

### Cross-cutting

| ID | ระดับ | หลักฐาน | การแก้ | สถานะ |
| --- | --- | --- | --- | --- |
| X1 | 🔴 บน Windows clone ใหม่ | `spike_event_audit.csv` ไม่มี EOL attribute; autocrlf ทำ SHA เป็น `5D214F48` ต่างจาก `851E3AFA` ที่เก็บไว้ → `build_experiment_datasets.py:326` raise | `*.csv text eol=lf` ใน `.gitattributes` | ✅ |
| X2 | 🟠 | `to_csv()` ของ Phase 1 ใช้ line ending ของ OS → hash ที่เก็บไว้ใช้ได้บน Windows เท่านั้น | บันทึกว่า "Windows only"; การ pin `lineterminator` ต้อง regenerate ทุกอย่าง | 📝 ไม่แก้ |
| X3 | 🟠 | report ที่ track อ้าง `tmp/phase1-reproduction` ที่ ignore | คงขั้นตอน RUNBOOK §11 เป็นข้อบังคับ | 📝 |
| X4 | 🟠 | ไม่มี snapshot → 20 errors + 8 failures แทนที่จะ skip | เพิ่ม skip guard | ⏳ |
| X5 | 🟠 | `configs/modeling.json:695-696,700` ชี้ `src/modeling`, `src/models` และ layout ที่ไม่มี | แก้หรือลบ block | ⏳ ไม่แตะเพราะอาจกระทบ hash ของ config |
| X6 | 🟡 | ไม่มี test เทียบผล regenerate กับ hash ที่ตรวจรับ | — | ⏳ |
| X8–X10 | 🟡 | โฟลเดอร์ `.pyc` ตกค้าง; ไม่มี ruff/coverage; `pypdf` ไม่อยู่ใน lock; `scipy` ใช้ใน test แต่ไม่ประกาศ | ทำความสะอาด | ⏳ |

## ข้อวิเคราะห์ที่ต้องอยู่ในรายงาน

- **Time gap หลังตัดแถว:** `non_spike` มีช่องว่าง 7 ช่วง (26–53 วันทำการ) ไม่มีโค้ดใดคำนวณ lag/rolling ตอน train และโค้ดที่ขึ้นกับลำดับใช้ตำแหน่งบน timeline เดิม (`trainer.py:31-42`, `slp.py:112-118`, `mlp.py:154-157`, `stacking.py:45-52`) จึงถูกต้อง แต่ขนาด fold ไม่เท่ากันระหว่าง A กับ B
- **Drop หรือ clip:** คง drop เป็นค่าเริ่มต้น การ winsorize `return_1d` อย่างเดียวจะทำให้ rolling feature กับ target ไม่สอดคล้องกัน เว้นแต่คำนวณใหม่ทั้งชั้นข้อมูล
- **Walk-forward CV:** ไม่ควรเลือกโมเดลใหม่ทั้ง 19 ตัวในเวลาที่เหลือ ให้คงผู้ชนะเดิม แล้วเพิ่มตาราง robustness ภายใน Train (fold ของ `stacking` มี positive 353 แถว เทียบกับ 34 ใน Validation) พร้อม block-bootstrap CI
- **Regime:** Train มี Volmageddon 2018, 2018-Q4, COVID 2020, ตลาดหมี 2022 · Validation มีแค่ carry unwind ส.ค. 2024 · Test มี tariff shock เม.ย. 2025

## สิ่งที่ตรวจไม่ได้

ราคาจริงของปี 2026 · เหตุการณ์ของ spike 2026-06-05 (−4.80%) · พฤติกรรมบน Linux จริง (X2 สรุปจากโค้ดและ byte ที่เห็น) · coverage และ lint · notebook 01/02 ยังไม่ได้รันใหม่ (เขียน PNG 12 ไฟล์ลง `outputs/` ต้องรอเจ้าของอนุมัติ) · การลงทะเบียนกลุ่มใน Google Sheet
