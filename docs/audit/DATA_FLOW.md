# DATA FLOW — จำนวนแถวจริงทุกขั้น

> ตัวเลขทั้งหมดวัดจากไฟล์บนดิสก์เมื่อ 2026-10-11 · รายละเอียดการตรวจ: [`DATA_PIPELINE_AUDIT.md`](DATA_PIPELINE_AUDIT.md)

```mermaid
flowchart TD
    RAW["data/raw/qqq_daily.csv<br/>2,512 แถว · 2016-08-31 → 2026-08-28<br/>เรียงใหม่สุดก่อน · SHA 649e1db1…b6b13"]
    CLEAN["clean_data.py → qqq_clean.csv<br/>2,512 แถว · เรียงเก่าสุดก่อน<br/>ตัดออก 0 แถว"]
    FEAT["build_features.py → qqq_features.csv<br/>2,512 แถว · 8 features<br/>20 แถวแรกมี NaN (warm-up)"]
    TGT["build_targets.py → qqq_regression_target.csv<br/>2,512 แถว · target = std ของ return t+1…t+5 × √252<br/>5 แถวท้ายเป็น NaN"]
    DROP["split_data.py: dropna<br/>2,487 แถว (ตัด 20 หัว + 5 ท้าย)"]
    TRAIN["Train<br/>1,733 แถว<br/>2016-09-29 → 2023-08-18"]
    GAP1["purge gap 1<br/>5 แถว · 2023-08-21 → 08-25"]
    VAL["Validation<br/>371 แถว<br/>2023-08-28 → 2025-02-19"]
    GAP2["purge gap 2<br/>5 แถว · 2025-02-20 → 02-26"]
    TEST["Test<br/>373 แถว<br/>2025-02-27 → 2026-08-21"]
    Q75["Q75 จาก Train เท่านั้น = 0.2530580184684854<br/>label = target &gt; Q75"]
    TL["train_labeled.csv<br/>0/1 = 1,300 / 433"]
    VL["validation_labeled.csv<br/>0/1 = 337 / 34"]
    TEL["test_labeled.csv<br/>0/1 = 300 / 73"]
    SPK["detect_spikes.py: threshold จาก Train เท่านั้น<br/>abs(return_1d) &gt; Q3 + 3·IQR = 0.0465436445413787<br/>spike 19 (Train) / 0 (Val) / 4 (Test)"]
    MASK["spike_contract.py: หน้าต่าง [s−5, s+19]<br/>Train affected 213 แถว"]
    A["with_spikes/train.csv<br/>1,733 แถว · byte-identical กับ train_labeled"]
    B["non_spike/train.csv<br/>1,520 แถว · 0/1 = 1,240 / 280"]
    VF["diagnostics/validation_flagged.csv<br/>371 แถว · affected 0"]
    TF["diagnostics/test_flagged.csv<br/>373 แถว · affected 54"]
    ML["src/ml: โหลด Train ตาม variant<br/>+ validation_labeled.csv เดียวกัน<br/>Test ยังไม่ถูกอ่าน"]

    RAW --> CLEAN --> FEAT --> TGT --> DROP
    DROP --> TRAIN & GAP1 & VAL & GAP2 & TEST
    TRAIN --> Q75
    Q75 --> TL & VL & TEL
    TRAIN --> TL
    VAL --> VL
    TEST --> TEL
    TL --> SPK --> MASK
    MASK --> A & B
    VL --> VF
    TEL --> TF
    SPK -. flag เท่านั้น ไม่กรอง .-> VF & TF
    A & B & VL --> ML
```

## เลขคณิตของ split

| ขั้น | แถว |
| --- | --- |
| ทั้งหมด | 2,512 |
| − NaN (20 warm-up + 5 ไม่มี target) | 2,487 |
| − purge gap 2 × 5 | 2,477 |
| Train = floor(2,477 × 0.70) | 1,733 |
| Validation = floor(2,477 × 0.15) | 371 |
| Test = ส่วนที่เหลือ | 373 |

Purge gap 5 แถวเท่ากับ horizon ของ target พอดี (ไม่มีส่วนเผื่อ): แถวสุดท้ายของ Train (2023-08-18)
ใช้ return ถึง 2023-08-25 และ Validation เริ่ม 2023-08-28 · feature มองย้อนหลังได้ถึง 20 วันข้าม boundary
ซึ่งไม่ใช่ leakage เพราะเป็นข้อมูลอดีต
