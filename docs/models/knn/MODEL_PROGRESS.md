# knn — classification · ✅ เสร็จ

**เจ้าของ:** Claude · **สเปก:** [card #7](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11
**ปลดล็อก:** `stacking` ของ Codex ใช้ตัวนี้เป็น base learner

## ผล (validation)

| variant | k | weights | Acc | Prec | Sens | Spec | F1 | **AUC** | confusion |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_spike` | 15 | distance | 0.8922 | 0.3125 | 0.1471 | 0.9674 | 0.2000 | **0.6987** | `[[326,11],[29,5]]` |
| `non_spike` | 9 | distance | 0.8868 | 0.3000 | 0.1765 | 0.9585 | 0.2222 | **0.6455** | `[[323,14],[28,6]]` |

reload ✅ ทั้งสอง variant · preprocess `standardize` (ระยะยุคลิดไวต่อสเกลโดยตรง)

## ข้อสังเกตสำคัญสำหรับรายงาน

**1. Accuracy เป็น metric ที่หลอกตาในชุดนี้ — ต้องชี้ให้ชัดในรายงาน**

validation มี 337 neg / 34 pos (**positive แค่ 9.2%**) → ทายว่า "ไม่ผันผวน" ทุกแถวได้
accuracy = **0.9084** ซึ่ง **สูงกว่า kNN ที่ได้ 0.8922**

โมเดลจึงแย่กว่า majority baseline เมื่อวัดด้วย accuracy แต่ AUC 0.6987 บอกว่ามีสัญญาณจริง
→ รายงานต้องใช้ **AUC กับ F1 เป็นตัวหลัก** และอธิบายว่าทำไม accuracy ใช้ไม่ได้กับ imbalanced data

**2. ทิศทาง A↔B ของ classification กลับทางกับ regression**

| task | with_spike | non_spike | ใครดีกว่า |
| --- | --- | --- | --- |
| regression (R²) | ติดลบ | พลิกบวก | **non_spike** |
| classification (AUC) | 0.6987 | 0.6455 | **with_spike** |

การตัด spike ออกช่วย regression แต่**ทำร้าย** classification — สมเหตุสมผลเพราะ target
ของ classification คือ "ผันผวนสูงหรือไม่" ซึ่ง spike **คือ** ตัวอย่างบวกที่โมเดลต้องเรียนรู้
ตัดออกไปก็เหลือตัวอย่างบวกน้อยลง นี่เป็นเนื้อหา discussion ชั้นดีและต้องรอดูว่าโมเดลอื่นยืนยันไหม

**3. ทั้งสอง variant เลือก `weights="distance"`** — การถ่วงน้ำหนักตามระยะช่วยจริงในชุดนี้

## Parity

exact ครบทั้ง 8 candidates (k ∈ {3,5,9,15} × weights ∈ {uniform, distance})
ทดสอบบน **held-out points** ซึ่งเป็นสิ่งที่ trainer ทำจริง (fit บน train → predict บน validation)

## ปัญหาที่เจอและแก้แล้ว

### sklearn คำนวณระยะจากจุดถึงตัวเองได้ไม่เท่ากับ 0

parity พัง 4 เคส (`distance` weights ทุกค่า k) ด้วย max diff **4.8e-08** ตอนทดสอบโดย
predict บน training set ตัวเอง

**สาเหตุจริง:** sklearn ใช้สูตรกระจาย `‖a‖² + ‖b‖² − 2a·b` ซึ่งเกิด catastrophic cancellation
วัดได้ว่าระยะจากจุด 142 ถึงตัวเอง = `4.2146848510894035e-08` ไม่ใช่ 0 → ไม่ trigger inf-mask
ของ sklearn → คิด `1/d = 2.37e7` ซึ่งครอบงำแต่ไม่กลบสนิท ได้ proba `4.84e-08`
ส่วน `np.linalg.norm(x−x)` ของเราได้ **0 พอดี** → น้ำหนักเต็ม → proba `0.0`

**scratch แม่นกว่า sklearn ในเคสนี้** (เหมือนกรณี `svm` ที่ objective ต่ำกว่า)

**แก้:** ย้าย parity test ไปทดสอบบน held-out points ซึ่งเป็นสถานการณ์เดียวที่ pipeline สร้างจริง
(trainer fit บน train แล้ว predict บน validation ไม่เคย predict ทับ train) → ตรงที่ **1e-16**
แล้วเพิ่ม test เฉพาะที่บันทึกพฤติกรรม self-distance ไว้ พร้อม test ว่า label ยังตรง sklearn
ครบทุกแถวแม้บน training set

**ไม่ใช่การลดเกณฑ์** — เป็นการทดสอบสถานการณ์ที่ถูกต้อง

### tie ที่ proba = 0.5 พอดี

`predict` เดิมใช้ `>= 0.5` ได้คลาส 1 แต่ sklearn ใช้ `argmax` ซึ่งเลือก **คลาสที่ index ต่ำกว่า**
เมื่อเสมอ → ต้องใช้ `> 0.5`

## การตัดสินใจ

- `np.argsort(kind="stable")` เพื่อให้ tie ของระยะทางตัดสินด้วย training index เหมือน sklearn
- `algorithm="brute"` ใน reference เพื่อตัดความต่างจาก KD-tree ออก
- zero distance → น้ำหนักเต็มกับจุดนั้น (ไม่ใช่ `inf`) ตามที่ sklearn ตั้งใจทำ
- `state_dict` เก็บ train set ทั้งก้อนเพราะ kNN ไม่มีพารามิเตอร์ที่เรียนรู้ — ตัวข้อมูลคือโมเดล

## ไฟล์

- `src/ml/classification/knn.py`
- `tests/ml/classification/test_knn.py` (21 tests)

## หมายเหตุจากการตรวจครั้งสุดท้าย (2026-10-11)

tie-break ด้วย index ทำให้ผล deterministic แต่**ไม่รับประกันว่าตรง sklearn เมื่อระยะทางเท่ากันพอดี** —
sklearn ใช้ partial sort ที่เลือกในกลุ่ม tie ไม่แน่นอน. สุ่มทดสอบบนข้อมูลที่ปัดทศนิยมให้มีค่าซ้ำ เจอ 2 จาก 60 แถว
ที่เพื่อนบ้านตัวที่ k เสมอกันและสองฝั่งเลือกต่างกัน. ข้อมูลจริงเป็นค่าต่อเนื่อง ไม่มี tie: ที่ค่า production
probability ต่างจาก sklearn ≤ 7.8e-16 และ label ต่าง 0. แก้ docstring ที่เคยเขียนว่า "matching sklearn" แล้ว.

## ค้างอยู่

- card #7 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/knn.md -->
### ผลจริงของ `knn`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"k": 15, "weights": "distance"} | 0.2037 | 0.6987 | 0.2000 | 0.1471 | 0.3125 | 0.9674 | 0.8922 |
| with_spike | test (one-shot) | {"k": 15, "weights": "distance"} | 0.5472 | 0.7847 | 0.5512 | 0.4795 | 0.6481 | 0.9367 | 0.8472 |
| non_spike | validation (selection score) | {"k": 9, "weights": "distance"} | 0.2082 | 0.6455 | 0.2222 | 0.1765 | 0.3000 | 0.9585 | 0.8868 |
| non_spike | test (one-shot) | {"k": 9, "weights": "distance"} | 0.5389 | 0.8008 | 0.4237 | 0.3425 | 0.5556 | 0.9333 | 0.8177 |
<!-- END GENERATED -->
