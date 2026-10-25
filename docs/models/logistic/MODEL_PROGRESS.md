# logistic — classification · ✅ เสร็จ

**ผู้รับผิดชอบ:** พิทวัส · **สเปก:** [card #5](../../design/MODEL_CARDS.md) · **วันที่:** 2026-10-11
**ปลดล็อก:** `stacking` ของวิชญา ครบแล้ว (ต้องการ `logistic` + `knn` + `decision_tree`)

## ผล (validation)

| variant | l2 | class_weight | Acc | Prec | Sens | Spec | F1 | **AUC** | confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_spike` | 0.01 | None | 0.9084 | 0.5000 | 0.1765 | 0.9822 | 0.2609 | 0.6732 | `[[331,6],[28,6]]` |
| `non_spike` | 0 | None | 0.9057 | 0.4615 | 0.1765 | 0.9792 | 0.2553 | **0.6824** | `[[330,7],[28,6]]` |

reload ✅ ทั้งคู่ · preprocess `standardize`

## ข้อสังเกตสำหรับรายงาน

1. **accuracy 0.9084 ของ `with_spike` เท่ากับ majority baseline พอดี** — โมเดลจับ positive
   ได้แค่ 6 จาก 34 (sensitivity 0.18) แต่ precision 0.50 แปลว่าที่ทายว่าบวกถูกครึ่งหนึ่ง
   เป็นตัวอย่างชัดว่าทำไมต้องรายงาน AUC/F1 ไม่ใช่ accuracy
2. **`class_weight="balanced"` แพ้ทั้งสอง variant** — ถึงข้อมูลจะ imbalance 9.2% แต่เกณฑ์เลือก
   candidate คือ average_precision → roc_auc ซึ่งไม่ได้ดีขึ้นจากการถ่วงน้ำหนัก
   (balanced ช่วย recall แต่แลกกับ precision)
3. **ทิศทาง A↔B สวนทางกับ `knn`** — logistic ชอบ `non_spike` (AUC +0.009) ส่วน knn ชอบ
   `with_spike` (+0.053) → **ยังสรุปไม่ได้ว่า classification มีทิศทางเดียว** ต้องรอโมเดลที่เหลือ
   อย่าเพิ่งเขียนข้อสรุปลงรายงาน

## Parity

exact ครบ 6 candidates (class_weight ∈ {None, balanced} × l2 ∈ {0, 0.001, 0.01})
`max|Δcoef| = 2.9e-08` · `max|Δproba| = 7.8e-09`

**reference คือ `LogisticRegression` ไม่ใช่ `SGDClassifier` ตามที่ config เดิมระบุ** —
SGD อัปเดตทีละตัวอย่าง เดินคนละ path กับ full-batch GD ตลอดกาล แต่ objective เป็น convex
มี optimum เดียว LBFGS กับ GD จึงลู่เข้าจุดเดียวกัน (override นี้บันทึกไว้ใน `MODEL_CARDS.md` §0.1.2)

### สูตรแปลง C

```text
ของเรา   J   = Σ wᵢ·BCE(yᵢ,pᵢ) / Σ wᵢ  +  (λ/2)·‖w‖²
sklearn  J   = C · Σ wᵢ·logloss(yᵢ)     +  0.5·‖w‖²
หาร J ของเราด้วย λ แล้วจับคู่  →  C = 1 / (λ · Σ wᵢ)
λ = 0 → C = np.inf
```

`class_weight` ทั้ง None และ `balanced` ให้ `Σ wᵢ = n` เท่ากัน เพราะ balanced ใช้ `n/(2·nᶜ)`

## ปัญหาที่เจอและแก้แล้ว

**parity พัง 6/6 ตอนแรกที่ระดับ 3e-7** — ไม่ใช่อัลกอริทึมผิด แต่เป็น **convergence budget**

วัดผลของ `tol` บน fixture เดียวกัน (l2=0.01, lr=0.5):

| tol | หยุดที่ iter | max\|Δcoef\| | max\|Δproba\| |
| --- | ---: | --- | --- |
| 1e-6 (ค่าใน config) | 550 | 2.67e-05 | 3.19e-06 |
| 1e-9 | 900 | **2.86e-08** | 7.79e-09 |
| 1e-12 | 1300 | 1.05e-08 | 7.43e-09 |
| 0 (รันเต็ม 60000) | 59999 | 1.05e-08 | 7.43e-09 |

ลด tol แล้วเข้าใกล้ sklearn เรื่อย ๆ จนชนพื้น 1.0e-08 ซึ่งเป็นความแม่นของ sklearn เอง
→ ยืนยันว่า gradient ถูก

**แก้:** parity test รันด้วย `tol=1e-9` เพื่อถามคำถามที่ถูก (GD ลู่เข้าจุดเดียวกับ LBFGS ไหม)
ส่วน **production ยังใช้ `tol=1e-6` ตาม config** · เพิ่ม test ที่ล็อกค่า default ไว้ไม่ให้ drift
และ test ที่พิสูจน์ว่า tol แน่นขึ้นแล้วเข้าใกล้ sklearn (ไม่ใช่ห่างออก)

### หมายเหตุ (2026-10-11): บนข้อมูลจริง ค่า production หยุดเพราะครบ `max_iter` ไม่ใช่เพราะถึง `tol`

ตัวเลขข้างบนวัดบน fixture ซึ่งหยุดที่รอบ 550. บนข้อมูลจริงด้วยค่าตาม config (`learning_rate=0.05`,
`max_iter=5000`, `tol=1e-6`) **ทั้ง 12 การรัน (6 candidates × 2 variants) หยุดที่รอบ 5,000** โดย gradient
ยังอยู่ที่ 4.5e-06 ถึง 4.3e-04. โมเดลที่บันทึกจึงอยู่ใกล้ optimum แต่ไม่ใช่ optimum พอดี.

เทียบกับโมเดลเดียวกันที่รันจนลู่เข้า (`learning_rate=0.5, max_iter=60000, tol=1e-9`) บน Validation:

| | แย่ที่สุดใน 12 การรัน | ผู้ชนะ `with_spike` (l2=0.01) | ผู้ชนะ `non_spike` (l2=0) |
| --- | ---: | ---: | ---: |
| probability ต่างสูงสุด | 0.0079 | 0.0002 | 0.0010 |
| AP ต่าง | 0.0007 | 0.0000 | 0.0000 |
| AUC ต่าง | 0.0012 | 0.0000 | 0.0000 |
| label ที่ต่าง (จาก 371 แถว) | 1 | 0 | 0 |

**ตัดสินใจไม่แก้โมเดล** — ผลไม่เปลี่ยนถึงทศนิยม 4 ตำแหน่งสำหรับผู้ชนะ และลำดับผู้ชนะไม่เปลี่ยน.
สิ่งที่ต้องระวังเมื่อเขียนรายงาน: คำว่า "parity กับ sklearn ที่ 2.9e-08" เป็นของอัลกอริทึมเมื่อรันจนลู่เข้า
ไม่ใช่ของ artifact ที่ train ด้วยค่า production ซึ่งห่างจาก sklearn ไม่เกิน 0.008 ใน probability.
เรื่องนี้พบตอนทำ `stacking` ซึ่งใช้ logistic เป็น base บน fold ที่เล็กกว่าและห่างจาก optimum มากกว่า (0.03–0.07)
จึงรัน base จนลู่เข้าเอง — ดู [`stacking`](../stacking/MODEL_PROGRESS.md).

## การตัดสินใจ

- sigmoid แยก branch ตามเครื่องหมายเพื่อไม่ให้ `exp` overflow
- loss ใช้ `logaddexp(0, margin) − y·margin` แทน `log(sigmoid)` ที่ underflow ได้
- intercept ไม่ถูก penalize (ตรงกับ `penalize_intercept: false` ใน config)
- `predict` ใช้ `> 0.5` ให้ตรง argmax ของ sklearn

## ไฟล์

- `src/ml/classification/logistic.py`
- `tests/ml/classification/test_logistic.py` (19 tests)

## ค้างอยู่

- card #5 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/logistic.md -->
### ผลจริงของ `logistic`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"class_weight": null, "l2": 0.01, "learning_rate": 0.05, "max_iter": 5000, "tol": 1e-06} | 0.2330 | 0.6732 | 0.2609 | 0.1765 | 0.5000 | 0.9822 | 0.9084 |
| with_spike | test (one-shot) | {"class_weight": null, "l2": 0.01, "learning_rate": 0.05, "max_iter": 5000, "tol": 1e-06} | 0.5518 | 0.7822 | 0.5203 | 0.4384 | 0.6400 | 0.9400 | 0.8418 |
| non_spike | validation (selection score) | {"class_weight": null, "l2": 0, "learning_rate": 0.05, "max_iter": 5000, "tol": 1e-06} | 0.2513 | 0.6824 | 0.2553 | 0.1765 | 0.4615 | 0.9792 | 0.9057 |
| non_spike | test (one-shot) | {"class_weight": null, "l2": 0, "learning_rate": 0.05, "max_iter": 5000, "tol": 1e-06} | 0.5510 | 0.7782 | 0.5041 | 0.4247 | 0.6200 | 0.9367 | 0.8365 |
<!-- END GENERATED -->
