# logistic — classification · ✅ เสร็จ

**เจ้าของ:** Claude · **สเปก:** [card #5](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11
**ปลดล็อก:** `stacking` ของ Codex ครบแล้ว (ต้องการ `logistic` + `knn` + `decision_tree`)

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
