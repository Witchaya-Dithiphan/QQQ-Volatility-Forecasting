# perceptron — classification · ✅ เสร็จ (ผลแย่ แต่ถูกต้อง)

**เจ้าของ:** Claude · **สเปก:** [card #8](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | lr | Acc | Prec | Sens | Spec | F1 | **AUC** | confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_spike` | 0.001 | 0.8760 | 0.0000 | 0.0000 | 0.9644 | 0.0000 | **0.2993** | `[[325,12],[34,0]]` |
| `non_spike` | 0.001 | 0.8356 | 0.1143 | 0.1176 | 0.9080 | 0.1159 | **0.3383** | `[[306,31],[30,4]]` |

reload ✅ · preprocess `standardize`

## ⚠️ AUC < 0.5 — ตรวจแล้วว่าไม่ใช่บั๊ก

AUC 0.2993 ต่ำกว่าการสุ่ม แปลว่าคะแนนที่โมเดลให้**สัมพันธ์ผกผัน**กับ target
ก่อนบันทึกได้ตรวจกับ sklearn บนข้อมูลจริงชุดเดียวกัน:

```text
lr=0.001  coef ตรง sklearn: True | AUC scratch 0.2993 | AUC sklearn 0.2993 | epochs 1000  mistake_rate 0.381
lr=0.01   coef ตรง sklearn: True | AUC scratch 0.2993 | AUC sklearn 0.2993 | epochs 1000  mistake_rate 0.381
lr=0.1    coef ตรง sklearn: True | AUC scratch 0.2993 | AUC sklearn 0.2993 | epochs 1000  mistake_rate 0.381
```

**sklearn ได้ค่าเดียวกันเป๊ะ** → implementation ถูกต้อง ผลแย่เป็นคุณสมบัติจริงของอัลกอริทึมบนข้อมูลนี้

### ทำไมถึงแย่ — ใช้เขียนรายงานได้ตรง ๆ

1. **ข้อมูลไม่ linearly separable** → Perceptron ไม่มีการรับประกันการลู่เข้า
   รันครบ 1000 epochs โดยยังผิด 38.1% และ**จบที่ weight ของการ update ครั้งสุดท้าย**
   ซึ่งไม่ใช่จุดที่ดีที่สุดที่เคยผ่านมา
2. **ไม่มี objective ให้ลด** — ต่างจาก logistic ที่ลู่เข้า optimum ของ convex loss
   Perceptron แก้เฉพาะตัวอย่างที่ผิดทีละตัว ไม่มีแนวคิดเรื่อง "คำตอบที่ดีที่สุด"
3. เทียบโดยตรง: logistic บนข้อมูลเดียวกันได้ AUC **0.6732** — ความต่างนี้คือคุณค่าของการมี
   loss function ที่หาอนุพันธ์ได้ **เป็นเนื้อหา discussion ชั้นดี**

### lr ทั้ง 3 ค่าให้ผลเหมือนกันทุกตัว — อธิบายได้ทางทฤษฎี

เริ่มจาก `w = 0` และทุก update เป็น `lr·y·x` ดังนั้น `w(lr) = lr · w(1)` พอดี
→ decision function ถูกสเกลด้วยค่าคงที่บวก → **เครื่องหมายและลำดับไม่เปลี่ยน**
→ prediction และ AUC เหมือนกันทุก lr

**grid `learning_rate` ของโมเดลนี้จึงไม่มีผลใด ๆ** (3 candidates ที่ให้ผลเท่ากัน)
ควรบันทึกไว้ในรายงานว่าเป็นข้อสังเกตเชิงทฤษฎีที่ยืนยันด้วยการทดลอง

## Parity

exact · coefficient ตรง sklearn ที่ **1e-10** ครบทั้ง 3 candidates

**`tol=None` คือกุญแจ** — ค่า default ของ sklearn คือ `tol=1e-3` ซึ่งจะ early-stop
คนละ epoch กับเรา ทำให้ weight จบคนละที่และ parity พังทันที

## การตัดสินใจ

- label แปลงเป็น {−1, +1} ตอนคำนวณ margin ตาม config แล้วแปลงกลับเป็น {0,1} ตอนคืนค่า
- `shuffle=False` ทั้งสองฝั่ง — ลำดับการ update เปลี่ยนผลทั้งหมด
- `predict_proba` raise `NotImplementedError` ตาม `output: decision_score` ใน config
  (Perceptron ไม่ให้ความน่าจะเป็นที่ calibrate แล้ว) → trainer ใช้ `decision_function` แทนอัตโนมัติ
- loss ที่บันทึกคือ perceptron criterion `mean(max(0, −margin))` ตาม config

## ไฟล์

- `src/ml/classification/perceptron.py`
- `tests/ml/classification/test_perceptron.py` (13 tests)

## ค้างอยู่

- card #8 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง
