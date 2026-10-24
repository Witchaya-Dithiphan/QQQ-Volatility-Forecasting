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
   ตรวจอสมการ `y_i(w·x_i+b) >= 1` ด้วย `scipy.optimize.linprog` บน train ที่ standardize แล้ว: solver รายงาน `infeasible` ทั้ง `with_spike` และ `non_spike` (มี test ตรวจซ้ำใน `tests/ml/classification/test_perceptron.py`)
   รันครบ 1000 epochs โดยยังผิด 38.1% ใน `with_spike` และ**จบที่ weight ของการ update ครั้งสุดท้าย**
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

- หากอาจารย์ต้องการให้โมเดลบนข้อมูล QQQ ลู่เข้าแบบไม่มีการแก้น้ำหนัก ต้องเปลี่ยนโจทย์ข้อมูลหรือเปลี่ยนอัลกอริทึม; Perceptron แบบดั้งเดิมบน train สองชุดนี้ทำไม่ได้

## ตรวจการลู่เข้าเพิ่มเติม (branch `fix/perceptron-convergence`)

ใช้ train ที่ `src.ml.core.data.load()` ยืนยัน hash และ `Standardizer` ที่ fit เฉพาะ train เช่นเดียวกับ trainer จากนั้นตรวจความเป็นไปได้ของเส้นแบ่งเชิงเส้นด้วย linear programming โดยปล่อย `w` และ `b` ไม่จำกัดเครื่องหมาย ผล `infeasible` ทั้งสอง variants แปลว่าไม่มีเส้นแบ่งที่จัดทุกแถวได้ถูกต้องพร้อม margin บวก ดังนั้น epoch ที่ไม่มีการอัปเดตไม่อาจเกิดกับข้อมูลฝึกเดิมได้

| variant | train rows | train positives | LP status | updates epoch แรก | updates epoch สุดท้าย | ช่วง updates/epoch | train error หลังรอบ 1000 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `with_spike` | 1,733 | 433 | infeasible | 274 | 274 | 247–301 | 38.1% |
| `non_spike` | 1,520 | 280 | infeasible | 209 | 209 | 195–237 | 42.4% |

เพิ่ม `history_["updates"]` เพื่อให้ตรวจจำนวนการอัปเดตทุก epoch ได้ ผลฝึกซ้ำใน output root แยกมี validation AP/ROC-AUC/recall เท่าผลเดิมทั้งสอง variants และผ่าน reload check ไม่ได้แตะ Test หรือเขียนทับ artifact ที่ล็อกไว้

ทดลองเก็บน้ำหนักที่มี train error ต่ำสุดเมื่อจบแต่ละ epoch (pocket) และเฉลี่ยน้ำหนักเมื่อจบแต่ละ epoch แบบสำรวจ โดยคงลำดับข้อมูลและ 1,000 epochs เดิม: validation ROC-AUC ของ pocket = 0.3574 / 0.3291 และของค่าเฉลี่ย = 0.3061 / 0.3254 (`with_spike` / `non_spike`); AP ของ pocket = 0.0718 / 0.0671 ส่วน AP ของค่าเฉลี่ย = 0.0645 / 0.0681 ไม่มีวิธีใดแก้การไม่ลู่เข้าแบบ Perceptron ดั้งเดิม และผลจัดอันดับไม่ดีพอให้แทนโมเดลเดิม จึงไม่เพิ่มวิธีเหล่านี้ใน pipeline

การทำซ้ำหลักฐาน: `python -m pytest tests/ml/classification/test_perceptron.py -q` ตรวจ LP และกฎหยุด; รัน `python -m src.ml.run train --model perceptron --variant with_spike --output-root <temporary-directory>` และเปลี่ยน variant เป็น `non_spike` เพื่อดู `history.json` ใน output root แยก

ทดลอง RBF Kernel Perceptron เพิ่มแบบแยกจากโมเดลบังคับแล้ว: ทั้งสอง variants หยุดโดยไม่มีการแก้น้ำหนักและทาย Train ถูกทุกแถว ผลและคำสั่งทำซ้ำอยู่ใน [แผนการลู่เข้า](../../plan/PERCEPTRON_CONVERGENCE_PLAN.md#การทดลองเสริม-rbf-kernel-perceptron) ห้ามนำผลนี้ไประบุว่า Perceptron แบบดั้งเดิมลู่เข้าแล้ว

ผล Validation และ Test ที่ประเมินหลังส่งงานของโมเดลเสริม รวม metric, confusion matrix และ receipt อยู่ใน [บันทึก RBF Kernel Perceptron](KERNEL_EXPERIMENT.md) ผล Test นี้ไม่ใช่ผล Test ที่ล็อกไว้ล่วงหน้าของ Perceptron แบบดั้งเดิม

การวิเคราะห์เสริมเสร็จแล้ว: walk-forward 4 fold ลู่เข้าทั้งสอง variants, มีช่วง 95% แบบ block bootstrap, ผลต่าง paired, feature importance และ Test แยกช่วง บันทึกใน `outputs/experiments/kernel_perceptron/report/` การตรวจเทียบ RBF ของไลบรารีผ่านบนโมเดลที่บันทึกไว้ แต่การฝึกอิสระของ `with_spike` หยุดคนละ epoch เพราะการคำนวณใกล้ศูนย์ไวต่อลำดับเลขทศนิยม ดูผลและข้อจำกัดใน [บันทึกการทดลอง](KERNEL_EXPERIMENT.md)

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/perceptron.md -->
### ผลจริงของ `perceptron`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"learning_rate": 0.001, "max_epochs": 1000} | 0.0648 | 0.2993 | 0.0000 | 0.0000 | 0.0000 | 0.9644 | 0.8760 |
| with_spike | test (one-shot) | {"learning_rate": 0.001, "max_epochs": 1000} | 0.1374 | 0.2749 | 0.0577 | 0.0411 | 0.0968 | 0.9067 | 0.7373 |
| non_spike | validation (selection score) | {"learning_rate": 0.001, "max_epochs": 1000} | 0.0742 | 0.3383 | 0.1159 | 0.1176 | 0.1143 | 0.9080 | 0.8356 |
| non_spike | test (one-shot) | {"learning_rate": 0.001, "max_epochs": 1000} | 0.1328 | 0.2626 | 0.0813 | 0.0685 | 0.1000 | 0.8500 | 0.6971 |
<!-- END GENERATED -->
