# multiple_linear — regression · ✅ เสร็จ

**สถานะ:** เสร็จ 2026-10-10 (pilot ที่ใช้พิสูจน์ framework ทั้งลูป)

## ผล (validation)

| variant | RMSE | MAE | R² |
| --- | --- | --- | --- |
| `with_spike` | 0.068864 | 0.054186 | **−0.0446** |
| `non_spike` | รันแล้ว `reload=ok` | | ยังไม่บันทึกตัวเลข |

**R² ติดลบเป็นผลจริง ไม่ใช่บั๊ก** — linear model ทำนายแย่กว่าเดาค่าเฉลี่ย
ใช้เป็นหลักฐานตอบ *"why must use ML/DL?"* ของโจทย์ และเป็น baseline ให้ตัวอื่นเทียบ

## Parity

exact · max abs diff **1.11e-15** กับ `LinearRegression` บน train set เต็ม (1,733 × 8)

## การตัดสินใจ

ใช้ `np.linalg.lstsq` (SVD) ไม่ใช่ normal equation `inv(X'X)` — normal equation ยกกำลังสอง
condition number และ sklearn ก็ใช้ SVD อยู่แล้ว parity จึงมาเอง

## ไฟล์

- `src/ml/regression/multiple_linear.py`
- `tests/ml/regression/test_multiple_linear.py` (8 tests)

## ปัญหาที่เจอและแก้แล้ว

test `test_handles_collinear_features` พังตอนแรก — ผมสร้างข้อมูล collinear เกินจริง
(cond 1.97e6) ทำให้ระบบ rank-deficient คำตอบ least-squares ไม่ unique → solver คนละตัว
ให้คนละคำตอบโดยไม่มีใครผิด **แก้ test ให้ใช้ |r| = 0.8 ตามข้อมูลจริง (cond(X) = 6.0)**

บทเรียน: อย่าเขียน test ที่ ill-conditioned เกินข้อมูลจริง จะได้ทดสอบโมเดล ไม่ใช่ทดสอบ rcond cutoff

## ค้างอยู่

- ยังไม่บันทึกตัวเลข `non_spike` ลงเอกสาร (รันผ่านแล้ว แต่ยังไม่ได้อ่านค่า)
- card ใน `MODEL_CARDS.md` #2 ยังเป็นค่าที่วางแผนไว้ ยังไม่ใส่ผลจริง

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/multiple_linear.md -->
### ผลจริงของ `multiple_linear`

| variant | split | params | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"fit_intercept": true} | 0.0689 | 0.0542 | -0.0446 |
| with_spike | test (one-shot) | {"fit_intercept": true} | 0.1158 | 0.0722 | 0.2959 |
| non_spike | validation (selection score) | {"fit_intercept": true} | 0.0668 | 0.0530 | 0.0165 |
| non_spike | test (one-shot) | {"fit_intercept": true} | 0.1180 | 0.0721 | 0.2684 |
<!-- END GENERATED -->
