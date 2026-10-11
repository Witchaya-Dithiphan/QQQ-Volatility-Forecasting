# gaussian_nb — classification · ✅ เสร็จ

**เจ้าของ:** Claude · **สเปก:** [card #6](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | var_smoothing | Acc | Prec | Sens | Spec | **F1** | AUC | confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_spike` | 1e-7 | 0.8760 | 0.3000 | 0.2647 | 0.9377 | **0.2812** | 0.6665 | `[[316,21],[25,9]]` |
| `non_spike` | 1e-8 | 0.8275 | 0.2000 | 0.2941 | 0.8813 | 0.2381 | 0.6789 | `[[297,40],[24,10]]` |

reload ✅ · preprocess **`none`** (ตาม config — เป็นโมเดลแรกที่ใช้เส้นทางนี้หลังแก้ trainer)

## ข้อสังเกต

- **F1 0.2812 ดีที่สุดในบรรดา 3 classifier ที่ทำแล้ว** (knn 0.2000 · logistic 0.2609)
  และ **sensitivity สูงสุด** (0.2647 vs 0.1765 ของอีกสองตัว) — จับ positive ได้ 9/34
  แลกกับ accuracy ที่ต่ำลง ซึ่งเป็น trade-off ที่ถูกต้องสำหรับปัญหา imbalanced นี้
- AUC 0.6665 ต่ำสุดในสามตัว → เรียงอันดับได้แย่กว่า แต่ที่ threshold 0.5 ให้ผลใช้งานได้ดีกว่า
  (เป็นตัวอย่างชัดว่า AUC กับ F1 วัดคนละเรื่อง ใช้ในรายงาน §7.2 ได้)
- ทิศทาง A↔B: AUC ชอบ `non_spike` (+0.012) แต่ F1 ชอบ `with_spike` (+0.043) — **ขัดกันเอง**

## Parity

exact ครบ 3 candidates · `theta_` และ `var_` ตรงที่ **1e-12** · proba ตรงที่ 1e-10
ทดสอบเพิ่มบนข้อมูล 8 features ที่สเกลต่างกัน 1000 เท่า (จำลองข้อมูลจริงที่ไม่ standardize)

## การตัดสินใจ

- `var_smoothing` คูณกับ `np.var(X, axis=0).max()` ของ **ทั้งชุด** ไม่ใช่ของแต่ละคลาส
  (จุดที่ card เตือนไว้ และเป็นจุดเดียวที่ทำให้ค่าเพี้ยนแบบสม่ำเสมอถ้าพลาด)
- `ddof=0` ตรงกับ sklearn
- คำนวณใน log space ตลอด + log-sum-exp ที่ shift ด้วย row max → ไม่ underflow แม้ 60 features
- `predict` ใช้ `argmax` ของ joint log-likelihood ตรง ๆ (ไม่ผ่าน threshold 0.5) ให้ตรง sklearn

## ไฟล์

- `src/ml/classification/gaussian_nb.py`
- `tests/ml/classification/test_gaussian_nb.py` (13 tests)

## ปัญหาที่เจอ

ไม่มี — ผ่าน parity รอบแรก เป็นโมเดลที่ closed form ล้วนและไม่มี RNG ตามที่ card คาดไว้

## ค้างอยู่

- card #6 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/gaussian_nb.md -->
### ผลจริงของ `gaussian_nb`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"var_smoothing": 1e-07} | 0.2218 | 0.6665 | 0.2812 | 0.2647 | 0.3000 | 0.9377 | 0.8760 |
| with_spike | test (one-shot) | {"var_smoothing": 1e-07} | 0.5538 | 0.7703 | 0.5806 | 0.6164 | 0.5488 | 0.8767 | 0.8257 |
| non_spike | validation (selection score) | {"var_smoothing": 1e-08} | 0.2292 | 0.6789 | 0.2381 | 0.2941 | 0.2000 | 0.8813 | 0.8275 |
| non_spike | test (one-shot) | {"var_smoothing": 1e-08} | 0.5884 | 0.7779 | 0.5824 | 0.7260 | 0.4862 | 0.8133 | 0.7962 |
<!-- END GENERATED -->
