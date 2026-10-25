# xgboost — classification · 🟡 รอ leaderboard

**ผู้รับผิดชอบ:** วิชญา (ตั้งแต่ขั้น T5 พิทวัสทำต่อ) · **สเปก:** [card #14](../../design/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | n | depth | lr | λ | γ | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 100 | 1 | 0.03 | 1 | 0 | 0.250802 | 0.669314 | 0.244898 | 0.900270 | 0.176471 | 0.973294 | `[[328,9],[28,6]]` |
| `non_spike` | 100 | 2 | 0.03 | 10 | 0.1 | 0.265178 | 0.685285 | 0.280000 | 0.902965 | 0.205882 | 0.973294 | `[[328,9],[27,7]]` |

เลือกจาก 8 candidates ด้วย AP ก่อน AUC (`xgboost-02` และ `xgboost-07`). ไฟล์อยู่ใน `outputs/modeling/<variant>/classification/xgboost/`; `load_verification.json` เป็น `"passed": true` ทั้งคู่. log loss บน Train: 0.6806 → 0.3871 (`with_spike`), 0.6774 → 0.3298 (`non_spike`) — เริ่มที่ ≈ ln 2 เพราะ `F0 = 0`.

ทั้ง grid (AP / AUC):

| # | n | depth | lr | λ | γ | `with_spike` | `non_spike` |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 50 | 1 | 0.1 | 1 | 0 | 0.2478 / 0.6800 | 0.2526 / 0.6814 |
| 1 | 50 | 2 | 0.1 | 1 | 0 | 0.2268 / 0.6759 | 0.2334 / 0.6639 |
| 2 | 100 | 1 | 0.03 | 1 | 0 | **0.2508** / 0.6693 | 0.2515 / 0.6853 |
| 3 | 100 | 2 | 0.03 | 1 | 0 | 0.2341 / 0.6754 | 0.2468 / 0.6750 |
| 4 | 100 | 2 | 0.1 | 1 | 0 | 0.2127 / 0.6670 | 0.2443 / 0.6589 |
| 5 | 100 | 2 | 0.1 | 10 | 0 | 0.2254 / 0.6816 | 0.2522 / 0.6756 |
| 6 | 100 | 2 | 0.1 | 1 | 0.1 | 0.2127 / 0.6670 | 0.2443 / 0.6589 |
| 7 | 100 | 2 | 0.03 | 10 | 0.1 | 0.2459 / 0.6733 | **0.2652** / 0.6853 |

ข้อสังเกตจากตัวเลข: candidate 6 (γ = 0.1) ให้ผล**เท่ากับ** candidate 4 (γ = 0) ทุกหลักในทั้งสอง variant — γ = 0.1 ไม่ได้ตัดกิ่งใดเลย เพราะ split ที่เกิดขึ้นได้ gain เกิน 0.1 ทั้งหมด · λ = 10 ช่วยทั้งคู่เมื่อเทียบ candidate 5 กับ 4 · Validation มี positive 34 แถว ความต่างระดับนี้ยังไม่ได้ทดสอบนัยสำคัญ.

## Parity

เกณฑ์ strict ปกติ (label 100% + score `allclose(rtol=1e-5)`) กับ `xgboost` **3.4.1** และเทียบทีละต้น. วัดครบ 8 candidates × 2 variants บน Train และ Validation (1,400 ต้น):

- โครงสร้างตรงทุกต้น: จำนวน node, feature, threshold (float32 ตรงทุก bit) — 0 จาก 1,400 ต้นที่ต่าง
- label ต่าง 0 แถว · margin ต่างสูงสุด 4.8e-7 · probability ต่างสูงสุด 6.0e-8
- 13 จาก 16 ชุดตรงระดับ bit ทั้งหมด; อีก 3 ชุดมีค่าใบต่าง 1 ulp ของ float32 ในบางต้น (20 จาก 1,400 ต้น, relative สูงสุด 1.1e-6)

**ที่มาของ 1 ulp ที่เหลือ — พิสูจน์แล้ว ไม่ใช่ข้อสันนิษฐาน:** `expf` ของ C runtime ที่ library เรียกปัดผลต่างจากค่าที่ปัดถูกต้อง 40 จาก 400,000 input (1 ใน 10,000). เทียบ root ของ 720 ต้น: สูตรของเราตรง 717; เมื่อสลับไปเรียก `expf` ของระบบผ่าน ctypes เพื่อวินิจฉัย ตรง **720 จาก 720**. ไม่ได้ใส่ ctypes ในโมเดลเพราะผูกกับ Windows และไม่ใช่ NumPy.

stress test นอก grid (240 รัน: depth ถึง 4, λ ∈ {0, 1, 10}, γ ถึง 0.5, ข้อมูลมีค่าซ้ำ, reference 8 threads): label ต่าง 0, score ต่างสูงสุด 2.4e-7. `test_xgboost.py` 36 tests ผ่าน; ทั้งชุด 854 tests ผ่าน.

## การตัดสินใจ

- **ตาม library เมื่อ card ต่างจาก library.** card เขียน `Gain = 0.5·[…] − γ` และ "แบ่งเมื่อ Gain > 0". ของจริงใน 3.4.1: `loss_chg = score_L + score_R − score_parent` **ไม่มี 0.5**; โตต้นไม้จนสุด depth โดยไม่สน γ (ขอแค่ `loss_chg > 1e-6`) แล้วค่อย prune จากล่างขึ้นบนเมื่อ `loss_chg < γ` และลูกทั้งสองเป็นใบ. ผลคือ split ที่อ่อนอยู่รอดได้ถ้ามีอะไรข้างใต้ที่ควรเก็บ (มี test).
- **คง float32 ไว้ตาม library** สำหรับ gradient, gain, threshold, ค่าใบ, margin; ผลรวม gradient เป็น float64. เขียนเป็น float64 ทั้งหมดจะสะอาดกว่าทางคณิตศาสตร์ แต่เลือก split ต่างกันเมื่อ gain ใกล้กัน.
- **`reference()` ตั้ง `n_jobs=1`** เพิ่มจาก card เพื่อให้ลำดับการบวก gradient คงที่. ไม่เปลี่ยนอัลกอริทึม; ทดสอบกับ 8 threads แล้ว label ยังตรงทุกรัน.
- **grid แบบ `tuples` แก้ที่ trainer ไม่ใช่ที่โมเดล.** grid ใน config เป็นรายการชุดค่า ไม่ใช่ผลคูณของช่วง และ `expand_grid` เดิมทำได้แต่ผลคูณ จึงส่ง `{"tuples": [...]}` มาเป็น kwarg. รอบแรกแก้ทางอ้อมโดยให้ constructor รับ `tuples`; เจ้าของโปรเจกต์ให้ย้ายไปแก้ที่ต้นทางแทน: `expand_grid` รู้จัก `{tuples, tuple_fields}` และ `_load_grid` ดึงชื่อ field จาก `parameters.tuple_fields`. constructor ของ `XGBoost` จึงมีแต่ hyperparameter ของอัลกอริทึม และ `search_results.json` บันทึก 5 field มีชื่อกำกับ.
- **`predict` ตัดที่ probability float32 `> 0.5`** ไม่ใช่ `margin > 0`: margin บวกเล็กมากที่ sigmoid ปัดเป็น 0.5 พอดีต้องได้คลาส 0.
- `random_state` ไม่ถูกใช้ในอัลกอริทึม (ปิด sampling ทั้งหมด) เก็บไว้ส่งให้ reference.

## ไฟล์

- `src/ml/classification/xgboost.py`, `src/ml/registry.py`
- `src/ml/core/trainer.py` (`expand_grid`), `src/ml/run.py` (`_load_grid`) — รองรับ grid แบบ `tuples`
- `tests/ml/classification/test_xgboost.py`, `tests/ml/core/test_trainer.py`, `tests/ml/test_run.py`

## ปัญหาที่เจอและแก้แล้ว

**1. อ่าน source ผิดเวอร์ชัน.** อาการ: gain ต่างจาก reference ที่หลักท้ายของ float32 (6.1518345 vs 6.1518364) และ 1 ใน 240 รัน stress เลือกฟีเจอร์ต่างกันจน margin ต่าง 0.24. สาเหตุ: เขียนตาม source ของ v3.1.1 (`G²/(H+λ)` หารใน float32) แต่ที่ติดตั้งคือ 3.4.1 ซึ่ง split evaluator ถูกเขียนใหม่ — คำนวณ `w = float32(−G/(H+λ))` ก่อน แล้วแทนกลับ `−(2·G·w + (H+λ)·w²)` โดย `w²` คูณใน float32. พีชคณิตเท่ากัน การปัดเศษต่างกัน. 3.4.1 ยังถือว่า split ที่ฝั่งใดมี hessian ≤ 0 ไม่ถูกต้อง. แก้ตาม 3.4.1 แล้วกรณีโครงสร้างต่างหายหมด. **บทเรียน: ดึง source ตาม tag ที่ตรงกับ `xgboost.__version__` เสมอ.**

**2. `np.exp` แบบ float32 ไม่ตรง `expf`.** อาการ: ค่าใบต่าง 1 ulp ใน 43 จาก 50 ต้น. สาเหตุ: `np.exp` บน float32 ต่างจาก `expf` ของ library ถึง 633 จาก 1,520 ค่า. แก้โดยคำนวณ `exp` ใน float64 แล้วปัดเป็น float32 ซึ่งตรงกว่ามาก (เหลือ 1 ใน 10,000 ตามหัวข้อ Parity).

**3. test มือผิดสองตัว (ผิดที่ test ไม่ใช่โมเดล).** คิด `G` ของ root ผิด (`[0,0,0,1]` ให้ `G = 1` ไม่ใช่ 1/2) และใช้ XOR สมดุลทดสอบ prune ซึ่ง root gain เป็น 0 จึงไม่แตกเลย. แก้เป็นชุด 14 แถวที่ root gain 8/21 และลูกได้ 8 กับ 16/3 ตรวจมือได้ทั้งหมด.

## ค้างอยู่

- leaderboard: มีแล้ว (อัปเดต 2026-10-11) — `python -m src.ml.run report --variant <variant>` เขียน `outputs/modeling/report/<variant>/leaderboard_classification.{csv,md}` ซึ่งมีแถวของโมเดลนี้ทั้งสอง variant
- ค่าใบต่าง 1 ulp ใน ~1.4% ของต้นไม้จาก `expf` ของระบบ — อธิบายได้และอยู่ในเกณฑ์ strict แต่ไม่ใช่ bit-exact; บนเครื่อง/OS อื่นสัดส่วนอาจต่าง
- ยังไม่แตะ Test set

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/xgboost.md -->
### ผลจริงของ `xgboost`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"gamma": 0, "learning_rate": 0.03, "max_depth": 1, "n_estimators": 100, "random_state": 42, "reg_lambda": 1} | 0.2508 | 0.6693 | 0.2449 | 0.1765 | 0.4000 | 0.9733 | 0.9003 |
| with_spike | test (one-shot) | {"gamma": 0, "learning_rate": 0.03, "max_depth": 1, "n_estimators": 100, "random_state": 42, "reg_lambda": 1} | 0.5305 | 0.7716 | 0.4553 | 0.3836 | 0.5600 | 0.9267 | 0.8204 |
| non_spike | validation (selection score) | {"gamma": 0.1, "learning_rate": 0.03, "max_depth": 2, "n_estimators": 100, "random_state": 42, "reg_lambda": 10} | 0.2652 | 0.6853 | 0.2800 | 0.2059 | 0.4375 | 0.9733 | 0.9030 |
| non_spike | test (one-shot) | {"gamma": 0.1, "learning_rate": 0.03, "max_depth": 2, "n_estimators": 100, "random_state": 42, "reg_lambda": 10} | 0.5823 | 0.7887 | 0.5469 | 0.4795 | 0.6364 | 0.9333 | 0.8445 |
<!-- END GENERATED -->
