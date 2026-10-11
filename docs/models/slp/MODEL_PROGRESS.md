# slp — classification · ✅ เสร็จ

**เจ้าของ:** Claude · **สเปก:** [card #9](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | batch | l2 | lr | Acc | Prec | Sens | Spec | F1 | **AUC** | confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_spike` | 32 | 0 | 0.001 | 0.9084 | 0.5000 | 0.1765 | 0.9822 | 0.2609 | **0.6803** | `[[331,6],[28,6]]` |
| `non_spike` | full | 0 | 0.01 | 0.8706 | 0.2500 | 0.2059 | 0.9377 | 0.2258 | 0.6778 | `[[316,21],[27,7]]` |

reload ✅ · preprocess `standardize`

## ข้อสังเกต

- **`with_spike` ให้ตัวเลขเหมือน `logistic` เป๊ะทุกช่อง** (Acc/Prec/Sens/Spec/F1 และ confusion
  `[[331,6],[28,6]]` ตรงกัน) ต่างกันแค่ AUC (0.6803 vs 0.6732) ซึ่งสมเหตุสมผล —
  **SLP ที่ไม่มีชั้นซ่อนคือ logistic regression ตัวเดียวกัน** ต่างแค่วิธี optimise
  (mini-batch + early stopping vs LBFGS) เป็นจุดที่ควรชี้ในรายงานว่าสองโมเดลนี้
  เป็นโมเดลเดียวกันในเชิงคณิตศาสตร์
- AUC ดีกว่า logistic เล็กน้อย — early stopping ทำหน้าที่เป็น regularizer โดยหยุดก่อน overfit
- `l2=0` ชนะทั้งสอง variant → early stopping คุม overfit ได้อยู่แล้ว ไม่ต้องพึ่ง L2 ซ้ำ

## Parity

**ลู่เข้าจุดเดียวกับ `LogisticRegression` ที่ `max|Δcoef| = 1.3e-08`** (l2 = 0 และ 1e-4)
เมื่อปิด early stopping และปล่อยให้รันจนลู่เข้า · label ตรงกัน 300/300

**ทำไม reference ถึงเป็น logistic ไม่ใช่ MLPClassifier** — sklearn ไม่รับ
`hidden_layer_sizes=()` และ `SGDClassifier` อัปเดตทีละตัวอย่าง (เดินคนละ path)
สิ่งที่เทียบได้จริงคือ **objective เดียวกัน** ซึ่ง LBFGS แก้ได้แม่นที่สุด

นอกจากนี้ `test_full_batch_matches_a_hand_rolled_gradient_descent` ยังพิสูจน์ training loop
แบบ exact โดยไล่ gradient descent ด้วยมือทีละ epoch แล้วเทียบ weight ที่ 1e-10

## การตัดสินใจ

- **weight init ใช้ `np.random.RandomState`** (Glorot uniform, bound = √(6/(fan_in+1)))
  ไม่ใช่ `default_rng` — มี test ยืนยันว่าสอง stream ให้คนละค่า
- **batch สุดท้ายที่ไม่เต็มต้องนับ** ตาม `last_short_batch: true` — 100 แถว batch 32
  ได้ `[32, 32, 32, 4]` มี test ล็อกไว้
- batch เรียงตามเวลาแบบต่อเนื่อง ไม่สลับ (`contiguous_chronological`, `shuffle: false`)
- **patience นับจาก epoch ที่ดีที่สุด** ไม่ใช่นับต่อเนื่องจาก epoch ล่าสุด · tie เลือก epoch แรก
- early stopping ดูจาก **tail 15% ของ train** แล้ว **refit บน train เต็ม** ด้วยจำนวน epoch
  ที่ดีที่สุด ตาม `refit_full_train: true` + `refit_epochs: selected_best_epoch`
- **purge 5 วันทำการก่อน tail** (เพิ่ม 2026-10-11 ตาม `neural.purge_original_trading_days: 5`):
  target มองไปข้างหน้า 5 วัน แถวของชุดเทรนย่อยที่อยู่ภายใน 5 วันก่อน tail จึงมี label ที่คำนวณจาก
  ช่วงเวลาของ tail เอง — ทิ้งแถวเหล่านั้นออกจากการเทรนรอบ early stopping (refit ยังใช้ train เต็ม).
  นับวันบน timeline เดิมผ่าน `positions` ที่ trainer ส่งให้ ไม่ใช่นับแถว. ทั้งสอง variant ถูกทิ้ง 5 แถว
  (1,473 → 1,468 และ 1,292 → 1,287)

## ไฟล์

- `src/ml/classification/slp.py`
- `tests/ml/classification/test_slp.py` (21 tests)

## ปัญหาที่เจอ

ไม่มี parity failure — แต่ต้องคิดเรื่อง reference ก่อนว่าจะเทียบกับอะไร เพราะไม่มี
sklearn estimator ที่ตรงรูปแบบนี้ สรุปว่าเทียบที่ objective ซึ่งเป็นคำถามที่ถูกต้อง

## ผลขึ้นกับโควตา epoch ไม่ใช่ early stopping (ตรวจ 2026-10-11)

ข้อสังเกตข้างบนที่ว่า "early stopping ทำหน้าที่เป็น regularizer โดยหยุดก่อน overfit" **ไม่ตรงกับสิ่งที่เกิดจริง**:
ผู้ชนะทั้งสอง variant บันทึก `n_epochs = best_epoch = 1000` คือวิ่งครบ `max_epochs` โดย loss บน tail
ยังลดอยู่จนรอบสุดท้าย — early stopping ไม่เคยทำงาน. สิ่งที่หยุดการเทรนคือโควตา 1,000 epochs.

ผลคือโมเดลที่บันทึกยังห่างจาก logistic regression ที่ลู่เข้า: probability บน Validation ต่างสูงสุด
0.069 (`with_spike`) และ 0.287 (`non_spike`), label ต่าง 0.3% และ 4.6%. AUC ที่สูงกว่า `logistic`
เล็กน้อยจึงเป็นผลของการหยุดที่ 1,000 epochs ซึ่งเป็นค่าที่ config กำหนด ไม่ใช่จุดที่ข้อมูลเลือก.
ถ้าเพิ่ม `max_epochs` ผลจะเคลื่อนเข้าหา `logistic`. ไม่ใช่บั๊ก แต่รายงานควรเขียนตามนี้.

## ผลของ purge (2026-10-11)

train ใหม่ทั้งสอง variant หลังเพิ่ม purge: **ผู้ชนะและตัวเลขในตารางข้างบนไม่เปลี่ยนเลย**
(`slp-00` และ `slp-05` เดิม, metric เท่าเดิมทุกหลัก — ทั้งคู่วิ่งครบ 1,000 epochs ไม่ว่ามี purge หรือไม่).
candidates อื่นขยับเล็กน้อย: `with_spike` 2 จาก 8 ตัว (AP ต่างไม่เกิน 0.0005),
`non_spike` 4 จาก 8 ตัว (AP ต่างไม่เกิน 0.0016). parity ไม่กระทบเพราะเทียบที่ขั้นลู่เข้าซึ่งไม่ใช้ early stopping.
`test_slp.py` เพิ่มเป็น 21 tests (purge 5 ตัว รวม probe ว่าแถวที่ถูก purge เปลี่ยน label แล้วจำนวน epoch ไม่ขยับ).

## ค้างอยู่

- card #9 ใน `MODEL_CARDS.md` ยังไม่ได้ใส่ผลจริง
- `neural.scaler_fit: subtrain` ยังไม่ได้ทำ: trainer standardize บน train เต็มก่อนส่งให้โมเดล
  จึงยังมีค่าเฉลี่ย/ส่วนเบี่ยงเบนของ tail ปนอยู่ในการเลือกจำนวน epoch (ไม่เกี่ยวกับ Validation/Test)
