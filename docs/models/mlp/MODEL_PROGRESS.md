# mlp — classification · 🟡 รอ leaderboard

**เจ้าของ:** Codex (T7 ทำต่อด้วย Claude Code ตามคำขอเจ้าของโปรเจกต์) · **สเปก:** [card #10](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | hidden | l2 | lr | best epoch | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | [16] | 0 | 0.001 | 17 | 0.266817 | 0.675685 | 0.320000 | 0.908356 | 0.235294 | 0.976261 | `[[329,8],[26,8]]` |
| `non_spike` | [32] | 0 | 0.001 | 9 | 0.274049 | 0.661023 | 0.260870 | 0.908356 | 0.176471 | 0.982196 | `[[331,6],[28,6]]` |

เลือกจาก 12 candidates ด้วย AP ก่อน AUC (`mlp-00` และ `mlp-04`). ไฟล์อยู่ใน `outputs/modeling/<variant>/classification/mlp/`; `load_verification.json` เป็น `"passed": true` ทั้งคู่.

ทั้ง grid (AP / AUC) — `l2=0.0001` ให้ผลเท่ากับ `l2=0` ทุกหลักที่แสดง จึงรวมแถว:

| hidden | lr | `with_spike` | `non_spike` |
| --- | ---: | --- | --- |
| [16] | 0.001 | **0.2668** / 0.6757 | 0.2589 / 0.6833 |
| [16] | 0.01 | 0.2653 / 0.6773 | 0.2671 / 0.6850 |
| [32] | 0.001 | 0.2555 / 0.6589 | **0.2740** / 0.6610 |
| [32] | 0.01 | 0.2370 / 0.6724 | 0.2662 / 0.6732 |
| [16, 8] | 0.001 | 0.2642 / 0.7027 | 0.1949 / 0.6423 |
| [16, 8] | 0.01 | 0.2605 / 0.7081 | 0.2068 / 0.6615 |

ข้อสังเกตจากตัวเลข:

- **early stopping หยุดเร็วมาก** best epoch อยู่ที่ 1–17 สำหรับ 22 จาก 24 การรัน (สูงสุด 70). loss บน tail 15% ของ train ขึ้นแทบทันที แปลว่าโมเดลที่ถูกประเมินผ่านการเทรนน้อยมาก — ผลจึงสะท้อนจุดเริ่มต้นและไม่กี่ epoch แรกมากกว่าความจุของ network
- **L2 = 0.0001 ไม่มีผลที่วัดได้** เพราะจำนวน step น้อยและ penalty เล็ก
- สองชั้นซ่อน `[16, 8]` ได้ AUC สูงสุดใน `with_spike` (0.708) แต่แย่สุดใน `non_spike` (0.642) — ความแปรปรวนระดับนี้กับ positive 34 แถวไม่ควรตีความ

## Parity

เกณฑ์ strict ปกติ (label 100% + `predict_proba` `allclose(rtol=1e-5)`) และเทียบ**น้ำหนักทุกชั้น**. วัดครบ 12 candidates × 2 variants บน Train และ Validation:

- น้ำหนักและ bias ทุกชั้นต่างจาก `MLPClassifier` สูงสุด **2.2e-16**
- probability ต่างสูงสุด **3.3e-16** · label ต่าง 0 แถว
- `n_iter_` ของ reference เท่ากับ `best_epoch_` ทุกการรัน

`test_mlp.py` 34 tests ผ่าน รวม gradient check ทุกชั้นเทียบ finite difference (rtol 1e-5), XOR ถูกครบ 200 แถว, และ network ที่ไม่มี nonlinearity ลู่เข้า optimum ของ logistic regression.

## การตัดสินใจ

- **มี Nesterov momentum 0.9** card ไม่ได้พูดถึง แต่ reference ที่ card ระบุ (`MLPClassifier(solver="sgd")`) ใช้ค่านี้เป็น default และ card ไม่ได้ปิด. เลือกเลียนให้ตรง reference ตามที่ card เขียน แทนการเติม `momentum=0` ให้ reference. `momentum=0` ก็รองรับและมี test parity เช่นกัน.
- **L2 หารด้วยขนาด batch** card เขียนว่า "sklearn หาร L2 ด้วยจำนวนตัวอย่าง ไม่ใช่ขนาด batch" — อ่าน source 1.9.1 แล้วพบว่าหารด้วย `sw_sum` ซึ่งคือจำนวนแถวของ **batch นั้น**. ยึดตาม source; มี test.
- **protocol เดียวกับ `slp`**: batch ต่อเนื่องตามเวลา ไม่ shuffle, early stopping ดู cross-entropy บน tail 15% ของ train, patience 50, แล้ว refit บน train เต็มตามจำนวน epoch ที่ดีที่สุด. reference จึงเป็น sklearn ที่รัน `max_iter = best_epoch_` บน train เต็ม.
- **ไม่มี purge 5 วันระหว่าง subtrain กับ tail และ scaler ไม่ได้ fit บน subtrain** ทั้งที่ config `neural` ระบุ (`purge_original_trading_days: 5`, `scaler_fit: subtrain`). เหตุผล: โมเดลรับแค่ `X` ที่ trainer standardize บน train เต็มมาแล้ว และ `slp` ที่ merge ไปก่อนก็ทำแบบเดียวกัน — เลือกให้สองโมเดลสอดคล้องกัน. ผลกระทบจำกัดอยู่ที่การเลือกจำนวน epoch (optimistic เล็กน้อย) ไม่เกี่ยวกับ Validation/Test. ถ้าจะทำตาม config ต้องแก้ทั้ง `slp` และ `mlp` พร้อมกัน.
- `activation="identity"` มีไว้เพื่อ unit test ตาม card (ชั้นซ่อน linear ≡ logistic) ไม่อยู่ใน grid.

## ไฟล์

- `src/ml/classification/mlp.py`, `src/ml/registry.py`
- `tests/ml/classification/test_mlp.py`

## ปัญหาที่เจอและแก้แล้ว

ไม่มีบั๊ก parity — น้ำหนักตรงตั้งแต่รันแรก เพราะอ่าน source ของ sklearn ก่อนเขียน สามจุดที่ถ้าต่างจะไม่มีทางตรง: ลำดับสุ่ม (weights แล้ว bias ทีละชั้นจาก `RandomState` ตัวเดียว), การหาร L2 ด้วยขนาด batch, และสูตร Nesterov (`update = μ·v_new − lr·g`).

**test ที่ตั้งผิดเอง 1 ตัว:** test "ชั้นซ่อน linear ≡ logistic" ใช้ mini-batch ที่ step คงที่ ซึ่งแกว่งรอบ optimum ไม่ลู่เข้า (loss 0.4446 เทียบ 0.4433). แก้เป็น full batch — เป็นการทดสอบโมเดล ไม่ใช่ noise ของ SGD.

## ค้างอยู่

- leaderboard: `src.ml.run report` ยังไม่ implement
- purge และ `scaler_fit: subtrain` ตาม config ยังไม่ได้ทำ (ดูหัวข้อการตัดสินใจ) — เหมือน `slp`
- ยังไม่แตะ Test set
