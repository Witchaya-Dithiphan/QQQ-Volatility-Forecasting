# mlp — classification · 🟡 รอ leaderboard

**เจ้าของ:** Codex (T7 ทำต่อด้วย Claude Code ตามคำขอเจ้าของโปรเจกต์) · **สเปก:** [card #10](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | hidden | l2 | lr | best epoch | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | [16] | 0 | 0.001 | 17 | 0.266817 | 0.675685 | 0.320000 | 0.908356 | 0.235294 | 0.976261 | `[[329,8],[26,8]]` |
| `non_spike` | [16, 8] | 0 | 0.01 | 5 | 0.280762 | 0.699249 | 0.307692 | 0.902965 | 0.235294 | 0.970326 | `[[327,10],[26,8]]` |

เลือกจาก 12 candidates ด้วย AP ก่อน AUC (`mlp-00` และ `mlp-09`). ไฟล์อยู่ใน `outputs/modeling/<variant>/classification/mlp/`; `load_verification.json` เป็น `"passed": true` ทั้งคู่. ตัวเลขนี้เป็นของรุ่นที่มี purge (ดูหัวข้อ "ผลของ purge").

ทั้ง grid (AP / AUC) — `l2=0.0001` ต่างจาก `l2=0` ไม่เกิน 0.0001 จึงรวมแถว:

| hidden | lr | `with_spike` | `non_spike` |
| --- | ---: | --- | --- |
| [16] | 0.001 | **0.2668** / 0.6757 | 0.2582 / 0.6832 |
| [16] | 0.01 | 0.2653 / 0.6773 | 0.2671 / 0.6850 |
| [32] | 0.001 | 0.2555 / 0.6589 | 0.2734 / 0.6621 |
| [32] | 0.01 | 0.2370 / 0.6724 | 0.2662 / 0.6732 |
| [16, 8] | 0.001 | 0.2649 / 0.7061 | 0.1949 / 0.6423 |
| [16, 8] | 0.01 | 0.2605 / 0.7081 | **0.2808** / 0.6992 |

ข้อสังเกตจากตัวเลข:

- **early stopping หยุดเร็วมาก** best epoch ไม่เกิน 17 ใน 20 จาก 24 การรัน (ที่เหลือ 23 และ 76). loss บน tail 15% ของ train ขึ้นแทบทันที แปลว่าโมเดลที่ถูกประเมินผ่านการเทรนน้อยมาก — ผลจึงสะท้อนจุดเริ่มต้นและไม่กี่ epoch แรกมากกว่าความจุของ network
- **การเลือกผู้ชนะเปราะ** ดูหัวข้อ "ผลของ purge": ทิ้งแถวเทรน 5 จาก 1,292 แถวแล้วผู้ชนะของ `non_spike` เปลี่ยน
- **L2 = 0.0001 ไม่มีผลที่วัดได้** เพราะจำนวน step น้อยและ penalty เล็ก
- Validation มี positive 34 แถว ความต่างระหว่าง candidates ยังไม่ได้ทดสอบนัยสำคัญ

## ผลของ purge (2026-10-11)

เพิ่ม purge 5 วันทำการก่อน tail ตาม `neural.purge_original_trading_days: 5` หลัง merge รุ่นแรก. เทียบก่อน/หลังบน artifact จริง:

| variant | ผู้ชนะก่อน | ผู้ชนะหลัง | candidates ที่ AP เปลี่ยน |
| --- | --- | --- | ---: |
| `with_spike` | `[16]`, lr 0.001 — AP 0.2668 | เดิม ตัวเลขเท่าเดิมทุกหลัก | 2 จาก 12 (≤ 0.0007) |
| `non_spike` | `[32]`, lr 0.001 — AP 0.2740 | `[16, 8]`, lr 0.01 — AP 0.2808 | 6 จาก 12 |

ใน `non_spike` candidate `[16, 8], lr 0.01` เลื่อน best epoch จาก 1 เป็น 5 และ AP จาก 0.2068 เป็น 0.2808 จึงขึ้นมาชนะ; อีก 4 ตัวขยับไม่เกิน 0.0007.

**ไม่ควรอ่านว่า purge ทำให้โมเดลดีขึ้น.** สิ่งที่ตัวเลขนี้บอกคือการเลือกจำนวน epoch ของ network ที่เทรนแค่ 1–5 epochs ไวต่อข้อมูลมาก: เอาแถวเทรนออก 5 แถวจาก 1,292 ผลของ candidate หนึ่งเปลี่ยน 0.07 AP. เหตุผลที่ทำ purge คือความถูกต้องของ protocol (label ของ 5 วันสุดท้ายซ้อนช่วงเวลากับ tail) ไม่ใช่ตัวเลข.

## Parity

เกณฑ์ strict ปกติ (label 100% + `predict_proba` `allclose(rtol=1e-5)`) และเทียบ**น้ำหนักทุกชั้น**. วัดครบ 12 candidates × 2 variants บน Train และ Validation:

- น้ำหนักและ bias ทุกชั้นต่างจาก `MLPClassifier` สูงสุด **2.2e-16** (วัดซ้ำหลังเพิ่ม purge แล้วได้ค่าเดิม)
- probability ต่างสูงสุด **3.3e-16** · label ต่าง 0 แถว
- `n_iter_` ของ reference เท่ากับ `best_epoch_` ทุกการรัน

`test_mlp.py` 39 tests ผ่าน รวม gradient check ทุกชั้นเทียบ finite difference (rtol 1e-5), XOR ถูกครบ 200 แถว, และ network ที่ไม่มี nonlinearity ลู่เข้า optimum ของ logistic regression.

## การตัดสินใจ

- **มี Nesterov momentum 0.9** card ไม่ได้พูดถึง แต่ reference ที่ card ระบุ (`MLPClassifier(solver="sgd")`) ใช้ค่านี้เป็น default และ card ไม่ได้ปิด. เลือกเลียนให้ตรง reference ตามที่ card เขียน แทนการเติม `momentum=0` ให้ reference. `momentum=0` ก็รองรับและมี test parity เช่นกัน.
- **L2 หารด้วยขนาด batch** card เขียนว่า "sklearn หาร L2 ด้วยจำนวนตัวอย่าง ไม่ใช่ขนาด batch" — อ่าน source 1.9.1 แล้วพบว่าหารด้วย `sw_sum` ซึ่งคือจำนวนแถวของ **batch นั้น**. ยึดตาม source; มี test.
- **protocol เดียวกับ `slp`**: batch ต่อเนื่องตามเวลา ไม่ shuffle, early stopping ดู cross-entropy บน tail 15% ของ train, patience 50, แล้ว refit บน train เต็มตามจำนวน epoch ที่ดีที่สุด. reference จึงเป็น sklearn ที่รัน `max_iter = best_epoch_` บน train เต็ม.
- **purge 5 วันทำการก่อน tail** ตาม `neural.purge_original_trading_days: 5`: target มองไปข้างหน้า 5 วัน แถวของชุดเทรนย่อยที่อยู่ภายใน 5 วันก่อน tail จึงมี label ที่คำนวณจากช่วงเวลาของ tail เอง — ทิ้งแถวเหล่านั้นออกจากรอบ early stopping (refit ยังใช้ train เต็ม จึงไม่กระทบ parity). นับวันบน timeline เดิมผ่าน `positions` ที่ trainer ส่งให้. ทั้งสอง variant ถูกทิ้ง 5 แถว (1,473 → 1,468 และ 1,292 → 1,287). ทำพร้อมกับ `slp` เพื่อให้ protocol ตรงกัน. มี probe: กลับ label ของแถวที่ถูก purge แล้ว history และ best epoch ต้องเท่าเดิม.
- **`scaler_fit: subtrain` ยังไม่ได้ทำ** trainer standardize บน train เต็มก่อนส่งให้โมเดล ค่าเฉลี่ย/ส่วนเบี่ยงเบนของ tail จึงยังปนอยู่ในการเลือกจำนวน epoch. แก้ต้องแตะ trainer; ยังไม่ได้วัดผลกระทบ.
- `activation="identity"` มีไว้เพื่อ unit test ตาม card (ชั้นซ่อน linear ≡ logistic) ไม่อยู่ใน grid.

## ไฟล์

- `src/ml/classification/mlp.py`, `src/ml/registry.py`
- `tests/ml/classification/test_mlp.py`

## ปัญหาที่เจอและแก้แล้ว

ไม่มีบั๊ก parity — น้ำหนักตรงตั้งแต่รันแรก เพราะอ่าน source ของ sklearn ก่อนเขียน สามจุดที่ถ้าต่างจะไม่มีทางตรง: ลำดับสุ่ม (weights แล้ว bias ทีละชั้นจาก `RandomState` ตัวเดียว), การหาร L2 ด้วยขนาด batch, และสูตร Nesterov (`update = μ·v_new − lr·g`).

**test ที่ตั้งผิดเอง 1 ตัว:** test "ชั้นซ่อน linear ≡ logistic" ใช้ mini-batch ที่ step คงที่ ซึ่งแกว่งรอบ optimum ไม่ลู่เข้า (loss 0.4446 เทียบ 0.4433). แก้เป็น full batch — เป็นการทดสอบโมเดล ไม่ใช่ noise ของ SGD.

## ค้างอยู่

- leaderboard: `src.ml.run report` ยังไม่ implement
- `scaler_fit: subtrain` ตาม config ยังไม่ได้ทำ (ดูหัวข้อการตัดสินใจ) — เหมือน `slp`
- ยังไม่แตะ Test set
