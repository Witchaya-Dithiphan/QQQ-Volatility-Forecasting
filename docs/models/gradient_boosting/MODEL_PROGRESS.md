# gradient_boosting — classification · 🟡 รอ leaderboard

**เจ้าของ:** Codex (T4 ทำต่อด้วย Claude Code ตามคำขอเจ้าของโปรเจกต์) · **สเปก:** [card #13](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | lr | depth | trees | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 0.1 | 1 | 50 | 0.253083 | 0.675380 | 0.244898 | 0.900270 | 0.176471 | 0.973294 | `[[328,9],[28,6]]` |
| `non_spike` | 0.03 | 2 | 50 | 0.252529 | 0.673154 | 0.266667 | 0.911051 | 0.176471 | 0.985163 | `[[332,5],[28,6]]` |

เลือกจาก 8 candidates ด้วย AP ก่อน AUC (`gradient_boosting-04` และ `-02`). ไฟล์อยู่ใน `outputs/modeling/<variant>/classification/gradient_boosting/`; `load_verification.json` เป็น `"passed": true` ทั้งคู่. log loss บน Train: 0.5399 → 0.3613 (`with_spike`), 0.4697 → 0.3407 (`non_spike`).

ทั้ง grid (AP / AUC):

| lr | depth | n | `with_spike` | `non_spike` |
| ---: | ---: | ---: | --- | --- |
| 0.03 | 1 | 50 | 0.2425 / 0.6613 | 0.2461 / 0.6576 |
| 0.03 | 1 | 100 | 0.2462 / 0.6714 | 0.2484 / 0.6713 |
| 0.03 | 2 | 50 | 0.2407 / 0.6680 | **0.2525** / 0.6732 |
| 0.03 | 2 | 100 | 0.2225 / 0.6687 | 0.2379 / 0.6766 |
| 0.1 | 1 | 50 | **0.2531** / 0.6754 | 0.2385 / 0.6721 |
| 0.1 | 1 | 100 | 0.2388 / 0.6779 | 0.2467 / 0.6731 |
| 0.1 | 2 | 50 | 0.2141 / 0.6682 | 0.2353 / 0.6739 |
| 0.1 | 2 | 100 | 0.2125 / 0.6665 | 0.2200 / 0.6575 |

ข้อสังเกตจากตัวเลข: ผู้ชนะสอง variant แทบไม่ต่างกัน (AP ต่าง 0.0006) · `depth=2, n=100` แย่ที่สุดในทั้งคู่ — โมเดลที่ซับซ้อนกว่าไม่ได้ช่วย · candidate `0.03/1/50` ของ `non_spike` ได้ F1 = 0 เพราะไม่มีแถวใดคะแนนถึง 0 (ยังไม่ขยับจาก prior พอ) ทั้งที่ AP ใกล้ตัวอื่น — ตัวอย่างว่าทำไมไม่ใช้ F1 ที่ threshold เดียวเป็นตัวเลือกโมเดล. Validation มี positive 34 แถว ความต่างระดับนี้ยังไม่ได้ทดสอบนัยสำคัญ.

## Parity

เกณฑ์ strict ปกติ และเทียบ**ทีละ stage**. วัดครบ 8 candidates × 2 variants (16 ชุด) บน Train และ Validation:

- ทุกต้นไม้ตรง sklearn 1.9.1: จำนวน node, feature, threshold
- `staged_decision_function` ทุก stage และ `predict_proba`: max abs diff = **0** พอดี · label ต่าง 0 แถว
- `history_["loss"] × 2` ตรง `train_score_` (sklearn รายงานเป็น deviance = 2 × log loss)

stress test นอก grid (240 รัน: 40 seeds × {ข้อมูลต่อเนื่อง depth 2/3/4, ข้อมูลปัดทศนิยมให้มีค่าซ้ำ depth 1/2/3}): โครงสร้างตรงทั้งหมดและ score ตรงระดับ bit ทั้งหมด. `test_gradient_boosting.py` 31 tests ผ่าน; ทั้งชุด 792 tests ผ่าน.

## การตัดสินใจ

- **regression tree อยู่ในไฟล์นี้เอง** ไม่ได้แก้ `decision_tree.py`. ตรรกะลำดับเยี่ยมฟีเจอร์ (xorshift) จึงซ้ำกับใน `DecisionTree.fit` ประมาณ 35 บรรทัด. เคยลองแยกเป็น class ร่วมใน `decision_tree.py` แต่การแก้ไฟล์ร่วมถูกปฏิเสธ จึงคงไว้แบบนี้ — ถ้าเจ้าของ T1 อยากรวม ให้ย้าย `_FeatureOrder` ไปที่นั่น.
- **`predict` ใช้ `>= 0`** ตาม sklearn ของ gradient boosting (คะแนนศูนย์ → คลาส 1) ซึ่ง**ตรงข้าม**กับโมเดลที่ใช้ argmax (`> 0.5`).
- **`min_samples_leaf=5`** เป็น default ตาม reference ใน card; ไม่อยู่ใน grid.
- **`history_["loss"]` เป็น log loss** ตาม config (`binary_logistic_loss`) ไม่ใช่ deviance.
- เก็บต้นไม้ทั้งหมดเป็น array ต่อกัน + `tree_sizes` แทน key แยกต่อต้น เพื่อให้ `model.npz` มี 8 key ไม่ว่ากี่ต้น.

## ไฟล์

- `src/ml/classification/gradient_boosting.py`, `src/ml/registry.py`
- `tests/ml/classification/test_gradient_boosting.py`

## ปัญหาที่เจอและแก้แล้ว

**1. `F0` ต่าง 1 ulp แล้วไปพลิกฟีเจอร์ในต้นที่ 37.** อาการ: synthetic depth 3 ต้นที่ 37 เลือกฟีเจอร์ 4 ขณะ sklearn เลือก 0 ที่ node 10 แถว; ต้นก่อนหน้าตรงหมด. สองฟีเจอร์นั้นแบ่งแถวชุดเดียวกันเป๊ะ จึงเสมอกันจริง. สาเหตุจริง**ไม่ใช่ต้นไม้**: raw score ต่างจาก sklearn `5.5e-17` ตั้งแต่ stage 0 ทุกแถว เพราะ sklearn ใช้ `scipy.special.logit` ซึ่งในช่วง `0.3 ≤ p ≤ 0.65` คำนวณเป็น `log1p(s) − log1p(−s)` โดย `s = 2(p − 0.5)` ไม่ใช่ `log(p/(1−p))`. residual จึงต่าง 1 ulp และ tie ถูกตัดสินต่างกัน. พิสูจน์โดยให้ `DecisionTreeRegressor` ของ sklearn สร้างต้นไม้บน residual ของเรา — ได้ฟีเจอร์ 4 เหมือนเรา. แก้ด้วย `_log_odds` ที่สลับสูตรที่จุดเดียวกัน; test เทียบกับ scipy ทุกเศษส่วน k/n ถึง n = 199 ตรงทุก bit. ข้อมูลจริงไม่เจอเพราะ prior = 0.25 อยู่นอกช่วงนั้น.

**2. node ที่ residual เท่ากันทุกแถวถูกตัดสินต่างกัน.** อาการ: synthetic depth 3 ต้นที่ 2 — sklearn แตก node ต่อ (impurity `2.5e-16`, เกิน eps นิดเดียว) แต่เราหยุด. สาเหตุ: impurity ของ node ลูกเป็นผลลบของเลขเกือบเท่ากัน ค่าที่ได้ขึ้นกับ**ลำดับการบวก**. sklearn บวกแบบ sequential ตามลำดับแถวภายใน ซึ่งถูกสลับด้วย sort ของฟีเจอร์ที่เยี่ยมล่าสุดและ two-pointer partition. แก้โดยเดินตามลำดับเดียวกัน: `np.cumsum` แทน `np.sum` (ซึ่งเป็น pairwise), เก็บลำดับแถวหลัง sort/partition ส่งต่อให้ลูก, คำนวณ impurity ของลูกจากฝั่งพ่อด้วยสูตรเดียวกับ `MSE.children_impurity`, และตรวจ `improvement + eps < 0`.

**3. `train_score_` เป็นสองเท่าของ loss เรา.** ไม่ใช่บั๊ก — sklearn รายงาน binomial deviance. แก้ที่ test ให้เทียบ `2 × loss` อย่างชัดเจน.

## ค้างอยู่

- leaderboard: มีแล้ว (อัปเดต 2026-10-11) — `python -m src.ml.run report --variant <variant>` เขียน `outputs/modeling/report/<variant>/leaderboard_classification.{csv,md}` ซึ่งมีแถวของโมเดลนี้ทั้งสอง variant
- การ sort ของ sklearn เป็น introsort (ไม่ stable) ส่วนเราใช้ stable sort; ลำดับของค่าที่ซ้ำกันจึงอาจต่าง และกระทบ roundoff ได้ในทางทฤษฎี. stress test 120 รันบนข้อมูลที่มีค่าซ้ำยังไม่เจอ แต่ไม่ได้พิสูจน์ว่าเป็นไปไม่ได้
- ยังไม่แตะ Test set
