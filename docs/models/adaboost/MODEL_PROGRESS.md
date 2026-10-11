# adaboost — classification · 🟡 รอ leaderboard

**เจ้าของ:** Codex (T3 ทำต่อด้วย Claude Code ตามคำขอเจ้าของโปรเจกต์) · **สเปก:** [card #15](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

**Dependency:** weighted stump จาก `decision_tree` (T1) — ใช้ `DecisionTree(max_depth=1, min_samples_leaf=1)` ตรง ๆ ไม่ได้แก้ไฟล์นั้น

## ผล (validation)

| variant | lr | stumps | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 0.5 | 100 | 0.236309 | 0.676776 | 0.222222 | 0.886792 | 0.176471 | 0.958457 | `[[323,14],[28,6]]` |
| `non_spike` | 0.5 | 100 | 0.240748 | 0.640862 | 0.235294 | 0.894879 | 0.176471 | 0.967359 | `[[326,11],[28,6]]` |

เลือกจาก 6 candidates ด้วย AP ก่อน AUC; ผู้ชนะคือ `adaboost-02` ทั้งสอง variant. accuracy ต่ำกว่าการทาย negative ทุกแถว (0.9084) — ใช้ AP/AUC/F1 ตัดสิน. ไฟล์อยู่ใน `outputs/modeling/<variant>/classification/adaboost/`; `load_verification.json` เป็น `"passed": true` ทั้งคู่.

ทั้ง grid (AP / AUC):

| lr | n | `with_spike` | `non_spike` |
| ---: | ---: | --- | --- |
| 0.5 | 25 | 0.2068 / 0.6555 | 0.1846 / 0.6442 |
| 0.5 | 50 | 0.2149 / 0.6718 | 0.1968 / 0.6438 |
| 0.5 | 100 | **0.2363** / 0.6768 | **0.2407** / 0.6409 |
| 1 | 25 | 0.2113 / 0.6997 | 0.1868 / 0.6750 |
| 1 | 50 | 0.2311 / 0.6872 | 0.1910 / 0.6772 |
| 1 | 100 | 0.2169 / 0.6894 | 0.1946 / 0.6563 |

### สมมติฐาน outlier — ผลจริง: **ไม่สนับสนุน**

card ทำนายว่า AdaBoost จะฟื้นตัวชัดที่สุดเมื่อตัด spike. ที่วัดได้:

- ผู้ชนะ: AP เพิ่มเพียง +0.0044 (0.2363 → 0.2407) ขณะที่ AUC **ลด** 0.0359 (0.6768 → 0.6409)
- ทั้ง grid: AUC ของ `non_spike` ต่ำกว่าทั้ง 6 candidates และ AP ต่ำกว่า 5 จาก 6
- เทียบ `random_forest` ที่ AP เพิ่ม +0.0462 (0.2548 → 0.3010) — AdaBoost ไม่ได้ฟื้นตัวมากที่สุด
- exponential loss บน Train จบที่ 0.5791 (`with_spike`) และ 0.5526 (`non_spike`) หลัง 100 รอบ — train fit ดีขึ้นเล็กน้อย แต่ไม่ส่งผลถึง Validation

ข้อควรระวังในการตีความ: Validation มี positive เพียง 34 แถว ความต่างระดับ 0.004 ของ AP อยู่ในระดับ noise และยังไม่ได้ทำ paired test. kurtosis 16.26 เป็นของ target ต่อเนื่อง แต่ AdaBoost ตัวนี้เห็นแค่ label 0/1 จึงไม่ได้รับ fat tail โดยตรง — นี่เป็นคำอธิบายที่เป็นไปได้ ยังไม่ได้พิสูจน์.

## Parity

เกณฑ์ strict ปกติ (label 100% + score `allclose(rtol=1e-5)`) และเทียบทีละรอบ. วัดครบ 6 candidates × 2 variants บน Train แล้วเทียบ score บน Validation:

- feature และ threshold ของ stump ตรง sklearn ทุกตัว (สูงสุด 100 ตัวต่อ candidate)
- `estimator_weights_`, `estimator_errors_`, `decision_function`: max abs diff = **0** พอดี
- `predict_proba`: max abs diff ≤ 1.1e-16 · label ต่าง 0 แถว
- ไม่มี candidate ใดหยุดก่อนกำหนด (`status_ = "completed"` ครบทุกรอบ)

synthetic 3 seeds × 2 learning rates (40 รอบ, เทียบทั้ง train และ held-out) ผ่านเช่นกัน. `tests/ml/classification/test_adaboost.py` 28 tests ผ่าน; ทั้งชุด 761 tests ผ่าน.

## การตัดสินใจ

- **ตาม sklearn เมื่อ config section `adaboost` ขัดกับ strict parity** — section นั้นมาจากดีไซน์เดิม: `alpha_factor: 0.5` และ `zero_vote_label: 1`. SAMME ของ sklearn ใช้ α เต็ม (`lr · ln((1-err)/err)` ตามไฟล์ task) และ `predict = decision > 0` ซึ่งให้คลาส 0 เมื่อคะแนนเป็นศูนย์. ค่า α ครึ่งเดียวให้ stump และ label ชุดเดียวกัน ต่างแค่ scale. ส่วนที่ใช้ตาม config: `epsilon = 1e-12`, perfect → หยุด, first `err ≥ 0.5` → `ValueError`, later → `completed_early`.
- **`decision_function` คืน `2·Σαh / Σα`** (ช่วง [-2, 2]) เหมือน sklearn และ `predict_proba = sigmoid(decision)` — ไม่ใช่ความน่าจะเป็นที่ calibrate แล้ว แต่ต้องมีเพื่อเทียบ score กับ reference.
- **`history_["loss"]` = mean `exp(-yF/2)`** บน Train. α ของ SAMME เป็นสองเท่าของตำรา จึงเป็น loss ที่ลดจริง; ที่ `lr=1` เท่ากับผลคูณของ `Z_m = 2√(err(1-err))` (มี test ยืนยัน).
- เก็บ `sample_weight_` ที่ normalize แล้วหลังรอบสุดท้ายไว้ให้รายงานใช้ (sklearn ข้ามการอัปเดตรอบสุดท้าย ซึ่งไม่กระทบโมเดล).

## ไฟล์

- `src/ml/classification/adaboost.py`, `src/ml/registry.py`
- `tests/ml/classification/test_adaboost.py`

## ปัญหาที่เจอและแก้แล้ว

ไม่มีบั๊ก parity — ผ่านตั้งแต่รันแรก เพราะทำตามลำดับการคำนวณของ sklearn สามจุดที่ถ้าต่างจะเพี้ยนสะสม:

1. clip น้ำหนักที่ `float64 eps` ต้นทุกรอบ ก่อน fit stump
2. อัปเดตด้วย `exp(log(w) + α·incorrect)` แล้วหารผลรวม (ไม่ใช่คูณ `exp(α)` ตรง ๆ)
3. สุ่ม seed ของ stump ทีละรอบจาก `RandomState(random_state).randint(int32 max)` — `DecisionTree` ใช้ seed นี้เรียงฟีเจอร์เพื่อ tie-break

sklearn 1.9.1: `AdaBoostClassifier` ไม่มีพารามิเตอร์ `algorithm` แล้ว จึงตัดออกจาก `reference()` (มี test ยืนยัน).

## ค้างอยู่

- leaderboard: มีแล้ว (อัปเดต 2026-10-11) — `python -m src.ml.run report --variant <variant>` เขียน `outputs/modeling/report/<variant>/leaderboard_classification.{csv,md}` ซึ่งมีแถวของโมเดลนี้ทั้งสอง variant
- เคส `completed_early` (`err ≥ 0.5` หลังรอบแรก) ยังไม่มี test เพราะสร้างข้อมูลที่ทำให้เกิดไม่ได้ง่าย ๆ ด้วย Gini stump; มีแต่ test ของ perfect และ first-failed
- ยังไม่ได้ทำ paired test ของสมมติฐาน outlier · ยังไม่แตะ Test set

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/adaboost.md -->
### ผลจริงของ `adaboost`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"learning_rate": 0.5, "n_estimators": 100, "random_state": 42} | 0.2363 | 0.6768 | 0.2222 | 0.1765 | 0.3000 | 0.9585 | 0.8868 |
| with_spike | test (one-shot) | {"learning_rate": 0.5, "n_estimators": 100, "random_state": 42} | 0.4858 | 0.7614 | 0.4853 | 0.4521 | 0.5238 | 0.9000 | 0.8123 |
| non_spike | validation (selection score) | {"learning_rate": 0.5, "n_estimators": 100, "random_state": 42} | 0.2407 | 0.6409 | 0.2353 | 0.1765 | 0.3529 | 0.9674 | 0.8949 |
| non_spike | test (one-shot) | {"learning_rate": 0.5, "n_estimators": 100, "random_state": 42} | 0.5443 | 0.7735 | 0.5000 | 0.4110 | 0.6383 | 0.9433 | 0.8391 |
<!-- END GENERATED -->
