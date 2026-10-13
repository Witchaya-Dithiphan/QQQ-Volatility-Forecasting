# svm — classification · 🟡 รอ leaderboard

**เจ้าของ:** Codex (T6 ทำต่อด้วย Claude Code ตามคำขอเจ้าของโปรเจกต์) · **สเปก:** [card #17](../../plan/MODEL_CARDS.md) · **วันที่:** 2026-10-11

## ผล (validation)

| variant | C | PCA | AP | AUC | F1 | accuracy | sensitivity | specificity | confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `with_spike` | 0.1 | 4 | 0.251395 | 0.678565 | 0.255319 | 0.905660 | 0.176471 | 0.979228 | `[[330,7],[28,6]]` |
| `non_spike` | 10 | 4 | 0.250110 | 0.680398 | 0.272727 | 0.913747 | 0.176471 | 0.988131 | `[[333,4],[28,6]]` |

เลือกจาก 12 candidates (C 3 ค่า × ไม่มี PCA / PCA 2, 4, 6) ด้วย AP ก่อน AUC. metric คิดจาก `decision_function` เพราะ SVM ไม่ให้ probability (`score_kind: decision`). ไฟล์อยู่ใน `outputs/modeling/<variant>/classification/svm/`; `load_verification.json` เป็น `"passed": true` ทั้งคู่.

ทั้ง grid (AP / AUC):

| PCA | C | `with_spike` | `non_spike` |
| ---: | ---: | --- | --- |
| ไม่มี | 0.1 | 0.2151 / 0.6478 | 0.2172 / 0.6455 |
| ไม่มี | 1 | 0.2136 / 0.6440 | 0.2118 / 0.6361 |
| ไม่มี | 10 | 0.2137 / 0.6435 | 0.2113 / 0.6346 |
| 2 | 0.1 | 0.2458 / 0.6765 | 0.2483 / 0.6789 |
| 2 | 1 | 0.2456 / 0.6760 | 0.2485 / 0.6792 |
| 2 | 10 | 0.2456 / 0.6760 | 0.2485 / 0.6792 |
| 4 | 0.1 | **0.2514** / 0.6786 | 0.2500 / 0.6803 |
| 4 | 1 | 0.2510 / 0.6782 | 0.2493 / 0.6800 |
| 4 | 10 | 0.2511 / 0.6787 | **0.2501** / 0.6804 |
| 6 | 0.1 | 0.2475 / 0.6776 | 0.2445 / 0.6746 |
| 6 | 1 | 0.2475 / 0.6776 | 0.2452 / 0.6751 |
| 6 | 10 | 0.2473 / 0.6774 | 0.2440 / 0.6741 |

ข้อสังเกตจากตัวเลข: **PCA ช่วยทุกขนาดในทั้งสอง variant** (AP จาก ~0.21 เป็น ~0.25) ขณะที่ **C แทบไม่มีผล** — ภายใน PCA ขนาดเดียวกัน AP ต่างกันไม่เกิน 0.001. "ผู้ชนะ" C=0.1 กับ C=10 จึงเป็นผลของกฎเลือก ไม่ใช่หลักฐานว่า C ต่างกัน. Validation มี positive 34 แถว ยังไม่ได้ทดสอบนัยสำคัญ.

## Parity

เกณฑ์ของ SVM ตาม ADR-002 ใน `docs/plan/ARCHITECTURE.md` (ข้อยกเว้นเดียวของโปรเจกต์): เทียบที่ **objective** ไม่ใช่น้ำหนัก เพราะ hinge ไม่เรียบ.

วัดครบ 24 candidates จริง (3 C × 4 PCA × 2 variants):

- relative gap ของ objective สูงสุด **5.6e-06** (เกณฑ์ < 1e-3)
- scratch ไม่เคยแย่กว่า reference เกิน 5.6e-06 (เกณฑ์ ≤ 1e-3)
- label ตรงกันต่ำสุด **99.94%** บน Train และ Validation (เกณฑ์ ≥ 99%)

synthetic 3 seeds × 3 ค่า C ผ่านเกณฑ์เดียวกัน. กรณีที่ optimum well-conditioned (C=0.1, 200,000 รอบ) น้ำหนักตรง reference ที่ 1e-4. `test_svm.py` 38 tests ผ่าน.

## การตัดสินใจ

- **reference เป็น `SVC(kernel="linear")` ไม่ใช่ `LinearSVC` ตาม card — มีหลักฐาน.** `LinearSVC(intercept_scaling=1000, max_iter=2_000_000)` **ไม่ลู่เข้า** บนข้อมูลจริง: วิ่งชน max_iter และได้ objective แย่กว่าเรา (C=1: 705.65 เทียบ 704.14 · C=10: 17,517 เทียบ 7,036). สาเหตุคือ liblinear penalize intercept และ `intercept_scaling` ที่ใหญ่พอจะกลบผลนั้นทำให้ dual coordinate descent ช้ามาก. `SVC(kernel="linear")` ของ libsvm แก้ objective เดียวกับเราพอดี (intercept ไม่ถูก penalize) และลู่เข้าใน < 1 วินาที. ถ้ายึด card เกณฑ์จะ "ไม่ผ่าน" เพราะ reference ผิด ไม่ใช่เพราะเราผิด.
- **คืนจุดที่ objective ต่ำสุดที่เคยผ่าน** ไม่ใช่จุดสุดท้าย. subgradient step ไม่ใช่ descent step — objective ขึ้นได้. card เขียนว่า "objective ลดลง monotone" ซึ่งไม่จริงสำหรับวิธีนี้; ที่ monotone คือ best-so-far. คำนวณ objective ของทุกจุดได้ฟรีเพราะ margin ถูกคิดไว้แล้วสำหรับ step ถัดไป.
- **`max_iter=20000`** default: ที่ 2,000 รอบ C=10 ยังห่าง 17%; ที่ 20,000 รอบ gap ≤ 3e-6 ทุกค่า C (0.6 วินาทีต่อ fit).
- step size `1/t` ตาม card. objective เป็น 1-strongly convex จากพจน์ norm.
- **center feature ภายใน solver** (เพิ่ม 2026-10-11) แล้วเลื่อน intercept กลับตอนจบ. intercept ไม่ถูก penalize
  ดังนั้น `w·(x − mean) + b` กับ `w·x + b'` เป็นโมเดลเดียวกันและมี optimum เดียวกัน — ต่างแค่ความเร็วลู่เข้า.
  ทำเฉพาะเมื่อ `fit_intercept=True`.
- **12 candidates ต้องแก้ trainer**: svm ไขว้ทุก C กับทุกขนาด PCA ต่างจาก random_forest ที่ลอง PCA เฉพาะค่าที่ชนะ.

## ไฟล์

- `src/ml/classification/svm.py`, `src/ml/registry.py`
- `src/ml/core/trainer.py` — PCA search สำหรับ svm
- `tests/ml/classification/test_svm.py`, `tests/ml/core/test_trainer.py`

## ปัญหาที่เจอและแก้แล้ว

**1. reference ใน card ไม่ลู่เข้า.** อาการ: prototype ได้ rel_gap −2e-3 (C=1) และ −60% (C=10) เทียบ `LinearSVC` — เราได้ objective **ต่ำกว่า**. แทนที่จะสรุปว่าเราดีกว่าหรือขยายเกณฑ์ ตรวจ `n_iter_` ของ reference พบว่าชน 2,000,000 ทุกค่า C. ลอง `intercept_scaling` เล็กลง (1, 10) ลู่เข้าแต่แก้โจทย์ที่ต่างเล็กน้อย. เปลี่ยนเป็น `SVC(kernel="linear")` แล้ว gap เหลือ ≤ 6e-6. **บทเรียน: เมื่อ scratch ชนะ reference ให้สงสัย reference ก่อน และดู `n_iter_`.**

**2. ข้อมูลที่ไม่ได้ center ทำให้ได้คำตอบผิดเงียบ ๆ (พบจากการตรวจครั้งสุดท้าย).** อาการ: สุ่มทดสอบบนข้อมูลที่มี
คอลัมน์คงที่ค่า 3.0 แล้ว objective ห่างจาก libsvm 4.9% ที่ 60,000 รอบ และยังห่าง 2.3% ที่ 3,000,000 รอบ.
วัดแยก: เลื่อนค่าเฉลี่ยของทุก feature ไป 5 หน่วย gap เป็น 3.2% (C=1) และ 120% (C=10) ที่ 20,000 รอบ.
สาเหตุ: offset ทำให้ intercept (ซึ่งไม่มี penalty จึงไม่ strongly convex) ผูกกับน้ำหนักทุกตัว และ step `1/t` เดินช้ามาก.
**ผล production ไม่เคยกระทบ** เพราะ trainer standardize ให้เสมอและ PCA score มี mean 0 (วัดได้ 4e-06).
แก้โดย center ภายใน. วัดซ้ำหลังแก้ (shift 0, 1, 5, 50 × C 1, 10): |gap| ≤ 1.3e-05 ที่ shift ≤ 5; ที่ shift 50 gap เป็น −5.5e-05 และ −6.4e-04 คือ scratch ได้ objective **ต่ำกว่า** libsvm เล็กน้อย (libsvm เองเสียความแม่นเมื่อ feature อยู่ไกลจากศูนย์) — ทุกกรณีอยู่ในเกณฑ์ 1e-3.

## ค้างอยู่

- leaderboard: มีแล้ว (อัปเดต 2026-10-11) — `python -m src.ml.run report --variant <variant>` เขียน `outputs/modeling/report/<variant>/leaderboard_classification.{csv,md}` ซึ่งมีแถวของโมเดลนี้ทั้งสอง variant
- kernel SVM ไม่ได้ทำ (config กำหนด `kernel: linear`); รายงานอธิบาย kernel trick ได้โดยไม่ implement ตามแผนสำรองของ card
- ยังไม่แตะ Test set

## ผลจริงจากไฟล์ผล (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

ตารางนี้ถูกเติมโดย `python -m src.ml.docs_sync` จาก `validation_metrics.json` และ `test_metrics.json` ของโมเดลนี้ ตารางด้านบนของไฟล์เป็นบันทึกระหว่างพัฒนาและเก็บไว้ตามเดิม

<!-- BEGIN GENERATED: outputs/modeling/report/cards/svm.md -->
### ผลจริงของ `svm`

| variant | split | params | average_precision | roc_auc | f1 | recall | precision | specificity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_spike | validation (selection score) | {"C": 0.1, "fit_intercept": true, "max_iter": 20000, "random_state": 42} | 0.2514 | 0.6786 | 0.2553 | 0.1765 | 0.4615 | 0.9792 | 0.9057 |
| with_spike | test (one-shot) | {"C": 0.1, "fit_intercept": true, "max_iter": 20000, "random_state": 42} | 0.5716 | 0.7835 | 0.4874 | 0.3973 | 0.6304 | 0.9433 | 0.8365 |
| non_spike | validation (selection score) | {"C": 10, "fit_intercept": true, "max_iter": 20000, "random_state": 42} | 0.2501 | 0.6804 | 0.2727 | 0.1765 | 0.6000 | 0.9881 | 0.9137 |
| non_spike | test (one-shot) | {"C": 10, "fit_intercept": true, "max_iter": 20000, "random_state": 42} | 0.5805 | 0.7853 | 0.4587 | 0.3425 | 0.6944 | 0.9633 | 0.8418 |
<!-- END GENERATED -->
