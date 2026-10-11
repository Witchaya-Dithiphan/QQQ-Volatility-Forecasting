# MASTER PROGRESS INDEX — 19 โมเดล

> **ดัชนีเท่านั้น** — รายละเอียดอยู่ใน `docs/models/<name>/MODEL_PROGRESS.md` ของแต่ละตัว
> วิธีทำงาน: invoke skill `qqq-model` (กลั่นจาก `AGENTS.md` + `MODEL_CARDS.md` แล้ว ไม่ต้องอ่านต้นฉบับซ้ำ)
> อัปเดตตารางนี้ + ไฟล์ของโมเดลนั้น หลังจบทุกตัว · **ห้ามลบผลเก่า**

**สถานะรวม:** ✅ **11/11 ของฝั่ง Claude ครบแล้ว** · test suite `657 passed, 0 failed`

## ฝั่ง Claude (11 ตัว — ของเรา)

| # | โมเดล | task | สถานะ | parity | ผลหลัก (validation, with_spike) | บันทึก |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `multiple_linear` | reg | ✅ | exact 1.1e-15 | RMSE 0.068864 · R² **−0.0446** | [↗](../models/multiple_linear/MODEL_PROGRESS.md) |
| 2 | `simple_linear` | reg | ✅ | exact (8/8 ฟีเจอร์) | RMSE 0.069038 · R² **−0.0498** | [↗](../models/simple_linear/MODEL_PROGRESS.md) |
| 3 | `knn` | cls | ✅ | exact (8/8, held-out) | AUC **0.6987** · F1 0.2000 · Acc 0.8922 | [↗](../models/knn/MODEL_PROGRESS.md) · ปลดล็อก `stacking` แล้ว |
| 4 | `logistic` | cls | ✅ | exact (6/6) Δcoef 2.9e-08 | AUC 0.6732 · F1 0.2609 · Acc 0.9084 | [↗](../models/logistic/MODEL_PROGRESS.md) · **ปลดล็อก `stacking` ครบแล้ว** |
| 5 | `gaussian_nb` | cls | ✅ | exact 1e-12 | **F1 0.2812** (ดีสุด) · AUC 0.6665 | [↗](../models/gaussian_nb/MODEL_PROGRESS.md) |
| 6 | `polynomial` | reg | ✅ | exact (16/16) | RMSE 0.069065 · R² −0.0507 | [↗](../models/polynomial/MODEL_PROGRESS.md) |
| 7 | `perceptron` | cls | ✅ | exact 1e-10 | **AUC 0.2993** (< สุ่ม) · F1 0.0 | [↗](../models/perceptron/MODEL_PROGRESS.md) ตรวจแล้ว sklearn ได้เท่ากัน |
| 8 | `slp` | cls | ✅ | converged 1.3e-08 vs LogReg | AUC **0.6803** (ดีสุดในกลุ่ม cls) · F1 0.2609 | [↗](../models/slp/MODEL_PROGRESS.md) |
| 9 | `elastic_net` ★ | reg | ✅ | exact (12/12) 1e-6 | **RMSE 0.066806 · R² +0.0169** ชนะทุกตัว | [↗](../models/elastic_net/MODEL_PROGRESS.md) **หลักฐาน +5** |
| 10 | `kmeans` | clustering | ✅ | partition ตรง 5/5 · inertia ≤ sklearn | k=2 · silhouette **0.3666** | [↗](../models/kmeans/MODEL_PROGRESS.md) |
| 11 | `agglomerative` | clustering | ✅ | partition ตรง 10/10 | k=2 ward · silhouette 0.3602 | [↗](../models/agglomerative/MODEL_PROGRESS.md) |

## ฝั่ง Codex (8 ตัว — เจ้าของดูแลเอง)

| โมเดล | สถานะ | ผลหลัก (validation) | บันทึก |
| --- | --- | --- | --- |
| `decision_tree` | 🟡 parity/training/reload ผ่าน; รอ leaderboard | AUC 0.6418 (`with_spike`) · 0.6924 (`non_spike`) | [↗](../models/decision_tree/MODEL_PROGRESS.md) |
| `random_forest` | 🟡 parity/training/reload ผ่าน; รอ leaderboard | AP 0.2548 (`with_spike`) · 0.3010 (`non_spike`) | [↗](../models/random_forest/MODEL_PROGRESS.md) |
| `adaboost` ★ | 🟡 parity/training/reload ผ่าน; รอ leaderboard | AP 0.2363 (`with_spike`) · 0.2407 (`non_spike`) | [↗](../models/adaboost/MODEL_PROGRESS.md) |
| `gradient_boosting` | 🟡 parity/training/reload ผ่าน; รอ leaderboard | AP 0.2531 (`with_spike`) · 0.2525 (`non_spike`) | [↗](../models/gradient_boosting/MODEL_PROGRESS.md) |
| `xgboost` | 🟡 parity/training/reload ผ่าน; รอ leaderboard | AP 0.2508 (`with_spike`) · 0.2652 (`non_spike`) | [↗](../models/xgboost/MODEL_PROGRESS.md) |
| `svm` | 🟡 parity/training/reload ผ่าน; รอ leaderboard | AP 0.2514 (`with_spike`) · 0.2501 (`non_spike`) | [↗](../models/svm/MODEL_PROGRESS.md) |
| `mlp` | 🟡 parity/training/reload ผ่าน; รอ leaderboard | AP 0.2668 (`with_spike`) · 0.2808 (`non_spike`) | [↗](../models/mlp/MODEL_PROGRESS.md) |
| `stacking` | 🟡 parity/training/reload ผ่าน; รอ leaderboard | AP 0.2262 (`with_spike`) · 0.2311 (`non_spike`) | [↗](../models/stacking/MODEL_PROGRESS.md) |

เจ้าของดูแล progress เอง บน branch `feat/models-tree-family`; T1/T2 ยังไม่ครบ DoD ข้อ leaderboard

## Dependency

```text
ทุกตัว ──► src/ml/core/        เสร็จแล้ว ไม่บล็อกใคร
Codex:  decision_tree ──► random_forest · adaboost · gradient_boosting ──► xgboost
ข้ามฝั่ง: stacking (Codex) ต้องใช้ logistic + knn (เรา)   ← เหตุผลที่ดึง 2 ตัวนี้ขึ้นมาก่อน
ฝั่งเรา: ไม่มี dependency ภายใน — ข้ามตัวที่ติดปัญหาไปทำตัวอื่นได้อิสระ
```

## สถานะ core framework

merge เข้า `main` แล้ว (`385adfc`) — `base` · `compare` · `data` · `preprocess` · `metrics` ·
`persist` · `trainer` · `figures` · `registry` · `run`

**ยังไม่ทำ (Phase C, รอโมเดลครบทั้งสองฝั่ง):** `run.py report` · `run.py finalize` ·
paired comparison A↔B · notebooks 03/04/05

## ถัดไป

**โมเดลฝั่งเราครบ 11/11 แล้ว** → เข้า Phase C ซึ่งต้องรอฝั่ง Codex ให้ครบ 8 ตัวก่อนบางส่วน

| ลำดับ | งาน | รอฝั่งอื่นไหม |
| --- | --- | --- |
| C1 | `run.py report` — leaderboard + ROC ซ้อน + confusion + performance curve | รอ (ต้องมีครบ 19) |
| C2 | paired comparison A↔B เต็มรูปแบบ | รอ |
| C3 | `run.py finalize` — **แตะ Test ครั้งเดียว** | รอ |
| C4 | notebooks 03 / 04 / 05 | เริ่มโครงได้เลย |
| C5 | report + slides ตาม `REPORT_AND_SLIDES_OUTLINE.md` | **เริ่มได้เลย** — ข้อมูลฝั่งเราพอสำหรับ §2, §3, §6.1 |
| C6 | ใส่ผลจริงลง `MODEL_CARDS.md` ทั้ง 11 ใบ | **เริ่มได้เลย** |

## ข้อค้นพบสะสม (ใช้ในรายงาน §8.2)

ทั้ง 2 โมเดล regression ที่ทำแล้วให้ **pattern เดียวกัน**: `with_spike` R² ติดลบ ·
`non_spike` R² พลิกเป็นบวก

**Regression — ตัด spike แล้วดีขึ้น**

| โมเดล | R² with_spike | R² non_spike | Δ |
| --- | --- | --- | --- |
| `multiple_linear` | −0.044554 | +0.016536 | **+0.061** |
| `simple_linear` | −0.049847 | +0.026542 | **+0.076** |
| `polynomial` | −0.050652 | +0.038890 | **+0.090** |
| **`elastic_net` ★** | **+0.016942** | **+0.090577** | **+0.074** |

**`elastic_net` เป็นโมเดล regression ตัวเดียวที่ R² บวกบน `with_spike`** และชนะทุกตัวทั้งสอง variant
เทียบ `multiple_linear`: RMSE ลดลง 2.99% (A) / 3.84% (B) · R² −0.0446 → +0.0169 (A)
→ **หลักฐานตรงสำหรับคะแนนพิเศษ +5 "much better than existing-taught models"**

**Classification — กลับทาง ตัด spike แล้วแย่ลง**

| โมเดล | AUC with_spike | AUC non_spike | Δ |
| --- | --- | --- | --- |
| `knn` | 0.6987 | 0.6455 | **−0.053** |
| `logistic` | 0.6732 | 0.6824 | **+0.009** |
| `gaussian_nb` | 0.6665 | 0.6789 | **+0.012** |
| `perceptron` | 0.2993 | 0.3383 | **+0.039** |
| `slp` | 0.6803 | 0.6778 | −0.003 |

**ยังสรุปทิศทางของ classification ไม่ได้** — knn ชอบ `with_spike` ชัดเจน แต่ logistic
กลับชอบ `non_spike` เล็กน้อย ต่างจากฝั่ง regression ที่สอดคล้องกันทั้งสองตัว
สมมติฐานที่ยังต้องทดสอบ: spike **คือ** ตัวอย่างบวกของงาน classification ตัดออกจึงเสียข้อมูล
แต่ผลของมันอาจขึ้นกับว่าโมเดลเป็นแบบ local (knn) หรือ global (logistic)
→ **ห้ามเขียนข้อสรุปลงรายงานจนกว่าจะครบอย่างน้อย 5-6 classifier**

**Accuracy ใช้ไม่ได้กับชุดนี้** — validation 337 neg / 34 pos (9.2% positive)
majority baseline = **0.9084** สูงกว่าโมเดลจริงหลายตัว ต้องรายงาน AUC/F1 เป็นหลัก

## ตรวจครั้งสุดท้ายทั้ง 19 โมเดล (2026-10-11)

ตรวจ 4 ชั้นโดยไม่พึ่ง test เดิม: (1) อ่านโค้ดทุกไฟล์เทียบสูตรกับ card · (2) train ใหม่ทั้ง 38 artifact ด้วยค่าที่
trainer เลือกจริง แล้วเทียบกับ library, ตรวจว่าทำซ้ำได้, และคำนวณ metric ซ้ำด้วย `sklearn.metrics` ·
(3) สุ่มทดสอบ 255 ครั้งด้วย hyperparameter นอก grid และข้อมูลที่มีค่าซ้ำ/คอลัมน์คงที่/คลาสไม่สมดุล ·
(4) ตรวจกฎโปรเจกต์ (ไม่แตะ Test, ไม่มี library ใน `fit`/`predict`, ไม่มี `default_rng`, deterministic).

**ไม่พบสูตรหรือ logic ผิดในโมเดลใด.** ทั้ง 38 artifact ทำซ้ำได้พอดี, reload ผ่านทั้งหมด, metric ตรง sklearn ที่ 1e-16.

ห่างจาก library ที่ค่า production (บน Validation):

| โมเดล | ผลต่าง |
| --- | --- |
| simple_linear, multiple_linear, polynomial, gaussian_nb, knn, adaboost, mlp | ≤ 1e-15 |
| perceptron, decision_tree, random_forest, gradient_boosting, xgboost | 0 |
| kmeans, agglomerative | partition ตรง 100% |
| elastic_net | ≤ 2e-06 (กฎหยุดต่างจาก sklearn) |
| svm | objective ≤ 3e-06 |
| logistic | ≤ 1e-03 (หยุดที่ 5,000 รอบ) |
| slp | 0.07–0.29 (หยุดที่โควตา 1,000 epochs; reference คือ logistic ที่ลู่เข้า) |
| stacking | ไม่มี library เทียบ — ตรวจด้วย rebuild fold ต่อ fold และ leakage probe |

สิ่งที่พบและแก้แล้ว:

1. **`svm` ได้คำตอบผิดเงียบ ๆ บนข้อมูลที่ไม่ได้ center** (ไม่กระทบ production เพราะ trainer standardize เสมอ) → center ภายใน solver
2. **artifact ล้าสมัย:** `preprocessor.npz` ของ 10 โมเดลฝั่ง linear ไม่มี key `constant` และ metadata ของ `multiple_linear`
   เขียน `preprocess: None` → train ใหม่ให้ตรงกับโค้ดปัจจุบัน (เนื้อหาโมเดลและ metric เท่าเดิม)
3. **docstring ของ `knn`** อ้างว่า tie-break ตรง sklearn → แก้ข้อความ (ไม่กระทบข้อมูลจริง)
4. **`ARCHITECTURE.md` ขั้นที่ 5** บอกว่า trainer เลือก threshold แต่โค้ดใช้ค่าคงที่ → แก้เอกสาร

### สามโมเดลที่ผลขึ้นกับโควตาการเทรน — ต้องเขียนในรายงานตามนี้

ทั้งสามจบเพราะหมดโควตา ไม่ใช่เพราะลู่เข้า ตัวเลขของมันจึงเป็นของ "อัลกอริทึม + โควตาที่ config กำหนด" ไม่ใช่ของอัลกอริทึมอย่างเดียว:

| โมเดล | สิ่งที่เกิดจริง | ผลต่อการตีความ |
| --- | --- | --- |
| `slp` | ผู้ชนะทั้งสอง variant วิ่งครบ 1,000 epochs; early stopping ไม่เคยทำงาน | AUC ที่สูงกว่า `logistic` เล็กน้อยมาจากการหยุดที่ 1,000 ไม่ใช่จาก regularization ที่ข้อมูลเลือก |
| `perceptron` | ข้อมูลแยกเชิงเส้นไม่ได้ จึงไม่มีวันหยุด; จบที่น้ำหนักของ update ครั้งสุดท้าย (ผิด 38–42% บน Train) | AUC 0.30 / 0.34 ต่ำกว่าสุ่ม — ตรง sklearn ทุก bit จึงเป็นธรรมชาติของอัลกอริทึม |
| `logistic` | ทั้ง 12 การรันหยุดที่ 5,000 รอบ | ห่างจาก optimum ไม่เกิน 0.008 ใน probability; ผู้ชนะไม่เปลี่ยน |

ขอบเขตที่การตรวจไม่ครอบคลุม: ชั้นข้อมูลและ feature · ข้อผิดพลาดที่เราและ library ตั้งค่าผิดเหมือนกัน (พึ่ง test คำนวณมือ) ·
เส้นทาง `completed_early` ของ `adaboost` · `scaler_fit: subtrain` ของ `slp`/`mlp` · นัยสำคัญทางสถิติระหว่างโมเดล (positive 34 แถว)
