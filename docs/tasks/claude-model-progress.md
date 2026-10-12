# MASTER PROGRESS INDEX — 19 โมเดล

> **ดัชนีเท่านั้น** — รายละเอียดอยู่ใน `docs/models/<name>/MODEL_PROGRESS.md` ของแต่ละตัว
> วิธีทำงาน: invoke skill `qqq-model` (กลั่นจาก `AGENTS.md` + `MODEL_CARDS.md` แล้ว ไม่ต้องอ่านต้นฉบับซ้ำ)
> อัปเดตตารางนี้ + ไฟล์ของโมเดลนั้น หลังจบทุกตัว · **ห้ามลบผลเก่า**

**สถานะรวม (อัปเดต 2026-10-11):** ✅ **ครบ 19/19 โมเดล ทั้งสอง variant** · Phase C เสร็จถึงขั้น finalize: leaderboard, เทียบ with_spike กับ non_spike, feature importance, walk-forward และผล Test (อ่านครั้งเดียว) อยู่ใต้ `outputs/modeling/` · ตารางท้ายไฟล์นี้สร้างจากไฟล์ผลโดยตรง ·
ชั้นข้อมูลผ่าน audit: [DATA_PIPELINE_AUDIT.md](../audit/DATA_PIPELINE_AUDIT.md)

## ฝั่ง Claude (11 ตัว — ของเรา)

| # | โมเดล | task | สถานะ | parity | ผลหลัก (validation, with_spike) | บันทึก |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `multiple_linear` | reg | ✅ | exact 1.1e-15 | RMSE 0.068864 · R² **−0.0446** | [↗](../models/multiple_linear/MODEL_PROGRESS.md) |
| 2 | `simple_linear` | reg | ✅ | exact (8/8 ฟีเจอร์) | RMSE 0.069038 · R² **−0.0498** | [↗](../models/simple_linear/MODEL_PROGRESS.md) |
| 3 | `knn` | cls | ✅ | exact (8/8, held-out) | AUC **0.6987** · F1 0.2000 · Acc 0.8922 | [↗](../models/knn/MODEL_PROGRESS.md) · ปลดล็อก `stacking` แล้ว |
| 4 | `logistic` | cls | ✅ | exact (6/6) Δcoef 2.9e-08 | AUC 0.6732 · F1 0.2609 · Acc 0.9084 | [↗](../models/logistic/MODEL_PROGRESS.md) · **ปลดล็อก `stacking` ครบแล้ว** |
| 5 | `gaussian_nb` | cls | ✅ | exact 1e-12 | **F1 0.2812** (ดีสุดในฝั่ง linear/probabilistic; `mlp` 0.3200 และ `decision_tree` 0.3099 สูงกว่า) · AUC 0.6665 | [↗](../models/gaussian_nb/MODEL_PROGRESS.md) |
| 6 | `polynomial` | reg | ✅ | exact (16/16) | RMSE 0.069065 · R² −0.0507 | [↗](../models/polynomial/MODEL_PROGRESS.md) |
| 7 | `perceptron` | cls | ✅ | exact 1e-10 | **AUC 0.2993** (< สุ่ม) · F1 0.0 | [↗](../models/perceptron/MODEL_PROGRESS.md) ตรวจแล้ว sklearn ได้เท่ากัน |
| 8 | `slp` | cls | ✅ | converged 1.3e-08 vs LogReg | AUC **0.6803** (`knn` 0.6987 สูงกว่า) · F1 0.2609 | [↗](../models/slp/MODEL_PROGRESS.md) |
| 9 | `elastic_net` ★ | reg | ✅ | exact (12/12) 1e-6 | **RMSE 0.066806 · R² +0.0169** ชนะทุกตัว | [↗](../models/elastic_net/MODEL_PROGRESS.md) **หลักฐาน +5** |
| 10 | `kmeans` | clustering | ✅ | partition ตรง 5/5 · inertia ≤ sklearn | k=2 · silhouette **0.3666** | [↗](../models/kmeans/MODEL_PROGRESS.md) |
| 11 | `agglomerative` | clustering | ✅ | partition ตรง 10/10 | k=2 ward · silhouette 0.3602 | [↗](../models/agglomerative/MODEL_PROGRESS.md) |

## ฝั่ง Codex (8 ตัว — เจ้าของดูแลเอง)

| โมเดล | สถานะ | ผลหลัก (validation) | บันทึก |
| --- | --- | --- | --- |
| `decision_tree` | ✅ parity/training/reload/leaderboard ครบ | AUC 0.6418 (`with_spike`) · 0.6924 (`non_spike`) | [↗](../models/decision_tree/MODEL_PROGRESS.md) |
| `random_forest` | ✅ parity/training/reload/leaderboard ครบ | AP 0.2548 (`with_spike`) · 0.3010 (`non_spike`) | [↗](../models/random_forest/MODEL_PROGRESS.md) |
| `adaboost` ★ | ✅ parity/training/reload/leaderboard ครบ | AP 0.2363 (`with_spike`) · 0.2407 (`non_spike`) | [↗](../models/adaboost/MODEL_PROGRESS.md) |
| `gradient_boosting` | ✅ parity/training/reload/leaderboard ครบ | AP 0.2531 (`with_spike`) · 0.2525 (`non_spike`) | [↗](../models/gradient_boosting/MODEL_PROGRESS.md) |
| `xgboost` | ✅ parity/training/reload/leaderboard ครบ | AP 0.2508 (`with_spike`) · 0.2652 (`non_spike`) | [↗](../models/xgboost/MODEL_PROGRESS.md) |
| `svm` | ✅ parity/training/reload/leaderboard ครบ | AP 0.2514 (`with_spike`) · 0.2501 (`non_spike`) | [↗](../models/svm/MODEL_PROGRESS.md) |
| `mlp` | ✅ parity/training/reload/leaderboard ครบ | AP 0.2668 (`with_spike`) · 0.2808 (`non_spike`) | [↗](../models/mlp/MODEL_PROGRESS.md) |
| `stacking` | ✅ parity/training/reload/leaderboard ครบ | AP 0.2262 (`with_spike`) · 0.2311 (`non_spike`) | [↗](../models/stacking/MODEL_PROGRESS.md) |

⚠️ แก้ไข 2026-10-11: branch `feat/models-tree-family` ถูก merge แล้ว และ DoD ข้อ leaderboard ครบทุกตัวหลัง `run.py report` ใช้งานได้

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

Phase C เสร็จแล้วถึงขั้น finalize (2026-10-11) รายละเอียดและลำดับงานอยู่ในแผน
[`2026-10-11-phase-c-report-and-finalize.md`](../superpowers/plans/2026-10-11-phase-c-report-and-finalize.md)

| งาน | สถานะ | ผลลัพธ์อยู่ที่ |
| --- | --- | --- |
| `run.py report` — leaderboard + รูป | ✅ | `outputs/modeling/report/<variant>/`, `outputs/figures/modeling/<variant>/` |
| `run.py compare` — with_spike เทียบ non_spike | ✅ | `outputs/modeling/report/comparison_a_vs_b.md` |
| `run.py analyze` — feature importance + walk-forward | ✅ | `outputs/modeling/report/<variant>/` |
| `run.py finalize` — อ่าน Test ครั้งเดียว | ✅ รันแล้ว ห้ามรันซ้ำ | `outputs/modeling/final/`, `outputs/modeling/finalize_receipt.json` |
| notebooks 03 / 04 / 05 | ✅ | `notebooks/` |
| รายงานและสไลด์ | ดู `docs/report/` และ `deliverables/` | |

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
| simple_linear, multiple_linear, polynomial, gaussian_nb, knn, adaboost, mlp | ≤ 2e-15 (⚠️ แก้ไข 2026-10-13: เดิมเขียน ≤ 1e-15 แต่วัดซ้ำได้สูงสุด 1.1e-15 ที่ `gaussian_nb` แบบ `non_spike`) |
| perceptron, decision_tree, random_forest, gradient_boosting, xgboost | 0 |
| kmeans, agglomerative | partition ตรง 100% |
| elastic_net | ≤ 3e-06 (กฎหยุดต่างจาก sklearn · ⚠️ แก้ไข 2026-10-13: เดิมเขียน ≤ 2e-06 แต่วัดซ้ำได้ 8.8e-07 ที่ `with_spike` และ 2.2e-06 ที่ `non_spike`) |
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

## Leaderboard (สร้างอัตโนมัติ ห้ามแก้ด้วยมือ)

เติมโดย `python -m src.ml.docs_sync` · ตัวเลข Validation เป็น selection score · ผล Test อ่านครั้งเดียวเมื่อ 2026-10-11

<!-- BEGIN GENERATED: outputs/modeling/report/with_spike/leaderboard_classification.md -->
### Validation leaderboard — classification — with_spike

| model | kind | average_precision_ci | roc_auc_ci | f1_ci | recall_ci | precision | specificity | accuracy | loss_name | loss | tied_with_leader | beats_best_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | model | 0.2668 [0.0579, 0.4724] | 0.6757 [0.3961, 0.8474] | 0.3200 [0.0000, 0.5556] | 0.2353 [0.0000, 0.4889] | 0.5000 | 0.9763 | 0.9084 | binary_cross_entropy | 0.3359 | yes | yes |
| random_forest | model | 0.2548 [0.0603, 0.4586] | 0.6862 [0.4634, 0.8323] | 0.2917 [0.0000, 0.5051] | 0.2059 [0.0000, 0.4255] | 0.5000 | 0.9792 | 0.9084 | log_loss | 0.3131 | yes | yes |
| gradient_boosting | model | 0.2531 [0.0548, 0.4746] | 0.6754 [0.4075, 0.8409] | 0.2449 [0.0000, 0.4330] | 0.1765 [0.0000, 0.3674] | 0.4000 | 0.9733 | 0.9003 | binary_logistic_loss | 0.3075 | yes | yes |
| svm | model | 0.2514 [0.0514, 0.4651] | 0.6786 [0.4339, 0.8404] | 0.2553 [0.0000, 0.4445] | 0.1765 [0.0000, 0.3674] | 0.4615 | 0.9792 | 0.9057 | hinge_loss | 0.2688 | yes | yes |
| xgboost | model | 0.2508 [0.0559, 0.4681] | 0.6693 [0.4080, 0.8368] | 0.2449 [0.0000, 0.4424] | 0.1765 [0.0000, 0.3674] | 0.4000 | 0.9733 | 0.9003 | binary_logistic_loss | 0.3263 | yes | yes |
| baseline_persistence | baseline | 0.2474 [0.0545, 0.5002] | 0.7214 [0.5384, 0.8407] | 0.2899 [0.0000, 0.5589] | 0.2941 [0.0000, 0.5601] | 0.2857 | 0.9258 | 0.8679 | n/a | n/a | yes | n/a |
| slp | model | 0.2427 [0.0545, 0.4432] | 0.6803 [0.4458, 0.8289] | 0.2609 [0.0000, 0.4615] | 0.1765 [0.0000, 0.3674] | 0.5000 | 0.9822 | 0.9084 | binary_cross_entropy | 0.3046 | yes | no |
| adaboost | model | 0.2363 [0.0626, 0.4381] | 0.6768 [0.4602, 0.8196] | 0.2222 [0.0000, 0.4226] | 0.1765 [0.0000, 0.3674] | 0.3000 | 0.9585 | 0.8868 | exponential_loss | 0.5771 | yes | no |
| logistic | model | 0.2330 [0.0526, 0.4301] | 0.6732 [0.4224, 0.8305] | 0.2609 [0.0000, 0.4615] | 0.1765 [0.0000, 0.3674] | 0.5000 | 0.9822 | 0.9084 | binary_cross_entropy | 0.3089 | yes | no |
| stacking | model | 0.2262 [0.0551, 0.4176] | 0.6641 [0.4218, 0.8155] | 0.2353 [0.0000, 0.4000] | 0.1765 [0.0000, 0.3674] | 0.3529 | 0.9674 | 0.8949 | log_loss | 0.3455 | yes | no |
| gaussian_nb | model | 0.2218 [0.0464, 0.4333] | 0.6665 [0.4217, 0.8214] | 0.2812 [0.0000, 0.5001] | 0.2647 [0.0000, 0.5228] | 0.3000 | 0.9377 | 0.8760 | log_loss | 0.6203 | yes | no |
| knn | model | 0.2037 [0.0610, 0.4015] | 0.6987 [0.5068, 0.8179] | 0.2000 [0.0000, 0.3509] | 0.1471 [0.0000, 0.3077] | 0.3125 | 0.9674 | 0.8922 | log_loss | 0.6345 | yes | no |
| decision_tree | model | 0.1732 [0.0494, 0.3562] | 0.6418 [0.4752, 0.7795] | 0.3099 [0.0000, 0.5094] | 0.3235 [0.0000, 0.5610] | 0.2973 | 0.9228 | 0.8679 | log_loss | 0.3350 | yes | no |
| baseline_majority | baseline | 0.0916 [0.0350, 0.1779] | 0.5000 [0.5000, 0.5000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | n/a | 1.0000 | 0.9084 | n/a | n/a | no | n/a |
| perceptron | model | 0.0648 [0.0310, 0.1334] | 0.2993 [0.1807, 0.4718] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.9644 | 0.8760 | perceptron_criterion | 0.0008 | yes | no |

> ตัวเลขบน Validation เป็น selection score: hyperparameter ของทุกโมเดลถูกเลือกบนแถวชุดนี้ จึงมองโลกในแง่ดีเกินจริง และโมเดลที่ `tied_with_leader = yes` แยกจากอันดับ 1 ไม่ได้ที่ระดับ 95% (moving-block bootstrap, block 20 วัน) ห้ามอ้างอันดับภายในกลุ่มนั้น
> 13 จาก 13 โมเดลอยู่ในกลุ่มที่แยกจากอันดับ 1 ไม่ได้
<!-- END GENERATED -->

<!-- BEGIN GENERATED: outputs/modeling/report/with_spike/leaderboard_regression.md -->
### Validation leaderboard — regression — with_spike

| model | kind | rmse_ci | mae | r2_ci | tied_with_leader | beats_best_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| elastic_net | model | 0.0668 [0.0610, 0.0739] | 0.0539 | 0.0169 [-0.2649, 0.2159] | yes | yes |
| multiple_linear | model | 0.0689 [0.0617, 0.0779] | 0.0542 | -0.0446 [-0.3994, 0.2085] | yes | yes |
| simple_linear | model | 0.0690 [0.0609, 0.0789] | 0.0557 | -0.0498 [-0.2829, 0.0997] | yes | yes |
| polynomial | model | 0.0691 [0.0606, 0.0797] | 0.0552 | -0.0507 [-0.2579, 0.0879] | yes | yes |
| baseline_train_mean | baseline | 0.0751 [0.0673, 0.0837] | 0.0623 | -0.2430 [-0.6975, -0.0225] | no | n/a |
| baseline_persistence | baseline | 0.0826 [0.0750, 0.0901] | 0.0673 | -0.5032 [-1.0576, -0.0677] | no | n/a |

> ตัวเลขบน Validation เป็น selection score: hyperparameter ของทุกโมเดลถูกเลือกบนแถวชุดนี้ จึงมองโลกในแง่ดีเกินจริง และโมเดลที่ `tied_with_leader = yes` แยกจากอันดับ 1 ไม่ได้ที่ระดับ 95% (moving-block bootstrap, block 20 วัน) ห้ามอ้างอันดับภายในกลุ่มนั้น
> 4 จาก 4 โมเดลอยู่ในกลุ่มที่แยกจากอันดับ 1 ไม่ได้
<!-- END GENERATED -->

<!-- BEGIN GENERATED: outputs/modeling/report/comparison_a_vs_b.md -->
### Paired comparison: non_spike − with_spike (Validation)

| model | task | metric | with_spike | non_spike | delta | delta_95 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| elastic_net | regression | rmse | 0.0668 | 0.0643 | -0.0026 | [-0.0053, +0.0019] | inconclusive |
| multiple_linear | regression | rmse | 0.0689 | 0.0668 | -0.0020 | [-0.0036, -0.0000] | non_spike better |
| polynomial | regression | rmse | 0.0691 | 0.0661 | -0.0030 | [-0.0073, +0.0016] | inconclusive |
| simple_linear | regression | rmse | 0.0690 | 0.0665 | -0.0026 | [-0.0066, +0.0026] | inconclusive |
| adaboost | classification | average_precision | 0.2363 | 0.2407 | 0.0044 | [-0.0307, +0.0627] | inconclusive |
| decision_tree | classification | average_precision | 0.1732 | 0.1887 | 0.0155 | [-0.0380, +0.0970] | inconclusive |
| gaussian_nb | classification | average_precision | 0.2218 | 0.2292 | 0.0074 | [-0.0007, +0.0187] | inconclusive |
| gradient_boosting | classification | average_precision | 0.2531 | 0.2525 | -0.0006 | [-0.0445, +0.0223] | inconclusive |
| knn | classification | average_precision | 0.2037 | 0.2082 | 0.0045 | [-0.0441, +0.0601] | inconclusive |
| logistic | classification | average_precision | 0.2330 | 0.2513 | 0.0183 | [-0.0110, +0.0401] | inconclusive |
| mlp | classification | average_precision | 0.2668 | 0.2808 | 0.0139 | [-0.0429, +0.0772] | inconclusive |
| perceptron | classification | average_precision | 0.0648 | 0.0742 | 0.0094 | [-0.0012, +0.0575] | inconclusive |
| random_forest | classification | average_precision | 0.2548 | 0.3010 | 0.0462 | [-0.0168, +0.1155] | inconclusive |
| slp | classification | average_precision | 0.2427 | 0.2474 | 0.0047 | [-0.0477, +0.0441] | inconclusive |
| stacking | classification | average_precision | 0.2262 | 0.2311 | 0.0049 | [-0.0386, +0.0477] | inconclusive |
| svm | classification | average_precision | 0.2514 | 0.2501 | -0.0013 | [-0.0108, +0.0082] | inconclusive |
| xgboost | classification | average_precision | 0.2508 | 0.2652 | 0.0144 | [-0.0148, +0.0386] | inconclusive |
| agglomerative | clustering | silhouette | 0.3602 | 0.4642 | 0.1040 | n/a | no interval |
| kmeans | clustering | silhouette | 0.3666 | 0.3298 | -0.0367 | n/a | no interval |

> ผลต่างในตารางนี้คือผลของ **นโยบายตัดแถวทั้งชุด** ไม่ใช่ผลของ spike เพียงอย่างเดียว: การตัด 213 แถวทำให้ class prior ของ Train เปลี่ยนจาก 25.0% เป็น 18.4%, จำนวนแถวลดจาก 1,733 เป็น 1,520 และ regime ที่ผันผวนสูงหายไปจาก Train พร้อมกัน ทั้งสาม confound นี้แยกจากกันไม่ได้ด้วยการทดลองนี้ delta = non_spike − with_spike บน Validation 371 แถวชุดเดียวกัน ช่วง 95% จาก paired moving-block bootstrap (block 20 วัน) และเมื่อเทียบ 17 โมเดลพร้อมกัน ควรคาดว่าจะมีราว 1 แถวที่ช่วงไม่คร่อมศูนย์โดยบังเอิญ
<!-- END GENERATED -->

<!-- BEGIN GENERATED: outputs/modeling/final/with_spike/leaderboard_classification.md -->
### Test leaderboard — classification — with_spike

| model | kind | headline | validation_average_precision | average_precision_ci | roc_auc_ci | f1_ci | recall_ci | tied_with_leader | beats_best_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | model | yes | 0.2668 | 0.5991 [0.2605, 0.7603] | 0.7930 [0.6103, 0.9031] | 0.5714 [0.2104, 0.6929] | 0.4932 [0.1429, 0.6505] | yes | yes |
| random_forest | model | no | 0.2548 | 0.5775 [0.2366, 0.7393] | 0.7866 [0.5922, 0.8920] | 0.5000 [0.1764, 0.6047] | 0.4247 [0.1333, 0.5424] | yes | yes |
| svm | model | no | 0.2514 | 0.5716 [0.2402, 0.7438] | 0.7835 [0.5935, 0.8955] | 0.4874 [0.1176, 0.6102] | 0.3973 [0.0789, 0.5536] | yes | yes |
| gaussian_nb | model | no | 0.2218 | 0.5538 [0.2390, 0.7289] | 0.7703 [0.5749, 0.8838] | 0.5806 [0.3055, 0.6892] | 0.6164 [0.3158, 0.7444] | yes | yes |
| slp | model | no | 0.2427 | 0.5536 [0.2473, 0.7470] | 0.7798 [0.5865, 0.8954] | 0.5041 [0.1665, 0.6480] | 0.4247 [0.1052, 0.6186] | yes | yes |
| logistic | model | no | 0.2330 | 0.5518 [0.2482, 0.7452] | 0.7822 [0.5936, 0.8978] | 0.5203 [0.2083, 0.6564] | 0.4384 [0.1351, 0.6211] | yes | yes |
| knn | model | no | 0.2037 | 0.5472 [0.2187, 0.7309] | 0.7847 [0.6241, 0.8634] | 0.5512 [0.1886, 0.6804] | 0.4795 [0.1350, 0.6579] | yes | yes |
| baseline_persistence | baseline | no | n/a | 0.5365 [0.1589, 0.7246] | 0.7255 [0.5280, 0.8308] | 0.4933 [0.1599, 0.6335] | 0.5068 [0.1633, 0.6552] | yes | n/a |
| gradient_boosting | model | no | 0.2531 | 0.5332 [0.2223, 0.7096] | 0.7665 [0.5798, 0.8803] | 0.4390 [0.1591, 0.6000] | 0.3699 [0.1190, 0.5539] | yes | no |
| xgboost | model | no | 0.2508 | 0.5305 [0.2021, 0.7067] | 0.7716 [0.5844, 0.8839] | 0.4553 [0.1795, 0.6078] | 0.3836 [0.1333, 0.5582] | yes | no |
| stacking | model | no | 0.2262 | 0.5131 [0.2395, 0.7154] | 0.7902 [0.6082, 0.8877] | 0.5714 [0.2898, 0.6842] | 0.5753 [0.2439, 0.7385] | yes | no |
| adaboost | model | no | 0.2363 | 0.4858 [0.2145, 0.6807] | 0.7614 [0.5710, 0.8785] | 0.4853 [0.2268, 0.6555] | 0.4521 [0.1852, 0.6775] | no | no |
| decision_tree | model | no | 0.1732 | 0.3792 [0.1524, 0.5643] | 0.7450 [0.6021, 0.8449] | 0.4800 [0.2000, 0.6667] | 0.4932 [0.1621, 0.7527] | no | no |
| baseline_majority | baseline | no | n/a | 0.1957 [0.0885, 0.3003] | 0.5000 [0.5000, 0.5000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | no | n/a |
| perceptron | model | no | 0.0648 | 0.1374 [0.0657, 0.2463] | 0.2749 [0.1567, 0.4688] | 0.0577 [0.0000, 0.1840] | 0.0411 [0.0000, 0.1690] | no | no |

> ประเมินบน Test ครั้งเดียวด้วยโมเดลที่เทรนจาก Train และเลือกบน Validation (ไม่ refit) โมเดลเด่นของแต่ละ task ถูกระบุไว้ใน selection_lock.json ก่อนเปิด Test แถวอื่นในตารางเป็นข้อมูลประกอบ ห้ามเลือกโมเดลจากตารางนี้
<!-- END GENERATED -->

<!-- BEGIN GENERATED: outputs/modeling/final/with_spike/leaderboard_regression.md -->
### Test leaderboard — regression — with_spike

| model | kind | headline | validation_rmse | rmse_ci | r2_ci | tied_with_leader | beats_best_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multiple_linear | model | no | 0.0689 | 0.1158 [0.0711, 0.1680] | 0.2959 [-0.1971, 0.3897] | yes | yes |
| elastic_net | model | yes | 0.0668 | 0.1174 [0.0709, 0.1746] | 0.2758 [-0.0402, 0.3258] | yes | yes |
| simple_linear | model | no | 0.0690 | 0.1356 [0.0754, 0.1985] | 0.0347 [-0.7702, 0.2310] | no | yes |
| baseline_train_mean | baseline | no | n/a | 0.1380 [0.0806, 0.2060] | -0.0006 [-0.2998, -0.0000] | yes | n/a |
| polynomial | model | no | 0.0691 | 0.1401 [0.0754, 0.2037] | -0.0314 [-1.1539, 0.2401] | no | no |
| baseline_persistence | baseline | no | n/a | 0.1502 [0.0919, 0.2222] | -0.1855 [-1.2071, 0.1280] | no | n/a |

> ประเมินบน Test ครั้งเดียวด้วยโมเดลที่เทรนจาก Train และเลือกบน Validation (ไม่ refit) โมเดลเด่นของแต่ละ task ถูกระบุไว้ใน selection_lock.json ก่อนเปิด Test แถวอื่นในตารางเป็นข้อมูลประกอบ ห้ามเลือกโมเดลจากตารางนี้
<!-- END GENERATED -->
