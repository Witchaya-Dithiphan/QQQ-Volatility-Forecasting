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

## ฝั่ง Codex (8 ตัว — ห้ามแตะ)

`decision_tree` → `random_forest` · `adaboost` ★ · `gradient_boosting` → `xgboost` ·
`stacking` · `svm` · `mlp`

เจ้าของดูแล progress เอง บน branch `feat/models-tree-family`

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
