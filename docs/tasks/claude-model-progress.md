# MASTER PROGRESS INDEX — 19 โมเดล

> **ดัชนีเท่านั้น** — รายละเอียดอยู่ใน `docs/models/<name>/MODEL_PROGRESS.md` ของแต่ละตัว
> วิธีทำงาน: invoke skill `qqq-model` (กลั่นจาก `AGENTS.md` + `MODEL_CARDS.md` แล้ว ไม่ต้องอ่านต้นฉบับซ้ำ)
> อัปเดตตารางนี้ + ไฟล์ของโมเดลนั้น หลังจบทุกตัว · **ห้ามลบผลเก่า**

**สถานะรวม:** 3/11 ของฝั่ง Claude · test suite `494 passed, 0 failed`

## ฝั่ง Claude (11 ตัว — ของเรา)

| # | โมเดล | task | สถานะ | parity | ผลหลัก (validation, with_spike) | บันทึก |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `multiple_linear` | reg | ✅ | exact 1.1e-15 | RMSE 0.068864 · R² **−0.0446** | [↗](../models/multiple_linear/MODEL_PROGRESS.md) |
| 2 | `simple_linear` | reg | ✅ | exact (8/8 ฟีเจอร์) | RMSE 0.069038 · R² **−0.0498** | [↗](../models/simple_linear/MODEL_PROGRESS.md) |
| 3 | `knn` | cls | ✅ | exact (8/8, held-out) | AUC **0.6987** · F1 0.2000 · Acc 0.8922 | [↗](../models/knn/MODEL_PROGRESS.md) · ปลดล็อก `stacking` แล้ว |
| 4 | `logistic` | cls | ⬜ | | | ⚠️ Codex `stacking` รออยู่ |
| 5 | `gaussian_nb` | cls | ⬜ | | | |
| 6 | `polynomial` | reg | ⬜ | | | |
| 7 | `perceptron` | cls | ⬜ | | | |
| 8 | `slp` | cls | ⬜ | | | |
| 9 | `elastic_net` ★ | reg | ⬜ | | | extracurricular — card ต้องลึก |
| 10 | `kmeans` | clustering | ⬜ | | | |
| 11 | `agglomerative` | clustering | ⬜ | | | |

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

`logistic` — class_weight ∈ {null, balanced} × l2 ∈ {0, 0.001, 0.01}
`C = 1/(λ·Σw)` · `l1_ratio=0` ไม่ใช่ `penalty="l2"` · **ปลดล็อก `stacking` ของ Codex ตัวสุดท้าย**

## ข้อค้นพบสะสม (ใช้ในรายงาน §8.2)

ทั้ง 2 โมเดล regression ที่ทำแล้วให้ **pattern เดียวกัน**: `with_spike` R² ติดลบ ·
`non_spike` R² พลิกเป็นบวก

**Regression — ตัด spike แล้วดีขึ้น**

| โมเดล | R² with_spike | R² non_spike | Δ |
| --- | --- | --- | --- |
| `multiple_linear` | −0.044554 | +0.016536 | **+0.061** |
| `simple_linear` | −0.049847 | +0.026542 | **+0.076** |

**Classification — กลับทาง ตัด spike แล้วแย่ลง**

| โมเดล | AUC with_spike | AUC non_spike | Δ |
| --- | --- | --- | --- |
| `knn` | 0.6987 | 0.6455 | **−0.053** |

สมมติฐานที่ต้องรอโมเดลอื่นยืนยัน: spike **คือ** ตัวอย่างบวกของงาน classification
(target = "ผันผวนสูงหรือไม่") ตัดออกจึงเหลือตัวอย่างบวกน้อยลงและโมเดลเรียนรู้ได้แย่ลง
ขณะที่ฝั่ง regression spike เป็น outlier ที่ลากเส้นให้เพี้ยน ตัดออกแล้วดีขึ้น
→ **เนื้อหา discussion ชั้นดีสำหรับรายงาน §8.2**

**Accuracy ใช้ไม่ได้กับชุดนี้** — validation 337 neg / 34 pos (9.2% positive)
majority baseline = **0.9084** สูงกว่าโมเดลจริงหลายตัว ต้องรายงาน AUC/F1 เป็นหลัก
