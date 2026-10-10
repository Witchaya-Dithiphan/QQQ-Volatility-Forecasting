# ROADMAP — 10 → 18 ตุลาคม 2026

> **กำหนดส่ง 18 ต.ค. 2026** (เลื่อนจาก 12 ต.ค. ที่พิมพ์ใน PDF — ยืนยันแล้ว)
> เหลือ 9 วันปฏิทิน · ทีม 2 คน · ต้องส่ง: report, slides + นำเสนอ on-site, .ipynb/.py, เอกสารอื่น

---

## 1. Critical path

```
core framework  ──►  pilot ผ่าน  ──►  merge main  ──┬──► โมเดล 11 ตัว (Claude)  ──┐
   (10-11)           (11)            (12)          └──► โมเดล 8 ตัว (Codex)   ──┤
                                                                                 ▼
                                                        variant B + leaderboard (16)
                                                                                 ▼
                                                        notebooks + figures (17)
                                                                                 ▼
                                                        report + slides + ซ้อม (17-18)
```

**จุดคอขวดเดียวคือ `core/` ที่ต้องเสร็จและ merge ให้ได้ภายในวันที่ 12** ทุกอย่างหลังจากนั้นขนานกันได้
ถ้า core ช้า 1 วัน ทั้งโปรเจกต์ช้า 1 วัน — จึงต้องไม่เพิ่ม scope ให้ core เกินที่ระบุใน `ARCHITECTURE.md` §2

---

## 2. แผนรายวัน

| วัน | Claude Code (คุณ) | Codex CLI (เพื่อน) | Gate ที่ต้องผ่านก่อนจบวัน |
| --- | --- | --- | --- |
| **10 ศ.** | แผน + audit ✅ · ล้าง `src/modeling` · ย้ายโค้ดคณิต · `base.py`, `data.py`, `compare.py` | อ่าน `AGENTS.md` + `MODEL_CARDS.md` · ตั้ง environment · ลองเขียน Decision Tree stump ไว้ก่อนแบบ local | `pytest -q` เขียวหลังย้ายโค้ด |
| **11 ส.** | `trainer.py`, `registry.py`, `run.py`, `figures.py` · **pilot Multiple Linear ครบลูป** | เตรียม decision tree ให้พร้อม merge ทันทีที่ core เข้า main | pilot ผลิต artifact ครบ 8 ไฟล์ + ROC/perf curve จริง |
| **12 อา.** | merge `feat/ml-core` → `main` · เริ่ม Linear, Multiple, Polynomial, Elastic Net★ | rebase · เริ่ม Decision Tree, Random Forest | core freeze · ทั้งคู่ rebase แล้ว |
| **13 จ.** | Logistic, Naive Bayes, k-NN | Gradient Boosting, AdaBoost★ | ≥ 4 โมเดลต่อคนผ่าน parity |
| **14 อ.** | Perceptron, SLP | XGBoost (ตัวเสี่ยงสุด เริ่มแต่เช้า) | XGBoost parity ผ่านหรือเข้าแผนสำรอง |
| **15 พ.** | k-Means, Agglomerative · เก็บตกของตัวเอง | SVM (+PCA), Stacking | **19/19 ผ่าน variant A** |
| **16 พฤ.** | รัน variant B ทั้งชุด · paired comparison A↔B · leaderboard | MLP · ช่วยเก็บตก | ตาราง 38 runs ครบ |
| **17 ศ.** | notebooks 03/04/05 · figures ครบ · **finalize Test ครั้งเดียว** · ร่าง report เต็ม | ตรวจทาน model cards ให้ตรงผลจริง · ช่วยรูป | รายงานมีตัวเลขจริงทุกช่อง |
| **18 ส.** | slides · final verification · PR + merge | ซ้อมนำเสนอ · ตรวจ deliverable ครบ | ✅ ส่ง |

วันที่ 10–11 เป็นงานของผมล้วน เพื่อนยังเริ่มโมเดลจริงไม่ได้เพราะ core ยังไม่นิ่ง — ช่วงนี้ให้เพื่อน
ตั้ง environment และอ่าน spec ให้จบ จะได้ลงมือได้ทันทีในวันที่ 12

---

## 3. งานที่ขนานได้ vs ต้องเรียงลำดับ

**ต้องเรียง (ห้ามข้าม):**
core เสร็จ → pilot ผ่าน → merge → เขียนโมเดล → variant B → leaderboard → report
และ **finalize Test ต้องเป็นขั้นตอนสุดท้ายหลังเลือกโมเดลครบทุกตัวแล้ว**

**ขนานได้:**
โมเดล 11 ตัว ↔ โมเดล 8 ตัว · notebook ↔ model cards · slides ↔ final verification ·
รูปแต่ละชุด ↔ กันเอง

---

## 4. Buffer และแผนเมื่อช้ากว่ากำหนด

มี buffer จริง **1 วัน** (18 ต.ค. เป็นวันส่ง ไม่ใช่วันทำงานเต็ม) ถ้าถึงเช้าวันที่ 16 แล้วยังไม่ครบ 19/19
ให้ตัดตามลำดับนี้ โดย**ห้ามตัดข้ามลำดับ**:

| ลำดับตัด | ตัดอะไร | ผลกระทบต่อคะแนน |
| --- | --- | --- |
| 1 | hyperparameter grid ให้เล็กลง (เหลือ 2-3 candidate ต่อโมเดล) | แทบไม่กระทบ — โจทย์ไม่ได้ให้คะแนนขนาด grid |
| 2 | optional stability runs (หลาย seed) | ไม่กระทบ ไม่ใช่ข้อบังคับ |
| 3 | kernel SVM → เหลือ linear SVM | กระทบเล็กน้อย ยังตอบ "SVM" ได้ |
| 4 | Stacking → ใช้ base learner น้อยลง | กระทบเล็กน้อย |
| 5 | XGBoost → ลดให้เหลือกลไกแกน (exact greedy, ไม่ทำ sparsity-aware/approximate split) | ยังนับเป็น XGBoost ได้ ต้องอธิบายขอบเขตในรายงาน |
| 6 | variant B เหลือเฉพาะโมเดลที่น่าสนใจ ไม่ครบ 19 | กระทบ paired comparison — **ต้องแจ้งเจ้าของโปรเจกต์ก่อน** |

**ห้ามตัด** (เป็นข้อบังคับของโจทย์โดยตรง): save/load + re-test · metric ครบชุด · ROC+AUC ·
performance curve · โมเดล extracurricular 2 ตัวพร้อมคำอธิบายกลไก · benchmarking + discussion

---

## 5. Checkpoint ที่ต้องรายงานสถานะจริง

จบวันที่ **12, 15 และ 17** ให้อัปเดต `PROJECT_STATUS.md` ด้วย
- จำนวนโมเดลที่ผ่าน DoD ครบ 6 ข้อ (ไม่ใช่ "เขียนเสร็จแล้ว")
- ผล `pytest -q` จริง
- โมเดลที่ติดปัญหาและเหตุผล

กฎเดิมยังใช้: **ห้ามเขียนว่าเสร็จถ้ายังไม่ได้รันแล้วเห็น output**
