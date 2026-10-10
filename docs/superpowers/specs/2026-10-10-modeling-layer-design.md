# Design Spec — ชั้น modeling ของ QQQ Volatility Forecasting

**วันที่:** 2026-10-10 · **สถานะ:** อนุมัติแล้ว · **กำหนดส่ง:** 2026-10-18

เอกสารนี้บันทึก **ข้อกำหนดและการตัดสินใจ** ที่ได้จากรอบ brainstorming
ส่วนดีไซน์เชิงเทคนิคอยู่ใน `ARCHITECTURE.md` · กฎปฏิบัติอยู่ใน `AGENTS.md`

---

## 1. ปัญหาที่กำลังแก้

โปรเจกต์วิชา ML ต้องส่ง 18 ต.ค. 2026 ชั้นข้อมูลเสร็จแล้ว (cleaning, features, targets, splits,
Q75 labels, spike detection, 2 dataset variants) แต่ **ชั้นโมเดลเป็นศูนย์** — ต้องสร้าง 19 โมเดล
จากศูนย์ด้วย NumPy ภายใน 8 วัน โดยทีม 2 คน พร้อมรายงานและสไลด์

ความพยายามรอบก่อนลง infrastructure 1,930 บรรทัดก่อนมีโมเดลแม้ตัวเดียว แล้วหยุดไป
สมมติฐานที่ตรวจสอบแล้วว่าจริง: over-engineering คือสาเหตุ (ดู `docs/audit/REALITY_AUDIT.md`)

---

## 2. ข้อกำหนดจากโจทย์ (สิ่งที่ต้องส่งและให้คะแนน)

| ข้อกำหนด | ที่มา |
| --- | --- |
| ทุกโมเดลสร้างเองจากศูนย์ library ใช้ประกอบได้ "เพื่อยืนยันผล" | PDF หน้า 2, 5 |
| build · training-testing iterations · batch runs · **save-load model แล้ว test** | PDF หน้า 2 |
| Regression: Linear, Multiple, Polynomial + 1 ตัวนอกห้องเรียน | PDF หน้า 2 |
| Classification: Logistic, DT, RF, Stacking, GB, XGBoost, NB, SVM, dim-reduction สำหรับ RF/SVM, kNN, k-Means, Agglomerative, Perceptron, SLP, MLP + 1 ตัวนอกห้องเรียน (ต่างจากฝั่ง regression) | PDF หน้า 2 |
| โมเดลนอกห้องเรียน 2 ตัวต้องทำ **ทั้ง scratch และ ready-to-use** + อธิบายกลไกภายในละเอียด | PDF หน้า 3, 5 |
| Metrics: Loss, Confusion matrix, Accuracy, Precision, Sensitivity, Specificity, TNR, F1, ROC+AUC, Performance curve, R² | PDF หน้า 3 |
| **Quantitative results, benchmarking, analysis, and discussion** | PDF หน้า 3 |
| ใช้ feature ≥ 5 (เรามี 8 ✅) · storytelling + feature insight · ตอบ "why must use ML/DL" | PDF หน้า 1, 5 |
| ส่ง: report · slides + นำเสนอ on-site · .ipynb และ/หรือ .py · เอกสารอื่น · demo (ถ้ามี) | PDF หน้า 1, 5 |
| คะแนน: 30 (problem/feature/data prep) + 40 (models) + 30 (evaluation) + 5 creative + 5 much-better | PDF หน้า 5 |

---

## 3. การตัดสินใจที่ตกลงแล้ว

| # | ประเด็น | มติ | เหตุผลย่อ |
| --- | --- | --- | --- |
| D1 | Dataset B สร้างอย่างไร | ใช้ `non_spike` train เดิม (1,520 rows) | ผ่านการตรวจรับแล้ว มี SHA-256 ครบ · เปลี่ยนตัวแปรเดียวจริง · ไม่ต้องแตะ data layer ที่ freeze |
| D2 | นิยาม scratch ตรงกับ reference | **Strict เท่ากันทุกโมเดล** | การแบ่ง tier เปิดช่องให้ลดเกณฑ์จนผ่าน ซึ่งทำลายคุณค่าของการมี reference → ADR-002 |
| D3 | Deadline | 18 ต.ค. (เลื่อนจาก 12 ต.ค. ใน PDF) | ยืนยันโดยเจ้าของโปรเจกต์ |
| D4 | `src/modeling/` | เลิกใช้ทั้ง package ย้ายเฉพาะโค้ดคณิต | มี deadlock จริง (`code_snapshot_hash` invalidate ผลเก่าทุกครั้งที่เพิ่มโมเดล) → ADR-001 |
| D5 | ทีม 2 คน | แบ่งตามกลุ่มอัลกอริทึม คนละ branch · ผมวาง core ก่อน | ไฟล์ไม่ทับกัน · เพื่อนใช้ **Codex CLI** จึงต้องมี `AGENTS.md` เป็น contract กลาง |
| D6 | โมเดลนอกห้องเรียน | **Elastic Net + AdaBoost** | มีใน sklearn → ทำ "scratch + ready-to-use" ได้โดยไม่แตะ frozen environment · มีเหตุผลเชิงข้อมูลหนุน → ADR-004 |
| D7 | Report/Slides | ภาษาไทย ศัพท์เทคนิคอังกฤษ · ผมร่างเต็ม ทีมขัดเกลา | ตรงกับ PDF ที่เป็นสองภาษาและเอกสารเดิมในรีโป |

---

## 4. ขอบเขต

**อยู่ในขอบเขต:** `src/ml/` ทั้งหมด · 19 โมเดล × 2 variants · metrics + figures + leaderboard ·
save/load + re-test · paired comparison A↔B · notebooks 03/04/05 · report + slides ·
การแก้เอกสารเดิมที่ขัดกับโค้ด

**นอกขอบเขต:** ชั้นข้อมูลทั้งหมด (frozen) · cross-validation · automated hyperparameter search ·
model registry/versioning · experiment tracking server · CI · resume/checkpoint · GARCH (stretch goal เท่านั้น)

---

## 5. เกณฑ์ความสำเร็จ

1. 38 runs ครบ และทุก run มี `load_verification.json` ที่ `passed: true`
2. `pytest -q` เขียว
3. ทุกโมเดลผ่าน strict parity หรือมี limitation ที่วิเคราะห์และบันทึกไว้อย่างตรงไปตรงมา
4. metric ครบตามโจทย์ทุกตัว
5. ตัวเลขทุกตัวในรายงานชี้กลับไปยังไฟล์ใน `outputs/` ได้
6. Test set ถูกแตะครั้งเดียวในขั้น finalize
7. ส่งทันวันที่ 18 ต.ค.

---

## 6. สมมติฐานที่ถ้าผิดต้องกลับมาทบทวนแผน

- Deadline คือ 18 ต.ค. จริง (ถ้ากลับเป็น 12 ต.ค. ต้อง triage ใหม่ทั้งหมดตาม `ROADMAP.md` §4)
- เพื่อนร่วมทีมเริ่มลงมือได้ตั้งแต่วันที่ 12 ตามแผน
- อาจารย์ไม่ได้สอน Elastic Net และ AdaBoost (ยืนยันโดยเจ้าของโปรเจกต์) จึงนับเป็น "beyond classroom"
- `configs/modeling.json` ที่มีอยู่ใช้เป็นแหล่ง hyperparameter grid ได้โดยไม่ต้องออกแบบใหม่
