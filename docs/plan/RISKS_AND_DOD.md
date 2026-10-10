# RISKS & DEFINITION OF DONE

---

## 1. Definition of Done

### 1.1 ระดับโมเดล (1 ใน 19)

ติ๊กครบ 6 ข้อเท่านั้นจึงนับว่าเสร็จ — รายละเอียดใน `AGENTS.md` §5

- [ ] ไฟล์ `src/ml/<task>/<name>.py` ทำตาม `BaseModel` ครบทุก method
- [ ] `tests/ml/<task>/test_<name>.py` ผ่าน: unit test กลไกภายใน + strict parity + save/load round-trip
- [ ] รันผ่าน trainer ได้ทั้ง `with_spike` และ `non_spike`
- [ ] artifact ครบใน `outputs/modeling/<variant>/<task>/<name>/`
- [ ] มีแถวใน leaderboard
- [ ] model card อัปเดตด้วย**ตัวเลขจริง**

### 1.2 ระดับโปรเจกต์

- [ ] 19 โมเดล × 2 variants = 38 runs ครบ
- [ ] `pytest -q` เขียวทั้งหมด
- [ ] metric ครบตามโจทย์: Loss · Confusion matrix · Accuracy · Precision · Sensitivity · Specificity · TNR · F1 · ROC+AUC · Performance curve · R²
- [ ] save → load → re-test พิสูจน์ได้ทุกโมเดล (`load_verification.json` ทุกตัว `passed: true`)
- [ ] paired comparison A↔B พร้อมบทวิเคราะห์
- [ ] notebooks 03/04/05 รันจบตั้งแต่ต้นจนจบโดยไม่ error
- [ ] report + slides เสร็จ มีตัวเลขจริงทุกช่อง
- [ ] Test ถูกแตะครั้งเดียวตอน finalize
- [ ] เอกสารทุกฉบับตรงกับโค้ด

---

## 2. Risk register

| # | ความเสี่ยง | โอกาส | ผลกระทบ | การป้องกัน | สัญญาณเตือน | แผนสำรอง |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | **XGBoost strict parity ไม่ผ่าน** — ต่างจาก sklearn/xgboost ที่ระดับ numerical detail | สูง | สูง | เริ่มวันที่ 14 แต่เช้า · ตั้ง `tree_method="exact"`, ปิด subsample/colsample, `reg_lambda` ให้ตรง · เทียบทีละ tree ไม่ใช่ทีละ prediction | ผ่านครึ่งวันแล้วยัง mismatch > 1e-3 | ลดเหลือกลไกแกน (exact greedy + gradient/hessian + leaf shrinkage) แล้วอธิบายขอบเขตในรายงานอย่างตรงไปตรงมา |
| R2 | **Core framework ช้ากว่ากำหนด** → ทั้งทีมรอ | กลาง | สูงมาก | จำกัด scope ตาม `ARCHITECTURE.md` §2 เท่านั้น · ห้ามเพิ่ม feature ระหว่างทาง | เช้าวันที่ 12 ยัง merge ไม่ได้ | ให้เพื่อนเริ่มเขียน Decision Tree แบบ standalone ไปก่อนด้วย interface ที่ระบุไว้ แล้วค่อยเสียบเข้า core |
| R3 | **kernel SVM จาก scratch** — SMO เขียนยากและ converge ช้า | กลาง | กลาง | เริ่มจาก linear SVM ให้ผ่านก่อน แล้วค่อยเพิ่ม kernel | SMO ไม่ converge ใน 2 ชม. | เหลือ linear SVM + อธิบายว่า kernel trick ทำงานอย่างไรในรายงานโดยไม่ implement |
| R4 | **merge conflict ระหว่าง 2 branch** | ต่ำ | กลาง | 1 โมเดล 1 ไฟล์ · core freeze หลัง merge · `registry.py` เรียงตามตัวอักษร | conflict ใน `core/` | แก้บน `main` แล้วให้ทั้งคู่ rebase ทันที |
| R5 | **Codex ไม่ทำตาม interface** เพราะไม่มี context ของการคุยนี้ | กลาง | กลาง | `AGENTS.md` เขียนครบและเป็นแหล่งเดียว · pilot เป็นตัวอย่างให้ดูของจริง | PR แรกของเพื่อนไม่ตรง interface | review PR แรกละเอียดเป็นพิเศษ แล้วแก้ `AGENTS.md` ให้ชัดขึ้นแทนที่จะแก้โค้ดให้เขา |
| R6 | **Test รั่วเข้า loop เลือกโมเดล** → ผลที่รายงานเป็นโมฆะ | ต่ำ | สูงมาก | `load_test()` default-deny · เรียกได้จาก `run.py finalize` เท่านั้น · finalize เป็นขั้นสุดท้ายในแผน | มีการ import `load_test` ในไฟล์โมเดล | ถ้าเกิดขึ้นจริงต้องประกาศในรายงาน ไม่ปกปิด |
| R7 | **เวลาเขียน report ไม่พอ** เพราะโมเดลกินเวลาหมด | กลาง | สูง | ร่างโครงรายงานตั้งแต่วันนี้ · ใส่ตัวเลขทีหลัง · figures ถูกสร้างอัตโนมัติจาก `figures.py` | เย็นวันที่ 16 ยังไม่ได้เริ่มร่าง | ตัด scope โมเดลตามลำดับใน `ROADMAP.md` §4 — รายงานสำคัญกว่าโมเดลตัวที่ 19 |
| R8 | **ตัวเลขในเอกสารไม่ตรงกับผลจริง** (เกิดมาแล้วรอบก่อน) | กลาง | สูง | ตัวเลขทุกตัวใน report ต้องมาจากไฟล์ใน `outputs/` · ห้ามพิมพ์มือ | เจอเลขที่หาไฟล์ต้นทางไม่เจอ | re-generate จาก artifact ก่อนส่ง |
| R9 | **Agglomerative ไม่มี predict สำหรับข้อมูลใหม่** โดยธรรมชาติ | สูง (เป็นคุณสมบัติของอัลกอริทึม) | ต่ำ | ประกาศให้ชัดว่าประเมินบน Train structure + ใช้ nearest-centroid extension สำหรับ out-of-sample | — | อธิบายข้อจำกัดในรายงาน ซึ่งเป็นเนื้อหาที่ได้คะแนน discussion อยู่แล้ว |
| R10 | **ไม่ได้ลงทะเบียนกลุ่ม/dataset ใน Google Sheet** ตามที่โจทย์หน้า 4 ระบุ | ต่ำ | สูง | ตรวจสอบกับ Google Classroom **ภายในวันที่ 11** | — | ติดต่ออาจารย์ทันที |

---

## 3. ความเสี่ยงที่ปิดไปแล้ว ✅

| เดิม | สถานะ |
| --- | --- |
| Environment ไม่ตรง / รัน pipeline ไม่ได้ | ปิด — Python 3.14.8 + 69 distributions ติดตั้งครบ `pip check` สะอาด |
| 6 tests fail | ปิด — 5 ข้อคือ cross-drive quirk (แก้ด้วย ADR-006), 1 ข้อคือ provenance ที่ซ่อมแล้ว ตอนนี้ **459 passed, 3 skipped, 0 failed** |
| ไม่รู้ว่าเอกสารเชื่อได้แค่ไหน | ปิด — `docs/audit/REALITY_AUDIT.md` |
| Deadline คลาดเคลื่อน 6 วัน | ปิด — ยืนยัน 18 ต.ค. |
| Governance layer บล็อกการเพิ่มโมเดล | ปิด — ADR-001 เลิกใช้ทั้ง package |

---

## 4. กฎการรายงานสถานะ

1. **ห้ามเคลมว่าเสร็จโดยไม่ได้รันคำสั่งและเห็น output** — เป็นกฎที่ออกมาจากบทเรียนรอบก่อนโดยตรง
2. เอกสารขัดกับโค้ด → **เชื่อโค้ด แล้วแก้เอกสาร**
3. ตัวเลขทุกตัวในรายงานต้องชี้กลับไปยังไฟล์ใน `outputs/` ได้
4. ถ้าโมเดลใดไม่ผ่าน parity ให้บันทึกเป็น **limitation ที่วิเคราะห์แล้ว** ไม่ใช่ซ่อน
   และห้ามเอาผลของ reference มาสวมแทน scratch เด็ดขาด
