# เอกสารทั้งหมดของโปรเจกต์

หน้านี้คือสารบัญของโฟลเดอร์ `docs/` ทางเข้าหลักของรีโปคือ [`README.md`](../README.md) ที่ root

---

## เริ่มจากตรงไหน

| ถ้าคุณคือ | อ่านตามลำดับนี้ |
| --- | --- |
| **คนที่ไม่เคยเรียน ML และอยากเข้าใจทั้งโปรเจกต์** | [`guides/PROJECT_GUIDE_TH.md`](guides/PROJECT_GUIDE_TH.md) (อธิบายจากศูนย์ มีอภิธานศัพท์) |
| **ผู้ตรวจที่อยากดูผล** | notebook [`03`](../notebooks/03_regression.ipynb) → [`04`](../notebooks/04_classification.ipynb) → [`05`](../notebooks/05_results_comparison.ipynb) → [`06`](../notebooks/06_spike_analysis.ipynb) และตารางใต้ `outputs/modeling/report/` |
| **คนที่อยากรู้ว่าเกณฑ์ spike คืออะไรและทำไมเลือกเกณฑ์นี้** | notebook [`06_spike_analysis.ipynb`](../notebooks/06_spike_analysis.ipynb) |
| **คนที่จะรันโค้ดเอง** | [`guides/RUNBOOK.md`](guides/RUNBOOK.md) |
| **คนที่อยากรู้ว่าโมเดลออกแบบอย่างไร** | [`plan/ARCHITECTURE.md`](plan/ARCHITECTURE.md) → [`plan/MODEL_CARDS.md`](plan/MODEL_CARDS.md) |
| **คนที่อยากรู้ว่าข้อมูลเชื่อได้แค่ไหน** | [`audit/DATA_PIPELINE_AUDIT.md`](audit/DATA_PIPELINE_AUDIT.md) |

---

## โครงสร้าง

### `guides/` — วิธีใช้งานและคำอธิบาย

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`PROJECT_GUIDE_TH.md`](guides/PROJECT_GUIDE_TH.md) | คู่มือฉบับเดียวจบจากศูนย์: โจทย์, การเตรียมข้อมูล 8 ขั้น, การทดลอง spike, โมเดล 19 ตัว, วิธีอ่าน metric, ผลบน Validation และ Test, ข้อสรุป, คำถามที่น่าจะถูกถาม, สูตรทุกสูตรพร้อมคำอ่าน, ตัวย่อและอภิธานศัพท์ |
| [`RUNBOOK.md`](guides/RUNBOOK.md) | ติดตั้ง environment, รัน data pipeline, เทรน, สรุปผล, finalize, แก้ปัญหาที่พบบ่อย |

### `audit/` — หลักฐานการตรวจสอบชั้นข้อมูล (2026-10-11)

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`DATA_PIPELINE_AUDIT.md`](audit/DATA_PIPELINE_AUDIT.md) | คำตัดสิน, findings รายข้อพร้อมหลักฐาน, ข้อที่ต้องเขียนในรายงาน |
| [`DATA_FLOW.md`](audit/DATA_FLOW.md) | แผนภาพ raw → split → spike datasets พร้อมจำนวนแถวจริงทุกขั้น |
| [`PATH_AUDIT.md`](audit/PATH_AUDIT.md) | path ทุกตัว, line ending กับ checksum |
| [`FILE_INVENTORY.md`](audit/FILE_INVENTORY.md) | ไฟล์ทั้งหมดพร้อมคำตัดสินรายไฟล์ |

### `plan/` — การออกแบบและสเปก

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`ARCHITECTURE.md`](plan/ARCHITECTURE.md) | สถาปัตยกรรม `src/ml/` และ ADR (การตัดสินใจพร้อมเหตุผล) |
| [`MODEL_CARDS.md`](plan/MODEL_CARDS.md) | สเปก 19 โมเดล: สมการ, grid, library ที่ใช้เทียบ, กับดัก parity |
| [`PROJECT_STRUCTURE.md`](plan/PROJECT_STRUCTURE.md) | โครงไฟล์ของรีโป |
| [`ROADMAP.md`](plan/ROADMAP.md) | แผนรายวันถึงกำหนดส่ง 18 ต.ค. 2026 |
| [`TIME_SERIES_TRANSFER_LEARNING_PLAN.md`](plan/TIME_SERIES_TRANSFER_LEARNING_PLAN.md) | แผนการทดลอง transfer learning หลังนำเสนอ (ผลอยู่ที่ `models/transfer_learning/`) |
| [`RISKS_AND_DOD.md`](plan/RISKS_AND_DOD.md) | ความเสี่ยงและ definition of done |
| [`REPORT_AND_SLIDES_OUTLINE.md`](plan/REPORT_AND_SLIDES_OUTLINE.md) | โครงรายงานและสไลด์ ผูกกับเกณฑ์คะแนน |

### `models/` — บันทึกการเขียนโมเดล

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`models/PROGRESS_INDEX.md`](models/PROGRESS_INDEX.md) | ดัชนีสถานะของโมเดลทั้ง 19 ตัว พร้อมผลเทียบกับ library |
| [`models/<name>/MODEL_PROGRESS.md`](models/) | บันทึกรายโมเดล: ผลจริง, ผลเทียบกับ library, ปัญหาที่เจอและวิธีแก้ |
| [`models/README.md`](models/README.md) | รูปแบบของบันทึกรายโมเดล |
| [`models/transfer_learning/EXPERIMENT.md`](models/transfer_learning/EXPERIMENT.md) | การทดลองเสริมหลังนำเสนอ: SSL → DAPT → fine-tune Transformer/GRU จาก ETF 10 ตัวสู่ QQQ (ทั้งสอง variants) |

### `report/` — ต้นฉบับของรายงาน

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`REPORT_DRAFT.md`](report/REPORT_DRAFT.md) | ต้นฉบับที่ใช้สร้าง `deliverables/report.docx` และ `report.pdf` |
| [`TRACEABILITY.md`](report/TRACEABILITY.md) | ตารางจับคู่ข้อกำหนดของโจทย์กับหัวข้อรายงาน สไลด์ และไฟล์หลักฐาน |
| [`README.md`](report/README.md) | วิธีสร้างรายงานจากต้นฉบับ |

### `archive/` — เอกสารที่ถูกแทนที่แล้ว

เก็บไว้เป็นประวัติ ถ้าขัดกับเอกสารด้านบนให้ยึดเอกสารด้านบน

| ไฟล์ | เดิมคืออะไร |
| --- | --- |
| [`PROJECT_STATUS.md`](archive/PROJECT_STATUS.md), [`HANDOFF.md`](archive/HANDOFF.md) | สถานะและการส่งต่องานของชั้นข้อมูล |
| [`REALITY_AUDIT.md`](archive/REALITY_AUDIT.md) | ผลตรวจรีโป ณ 2026-10-10 (ถูกแทนด้วย `audit/DATA_PIPELINE_AUDIT.md`) |
| [`MODEL_TRAINING_PLAN.md`](archive/MODEL_TRAINING_PLAN.md) | แผนเทรนรอบก่อน |
| [`README_2026-10-10.md`](archive/README_2026-10-10.md) | README ของรีโปฉบับวันที่ 10 ต.ค. 2026 ก่อนเขียนใหม่ |
| [`REPORT_DRAFT_previous_format.md`](archive/REPORT_DRAFT_previous_format.md) | ต้นฉบับรายงานรูปแบบเดิม ก่อนจัดเป็นรูปแบบรายงานวิชาการ (ไฟล์ที่สร้างแล้วอยู่ที่ `deliverables/previous_version/`) |
| [`PHASE2_SPIKE_READINESS.md`](archive/PHASE2_SPIKE_READINESS.md), [`spike-analysis-plan.md`](archive/spike-analysis-plan.md) | หลักฐาน Phase 2 (spike analysis) ที่เสร็จแล้ว |
