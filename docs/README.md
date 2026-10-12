# เอกสารทั้งหมดของโปรเจกต์

หน้านี้คือสารบัญของโฟลเดอร์ `docs/` ไฟล์ที่อยู่ root ของรีโปมีสามตัว:
[`README.md`](../README.md) (ทางเข้าหลัก) · [`AGENTS.md`](../AGENTS.md) (กฎการเขียนโค้ด) ·
[`CLAUDE.md`](../CLAUDE.md) (ชี้ไปที่ `AGENTS.md`)

---

## เริ่มจากตรงไหน

| ถ้าคุณคือ | อ่านตามลำดับนี้ |
| --- | --- |
| **คนที่ไม่เคยเรียน ML และอยากเข้าใจทั้งโปรเจกต์** | [`guides/PROJECT_GUIDE_TH.md`](guides/PROJECT_GUIDE_TH.md) (อธิบายจากศูนย์ มีอภิธานศัพท์) |
| **ผู้ตรวจที่อยากดูผล** | notebook [`03`](../notebooks/03_regression.ipynb) → [`04`](../notebooks/04_classification.ipynb) → [`05`](../notebooks/05_results_comparison.ipynb) และตารางใต้ `outputs/modeling/report/` |
| **คนที่จะรันโค้ดเอง** | [`guides/RUNBOOK.md`](guides/RUNBOOK.md) |
| **คนที่จะแก้โค้ด** | [`AGENTS.md`](../AGENTS.md) → [`plan/ARCHITECTURE.md`](plan/ARCHITECTURE.md) → [`plan/MODEL_CARDS.md`](plan/MODEL_CARDS.md) |
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
| [`RISKS_AND_DOD.md`](plan/RISKS_AND_DOD.md) | ความเสี่ยงและ definition of done |
| [`REPORT_AND_SLIDES_OUTLINE.md`](plan/REPORT_AND_SLIDES_OUTLINE.md) | โครงรายงานและสไลด์ ผูกกับเกณฑ์คะแนน |

### `models/` และ `tasks/` — บันทึกการเขียนโมเดล

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`tasks/claude-model-progress.md`](tasks/claude-model-progress.md) | ดัชนีสถานะของโมเดลทั้ง 19 ตัว |
| [`models/<name>/MODEL_PROGRESS.md`](models/) | บันทึกรายโมเดล: ผลจริง, ผลเทียบกับ library, ปัญหาที่เจอและวิธีแก้ |
| [`models/README.md`](models/README.md) | กฎการเขียนบันทึกรายโมเดล |

### `superpowers/` — spec และแผนทำงานแบบ test-first

[`specs/`](superpowers/specs/) design spec · [`plans/`](superpowers/plans/) แผนทำงานรายขั้น
(แผน Phase C ที่ใช้ทำ report, compare, analyze, finalize, notebook และรายงานอยู่ที่นี่)

### `archive/` — เอกสารที่ถูกแทนที่แล้ว

เก็บไว้เป็นประวัติ **ห้ามใช้เป็นคำสั่งงาน** ถ้าขัดกับเอกสารด้านบนให้ยึดเอกสารด้านบน

| ไฟล์ | เดิมคืออะไร |
| --- | --- |
| [`claude-linear-family.md`](archive/claude-linear-family.md), [`codex-tree-family.md`](archive/codex-tree-family.md) | ใบงานช่วงเขียนโมเดล เมื่อทีมยังแบ่งงานสองสาย |
| [`PROJECT_STATUS.md`](archive/PROJECT_STATUS.md), [`HANDOFF.md`](archive/HANDOFF.md) | สถานะและการส่งต่องานของชั้นข้อมูล |
| [`REALITY_AUDIT.md`](archive/REALITY_AUDIT.md) | ผลตรวจรีโป ณ 2026-10-10 (ถูกแทนด้วย `audit/DATA_PIPELINE_AUDIT.md`) |
| [`MODEL_TRAINING_PLAN.md`](archive/MODEL_TRAINING_PLAN.md) | แผนเทรนรอบก่อน |
| [`PHASE2_SPIKE_READINESS.md`](archive/PHASE2_SPIKE_READINESS.md), [`spike-analysis-plan.md`](archive/spike-analysis-plan.md) | หลักฐาน Phase 2 (spike analysis) ที่เสร็จแล้ว |
