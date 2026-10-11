# เอกสารทั้งหมดของโปรเจกต์

ไฟล์ที่อยู่ root มีแค่ 3 ตัวเพราะเป็น convention ของเครื่องมือ:
[`README.md`](../README.md) · [`AGENTS.md`](../AGENTS.md) (Codex CLI อ่านชื่อนี้ที่ root) ·
[`CLAUDE.md`](../CLAUDE.md) (Claude Code) ที่เหลืออยู่ในโฟลเดอร์นี้

---

## เริ่มจากตรงไหน

| ถ้าคุณคือ | อ่านตามลำดับนี้ |
| --- | --- |
| **ผู้พัฒนาที่เพิ่งเข้าโปรเจกต์** | [`AGENTS.md`](../AGENTS.md) → [`guides/RUNBOOK.md`](guides/RUNBOOK.md) → task ของตัวเองใน [`tasks/`](tasks/) |
| **จะเขียนโมเดล** | [`tasks/claude-model-progress.md`](tasks/claude-model-progress.md) (ดัชนี) → [`models/<name>/MODEL_PROGRESS.md`](models/) → section ของโมเดลนั้นใน [`plan/MODEL_CARDS.md`](plan/MODEL_CARDS.md) |
| **อยากรู้ว่าทำไมออกแบบแบบนี้** | [`plan/ARCHITECTURE.md`](plan/ARCHITECTURE.md) (ADR-001…006) |
| **อยากรู้สถานะจริง** | [`tasks/claude-model-progress.md`](tasks/claude-model-progress.md) (โมเดล) และ [`audit/DATA_PIPELINE_AUDIT.md`](audit/DATA_PIPELINE_AUDIT.md) (ข้อมูล) ⚠️ แก้ไข 2026-10-11 |
| **จะเขียนรายงาน/สไลด์** | [`plan/REPORT_AND_SLIDES_OUTLINE.md`](plan/REPORT_AND_SLIDES_OUTLINE.md) |

---

## โครงสร้าง

### `tasks/` — งานที่ต้องทำ แบ่งตามคน

| ไฟล์ | สำหรับ |
| --- | --- |
| [`claude-model-progress.md`](tasks/claude-model-progress.md) | **ดัชนีหลัก** — สถานะ 19 โมเดล + dependency + ลำดับถัดไป |
| [`claude-linear-family.md`](tasks/claude-linear-family.md) | core framework + 11 โมเดล (linear / probabilistic / clustering) |
| [`codex-tree-family.md`](tasks/codex-tree-family.md) | 8 โมเดล (tree / ensemble / margin) — เขียนให้ Codex CLI ใช้ได้ทันที |

### `models/` — progress ต่อโมเดล (19 ไฟล์)

`models/<name>/MODEL_PROGRESS.md` — สถานะ · ผลจริง · parity · **ปัญหาที่เจอและแก้แล้ว** · สิ่งที่ค้าง
แยกไฟล์เพื่อให้หยิบงานต่อได้โดยไม่ต้องโหลดเอกสารทั้งโปรเจกต์ · กฎการเขียนอยู่ใน
[`models/README.md`](models/README.md)

### `plan/` — แผนและสเปก

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`ARCHITECTURE.md`](plan/ARCHITECTURE.md) | สถาปัตยกรรม `src/ml/` + ADR 6 ข้อพร้อมเหตุผล |
| [`MODEL_CARDS.md`](plan/MODEL_CARDS.md) | สเปก 19 โมเดล — สมการ, grid, reference, กับดัก parity, unit test |
| [`PROJECT_STRUCTURE.md`](plan/PROJECT_STRUCTURE.md) | โครงไฟล์เป้าหมาย + ตารางย้าย/ลบ |
| [`ROADMAP.md`](plan/ROADMAP.md) | แผนรายวันถึง 18 ต.ค. + ลำดับสิ่งที่ตัดได้เมื่อเวลาไม่พอ |
| [`RISKS_AND_DOD.md`](plan/RISKS_AND_DOD.md) | ความเสี่ยง R1–R10 + definition of done |
| [`REPORT_AND_SLIDES_OUTLINE.md`](plan/REPORT_AND_SLIDES_OUTLINE.md) | โครงรายงาน + 22 สไลด์ + ตาราง traceability ผูกกับเกณฑ์คะแนน |

### `guides/` — วิธีใช้งาน

[`RUNBOOK.md`](guides/RUNBOOK.md) — ติดตั้ง environment, รัน data pipeline, แก้ปัญหาที่พบบ่อย

### `status/` — สถานะงาน

[`PROJECT_STATUS.md`](status/PROJECT_STATUS.md) (สถานะ implementation) ·
[`HANDOFF.md`](status/HANDOFF.md) (รายละเอียดการส่งต่องานและ data contract)

### `audit/` — หลักฐานการตรวจสอบ

⚠️ แก้ไข 2026-10-11:

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`DATA_PIPELINE_AUDIT.md`](audit/DATA_PIPELINE_AUDIT.md) | **ผลตรวจชั้นข้อมูลล่าสุด (2026-10-11)** — คำตัดสิน, findings รายข้อพร้อมหลักฐาน, ข้อที่ต้องเขียนในรายงาน |
| [`DATA_FLOW.md`](audit/DATA_FLOW.md) | แผนภาพ raw → split → spike datasets พร้อมจำนวนแถวจริงทุกขั้น |
| [`PATH_AUDIT.md`](audit/PATH_AUDIT.md) | path ทุกตัว, line ending กับ checksum |
| [`FILE_INVENTORY.md`](audit/FILE_INVENTORY.md) | ไฟล์ทั้งหมดพร้อมคำตัดสินรายไฟล์ |
| [`REALITY_AUDIT.md`](audit/REALITY_AUDIT.md) | snapshot ทางประวัติศาสตร์ ณ 2026-10-10 (commit `779c78d`) — **ไม่ใช่สถานะปัจจุบันแล้ว** ตัวเลขชั้นข้อมูลยังถูก ส่วนสถานะโมเดลล้าสมัย |

### `superpowers/` — spec และ plan ที่สร้างโดย workflow skill

[`specs/`](superpowers/specs/) design spec · [`plans/`](superpowers/plans/) implementation plan
แบบ TDD ที่มีโค้ดจริงทุกขั้น

### `archive/` — เอกสารที่ถูกแทนที่แล้ว

[`MODEL_TRAINING_PLAN.md`](archive/MODEL_TRAINING_PLAN.md) (แผนรอบก่อน — inventory และ grid ยังใช้ได้
แต่ milestone/lifecycle เลิกใช้แล้ว) · [`PHASE2_SPIKE_READINESS.md`](archive/PHASE2_SPIKE_READINESS.md)
และ [`spike-analysis-plan.md`](archive/spike-analysis-plan.md) (หลักฐาน Phase 2 ที่เสร็จแล้ว)

> เอกสารใน `archive/` เก็บไว้เป็นหลักฐานและประวัติ **ห้ามใช้เป็นคำสั่งงาน** ถ้าขัดกับ `plan/` ให้ยึด `plan/`
