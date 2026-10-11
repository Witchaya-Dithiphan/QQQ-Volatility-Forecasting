# CLAUDE.md

**อ่าน [`AGENTS.md`](AGENTS.md) เป็นหลัก** — เป็น contract กลางฉบับเดียวของโปรเจกต์นี้
กฎทั้งหมดเก็บไว้ที่ไฟล์นั้นไฟล์เดียวเพื่อไม่ให้มีเอกสารสองชุดที่ขัดกันเอง
**ห้ามย้ายกฎมาเขียนซ้ำที่นี่**

**สถานะงาน (2026-10-11):** โมเดลทั้ง 19 ตัวเขียนและเทรนเสร็จแล้ว ตอนนี้อยู่ Phase C
(สรุปผล, notebook, รายงาน) ทีมทำงานร่วมกันบน branch เดียว ไม่มีการแบ่งเขตเจ้าของไฟล์แล้ว
แผนงานอยู่ที่ [`docs/superpowers/plans/2026-10-11-phase-c-report-and-finalize.md`](docs/superpowers/plans/2026-10-11-phase-c-report-and-finalize.md)

เอกสารประกอบ — ดัชนีเต็มที่ [`docs/README.md`](docs/README.md):

| ไฟล์ | ใช้เมื่อ |
| --- | --- |
| [`AGENTS.md`](AGENTS.md) | ก่อนเขียนโค้ดทุกครั้ง — interface, กฎ, artifact layout, คำสั่ง |
| [`ARCHITECTURE.md`](docs/plan/ARCHITECTURE.md) | อยากรู้ว่าทำไมออกแบบแบบนี้ (ADR) |
| [`MODEL_CARDS.md`](docs/plan/MODEL_CARDS.md) | ก่อนแก้โมเดลตัวใดตัวหนึ่ง |
| [`PROJECT_STRUCTURE.md`](docs/plan/PROJECT_STRUCTURE.md) | หาที่วางไฟล์ / รู้ว่าอะไรห้ามแตะ |
| [`RISKS_AND_DOD.md`](docs/plan/RISKS_AND_DOD.md) | เช็คก่อนบอกว่าเสร็จ |
| [`DATA_PIPELINE_AUDIT.md`](docs/audit/DATA_PIPELINE_AUDIT.md) | ผลตรวจชั้นข้อมูล ณ 2026-10-11 |
| [`RUNBOOK.md`](docs/guides/RUNBOOK.md) | คำสั่งติดตั้งและรัน pipeline |
| [`PROJECT_GUIDE_TH.md`](docs/guides/PROJECT_GUIDE_TH.md) | คำอธิบายทั้งโปรเจกต์จากศูนย์ |

## เตือนสั้น ๆ

- ชั้นข้อมูล (`data/**` และ data pipeline ใน `src/`) **เสร็จแล้ว ห้ามแก้**
- ทุกโมเดลต้องเขียนเองด้วย NumPy · library ใช้ได้เฉพาะเป็น reference เพื่อยืนยันผล
- **Test set อ่านได้ครั้งเดียว** ผ่าน `finalize --allow-test` และต้องได้รับอนุมัติจากเจ้าของโปรเจกต์ก่อน
  ถ้ามี `outputs/modeling/finalize_receipt.json` แปลว่าอ่านไปแล้ว ห้ามรันซ้ำ
- ห้ามเคลมว่าเสร็จโดยไม่ได้รันคำสั่งและแสดง output จริง
- ใช้ `.\.venv\Scripts\python.exe` เสมอ (Python 3.14.8)
