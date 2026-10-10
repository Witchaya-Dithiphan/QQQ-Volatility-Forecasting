# CLAUDE.md

**อ่าน [`AGENTS.md`](AGENTS.md) เป็นหลัก** — เป็น contract กลางฉบับเดียวของโปรเจกต์นี้
ทีมมี 2 คนใช้ agent คนละตัว (Claude Code และ Codex CLI) จึงตั้งใจเก็บกฎไว้ที่ไฟล์เดียว
เพื่อไม่ให้มีเอกสาร 2 ชุดที่ขัดกันเอง **ห้ามย้ายกฎมาเขียนซ้ำที่นี่**

เอกสารประกอบ:

| ไฟล์ | ใช้เมื่อ |
| --- | --- |
| [`AGENTS.md`](AGENTS.md) | ก่อนเขียนโค้ดทุกครั้ง — interface, กฎ, การแบ่งงาน, DoD |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | อยากรู้ว่าทำไมออกแบบแบบนี้ (ADR) |
| [`MODEL_CARDS.md`](MODEL_CARDS.md) | ก่อนลงมือเขียนโมเดลตัวใดตัวหนึ่ง |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | หาที่วางไฟล์ / รู้ว่าอะไรห้ามแตะ |
| [`ROADMAP.md`](ROADMAP.md) | ลำดับงานและสิ่งที่ตัดได้เมื่อเวลาไม่พอ |
| [`RISKS_AND_DOD.md`](RISKS_AND_DOD.md) | เช็คก่อนบอกว่าเสร็จ |
| [`docs/audit/REALITY_AUDIT.md`](docs/audit/REALITY_AUDIT.md) | สถานะจริง ณ 2026-10-10 (เอกสารเก่าหลายฉบับไม่ตรง) |
| [`RUNBOOK.md`](RUNBOOK.md) | คำสั่งติดตั้งและรัน pipeline |

## เตือนสั้น ๆ

- ชั้นข้อมูล (`data/**` และ data pipeline ใน `src/`) **เสร็จแล้ว ห้ามแก้**
- ทุกโมเดลต้องเขียนเองด้วย NumPy · library ใช้ได้เฉพาะเป็น reference เพื่อยืนยันผล
- ห้ามแตะ Test set จนถึงขั้น finalize
- ห้ามเคลมว่าเสร็จโดยไม่ได้รันคำสั่งและแสดง output จริง
- ใช้ `.\.venv\Scripts\python.exe` เสมอ (Python 3.14.8)
