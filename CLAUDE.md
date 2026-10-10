# CLAUDE.md

**อ่าน [`AGENTS.md`](AGENTS.md) เป็นหลัก** — เป็น contract กลางฉบับเดียวของโปรเจกต์นี้
ทีมมี 2 คนใช้ agent คนละตัว (Claude Code และ Codex CLI) จึงตั้งใจเก็บกฎไว้ที่ไฟล์เดียว
เพื่อไม่ให้มีเอกสาร 2 ชุดที่ขัดกันเอง **ห้ามย้ายกฎมาเขียนซ้ำที่นี่**

**งานของฝั่งนี้:** [`docs/tasks/claude-linear-family.md`](docs/tasks/claude-linear-family.md)
— core framework (คอขวดของทั้งทีม) + 11 โมเดล linear/probabilistic/clustering
ส่วนเพื่อนร่วมทีมใช้ Codex ทำ [`docs/tasks/codex-tree-family.md`](docs/tasks/codex-tree-family.md)

เอกสารประกอบ — ดัชนีเต็มที่ [`docs/README.md`](docs/README.md):

| ไฟล์ | ใช้เมื่อ |
| --- | --- |
| [`AGENTS.md`](AGENTS.md) | ก่อนเขียนโค้ดทุกครั้ง — interface, กฎ, การแบ่งงาน, DoD |
| [`ARCHITECTURE.md`](docs/plan/ARCHITECTURE.md) | อยากรู้ว่าทำไมออกแบบแบบนี้ (ADR) |
| [`MODEL_CARDS.md`](docs/plan/MODEL_CARDS.md) | ก่อนลงมือเขียนโมเดลตัวใดตัวหนึ่ง |
| [`PROJECT_STRUCTURE.md`](docs/plan/PROJECT_STRUCTURE.md) | หาที่วางไฟล์ / รู้ว่าอะไรห้ามแตะ |
| [`ROADMAP.md`](docs/plan/ROADMAP.md) | ลำดับงานและสิ่งที่ตัดได้เมื่อเวลาไม่พอ |
| [`RISKS_AND_DOD.md`](docs/plan/RISKS_AND_DOD.md) | เช็คก่อนบอกว่าเสร็จ |
| [`docs/audit/REALITY_AUDIT.md`](docs/audit/REALITY_AUDIT.md) | สถานะจริง ณ 2026-10-10 (เอกสารเก่าหลายฉบับไม่ตรง) |
| [`RUNBOOK.md`](docs/guides/RUNBOOK.md) | คำสั่งติดตั้งและรัน pipeline |

## เตือนสั้น ๆ

- ชั้นข้อมูล (`data/**` และ data pipeline ใน `src/`) **เสร็จแล้ว ห้ามแก้**
- ทุกโมเดลต้องเขียนเองด้วย NumPy · library ใช้ได้เฉพาะเป็น reference เพื่อยืนยันผล
- ห้ามแตะ Test set จนถึงขั้น finalize
- ห้ามเคลมว่าเสร็จโดยไม่ได้รันคำสั่งและแสดง output จริง
- ใช้ `.\.venv\Scripts\python.exe` เสมอ (Python 3.14.8)
