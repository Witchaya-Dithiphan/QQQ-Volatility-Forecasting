# รายงานและสไลด์

| ไฟล์ | คืออะไร |
| --- | --- |
| [`REPORT_DRAFT.md`](REPORT_DRAFT.md) | **ต้นฉบับของรายงาน** (ภาษาไทย) ตารางผลถูกเติมจากไฟล์ผลโดยอัตโนมัติ |
| [`TRACEABILITY.md`](TRACEABILITY.md) | ตารางจับคู่เกณฑ์การให้คะแนนของโจทย์กับตำแหน่งในรายงานและไฟล์หลักฐาน |
| [`../../deliverables/report.docx`](../../deliverables/report.docx) | รายงานฉบับ Word (แก้ไขต่อได้) สร้างจาก `REPORT_DRAFT.md` |
| [`../../deliverables/report.pdf`](../../deliverables/report.pdf) | รายงานฉบับ PDF ที่ส่งออกจากไฟล์ Word |
| [`../../deliverables/slides.pptx`](../../deliverables/slides.pptx) | สไลด์นำเสนอ 22 แผ่น พร้อมบทพูดใน speaker notes |
| [`tools/`](tools/) | สคริปต์ที่ใช้สร้างไฟล์ Word |

## สิ่งที่ต้องทำเองก่อนส่ง

- **กรอกชื่อและรหัสนักศึกษา** ในหน้าปกของ `report.docx` และสไลด์แผ่นแรกของ `slides.pptx` (ตอนนี้เป็นข้อความในวงเล็บเหลี่ยม)
- ถ้าแก้เนื้อหาใน Word โดยตรง ให้ส่งออก PDF ใหม่ด้วย File → Save As → PDF

## แก้รายงานอย่างไร

มีสองทาง เลือกทางใดทางหนึ่ง

**ทางที่ 1: แก้ใน Word โดยตรง** เปิด `deliverables/report.docx` แล้วแก้ได้เลย เหมาะกับการแก้สำนวนหรือจัดหน้าเล็กน้อย ข้อเสียคือถ้าสร้างไฟล์ใหม่จากต้นฉบับ การแก้ใน Word จะหายไป

**ทางที่ 2: แก้ที่ต้นฉบับแล้วสร้างใหม่** เหมาะเมื่อตัวเลขเปลี่ยนหรือแก้เนื้อหามาก

```bat
rem 1. แก้ docs\report\REPORT_DRAFT.md (ห้ามแก้ตัวเลขในตารางระหว่าง BEGIN GENERATED กับ END GENERATED)

rem 2. เติมตารางจากไฟล์ผล
.\.venv\Scripts\python.exe -m src.ml.docs_sync

rem 3. สร้างไฟล์ Word (ต้องมี Node.js; ติดตั้ง library ครั้งแรกครั้งเดียว)
cd docs\report\tools
npm install docx
node build_docx.js ..\..\.. ..\..\..\deliverables\report.docx
cd ..\..\..

rem 4. ให้ Word อัปเดตสารบัญและส่งออก PDF (ต้องมี Microsoft Word; ใช้ path เต็ม)
powershell -File docs\report\tools\word_finish.ps1 -DocxPath "%CD%\deliverables\report.docx" -PdfPath "%CD%\deliverables\report.pdf"
```

`build_docx.js` รองรับ Markdown เฉพาะส่วนที่รายงานใช้: หัวข้อ, ย่อหน้า, รายการ, ตาราง, รูป, blockquote, ตัวหนา (`**...**`) และ code (`` `...` ``) ตารางที่กว้างจะถูกวางบนหน้าแนวนอนโดยอัตโนมัติ

## สไลด์

`slides.pptx` แก้ใน PowerPoint ได้โดยตรง ทุกแผ่นมีบทพูดภาษาไทยใน speaker notes รูปทุกรูปมาจาก `outputs/figures/` และตัวเลขในตารางมาจาก `outputs/modeling/report/`

## ตรวจว่าเอกสารยังตรงกับไฟล์ผล

```bat
.\.venv\Scripts\python.exe -m pytest tests\test_docs_sync.py tests\test_docs_links.py -q
```

test จะ fail ถ้าตารางใน `REPORT_DRAFT.md` ไม่ตรงกับไฟล์ผล ถ้าลิงก์ในเอกสารขาด หรือถ้าไฟล์หลักฐานใน `TRACEABILITY.md` ไม่มีอยู่จริง
