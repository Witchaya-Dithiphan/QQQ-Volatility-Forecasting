# เอกสารประกอบโครงงาน

หน้านี้เป็นสารบัญของโฟลเดอร์ `docs/` ภาพรวมของโครงงาน ผลโดยสรุป และวิธีติดตั้งอยู่ใน [`README.md`](../README.md) ที่ root ของ repo ส่วนรายงานฉบับสมบูรณ์อยู่ที่ [`deliverables/report.pdf`](../deliverables/report.pdf)

## ควรเริ่มอ่านจากที่ใด

| ผู้อ่าน | เอกสารที่แนะนำ |
| --- | --- |
| ผู้ที่ยังไม่คุ้นกับ machine learning และต้องการเข้าใจงานทั้งหมด | [`guides/PROJECT_GUIDE_TH.md`](guides/PROJECT_GUIDE_TH.md) ซึ่งอธิบายตั้งแต่พื้นฐานและมีอภิธานศัพท์ |
| ผู้ตรวจที่ต้องการดูผล | notebook [`03`](../notebooks/03_regression.ipynb), [`04`](../notebooks/04_classification.ipynb), [`05`](../notebooks/05_results_comparison.ipynb) และ [`06`](../notebooks/06_spike_analysis.ipynb) ตามลำดับ และตารางใต้ `outputs/modeling/report/` |
| ผู้ที่ต้องการทราบว่าเกณฑ์ spike คืออะไรและเลือกมาอย่างไร | notebook [`06_spike_analysis.ipynb`](../notebooks/06_spike_analysis.ipynb) |
| ผู้ที่จะรันโค้ดเอง | [`guides/RUNBOOK.md`](guides/RUNBOOK.md) |
| ผู้ที่ต้องการทราบว่าระบบและโมเดลออกแบบอย่างไร | [`design/ARCHITECTURE.md`](design/ARCHITECTURE.md) แล้วต่อด้วย [`design/MODEL_CARDS.md`](design/MODEL_CARDS.md) |
| ผู้ที่ต้องการทราบว่าข้อมูลเชื่อถือได้เพียงใด | [`audit/DATA_PIPELINE_AUDIT.md`](audit/DATA_PIPELINE_AUDIT.md) |

## เนื้อหาในแต่ละโฟลเดอร์

### `guides/` คู่มือการใช้งานและคำอธิบาย

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`PROJECT_GUIDE_TH.md`](guides/PROJECT_GUIDE_TH.md) | คู่มือฉบับเดียวที่อธิบายงานตั้งแต่ต้นจนจบ ได้แก่ โจทย์ การเตรียมข้อมูล 8 ขั้น การทดลองเรื่อง spike โมเดล 19 ตัว วิธีอ่าน metric ผลบน Validation และ Test ข้อสรุป คำถามที่อาจถูกถาม สูตรพร้อมคำอ่าน และอภิธานศัพท์ |
| [`RUNBOOK.md`](guides/RUNBOOK.md) | การติดตั้ง environment การรัน data pipeline การฝึกโมเดล การสรุปผล การ finalize และการแก้ปัญหาที่พบบ่อย |

### `design/` การออกแบบ

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`ARCHITECTURE.md`](design/ARCHITECTURE.md) | โครงสร้างของ `src/ml/` และบันทึกการตัดสินใจด้านการออกแบบพร้อมเหตุผล (ADR) รวมถึงเกณฑ์การเทียบผลกับ library (ADR-002) |
| [`MODEL_CARDS.md`](design/MODEL_CARDS.md) | รายละเอียดของโมเดลทั้ง 19 ตัว ได้แก่ สมการ ช่วงค่าที่ลอง library ที่ใช้เทียบ และข้อควรระวังในการเทียบผล |

### `audit/` ผลการตรวจสอบชั้นข้อมูล (11 ตุลาคม 2026)

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`DATA_PIPELINE_AUDIT.md`](audit/DATA_PIPELINE_AUDIT.md) | ข้อสรุปของการตรวจ ประเด็นที่พบพร้อมหลักฐาน และสิ่งที่ต้องระบุในรายงาน |
| [`DATA_FLOW.md`](audit/DATA_FLOW.md) | แผนภาพการไหลของข้อมูลตั้งแต่ข้อมูลดิบ การแบ่งชุด จนถึงชุดข้อมูลสำหรับการทดลอง spike พร้อมจำนวนแถวจริงของแต่ละขั้น |
| [`PATH_AUDIT.md`](audit/PATH_AUDIT.md) | path ทั้งหมดที่ใช้ รูปแบบการขึ้นบรรทัดใหม่ และ checksum |
| [`FILE_INVENTORY.md`](audit/FILE_INVENTORY.md) | รายการไฟล์ทั้งหมดพร้อมผลตรวจรายไฟล์ |

### `models/` บันทึกการพัฒนาโมเดล

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`models/PROGRESS_INDEX.md`](models/PROGRESS_INDEX.md) | ดัชนีสถานะของโมเดลทั้ง 19 ตัว พร้อมผลการเทียบกับ library |
| [`models/<name>/MODEL_PROGRESS.md`](models/) | บันทึกของแต่ละโมเดล ได้แก่ ผลจริง ผลการเทียบกับ library ปัญหาที่พบ และวิธีแก้ |
| [`models/README.md`](models/README.md) | รูปแบบและหลักการเขียนบันทึกของแต่ละโมเดล |
| [`models/perceptron/KERNEL_EXPERIMENT.md`](models/perceptron/KERNEL_EXPERIMENT.md) | การทดลองเพิ่มเติมหลังส่งงาน: RBF Kernel Perceptron |
| [`models/transfer_learning/EXPERIMENT.md`](models/transfer_learning/EXPERIMENT.md) | การทดลองเพิ่มเติมหลังนำเสนอ: pretrain Transformer และ GRU บน ETF 10 ตัว แล้วปรับเข้ากับ QQQ |
| [`models/log_har/EXPERIMENT.md`](models/log_har/EXPERIMENT.md) | การทดลองเพิ่มเติมหลังนำเสนอ: Log-HAR และ pooled-prior transfer สำหรับงาน regression |

### `report/` ต้นฉบับของรายงาน

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`REPORT_DRAFT.md`](report/REPORT_DRAFT.md) | ต้นฉบับที่ใช้สร้าง `deliverables/report.docx` และ `report.pdf` |
| [`TRACEABILITY.md`](report/TRACEABILITY.md) | ตารางที่จับคู่ข้อกำหนดของโจทย์กับหัวข้อในรายงาน สไลด์ และไฟล์หลักฐาน |
| [`README.md`](report/README.md) | วิธีสร้างรายงานจากต้นฉบับ |

### `archive/` เอกสารฉบับเดิม

| ไฟล์ | เนื้อหา |
| --- | --- |
| [`REPORT_DRAFT_previous_format.md`](archive/REPORT_DRAFT_previous_format.md) | ต้นฉบับรายงานรูปแบบเดิม ก่อนปรับเป็นรูปแบบรายงานวิชาการ ไฟล์ที่สร้างจากต้นฉบับนี้อยู่ที่ `deliverables/previous_version/` |
