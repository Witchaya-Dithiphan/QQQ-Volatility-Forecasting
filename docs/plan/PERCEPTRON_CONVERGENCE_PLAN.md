# แผนตรวจและแก้การไม่ลู่เข้าของ Perceptron

Branch: `fix/perceptron-convergence`

## เป้าหมายและเกณฑ์ตัดสิน

หาสาเหตุที่ Perceptron จบที่ `max_epochs=1000` บนข้อมูล QQQ ทั้ง `with_spike` และ `non_spike` แล้วแก้ตามสาเหตุ พร้อมหลักฐานที่อธิบายให้อาจารย์ตรวจซ้ำได้ คำว่า **ลู่เข้าแบบ Perceptron ดั้งเดิม** ในแผนนี้หมายถึงมีหนึ่ง epoch ที่ไม่มีการแก้น้ำหนักเลย ไม่ใช้การครบจำนวนรอบหรือ loss ที่ดูนิ่งเป็นหลักฐานแทน

ผลเดิมที่ต้องเก็บไว้เทียบ: validation ROC-AUC เท่ากับ 0.2993 / 0.3383 และ test ROC-AUC เท่ากับ 0.2749 / 0.2626 สำหรับ `with_spike` / `non_spike` ตามลำดับ ทั้งสองแบบใช้ครบ 1,000 epochs ส่วนผล test เป็นผลที่เปิดแล้ว ห้ามนำมาใช้เลือกรูปแบบการแก้หรือ hyperparameter

## ขั้นตอน

1. **วัดอาการบนชุดฝึกเดิม** — ใช้ `src.ml.core.data.load()` และ `Standardizer` แบบเดียวกับ trainer โดย fit ตัวแปลงกับ train เท่านั้น บันทึกจำนวนการอัปเดตในแต่ละ epoch, training mistake rate และ criterion สำหรับทั้งสอง variants ในไฟล์วิเคราะห์แยกจาก artifact ที่ล็อกไว้ ตรวจด้วยว่าค่า `history_` ปัจจุบันบอกได้เพียง criterion ไม่ได้บอกจำนวนการอัปเดต
2. **ตรวจการแบ่งเชิงเส้น** — ใช้ `scipy.optimize.linprog` ซึ่งติดตั้งอยู่แล้ว ตรวจระบบอสมการ `y_i (w·x_i + b) >= 1` บน train ที่ standardize โดย `y_i ∈ {-1,+1}` และให้น้ำหนักกับ bias เป็นตัวแปรอิสระ หาก solver บอกว่าทำได้ ให้ตรวจ margin ขั้นต่ำของคำตอบอีกครั้งตาม tolerance เชิงตัวเลข หากผล solver ไม่ชัดเจน ให้รายงานว่าไม่สรุป แทนการเดาจากจำนวน epochs
3. **เลือกทางแก้จากผลข้อ 2**
   - **แบ่งได้:** ทดสอบเพิ่ม `max_epochs` แบบจำกัดขอบเขตและบันทึก epoch ที่ไม่มีการอัปเดต หากยังไม่หยุด ให้รายงานข้อจำกัดด้านเวลา/จำนวนรอบตามจริง ก่อนตัดสินใจเปลี่ยนการสเกลหรือลำดับข้อมูล
   - **แบ่งไม่ได้:** ระบุชัดว่า Perceptron ดั้งเดิมไม่สามารถลู่เข้าแบบศูนย์การอัปเดตบน train ชุดนี้ได้ หากโจทย์ยังต้องใช้ Perceptron ให้ทดลอง *pocket/averaged Perceptron* เพื่อให้คำตอบสุดท้ายเสถียรกว่าเดิม โดยใช้ validation เลือกวิธี และเรียกการหยุดของวิธีใหม่นี้ว่าเกณฑ์หยุดหรือเสถียรภาพ **ไม่เรียกว่า** การลู่เข้าของ Perceptron ดั้งเดิม หากอาจารย์ต้องการ optimizer ที่ลู่เข้าจริงบนข้อมูลทับซ้อน ให้เสนอเปลี่ยนเป็นโมเดลที่มี objective ชัดเจน เช่น logistic/SLP พร้อมระบุว่าเป็นคนละอัลกอริทึม
4. **แก้เฉพาะจุดที่ผลวิเคราะห์รองรับ** — จุดที่อาจต้องแก้คือ `src/ml/classification/perceptron.py`, `tests/ml/classification/test_perceptron.py` และ `configs/modeling.json`; ถ้าเพิ่มรูปแบบโมเดลใหม่ ให้ตรวจ `src/ml/registry.py` และ caller ใน `src/ml/core/trainer.py` ด้วย คงการโหลด/บันทึกโมเดลและการตรวจ parity ที่เกี่ยวข้องไว้ และแก้เอกสาร `docs/models/perceptron/MODEL_PROGRESS.md`, `docs/plan/MODEL_CARDS.md` กับส่วนรายงานที่กล่าวว่า "ข้อมูลไม่ linearly separable" ตามผลพิสูจน์จริง
5. **ตรวจผลก่อนสรุป** — รัน unit/parity/save-load tests, ฝึกทั้งสอง variants ไปยัง `--output-root` แยกจาก `outputs/modeling` เดิม แล้วเทียบจำนวนการอัปเดตต่อ epoch, AP, ROC-AUC, recall, precision บน validation กับผลเดิม ถ้าวิธีใหม่ไม่ดีขึ้นหรือยังไม่ลู่เข้า ให้รายงานตามจริง ไม่เลือกจาก test และไม่เขียนทับ selection lock, finalize receipt หรือผล test เดิม การสร้างผล test รอบใหม่ต้องมีเหตุผลและเปิดเผยว่าเป็นการประเมินซ้ำ

## สิ่งที่ถือว่าเสร็จ

- มีผลตรวจการแบ่งเชิงเส้นและตัวเลขจำนวนการอัปเดตต่อ epoch ของทั้งสอง variants ที่ทำซ้ำได้
- ข้อสรุปใช้คำว่า "ลู่เข้า" เฉพาะเมื่อเข้าเกณฑ์ที่นิยามไว้ พร้อมระบุข้อจำกัดหาก train แบ่งไม่ได้
- การเปลี่ยนโค้ด (ถ้าจำเป็น) ผ่าน tests ที่เกี่ยวข้อง และผล validation/เอกสารสอดคล้องกับ artifact ใหม่

ยังไม่เพิ่ม feature, dependency หรือแก้ผล test ก่อนรู้ผลการตรวจในขั้น 1–2

## ผลการดำเนินการ

- ขั้น 1–2: เสร็จ — LP รายงาน `infeasible` บน train ทั้งสอง variants; จำนวนการอัปเดตใน epoch สุดท้ายคือ 274 และ 209 ตามลำดับ ดูหลักฐานและวิธีตรวจซ้ำใน [บันทึก Perceptron](../models/perceptron/MODEL_PROGRESS.md)
- ขั้น 3: ทดลอง pocket และค่าเฉลี่ยน้ำหนักบน validation แล้ว ทั้งสองวิธีไม่ทำให้ Perceptron ดั้งเดิมลู่เข้า และไม่ให้ผลจัดอันดับที่คุ้มกับการเปลี่ยน pipeline จึงคงโมเดลเดิมเพื่อการเปรียบเทียบอย่างตรงไปตรงมา
- ขั้น 4–5: เพิ่ม `history_["updates"]` และ tests ตรวจการหยุดกับความเป็นไปได้ของเส้นแบ่ง ฝึกซ้ำไปยัง output root แยกทั้งสอง variants โดย reload ผ่านและ validation metrics เท่าเดิม ไม่เปิด Test ซ้ำ

**ข้อสรุป:** ไม่สามารถทำให้ Perceptron ดั้งเดิมลู่เข้าแบบศูนย์การอัปเดตบน train ชุดเดิมได้ การหยุดหลังครบ 1,000 epochs เป็นเพียงการหยุดตามเพดานรอบ หากข้อกำหนดของงานต้องมีโมเดลที่ลู่เข้าบนข้อมูลเดิมจริง ต้องใช้ objective/อัลกอริทึมอื่นและระบุชื่อโมเดลให้ถูกต้อง

## การทดลองเสริม: RBF Kernel Perceptron

เพิ่ม `src/ml/classification/kernel_perceptron.py` เป็น Perceptron ที่ยังอัปเดตเมื่อทายผิด แต่ใช้ RBF kernel เพื่อสร้างเส้นแบ่งที่ไม่เป็นเส้นตรงในฟีเจอร์เดิม รันแยกด้วย `python -m src.ml.kernel_perceptron_experiment` ไม่เพิ่มโมเดลใน registry ของ 19 โมเดลเดิม ไม่เปลี่ยนข้อมูลหลักหรือ Test และบันทึกผลไว้ใน `outputs/experiments/kernel_perceptron/summary.json` กับ artifact แยกตาม variant

ทดลอง `gamma ∈ {0.5, 0.75, 1.0, 1.5, 2.0}` กำหนดเพดาน 1,000 epochs ทุก candidate ลู่เข้าแบบหนึ่งรอบไม่มีการอัปเดต แล้วเลือกตัวที่มี validation average precision สูงสุดในแต่ละ variant:

| variant | gamma ที่เลือก | epoch ที่หยุด | Train ผิด | Validation AP | Validation ROC-AUC | Validation recall | Validation precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `with_spike` | 1.0 | 139 | 0 | 0.1732 | 0.6835 | 0.3235 | 0.1930 |
| `non_spike` | 0.75 | 142 | 0 | 0.1460 | 0.6533 | 0.0588 | 0.0870 |

เทียบกับ Perceptron เดิมบน validation: AP = 0.0648 / 0.0742 และ ROC-AUC = 0.2993 / 0.3383 (`with_spike` / `non_spike`) ผลใหม่จัดอันดับดีขึ้น แต่ recall ของ `non_spike` ที่ threshold 0 ลดจาก 0.1176 เป็น 0.0588 จึงไม่สรุปว่าดีกว่าทุกด้าน การลู่เข้าบน Train หมายถึงทาย Train ถูกทุกแถวเท่านั้น อาจเป็นการจำตัวอย่างและไม่รับประกันผลบนข้อมูลใหม่

การทดลองนี้เป็น **โมเดลเสริม** ไม่ใช่การแก้ Perceptron ดั้งเดิมหรือการแทน SLP/MLP ผล Test เดิมและ selection lock ของ 19 โมเดลยังคงเดิม

### ผล Test ของการทดลองเสริม (ประเมินหลังส่งงาน)

หลังเลือก gamma จาก Validation แล้ว โหลดโมเดลที่บันทึกไว้และประเมิน Test โดยไม่ refit ผล `with_spike` ได้ AP 0.4416, ROC-AUC 0.7826, recall 0.5890; `non_spike` ได้ AP 0.3728, ROC-AUC 0.7469, recall 0.1096 รายละเอียด metric ทั้งหมด, confusion matrix และข้อจำกัดอยู่ใน [บันทึกการทดลอง Kernel Perceptron](../models/perceptron/KERNEL_EXPERIMENT.md) การเปิด Test ครั้งนี้มีใบรับรองแยก `outputs/experiments/kernel_perceptron/test_receipt.json` และไม่รวมในผล Test ที่ล็อกไว้ก่อนส่งงาน

การตรวจเทียบให้ครบขั้นตอนของโมเดลหลักดำเนินการแล้วด้วย `python -m src.ml.kernel_perceptron_analysis`: walk-forward 4 fold, block-permutation importance, moving-block bootstrap 2,000 รอบ, ผล Test แยกช่วง spike, การตรวจคะแนนด้วย RBF kernel ของ scikit-learn และ leaderboard เสริมพร้อมช่วงความเชื่อมั่น ผลทั้งหมดอยู่ใต้ `outputs/experiments/kernel_perceptron/report/` และแสดงใน notebook 04/05 โดยไม่เพิ่มเข้า registry หรือแก้ selection lock ของ 19 โมเดลเดิม รายละเอียดและข้อจำกัดอยู่ในบันทึกการทดลองข้างต้น
